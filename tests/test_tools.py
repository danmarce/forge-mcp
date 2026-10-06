"""Tool-level: call each MCP tool by name through the server, against a fake Forge (no GPU, no network).

Pins what the helper tests can't: the declared hints, the status contract as a caller actually sees it, the
NSFW floor reaching the payload Forge receives, and edit_image's init-by-reference path.
"""

import asyncio
import base64
import io
import json

import httpx
import pytest
from PIL import Image

from forge_mcp import forge as forge_mod
from forge_mcp.config import Settings
from forge_mcp.server import _store_upload, build_server


def _png_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (64, 96), (120, 140, 160)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class FakeForge:
    """Answers the handful of A1111 endpoints the server uses; records every request."""

    def __init__(self, down: bool = False) -> None:
        self.down = down
        self.calls: list[tuple[str, str, dict | None]] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content) if req.content else None
        self.calls.append((req.method, req.url.path, body))
        if self.down:
            raise httpx.ConnectError("connection refused", request=req)
        path = req.url.path
        if path == "/sdapi/v1/options":
            return httpx.Response(200, json={"sd_model_checkpoint": "DreamShaperXL_Turbo_SFW [abc123]"})
        if path == "/sdapi/v1/sd-models":
            return httpx.Response(200, json=[{"model_name": "DreamShaperXL_Turbo_SFW"}])
        if path == "/sdapi/v1/loras":
            return httpx.Response(200, json=[{"name": "detail_tweaker"}])
        if path in ("/sdapi/v1/txt2img", "/sdapi/v1/img2img"):
            return httpx.Response(200, json={"images": [_png_b64()], "info": json.dumps({"seed": 4242})})
        return httpx.Response(200)  # refresh-* endpoints: empty body

    def posted(self, path: str) -> dict:
        return next(b for m, p, b in self.calls if m == "POST" and p == path)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_OUT_DIR", str(tmp_path))
    monkeypatch.setenv("FORGE_PUBLIC_URL", "http://gpu-host:8000")
    fake = FakeForge()
    real = httpx.AsyncClient
    monkeypatch.setattr(forge_mod.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(fake), **kw))
    return fake, Settings()


def _call(settings: Settings, name: str, args: dict) -> dict:
    res = asyncio.run(build_server(settings).call_tool(name, args))
    texts = [c for c in res.content if c.type == "text"]
    return json.loads(texts[-1].text)


def test_every_tool_declares_title_and_all_four_hints():
    tools = asyncio.run(build_server(Settings()).list_tools())
    assert {t.name for t in tools} == {"generate_image", "edit_image", "list_models"}
    by = {}
    for t in tools:
        wire = t.annotations.model_dump(by_alias=True)  # the camelCase JSON a client/directory actually reads
        for hint in ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"):
            assert isinstance(wire.get(hint), bool), f"{t.name}.{hint} unset"
        assert t.title and wire.get("title") == t.title, f"{t.name} missing title"
        by[t.name] = wire
    assert by["list_models"]["readOnlyHint"] and not by["generate_image"]["readOnlyHint"]
    assert not any(w["destructiveHint"] for w in by.values())


def test_generate_image_ok_echoes_seed_and_sends_prompt_verbatim(env):
    fake, s = env
    prompt = "a lighthouse at dusk, (warm light:1.2), <lora:detail_tweaker:0.6>"
    d = _call(s, "generate_image", {"prompt": prompt, "seed": -1})
    assert d["status"] == "ok"
    assert d["params"]["seed"] == 4242                     # the REAL seed, not -1
    assert d["full_res_url"].startswith("http://gpu-host:8000/img/")
    sent = fake.posted("/sdapi/v1/txt2img")
    assert sent["prompt"] == prompt                         # verbatim — weights + LoRA tag untouched
    assert "nsfw" in sent["negative_prompt"]                # the floor reaches Forge
    assert not any(p == "/sdapi/v1/options" and m == "POST" for m, p, _ in fake.calls)  # already loaded: no switch


def test_generate_image_nsfw_refused_without_touching_forge(env):
    fake, s = env
    d = _call(s, "generate_image", {"prompt": "score_9, rating_explicit"})
    assert d["status"] == "refused" and d["matched"]
    assert fake.calls == []


def test_edit_image_by_ref_sends_the_stored_image(env):
    fake, s = env
    ref = _store_upload(s, base64.b64decode(_png_b64()))
    d = _call(s, "edit_image", {"prompt": "same woman, red scarf", "init_image_ref": ref, "seed": 777})
    assert d["status"] == "ok"
    sent = fake.posted("/sdapi/v1/img2img")
    assert base64.b64decode(sent["init_images"][0]).startswith(b"\x89PNG")
    assert sent["denoising_strength"] == 0.45


@pytest.mark.parametrize("args", [
    {},                                                    # neither
    {"init_image_ref": "a.png", "init_image": "AAAA"},     # both
    {"init_image_ref": "expired-or-never-was.png"},        # unknown ref
    {"init_image_ref": "../../etc/passwd"},                # traversal attempt
])
def test_edit_image_bad_init_is_refused_not_raised(env, args):
    fake, s = env
    d = _call(s, "edit_image", {"prompt": "a portrait", **args})
    assert d["status"] == "refused" and d["kind"] == "input"
    assert not any(p == "/sdapi/v1/img2img" for _, p, _ in fake.calls)


def test_list_models_refresh_and_loras(env):
    fake, s = env
    d = _call(s, "list_models", {"refresh": True})
    assert d["status"] == "ok" and d["refreshed"] is True
    assert d["loaded"].startswith("DreamShaperXL_Turbo_SFW") and d["loras"] == ["detail_tweaker"]
    posts = [p for m, p, _ in fake.calls if m == "POST"]
    assert posts == ["/sdapi/v1/refresh-checkpoints", "/sdapi/v1/refresh-loras"]


def test_forge_down_is_error_not_refused(env):
    fake, s = env
    fake.down = True
    for name, args in [("generate_image", {"prompt": "a cat"}), ("list_models", {})]:
        d = _call(s, name, args)
        assert d["status"] == "error" and d["kind"] == "forge", name
