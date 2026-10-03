"""Server-level: it builds, and the status contract (ok / refused / error) has the right shape."""

import json

from forge_mcp.config import Settings
from forge_mcp.server import _error, _refused, build_server


def test_build():
    assert build_server(Settings()).name == "forge"


def test_refused_is_a_normal_structured_response():
    d = json.loads(_refused("prompt matched the SFW policy term 'nude'", matched="nude", advice="rephrase")[0].text)
    assert d["status"] == "refused"
    assert d["matched"] == "nude"
    assert "advice" in d


def test_error_is_distinct_from_refused():
    d = json.loads(_error("gpu_oom", "CUDA out of memory")[0].text)
    assert d["status"] == "error"
    assert d["kind"] == "gpu_oom"
    assert d["detail"] == "CUDA out of memory"


def test_fullres_save_purge_and_no_traversal(tmp_path, monkeypatch):
    import base64
    import os
    import time

    monkeypatch.setenv("FORGE_OUT_DIR", str(tmp_path))
    monkeypatch.setenv("FORGE_IMG_TTL", "600")
    from forge_mcp.config import Settings
    from forge_mcp.server import IMG_NAME_RE, _purge_expired, _save_fullres

    s = Settings()
    png_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"x" * 200).decode()
    name = _save_fullres(s, png_b64)
    assert IMG_NAME_RE.fullmatch(name)                      # capability-token filename
    p = os.path.join(str(tmp_path), name)
    assert os.path.isfile(p)

    # path-traversal / nested names are rejected by the serve guard
    assert not IMG_NAME_RE.fullmatch("../secret.png")
    assert not IMG_NAME_RE.fullmatch("a/b.png")

    # TTL GC removes an aged file
    old = time.time() - 700
    os.utime(p, (old, old))
    _purge_expired(s)
    assert not os.path.isfile(p)


def test_refresh_hits_both_rescans_then_lists_loras():
    import asyncio

    import httpx

    from forge_mcp.forge import ForgeClient

    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.method, req.url.path))
        if req.url.path == "/sdapi/v1/loras":
            return httpx.Response(200, json=[{"name": "zeta_style", "alias": "z"}, {"name": "alpha_detail"}])
        return httpx.Response(200, json=None)

    async def go():
        fc = ForgeClient(Settings())
        fc._client = httpx.AsyncClient(base_url="http://forge", transport=httpx.MockTransport(handler))
        await fc.refresh()
        return await fc.list_loras()

    assert asyncio.run(go()) == ["alpha_detail", "zeta_style"]
    assert seen[:2] == [("POST", "/sdapi/v1/refresh-checkpoints"), ("POST", "/sdapi/v1/refresh-loras")]


def _png_bytes(size=(64, 96)) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, (200, 180, 160)).save(buf, format="PNG")
    return buf.getvalue()


def test_upload_roundtrip_and_ref_resolution(tmp_path, monkeypatch):
    import base64

    import pytest

    from forge_mcp.server import _resolve_init_ref, _store_upload

    monkeypatch.setenv("FORGE_OUT_DIR", str(tmp_path))
    s = Settings()
    name = _store_upload(s, _png_bytes())
    # bare name and the full full_res_url form both resolve to the same image
    b64 = _resolve_init_ref(s, name)
    assert _resolve_init_ref(s, f"http://yuki:8646/img/{name}") == b64
    assert base64.b64decode(b64).startswith(b"\x89PNG")
    for bad in ["../../etc/passwd", r"..\secret.png", "nope.png", "x/../../a.png"]:
        with pytest.raises(ValueError):
            _resolve_init_ref(s, bad)
    with pytest.raises(ValueError):
        _store_upload(s, b"definitely not an image")


def test_upload_endpoint_is_bearer_gated(tmp_path, monkeypatch):
    import asyncio

    import httpx

    from forge_mcp.server import BearerAuth

    monkeypatch.setenv("FORGE_OUT_DIR", str(tmp_path))
    monkeypatch.setenv("FORGE_MCP_TOKEN", "t0k")
    monkeypatch.setenv("FORGE_UPLOAD_MAX_BYTES", "100000")

    async def inner(scope, receive, send):  # stands in for the MCP app; must never be reached by /upload
        raise AssertionError("upload leaked through to the MCP app")

    async def go():
        app = BearerAuth(inner, Settings())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            auth = {"Authorization": "Bearer t0k"}
            no_auth = await c.post("/upload", content=_png_bytes())
            ok = await c.post("/upload", content=_png_bytes(), headers=auth)
            junk = await c.post("/upload", content=b"junk", headers=auth)
            big = await c.post("/upload", content=b"x" * 200_000, headers=auth)
            get = await c.get("/upload", headers=auth)
            served = await c.get(f"/img/{ok.json()['ref']}")  # the stored ref is also fetchable like a full-res
            return no_auth, ok, junk, big, get, served

    no_auth, ok, junk, big, get, served = asyncio.run(go())
    assert no_auth.status_code == 401
    assert ok.status_code == 200 and ok.json()["status"] == "ok" and ok.json()["ref"].endswith(".png")
    assert junk.status_code == 400 and junk.json()["status"] == "refused"
    assert big.status_code == 413
    assert get.status_code == 405
    assert served.status_code == 200 and served.content.startswith(b"\x89PNG")
