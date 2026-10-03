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
