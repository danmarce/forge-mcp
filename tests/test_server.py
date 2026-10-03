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
