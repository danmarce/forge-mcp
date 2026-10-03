"""Runtime settings, all from environment variables (the only secret is the optional HTTP bearer)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# SDXL aspect buckets we allow (VRAM-safe envelope). ~1 MP each; anything bigger risks thrashing the
# GPU that a co-resident LLM may also be using. Named shortcuts map here; raw W/H is clamped to this set.
SHOTS: dict[str, tuple[int, int]] = {
    "portrait": (832, 1216),      # head-and-shoulders bias — ~95% of the creative work
    "full-figure": (832, 1216),   # same bucket, framed wide in the prompt
    "establishing": (1216, 832),  # landscape / scene
    "square": (1024, 1024),
}
ALLOWED_WH: set[tuple[int, int]] = {(832, 1216), (1216, 832), (1024, 1024)}
MAX_PIXELS = 1216 * 1216  # hard ceiling; reject anything larger

# Turbo defaults — Turbo checkpoints need these (they break on the usual 30-step/CFG-7 defaults).
DEFAULT_MODEL = "DreamShaperXL_Turbo_SFW"
DEFAULT_STEPS = 8
DEFAULT_CFG = 2.0
DEFAULT_SAMPLER = "DPM++ SDE"
DEFAULT_SCHEDULER = "Karras"
DEFAULT_SHOT = "portrait"

# Known checkpoints (discovery also queries Forge live; this is the friendly allow-list / hints).
KNOWN_MODELS = ("DreamShaperXL_Turbo_SFW", "DreamShaperXL_Turbo", "Juggernaut XL", "DreamShaper_8")


@dataclass(frozen=True)
class Settings:
    # The Forge / A1111 API base (the --api server). NOT the bare UI port unless --api is on it.
    forge_url: str = field(default_factory=lambda: os.environ.get("FORGE_URL", "http://127.0.0.1:7860").rstrip("/"))
    # Generous server->Forge timeout: covers a cold model-load + a Turbo gen. Stay under the MCP client's own timeout.
    gen_timeout: float = field(default_factory=lambda: float(os.environ.get("FORGE_GEN_TIMEOUT", "180")))
    # Inline preview: a full 1024^2 PNG base64 (~2 MB) strains some MCP clients -> return a modest JPEG preview.
    preview_max_px: int = field(default_factory=lambda: int(os.environ.get("FORGE_PREVIEW_MAX_PX", "768")))
    preview_quality: int = field(default_factory=lambda: int(os.environ.get("FORGE_PREVIEW_QUALITY", "82")))
    # Full-res keeper-save: the server writes the PNG here and serves it at /img/<name> for out-of-band download
    # (base64 in the result would flood the model's context). The CONSUMING REPO is still the real home.
    out_dir: str = field(default_factory=lambda: os.environ.get("FORGE_OUT_DIR", "out"))
    # The base URL clients use to reach THIS server (e.g. http://yuki:8646, or a ZeroTier IP for cross-site).
    # Required for include_full=True to return a downloadable link. Must be reachable from the saving machine.
    public_url: str = field(default_factory=lambda: os.environ.get("FORGE_PUBLIC_URL", "").rstrip("/"))
    # How long a served full-res link stays valid (TTL GC, not delete-on-first-GET — a dropped download retries).
    img_ttl: int = field(default_factory=lambda: int(os.environ.get("FORGE_IMG_TTL", "600")))
    default_model: str = field(default_factory=lambda: os.environ.get("FORGE_DEFAULT_MODEL", DEFAULT_MODEL))
    # Bearer token required on streamable-http when set (sent as an Authorization header).
    http_token: str | None = field(default_factory=lambda: os.environ.get("FORGE_MCP_TOKEN") or None)
