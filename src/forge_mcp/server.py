"""MCP server: image generation over Forge, frictionless + deterministic + a good shared-GPU citizen.

Design contract:
  * Returns a viewable PREVIEW inline + the FULL-RES PNG (base64) so the CALLING repo saves it where it wants
    (designs/<char>/, scenes/<slug>/). The server owns no files, no character DB, no anchors — the repo does.
  * Reproducibility by default: the resolved seed + every param used come back in the result (which IS the
    sidecar the client writes next to the PNG). seed=-1 -> the real seed is echoed so a keeper gets locked.
  * NEVER silently rewrites the prompt. The exact positive/negative actually sent are returned.
  * NSFW is hard-blocked server-side, unconditionally (see presets.py) — not a matter of caller judgment.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import secrets
import time
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ImageContent, TextContent, ToolAnnotations
from PIL import Image

from .config import (DEFAULT_CFG, DEFAULT_SAMPLER, DEFAULT_SCHEDULER, DEFAULT_SHOT, DEFAULT_STEPS,
                     ALLOWED_WH, MAX_PIXELS, SHOTS, Settings)
from .forge import ForgeClient, ForgeError, ForgeOOM
from .presets import apply_style, expand_negative, positive_is_blocked

INSTRUCTIONS = """\
Local SDXL image generation on a GPU box that may ALSO be running an LLM — so image gen shares the GPU
and is kept on a VRAM-safe envelope (fixed ~1 MP aspect buckets, no hires-fix). Two tools: generate_image
(txt2img) and edit_image (img2img). Defaults are the known-good DreamShaper XL Turbo recipe, so a bare
"generate X" just works; everything is overridable.

Each result returns a small inline PREVIEW (to judge in-context) plus, in the text block, the exact params
used. With include_full=True you also get `full_res_url` — a short-lived link to the full-res PNG: download it
out-of-band (the result shows a ready `download` command), e.g. `curl -o designs/<name>/<file> <full_res_url>`,
and write the returned `params` beside it as a sidecar .json. The full-res is a LINK, not base64, so it never
floods context. include_full=False returns preview + params only. The resolved `seed` is always echoed: lock
it to reproduce a character ("same seed, change only age/colour = same soul, different vessel").

edit_image (img2img) is a HOLISTIC transform — it re-derives the whole frame, so use it for variants that
should re-settle (wardrobe, pose, lighting, hair) at denoise ~0.3-0.45 (keep subject) to ~0.5-0.65 (restyle).
It is NOT a single-feature scalpel (fixing just the eyes also moves skin/age) — do surgical fixes in a PSD.

Prompts are sent VERBATIM (attention weights like `(grey eyes:1.3)` pass through). The server never rewrites
them. NSFW is blocked server-side regardless of prompt or profile.

Every result's text block has a `status`: "ok" (image returned), "refused" (a POLICY decision — SFW block
or an out-of-envelope resolution; `reason`/`matched`/`advice` say how to adjust — rephrase, don't retry
unchanged), or "error" (a genuine FAULT — `kind` like "gpu_oom"/"forge" + `detail`; infra, not policy, so a
prompt change won't help). These are three distinct outcomes — branch on `status`, never on the prose.
"""

TOOL = ToolAnnotations(readOnlyHint=False, openWorldHint=False)


def _resolve_wh(shot: str | None, width: int | None, height: int | None) -> tuple[int, int]:
    if width and height:
        wh = (int(width), int(height))
        if wh not in ALLOWED_WH or wh[0] * wh[1] > MAX_PIXELS:
            raise ValueError(
                f"resolution {wh[0]}x{wh[1]} is outside the VRAM-safe buckets "
                f"{sorted(ALLOWED_WH)} — pick one, or use shot=portrait|establishing|square."
            )
        return wh
    return SHOTS.get(shot or DEFAULT_SHOT, SHOTS[DEFAULT_SHOT])


def _preview_jpeg(png_b64: str, max_px: int, quality: int) -> str:
    img = Image.open(io.BytesIO(base64.b64decode(png_b64))).convert("RGB")
    img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s[:n].rstrip("-")) or "image"


def _text(payload: dict[str, Any]) -> list[Any]:
    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]


def _refused(reason: str, **extra: Any) -> list[Any]:
    """A POLICY refusal — a normal, successful response (not an error). The caller should adjust the
    request (rephrase / pick a valid shot), not retry unchanged or treat it as a fault."""
    return _text({"status": "refused", "reason": reason, **extra})


def _error(kind: str, detail: str) -> list[Any]:
    """A genuine FAULT (Forge down, OOM, bad response) — infra, not policy. The caller may retry later
    or surface it; it is NOT something a prompt change fixes."""
    return _text({"status": "error", "kind": kind, "detail": detail})


IMG_NAME_RE = re.compile(r"[A-Za-z0-9_-]+\.png")  # capability-token filenames only (no path traversal)


def _purge_expired(settings: Settings) -> None:
    """Delete served full-res files older than the TTL (lazy GC on each save)."""
    now = time.time()
    try:
        for fn in os.listdir(settings.out_dir):
            p = os.path.join(settings.out_dir, fn)
            if fn.endswith(".png") and now - os.path.getmtime(p) > settings.img_ttl:
                try:
                    os.remove(p)
                except OSError:
                    pass
    except OSError:
        pass


def _save_fullres(settings: Settings, png_b64: str) -> str:
    """Write the full-res PNG under an unguessable capability name; return that filename (served at /img/<name>)."""
    os.makedirs(settings.out_dir, exist_ok=True)
    _purge_expired(settings)
    name = f"{secrets.token_urlsafe(12)}.png"
    with open(os.path.join(settings.out_dir, name), "wb") as f:
        f.write(base64.b64decode(png_b64))
    return name


def build_server(settings: Settings) -> MCPServer:
    mcp = MCPServer(name="forge", instructions=INSTRUCTIONS)
    forge = ForgeClient(settings)

    async def _run(endpoint: str, payload: dict[str, Any], model: str, resolved: dict[str, Any],
                   include_full: bool, subject: str) -> list[Any]:
        resp = await forge.generate(endpoint, payload, model)
        images = resp.get("images") or []
        if not images:
            raise ForgeError("Forge returned no image.")
        png_b64 = images[0]
        info = {}
        try:
            info = json.loads(resp.get("info") or "{}")
        except (ValueError, TypeError):
            pass
        seed = info.get("seed", resolved.get("seed"))
        params = {**resolved, "model": model, "seed": seed}  # the sidecar content
        fname = f"{_slug(subject)}-{seed}.png"
        out: dict[str, Any] = {
            "status": "ok",
            "source": "forge",
            "params": params,
            "suggested_filename": fname,
        }
        if include_full:
            # Full-res is served as a short-lived LINK, never base64 — base64 floods/truncates the context.
            name = _save_fullres(settings, png_b64)
            if settings.public_url:
                url = f"{settings.public_url}/img/{name}"
                out["full_res_url"] = url
                out["full_res_ttl_seconds"] = settings.img_ttl
                out["download"] = f"curl -o designs/<name>/{fname} {url}"
                out["note"] = ("download full_res_url to your repo (designs/<name>/) and write `params` beside "
                               f"it as a sidecar .json. Link expires in ~{settings.img_ttl // 60} min; re-run "
                               "the same seed to regenerate if it lapses.")
            else:
                out["full_res_file"] = name  # saved on the server; no public URL configured
                out["note"] = ("full-res saved on the server but FORGE_PUBLIC_URL is not set, so no download "
                               "link can be given. Set it (e.g. http://<host>:<port>) to enable keeper-save.")
        else:
            out["note"] = "preview only (include_full=False). Re-run the same seed with include_full=True for the full-res link."
        preview = _preview_jpeg(png_b64, settings.preview_max_px, settings.preview_quality)
        return [
            ImageContent(type="image", data=preview, mimeType="image/jpeg"),
            TextContent(type="text", text=json.dumps(out, ensure_ascii=False)),
        ]

    @mcp.tool(annotations=TOOL, structured_output=False)
    async def generate_image(
        prompt: str,
        shot: str = DEFAULT_SHOT,
        model: str | None = None,
        seed: int = -1,
        steps: int = DEFAULT_STEPS,
        cfg: float = DEFAULT_CFG,
        sampler: str = DEFAULT_SAMPLER,
        scheduler: str = DEFAULT_SCHEDULER,
        negative: str | None = None,
        negative_profile: str = "sfw-strict",
        style: str | None = None,
        width: int | None = None,
        height: int | None = None,
        include_full: bool = True,
    ) -> list[Any]:
        """Generate an image (txt2img) on the local SDXL box. Returns an inline preview + (in the text block)
        the exact params and, with include_full=True, a short-lived `full_res_url` to download the full-res PNG
        into your repo (a link, not base64 — base64 floods context).

        Args:
            prompt: the positive prompt, sent VERBATIM (attention weights like `(grey eyes:1.3)` work).
            shot: aspect/framing shortcut — `portrait` (832x1216, default, face-focus), `full-figure`,
                `establishing` (1216x832 scene), `square` (1024). Raw width/height override this.
            model: checkpoint (default DreamShaperXL_Turbo_SFW). Switched only if different from loaded.
            seed: -1 = random (the resolved seed is returned so you can lock it); set it to reproduce.
            steps: default 8 (Turbo). cfg: default 2 (Turbo). sampler/scheduler: DPM++ SDE / Karras (Turbo).
            negative: extra negative text (combined with the profile; the NSFW block is always appended).
            negative_profile: named profile(s), e.g. "sfw-strict" (default), "mature", "candid+sfw-strict".
            style: optional positive preset, e.g. "photoreal".
            width/height: raw size, must be a VRAM-safe bucket (832x1216, 1216x832, 1024x1024).
            include_full: default True = also return `full_res_url` (a short-lived download link to the full-res
                PNG + a ready `download` curl command); False = preview + params only, lighter for rapid
                iteration (re-run the echoed seed with include_full=True to get the link for a keeper).
        """
        blocked = positive_is_blocked(prompt)
        if blocked:
            return _refused(f"prompt matched the SFW policy term {blocked!r}", matched=blocked,
                            advice="rephrase without that term — this tool is SFW-only (use the Forge UI directly for NSFW)")
        try:
            w, h = _resolve_wh(shot, width, height)
        except ValueError as e:
            return _refused(str(e), kind="resolution")
        positive = apply_style(prompt, style)
        neg = expand_negative(negative_profile, negative)
        model = model or settings.default_model
        payload = {
            "prompt": positive, "negative_prompt": neg, "seed": seed, "steps": steps, "cfg_scale": cfg,
            "sampler_name": sampler, "scheduler": scheduler, "width": w, "height": h,
            "override_settings_restore_afterwards": True,
        }
        resolved = {"prompt": positive, "negative_prompt": neg, "seed": seed, "steps": steps, "cfg": cfg,
                    "sampler": sampler, "scheduler": scheduler, "width": w, "height": h, "shot": shot}
        try:
            return await _run("/sdapi/v1/txt2img", payload, model, resolved, include_full, subject=prompt)
        except ForgeOOM as e:
            return _error("gpu_oom", str(e))
        except ForgeError as e:
            return _error("forge", str(e))

    @mcp.tool(annotations=TOOL, structured_output=False)
    async def edit_image(
        prompt: str,
        init_image: str,
        denoising_strength: float = 0.45,
        shot: str = DEFAULT_SHOT,
        model: str | None = None,
        seed: int = -1,
        steps: int = DEFAULT_STEPS,
        cfg: float = DEFAULT_CFG,
        sampler: str = DEFAULT_SAMPLER,
        scheduler: str = DEFAULT_SCHEDULER,
        negative: str | None = None,
        negative_profile: str = "sfw-strict",
        style: str | None = None,
        width: int | None = None,
        height: int | None = None,
        include_full: bool = True,
    ) -> list[Any]:
        """Edit/vary an image (img2img). ⚠ HOLISTIC — re-derives the WHOLE frame from the init, so use it for
        variants that should re-settle (wardrobe, pose, lighting, hair-mass), NOT single-feature fixes (those
        belong in a PSD — pushing just the eyes also moves skin/age/expression). For character consistency,
        pass the character's canonical keeper as `init_image` and its seed/prompt, then change only the delta.

        Args:
            init_image: the source image as base64 (your client reads the keeper PNG from the repo and passes
                it; also load its sidecar params so this render builds on the keeper's recipe).
            denoising_strength: 0.3-0.45 = keep the subject, fix details; 0.5-0.65 = restyle. Default 0.45.
            (all other args: see generate_image.)
        """
        blocked = positive_is_blocked(prompt)
        if blocked:
            return _refused(f"prompt matched the SFW policy term {blocked!r}", matched=blocked,
                            advice="rephrase without that term — this tool is SFW-only (use the Forge UI directly for NSFW)")
        try:
            w, h = _resolve_wh(shot, width, height)
        except ValueError as e:
            return _refused(str(e), kind="resolution")
        positive = apply_style(prompt, style)
        neg = expand_negative(negative_profile, negative)
        model = model or settings.default_model
        payload = {
            "prompt": positive, "negative_prompt": neg, "init_images": [init_image],
            "denoising_strength": denoising_strength, "seed": seed, "steps": steps, "cfg_scale": cfg,
            "sampler_name": sampler, "scheduler": scheduler, "width": w, "height": h,
            "override_settings_restore_afterwards": True,
        }
        resolved = {"prompt": positive, "negative_prompt": neg, "seed": seed, "steps": steps, "cfg": cfg,
                    "sampler": sampler, "scheduler": scheduler, "width": w, "height": h, "shot": shot,
                    "denoising_strength": denoising_strength}
        try:
            return await _run("/sdapi/v1/img2img", payload, model, resolved, include_full, subject=prompt)
        except ForgeOOM as e:
            return _error("gpu_oom", str(e))
        except ForgeError as e:
            return _error("forge", str(e))

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False), structured_output=False)
    async def list_models() -> str:
        """List the image models (checkpoints) available on the Forge box, and which one is loaded."""
        models = await forge.list_models()
        current = await forge.current_model()
        return json.dumps({"loaded": current, "available": models}, ensure_ascii=False)

    return mcp


# --- streamable-http bearer gate -------------------------

import hmac  # noqa: E402


class BearerAuth:
    """ASGI gate for streamable-http:
      * /healthz        — always open (Docker healthcheck).
      * /img/<name>     — open, serves a saved full-res PNG by its capability token (short-lived; no bearer so
                          the saving machine can `curl` it without the server secret). Path-traversal-safe.
      * everything else — the MCP app, requiring `Authorization: Bearer <token>` when a token is set.
    """

    def __init__(self, app, settings: Settings) -> None:
        self.app = app
        self.settings = settings
        self.expected = f"Bearer {settings.http_token}".encode() if settings.http_token else None

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http":
            path = scope["path"]
            if path == "/healthz":
                await _respond(send, 200, b"ok")
                return
            if path.startswith("/img/"):
                await self._serve_image(path[len("/img/"):], send)
                return
            if self.expected is not None:
                got = dict(scope["headers"]).get(b"authorization", b"")
                if not hmac.compare_digest(got, self.expected):
                    await _respond(send, 401, b"unauthorized")
                    return
        await self.app(scope, receive, send)

    async def _serve_image(self, name: str, send) -> None:
        if not IMG_NAME_RE.fullmatch(name):  # capability tokens only — blocks path traversal
            await _respond(send, 404, b"not found")
            return
        path = os.path.join(self.settings.out_dir, name)
        if not os.path.isfile(path) or time.time() - os.path.getmtime(path) > self.settings.img_ttl:
            await _respond(send, 404, b"not found or expired")
            return
        with open(path, "rb") as f:
            data = f.read()
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"image/png"), (b"content-length", str(len(data)).encode())]})
        await send({"type": "http.response.body", "body": data})


async def _respond(send, status: int, body: bytes) -> None:
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"text/plain"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})
