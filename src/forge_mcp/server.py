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
Pass the init image BY REFERENCE, never as base64 in context: upload the keeper out-of-band from the shell
(`curl -H "Authorization: Bearer $FORGE_MCP_TOKEN" --data-binary @keeper.png <server>/upload` -> {"ref": ...})
and call edit_image(init_image_ref=<ref>). A recent result's `full_res_url` also works as a ref, so you can
iterate on a render with no round-trip. Refs expire with the full-res TTL; re-upload if one lapses.

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
UPLOAD_MAX_PIXELS = 4096 * 4096  # an init image is resized to the bucket anyway; this just stops decompression bombs


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


def _store_upload(settings: Settings, data: bytes) -> str:
    """Validate an uploaded init image and store it under a capability name (same dir + TTL as full-res).
    Re-encoded as PNG through Pillow, so only a real decodable image ever lands on disk. ValueError if not."""
    if len(data) > settings.upload_max_bytes:
        raise ValueError(f"upload too large ({len(data)} bytes; max {settings.upload_max_bytes})")
    try:
        img = Image.open(io.BytesIO(data))
        if img.width * img.height > UPLOAD_MAX_PIXELS:
            raise ValueError(f"image too large ({img.width}x{img.height}; max {UPLOAD_MAX_PIXELS} pixels)")
        img.load()
    except ValueError:
        raise
    except Exception as e:  # Pillow raises a zoo of types for garbage / truncated input
        raise ValueError(f"not a decodable image: {e}") from e
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return _save_fullres(settings, base64.b64encode(buf.getvalue()).decode())


def _resolve_init_ref(settings: Settings, ref: str) -> str:
    """Turn an init_image_ref (an /upload ref, or a recent full_res_url / its name) into base64 for Forge.
    ValueError when it is malformed, missing, or expired — a caller-fixable input, not a fault."""
    name = ref.strip().rsplit("/", 1)[-1]  # accept the bare name or the whole http://host/img/<name> URL
    if not IMG_NAME_RE.fullmatch(name):  # capability tokens only — blocks path traversal
        raise ValueError(f"malformed init_image_ref {ref!r}")
    path = os.path.join(settings.out_dir, name)
    if not os.path.isfile(path) or time.time() - os.path.getmtime(path) > settings.img_ttl:
        raise ValueError(f"init_image_ref {name!r} not found or expired (refs live ~{settings.img_ttl // 60} min)")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


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
        init_image_ref: str | None = None,
        init_image: str | None = None,
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
            init_image_ref: the source image BY REFERENCE (preferred — keeps base64 out of your context). Either
                (a) the `ref` from uploading the keeper out-of-band, from the shell:
                    curl -H "Authorization: Bearer $FORGE_MCP_TOKEN" --data-binary @keeper.png <server>/upload
                or (b) a recent result's `full_res_url` (or its name) — iterate on a render with no round-trip.
                Refs expire with the full-res TTL (~10 min); re-upload if one lapses. Also load the keeper's
                sidecar params so this render builds on its recipe.
            init_image: the source image as raw base64 — only for small images; a full keeper floods context.
            denoising_strength: 0.3-0.45 = keep the subject, fix details; 0.5-0.65 = restyle. Default 0.45.
            (all other args: see generate_image.)
        """
        blocked = positive_is_blocked(prompt)
        if blocked:
            return _refused(f"prompt matched the SFW policy term {blocked!r}", matched=blocked,
                            advice="rephrase without that term — this tool is SFW-only (use the Forge UI directly for NSFW)")
        if bool(init_image_ref) == bool(init_image):
            return _refused("pass exactly one of init_image_ref (preferred) or init_image", kind="input")
        if init_image_ref:
            try:
                init_image = _resolve_init_ref(settings, init_image_ref)
            except ValueError as e:
                return _refused(str(e), kind="input",
                                advice="re-upload the keeper via POST /upload and pass the new ref")
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

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
              structured_output=False)
    async def list_models(refresh: bool = False) -> list[Any]:
        """List the checkpoints and LoRAs available on the Forge box, and which checkpoint is loaded.

        Args:
            refresh: rescan Forge's model + LoRA folders first — use after dropping new files on the box.
                Without it, Forge only sees files that were there at its last start/refresh.
        LoRAs are used in the prompt as `<lora:NAME:weight>` with a name from `loras`.
        """
        try:
            if refresh:
                await forge.refresh()
            models = await forge.list_models()
            loras = await forge.list_loras()
            current = await forge.current_model()
        except ForgeError as e:
            return _error("forge", str(e))
        return _text({"status": "ok", "refreshed": refresh, "loaded": current,
                      "available": models, "loras": loras})

    return mcp


# --- streamable-http bearer gate -------------------------

import hmac  # noqa: E402


class BearerAuth:
    """ASGI gate for streamable-http:
      * /healthz        — always open (Docker healthcheck).
      * /img/<name>     — open, serves a saved full-res PNG by its capability token (short-lived; no bearer so
                          the saving machine can `curl` it without the server secret). Path-traversal-safe.
      * POST /upload    — BEARER-gated (it writes): stores an init image for edit_image, returns {"ref": ...}.
                          The mirror of /img — the keeper travels out-of-band, never as base64 in context.
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
            if path == "/upload":
                await self._receive_upload(scope, receive, send)
                return
        await self.app(scope, receive, send)

    async def _receive_upload(self, scope, receive, send) -> None:
        if scope["method"] != "POST":
            await _respond(send, 405, b"POST the image bytes (curl --data-binary @file.png)")
            return
        cap = self.settings.upload_max_bytes
        body = bytearray()
        while True:
            msg = await receive()
            body += msg.get("body", b"")
            if len(body) > cap:  # stop reading early rather than buffer an unbounded body
                await _json(send, 413, {"status": "refused", "kind": "input",
                                        "reason": f"upload too large (max {cap} bytes)"})
                return
            if not msg.get("more_body"):
                break
        try:
            name = _store_upload(self.settings, bytes(body))
        except ValueError as e:
            await _json(send, 400, {"status": "refused", "kind": "input", "reason": str(e)})
            return
        await _json(send, 200, {"status": "ok", "ref": name, "ttl_seconds": self.settings.img_ttl,
                                "use": f'edit_image(init_image_ref="{name}", ...)'})

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


async def _json(send, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


async def _respond(send, status: int, body: bytes) -> None:
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"text/plain"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})
