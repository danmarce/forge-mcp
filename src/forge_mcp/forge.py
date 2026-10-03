"""Thin async client over the Forge / A1111 web API, with shared-GPU citizenship baked in.

Tenancy rules (this box may also run a co-resident LLM sharing the same GPU):
  * Serialize ALL GPU work behind one lock — never two generations, or a gen during a model switch, at once.
  * Don't reload the checkpoint needlessly — query the loaded model and switch ONLY when it differs
    (the reload is the VRAM spike that OOMs). We switch persistently (no restore-afterwards on the
    checkpoint) so the *next* call on the same model doesn't reload again.
  * Any per-request override_settings use restore_afterwards=True so nothing leaks to the next call.
  * Surface OOM as a clean, typed error — never a hang or an opaque 500.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from .config import Settings


class ForgeError(RuntimeError):
    """Forge is unreachable or returned an error."""


class ForgeOOM(ForgeError):
    """The GPU ran out of memory — the shared box was likely busy. Retriable later / use a smaller shot."""


def _looks_like_oom(text: str) -> bool:
    t = text.lower()
    return "out of memory" in t or "cuda oom" in t or "outofmemory" in t or "alloc" in t and "cuda" in t


class ForgeClient:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self._gpu = asyncio.Lock()  # the single-file queue for all GPU work
        self._client = httpx.AsyncClient(base_url=settings.forge_url, timeout=settings.gen_timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get(self, path: str) -> Any:
        try:
            r = await self._client.get(path)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise ForgeError(f"Forge GET {path} failed: {e}") from e

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            r = await self._client.post(path, json=body)
        except httpx.HTTPError as e:
            raise ForgeError(f"Forge POST {path} failed (is Forge up at {self.s.forge_url}?): {e}") from e
        if r.status_code >= 400:
            text = r.text
            if _looks_like_oom(text):
                raise ForgeOOM("GPU out of memory — the shared box was busy. Try again, or a smaller shot.")
            raise ForgeError(f"Forge POST {path} -> {r.status_code}: {text[:300]}")
        return r.json() if r.content else {}  # the refresh-* endpoints may answer with an empty body

    # --- discovery -------------------------------------------------------------------------------------

    async def list_models(self) -> list[str]:
        data = await self._get("/sdapi/v1/sd-models")
        return [m.get("model_name") or m.get("title", "") for m in data]

    async def list_loras(self) -> list[str]:
        """LoRA names as Forge accepts them in `<lora:NAME:weight>`."""
        data = await self._get("/sdapi/v1/loras")
        return sorted(m.get("name", "") for m in data)

    async def refresh(self) -> None:
        """Rescan the checkpoint + LoRA folders so newly dropped files show up without restarting Forge.
        A disk rescan, not GPU work — deliberately outside the GPU lock so it never queues behind a gen."""
        await self._post("/sdapi/v1/refresh-checkpoints", {})
        await self._post("/sdapi/v1/refresh-loras", {})

    async def current_model(self) -> str | None:
        opts = await self._get("/sdapi/v1/options")
        return opts.get("sd_model_checkpoint")

    async def _ensure_model(self, model: str | None) -> None:
        """Switch checkpoint ONLY if the request needs a different one than is loaded (anti-OOM)."""
        if not model:
            return
        current = await self.current_model()
        # A1111 'sd_model_checkpoint' is usually "name [hash]"; match on the leading name.
        if current and (current == model or current.split(" [")[0] == model):
            return
        # resolve against the live list so a friendly name still matches
        models = await self.list_models()
        match = next((m for m in models if m == model or m.split(" [")[0] == model), None)
        if match is None:
            raise ForgeError(f"model not found: {model!r}. Available: {', '.join(models) or '(none)'}")
        await self._post("/sdapi/v1/options", {"sd_model_checkpoint": match})

    # --- generation ------------------------------------------------------------------------------------

    async def generate(self, endpoint: str, payload: dict[str, Any], model: str | None) -> dict[str, Any]:
        """Serialized txt2img/img2img. Returns the raw A1111 response ({images, info, parameters})."""
        async with self._gpu:
            await self._ensure_model(model)
            return await self._post(endpoint, payload)
