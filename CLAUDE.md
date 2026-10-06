# CLAUDE.md — context for forge-mcp

An MCP server that fronts a local **Forge / AUTOMATIC1111** SDXL backend so image generation is a first-class
tool for any MCP client. Read this before changing the design — the constraints below are deliberate.

## What makes this one opinionated: it assumes a shared GPU

The common deployment is a box whose GPU is **also** running something else (a local LLM, other inference).
So image-gen is a **tenant on a busy GPU**, and that shapes the design:

- **Serialize all GPU work** behind one lock (`ForgeClient._gpu`) — never two generations, or a generation
  during a model switch, at once. This is the queue.
- **Don't reload the checkpoint needlessly** — `_ensure_model` queries the loaded model and switches ONLY
  when it differs. **The reload is the VRAM spike that OOMs.** The switch is persistent (no restore on the
  checkpoint) so the next call on the same model doesn't reload again.
- **VRAM-safe envelope** — only the ~1 MP aspect buckets (`config.ALLOWED_WH`); no hires-fix. A caller (or an
  over-eager model) cannot thrash the GPU.
- **OOM is a clean, typed error** (`ForgeOOM`) — never a hang or an opaque 500.

## The caller owns state — this server does not

Deliberately **no character database, no stored anchors, no profiles-of-record** inside the server — that
would become a second source of truth that drifts. The server is a **stateless proxy**. A client that wants
call-by-name character consistency keeps its own anchors (e.g. a committed `anchor.json` per character:
`{seed, model, canonical_positive, negative_profile, shot, keeper_png}`), resolves them itself, and passes the
resolved values in. The built-in negative **profiles** (`presets.NEGATIVE_PROFILES`) are a baseline the caller
can override by passing its own negative text.

## Determinism is a feature

- The **resolved seed** and **every parameter used** come back in every result — that text block is the
  sidecar a client writes next to the PNG. `seed=-1` still echoes the real seed so a keeper locks.
- **Prompts are sent verbatim.** ★ Never silently "enhance"/rewrite a prompt — it makes seeds irreproducible.
  If enhancement is ever added it must be opt-in and return the exact final text used.
- Consistency works by locking a seed and changing only small deltas (age/colour) — same seed, same "soul,"
  different vessel. img2img from a canonical keeper *anchors* a character, but it is **holistic** (re-derives
  the whole frame) — it is NOT a single-feature scalpel; document it honestly.

## Result contract: `status` = ok | refused | error

Every tool result's text block carries a `status` — branch on it, never on the prose:
- **ok** — image generated (preview inline; `full_res_url` present when `include_full`).
- **refused** — a POLICY decision (SFW block, or an out-of-envelope resolution); `reason`/`matched`/`advice`
  say how to adjust. The caller *rephrases* — it is NOT an error and NOT a retry-unchanged.
- **error** — a genuine FAULT (`kind` = "gpu_oom"/"forge" + `detail`); infra, not policy, so a prompt change
  won't help. (`_refused` / `_error` build these; `_run` tags success "ok".)
★ A policy refusal must **never be raised as an exception** — doing so made a false block, a correct block, and
a real crash indistinguishable to the caller (the opaque "Error executing tool"). Keep the three signals distinct.

## Full-res is a link, not base64 (topology-aware keeper-save)

The server and the consuming repo are typically on **different machines** (MCP next to the GPU; repo on a
workstation). So a full-res PNG can't be written to the repo's filesystem server-side, and base64 in the tool
result floods/truncates the model's context (~2.7 MB ≈ ~1M tokens — predicted in the brief, confirmed on first
contact). The fix: `include_full=True` saves the PNG under an unguessable capability name in `out_dir` and
returns a short-lived **`full_res_url`** (served at `/img/<name>` by the ASGI gate, open so the saving machine
can `curl` it without the bearer) + a ready `download` command. TTL GC (`FORGE_IMG_TTL`, default 10 min), not
delete-on-first-GET — a dropped download just retries. Needs `FORGE_PUBLIC_URL` set to a client-reachable base.
The `/img/<name>` serve is path-traversal-safe (capability tokens only). `include_full=False` = preview + params.

**The inbound mirror — init images by reference.** The same flood happens the other way: a keeper passed to
`edit_image` as base64 dumps ~1.8 MB into the caller's context (found on first real img2img test). So the caller
uploads it out-of-band — `POST /upload` (raw bytes; **bearer-gated**, since unlike `/img` it writes) → `{ref}` —
and passes `init_image_ref`. Uploads are re-encoded through Pillow (only a real image lands on disk), size- and
pixel-capped, and share `out_dir` + TTL with full-res, so a recent `full_res_url` is also a valid ref (iterate
with no round-trip). Raw base64 `init_image` remains for small images.

## Safety: NSFW is blocked server-side, unconditionally

`presets.NSFW_NEGATIVE` is appended to **every** request's negative prompt, and `positive_is_blocked` rejects
explicit positive prompts outright — regardless of profile, prompt, or calling model. **Rationale:** an MCP
cannot assume a smart/aligned caller (a weak or jailbroken client hits the same wall). Do not add a tool flag
that relaxes this.

⚠ The positive check matches on **word boundaries**, not raw substrings — else "cum" blocks "do**cum**entary"
and "sex" blocks "uni**sex**" (a real false-positive that blocked an innocent prompt). Don't regress it to
`term in prompt`. Pinned by `tests/test_presets.py`.

## Files

- `config.py` — env settings, the aspect buckets / Turbo defaults / known models.
- `presets.py` — negative profiles, style presets, shot vocab, **the NSFW floor** (safety-critical).
- `forge.py` — async A1111 client + the shared-GPU tenancy (lock, check-then-switch, OOM).
- `server.py` — the `MCPServer`, `generate_image` / `edit_image` / `list_models`, preview, `BearerAuth`.
- `__init__.py` — CLI (`serve --transport stdio|http`), `truststore` injection (for TLS-inspecting networks).

## Roadmap

1. **v1 (current):** generate_image + edit_image + reproducibility + presets/shots + tenancy + NSFW floor.
2. `upscale` (extras/upscale API) — keeper → higher-res. Never auto-upscale; the human picks keepers.
3. `variations` / batch — **async, queued-sequential, NEVER parallel** (shared GPU). Return as they land.
4. `inpaint` — **deferred** (an external editor does surgical fixes better). If built: region-name
   auto-masking (`region="eyes"`), not hand-supplied pixel masks.

## Conventions

- Python 3.12, `uv`, few dependencies (`httpx` / `mcp` / `Pillow` / `truststore`). License MPL-2.0.
- Keep it stateless and deterministic. Never rewrite prompts silently; never relax the NSFW floor via a flag.
- Tests: `uv run pytest` (NSFW denylist regression, the `status` contract shape, full-res TTL-GC + path-traversal,
  and `tests/test_tools.py`: every tool called by name against a fake Forge — hints, verbatim prompt, the floor in
  the sent payload, init-by-ref). Every tool declares a `title` + **all four** hints explicitly; a new tool must too.
  Behind a TLS-inspecting proxy, `uv` needs `--system-certs` (see README). Deployment (on a shared-GPU host) and
  its mcpo wiring live wherever this is deployed, not in this repo — this repo stays the generic, public tool.
