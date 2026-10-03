# forge-mcp

An [MCP](https://modelcontextprotocol.io) server that exposes a local **Forge / AUTOMATIC1111** SDXL backend
as image-generation tools — so any MCP client (Claude Code, Claude Desktop, Open WebUI, …) can generate and
edit images through a clean, typed interface instead of hand-rolled API calls.

It is built to be **frictionless, deterministic, and a good citizen on a shared GPU** — the common case where
the same box also runs a local LLM or other workloads.

## Features

- **`generate_image`** (txt2img) and **`edit_image`** (img2img) with sane, overridable defaults.
- **Deterministic by default** — the resolved seed and every parameter are returned with each result, so a
  keeper *is* its own recipe. Prompts are sent **verbatim**; the server never silently rewrites them.
- **Good shared-GPU citizen** — all GPU work is serialized; the checkpoint is switched only when it differs
  from the one already loaded (the reload is the VRAM spike that OOMs); generations are clamped to VRAM-safe
  aspect buckets; out-of-memory is surfaced as a clean, typed error.
- **Stateless** — it owns no database and no stored state; the calling client/repo stays the source of truth
  for any character anchors or profiles.
- **SFW by default** — a server-side NSFW block is applied unconditionally (it does not trust the prompt, the
  profile, or the calling model).
- **Returns a viewable preview inline** plus the full-resolution PNG, so the caller can save it wherever it wants.
- Named **negative/style profiles** and **shot** shortcuts (`portrait`, `establishing`, `square`) over raw
  width/height.

## Requirements

- A running **Forge** or **AUTOMATIC1111** instance with the API enabled (`--api`, and `--listen` if remote).
- Python 3.12+ and [uv](https://docs.astral.sh/uv/).

## Install & run

```bash
uv sync

# stdio (for a local MCP client that launches the server):
FORGE_URL=http://127.0.0.1:7860 uv run forge-mcp serve

# streamable-http (for a networked client / gateway), with a bearer token:
FORGE_URL=http://127.0.0.1:7860 FORGE_MCP_TOKEN=$(openssl rand -hex 32) \
  uv run forge-mcp serve --transport http --host 0.0.0.0 --port 8000
```

### Configuration (environment variables)

| Var | Default | Purpose |
|---|---|---|
| `FORGE_URL` | `http://127.0.0.1:7860` | Base URL of the Forge/A1111 **`--api`** server |
| `FORGE_MCP_TOKEN` | *(unset)* | If set, require `Authorization: Bearer <token>` on the HTTP transport |
| `FORGE_DEFAULT_MODEL` | `DreamShaperXL_Turbo_SFW` | Checkpoint used when a request doesn't name one |
| `FORGE_GEN_TIMEOUT` | `180` | Seconds to wait on Forge (covers a cold model-load + generation) |
| `FORGE_PREVIEW_MAX_PX` | `768` | Longest edge of the inline JPEG preview |

> **TLS note:** behind a TLS-inspecting proxy, `uv` may report `UnknownIssuer`; pass `--system-certs`
> (or set `UV_NATIVE_TLS=1`) so it trusts the OS certificate store. The server itself uses `truststore` at
> runtime for the same reason.

## Docker

```bash
docker build -t forge-mcp:latest .
docker run --rm -p 8000:8000 -e FORGE_URL=http://host.docker.internal:7860 \
  -e FORGE_MCP_TOKEN=... forge-mcp:latest
```

The server holds no GPU and no model — it brokers to your Forge instance and enforces the VRAM-safe envelope
and the NSFW block.

## Design principles

- **Deterministic over magic** — reproducibility (seed + params returned, verbatim prompts) is a feature, not
  a nicety. No silent "enhance."
- **The caller owns state** — no character database inside the server; it stays a stateless proxy.
- **A good tenant** — never thrash a GPU that something else is using.
- **SFW is enforced server-side** — because an MCP can't assume a well-behaved caller.

See [`CLAUDE.md`](CLAUDE.md) for the fuller design rationale and the roadmap (`upscale`, async batch
`variations`, and a deferred `inpaint`).

## Style

Python 3.12, `uv`, standard library + `httpx` / `mcp` / `Pillow` / `truststore` — few dependencies by design.
Keep it stateless and deterministic; do not add anything that rewrites prompts silently or relaxes the NSFW
floor via a tool flag.

## AI assistance

forge-mcp is developed openly with the help of Claude (Anthropic). We state this plainly: commits
Claude helped write carry a `Co-Authored-By: Claude` trailer. The code and design are open source so the
work can be inspected, reused, and given back.

## License

[MPL-2.0](LICENSE) © Daniel Núñez.
