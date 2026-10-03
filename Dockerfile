# forge-mcp — thin MCP proxy over a Forge / A1111 SDXL backend.
#   serve: forge-mcp serve --transport http --host 0.0.0.0   (FORGE_MCP_TOKEN set; FORGE_URL -> the Forge --api)
# The server holds NO GPU and NO model — it brokers requests to Forge and enforces the
# shared-GPU envelope + the unconditional NSFW block. Stateless; the consuming repo owns anchors/profiles.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/app/.venv
COPY pyproject.toml uv.lock LICENSE README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=60s --timeout=5s CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/healthz').status!=200)"
ENTRYPOINT ["forge-mcp"]
CMD ["serve", "--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
