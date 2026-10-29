FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.11 /uv /bin/uv

ENV UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    PATH="/srv/.venv/bin:$PATH"

WORKDIR /srv
COPY pyproject.toml uv.lock README.md ./
COPY app ./app
RUN uv sync --locked --no-dev --no-editable --no-cache

EXPOSE 8080
CMD ["mini-llm-gateway", "--config", "gateway.yaml"]
