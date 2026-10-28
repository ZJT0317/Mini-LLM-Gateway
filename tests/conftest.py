"""Build a gateway against a temp config and an injected upstream transport."""

from __future__ import annotations

import os
from pathlib import Path

import httpx

from app.config import load_settings
from app.main import create_app

YAML = """
server:
  addr: "127.0.0.1:8080"
  gateway_timeout: 60s
  max_body_bytes: 1048576
  shutdown_grace: 1s
  database: {database}
limits:
  max_tokens_cap: 128
  retry_max_attempts: {retries}
  repair_max_attempts: {repairs}
log:
  level: info
  debug_payload: false
  payload_max_runes: 50
providers:
  - name: openai
    base_url: http://127.0.0.1:9/v1
    api_key_env: OPENAI_API_KEY
    upstream_timeout: 90s
    allowed_hosts: ["127.0.0.1"]
    capabilities: [streaming, structured_output, tool_calling]
models:
{models}
keys:
  - id: demo
    secret_env: GATEWAY_DEMO_KEY
    tenant_id: demo
    status: active
    rpm: 60
    max_concurrency: {concurrency}
    allowed_models: []
  - id: off
    secret_env: GATEWAY_OFF_KEY
    tenant_id: demo
    status: disabled
    rpm: 60
    max_concurrency: 2
    allowed_models: []
"""

MODEL = """
  - logical_model: {logical}
    provider: openai
    resolved_model: {resolved}
    priority: {priority}
    input_cost_per_million: "1.00"
    output_cost_per_million: "2.00"
    capabilities: [{capabilities}]
"""


def write_config(
    directory: Path,
    *,
    models: list[dict] | None = None,
    retries: int = 2,
    repairs: int = 1,
    concurrency: int = 4,
) -> Path:
    os.environ["OPENAI_API_KEY"] = "sk-upstream-secret"
    os.environ["GATEWAY_DEMO_KEY"] = "gateway-demo-secret"
    os.environ["GATEWAY_OFF_KEY"] = "gateway-off-secret"
    specs = models or [
        {
            "logical": "demo",
            "resolved": "upstream-demo",
            "priority": 1,
            "capabilities": "streaming, structured_output, tool_calling",
        }
    ]
    rendered = "\n".join(MODEL.format(**item) for item in specs)
    path = directory / "gateway.yaml"
    path.write_text(
        YAML.format(
            database=(directory / "gateway.db").as_posix(),
            retries=retries,
            repairs=repairs,
            concurrency=concurrency,
            models=rendered,
        ),
        encoding="utf-8",
    )
    return path


def build_app(
    directory: Path,
    transport: httpx.AsyncBaseTransport,
    *,
    sleeper=None,
    rng=None,
    clock=None,
    prompts=None,
    models: list[dict] | None = None,
    retries: int = 2,
    repairs: int = 1,
    concurrency: int = 4,
):
    settings = load_settings(
        write_config(
            directory,
            models=models,
            retries=retries,
            repairs=repairs,
            concurrency=concurrency,
        )
    )
    return create_app(
        settings,
        transport=transport,
        sleeper=sleeper,
        rng=rng,
        clock=clock,
        prompts=prompts,
    )
