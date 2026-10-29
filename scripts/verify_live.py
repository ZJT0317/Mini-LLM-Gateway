"""Low-volume official SDK -> local HTTP gateway -> configured real upstream.

Credentials are read only from the named environment variable, never from CLI values.
Default: 8 small upstream requests, concurrency <= 2, SDK/gateway retries disabled.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict
from dotenv import load_dotenv

from app.config import (
    Settings,
    ServerSettings,
    LimitSettings,
    LogSettings,
    ProviderSettings,
    _check_url,
)
from app.core.auth import hash_secret
from app.core.models import APIKey, Capability, ModelSpec
from app.main import create_app
from validation_support import load, safe_error, serve


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool


async def run(args):
    api_key = os.environ.get(args.api_key_env, "")
    missing = []
    if not args.base_url:
        missing.append("UPSTREAM_BASE_URL or --base-url")
    if not args.model:
        missing.append("UPSTREAM_MODEL or --model")
    if not api_key:
        missing.append(args.api_key_env)
    if missing:
        return {
            "status": "blocked_missing_configuration",
            "measured_at": datetime.now(timezone.utc).isoformat(),
            "missing": missing,
            "paid_requests_made": 0,
        }
    host = urlparse(args.base_url).hostname or ""
    _check_url(args.base_url, (host,))
    cap = frozenset(Capability)
    secret = secrets.token_urlsafe(24)
    report = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "mode": "real_upstream",
        "openai_sdk": version("openai"),
        "upstream_origin": urlparse(args.base_url)
        ._replace(path="", query="", fragment="")
        .geturl(),
        "upstream_model": args.model,
        "notes": [
            "Synthetic non-sensitive prompts only.",
            "SDK and gateway retries disabled; no JSON repair; at most 8 upstream requests by default.",
            "Small sample verifies connectivity and contracts; it does not establish service capacity.",
        ],
    }
    with tempfile.TemporaryDirectory(prefix="mini-gateway-live-") as temporary:
        directory = Path(temporary)
        settings = Settings(
            ServerSettings(
                "127.0.0.1",
                0,
                timedelta(seconds=90),
                1_048_576,
                timedelta(seconds=2),
                directory / "ledger.db",
            ),
            LimitSettings(64, 1, 0),
            LogSettings("warning", False, 0),
            (
                ProviderSettings(
                    "live", args.base_url, api_key, timedelta(seconds=120), (host,), cap
                ),
            ),
            (
                ModelSpec(
                    "live-validation",
                    "live",
                    args.model,
                    args.base_url,
                    cap,
                    1,
                    Decimal(0),
                    Decimal(0),
                    timedelta(seconds=120),
                ),
            ),
            (
                APIKey(
                    "validation", hash_secret(secret), "validation", "active", 60, 2, ()
                ),
            ),
            directory / "prompts",
        )
        app = create_app(settings)
        async with (
            serve(app) as url,
            AsyncOpenAI(
                api_key=secret, base_url=url + "/v1", max_retries=0, timeout=100
            ) as sdk,
        ):
            report["ordinary"] = await load(sdk, "live-validation", 1, 1)
            report["stream"] = await load(sdk, "live-validation", 1, 1, stream=True)
            started = time.perf_counter()
            try:
                answer = await sdk.chat.completions.parse(
                    model="live-validation",
                    max_tokens=64,
                    messages=[
                        {
                            "role": "user",
                            "content": 'Return JSON with the field "ok" set to true.',
                        }
                    ],
                    response_format=Answer,
                )
                report["structured"] = {
                    "ok": answer.choices[0].message.parsed == Answer(ok=True),
                    "refused": bool(answer.choices[0].message.refusal),
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            except Exception as exc:
                report["structured"] = {"ok": False, "error": safe_error(exc)}
            try:
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "description": "Return the supplied text.",
                            "parameters": {
                                "type": "object",
                                "properties": {"text": {"type": "string"}},
                                "required": ["text"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ]
                messages = [
                    {
                        "role": "user",
                        "content": "Call echo with text OK, then briefly report the tool result.",
                    }
                ]
                tool_result = await sdk.chat.completions.create(
                    model="live-validation",
                    messages=messages,
                    tools=tools,
                    tool_choice={"type": "function", "function": {"name": "echo"}},
                    max_tokens=64,
                )
                calls = tool_result.choices[0].message.tool_calls or []
                if len(calls) != 1 or calls[0].function.name != "echo":
                    raise ValueError("provider did not return the required tool")
                arguments = json.loads(calls[0].function.arguments)
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": calls[0].id,
                                    "type": "function",
                                    "function": {
                                        "name": "echo",
                                        "arguments": calls[0].function.arguments,
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": calls[0].id,
                            "content": json.dumps({"text": arguments["text"]}),
                        },
                    ]
                )
                followup = await sdk.chat.completions.create(
                    model="live-validation", messages=messages, max_tokens=32
                )
                report["tool_roundtrip"] = {
                    "ok": bool(followup.choices[0].message.content)
                }
            except Exception as exc:
                report["tool_roundtrip"] = {"ok": False, "error": safe_error(exc)}
            report["small_concurrency_probe"] = await load(sdk, "live-validation", 3, 2)
            rows = app.state.ledger._conn.execute(
                "SELECT terminal, count(*) FROM calls GROUP BY terminal"
            ).fetchall()
            report["ledger"] = {
                "calls_by_terminal": {r[0]: r[1] for r in rows},
                "attempts": app.state.ledger._conn.execute(
                    "SELECT count(*) FROM attempts"
                ).fetchone()[0],
                "note": "Prices are zero because the validation command does not assume provider pricing.",
            }
        import sqlite3

        with sqlite3.connect(directory / "ledger.db") as connection:
            report["ledger"]["reopened_calls"] = connection.execute(
                "SELECT count(*) FROM calls"
            ).fetchone()[0]
    report["status"] = (
        "passed"
        if all(
            [
                report["ordinary"]["success"] == 1,
                report["stream"]["success"] == 1,
                report["structured"]["ok"],
                report["tool_roundtrip"]["ok"],
                report["small_concurrency_probe"]["success"] == 3,
            ]
        )
        else "failed"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        help="API root, for example https://provider.example/v1",
    )
    parser.add_argument("--model")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--output", type=Path, default=Path("docs/live-validation-results.json")
    )
    args = parser.parse_args()
    if args.env_file is not None:
        if not args.env_file.is_file():
            parser.error("--env-file does not exist")
        load_dotenv(args.env_file, override=False)
    args.base_url = args.base_url or os.environ.get("UPSTREAM_BASE_URL")
    args.model = args.model or os.environ.get("UPSTREAM_MODEL")
    report = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["status"] == "passed" else 1)
