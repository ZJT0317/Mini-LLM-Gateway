"""Measure the real SDK and two local HTTP hops; never contacts paid services.

Run: python scripts/benchmark.py --requests 100 --output docs/benchmark-results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import secrets
import sqlite3
import sys
import tempfile
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from app.config import (
    Settings,
    ServerSettings,
    LimitSettings,
    LogSettings,
    ProviderSettings,
)
from app.core.auth import hash_secret
from app.core.models import APIKey, Capability, ModelSpec
from app.main import create_app
from app.services.usage import Ledger
from validation_support import distribution, load, serve


def fixture():
    app = FastAPI()
    state = {"delay_ms": 0, "active": 0, "peak": 0}

    @app.post("/v1/chat/completions")
    async def complete(request: Request):
        body = await request.json()
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        if body.get("stream"):

            async def frames():
                try:
                    await asyncio.sleep(state["delay_ms"] / 1000)
                    for text in ("gateway-", "ok"):
                        yield (
                            "data: "
                            + json.dumps({"choices": [{"delta": {"content": text}}]})
                            + "\n\n"
                        )
                        await asyncio.sleep(0.005)
                    yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                    yield 'data: {"choices":[],"usage":{"prompt_tokens":8,"completion_tokens":3}}\n\n'
                    yield "data: [DONE]\n\n"
                finally:
                    state["active"] -= 1

            return StreamingResponse(frames(), media_type="text/event-stream")
        try:
            await asyncio.sleep(state["delay_ms"] / 1000)
            return {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "gateway-ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 8, "completion_tokens": 3},
            }
        finally:
            state["active"] -= 1

    return app, state


def settings(directory, upstream, secret):
    cap = frozenset(Capability)
    return Settings(
        ServerSettings(
            "127.0.0.1",
            0,
            timedelta(seconds=30),
            1_048_576,
            timedelta(seconds=2),
            directory / "gateway.db",
        ),
        LimitSettings(128, 1, 0),
        LogSettings("warning", False, 0),
        (
            ProviderSettings(
                "controlled",
                upstream + "/v1",
                "local-fixture",
                timedelta(seconds=60),
                ("127.0.0.1",),
                cap,
            ),
        ),
        (
            ModelSpec(
                "bench",
                "controlled",
                "fixture",
                upstream + "/v1",
                cap,
                1,
                Decimal("1"),
                Decimal("2"),
                timedelta(seconds=60),
            ),
        ),
        (APIKey("bench", hash_secret(secret), "bench", "active", 1_000_000, 32, ()),),
        directory / "prompts",
    )


def instrument(ledger):
    timings = {name: [] for name in ("save_call", "save_attempt", "save_rejection")}
    templates = {}
    for name in timings:
        original = getattr(ledger, name)

        def wrapper(*args, _name=name, _method=original, **kwargs):
            started = time.perf_counter()
            try:
                result = _method(*args, **kwargs)
                if _name != "save_rejection" and args:
                    templates[_name] = args[0]
                return result
            finally:
                timings[_name].append((time.perf_counter() - started) * 1000)

        setattr(ledger, name, wrapper)
    return timings, templates


def database_benchmark(directory, templates, count):
    ledger = Ledger(directory / "write-benchmark.db")
    durations = []
    started = time.perf_counter()
    try:
        journal_mode = ledger._conn.execute("PRAGMA journal_mode").fetchone()[0]
        synchronous = ledger._conn.execute("PRAGMA synchronous").fetchone()[0]
        for i in range(count):
            sample = time.perf_counter()
            ledger.save_attempt(
                replace(templates["save_attempt"], call_id=f"write-{i}", attempt=1)
            )
            ledger.save_call(replace(templates["save_call"], call_id=f"write-{i}"))
            durations.append((time.perf_counter() - sample) * 1000)
        elapsed = time.perf_counter() - started
    finally:
        ledger.close()
    return {
        "request_pairs": count,
        "row_writes": count * 2,
        "commits": count * 2,
        "pairs_per_second": round(count / elapsed, 2),
        "rows_per_second": round(2 * count / elapsed, 2),
        "pair_latency": distribution(durations),
        "journal_mode": journal_mode,
        "synchronous": synchronous,
        "database_bytes": (directory / "write-benchmark.db").stat().st_size,
    }


async def run(args):
    result = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "mode": "controlled_local_http",
        "host": {
            "os": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "sqlite": sqlite3.sqlite_version,
            "openai_sdk": version("openai"),
            "fastapi": version("fastapi"),
            "httpx": version("httpx"),
        },
        "notes": [
            "Single process; official SDK -> real local HTTP gateway -> real local HTTP fixture.",
            "Controlled fixed output; measures gateway overhead, not model inference capacity.",
            "SDK retries disabled; latency excludes waiting for the client-side concurrency semaphore.",
            "Database uses existing synchronous per-row commits; filesystem cache and host load affect results.",
        ],
    }
    with tempfile.TemporaryDirectory(prefix="mini-gateway-benchmark-") as temporary:
        directory = Path(temporary)
        upstream, state = fixture()
        secret = secrets.token_urlsafe(24)
        async with serve(upstream) as upstream_url:
            app = create_app(settings(directory, upstream_url, secret))
            timings, templates = instrument(app.state.ledger)
            async with (
                serve(app) as gateway_url,
                AsyncOpenAI(
                    api_key=secret,
                    base_url=gateway_url + "/v1",
                    max_retries=0,
                    timeout=30,
                ) as sdk,
            ):
                # Warm imports, SDK serializers, HTTP pools, and the database before recording.
                await load(sdk, "bench", 10, 1)
                cases = []
                for delay in (0, args.delay_ms):
                    state["delay_ms"] = delay
                    for concurrency in (1, 4, 16):
                        for values in timings.values():
                            values.clear()
                        state["peak"] = 0
                        measured = await load(sdk, "bench", args.requests, concurrency)
                        measured.update(
                            upstream_delay_ms=delay,
                            upstream_peak_concurrency=state["peak"],
                            writes={
                                name: distribution(values)
                                for name, values in timings.items()
                            },
                        )
                        cases.append(measured)
                state["delay_ms"] = args.delay_ms
                for concurrency in (1, 4):
                    measured = await load(
                        sdk, "bench", min(args.requests, 20), concurrency, stream=True
                    )
                    measured["upstream_delay_ms"] = args.delay_ms
                    cases.append(measured)
                result["cases"] = cases
                state["delay_ms"] = 50
                state["peak"] = 0
                burst_dir = directory / "burst"
                burst_settings = settings(burst_dir, upstream_url, secret)
                burst_settings = replace(
                    burst_settings,
                    keys=(replace(burst_settings.keys[0], max_concurrency=4),),
                )
                burst_app = create_app(burst_settings)
                async with (
                    serve(burst_app) as burst_url,
                    AsyncOpenAI(
                        api_key=secret, base_url=burst_url + "/v1", max_retries=0
                    ) as burst_sdk,
                ):
                    rejection = await load(burst_sdk, "bench", 32, 32)
                rejection["configured_limit"] = 4
                rejection["upstream_peak_concurrency"] = state["peak"]
                result["admission_burst"] = rejection
                counts = {
                    table: app.state.ledger._conn.execute(
                        f"SELECT count(*) FROM {table}"
                    ).fetchone()[0]
                    for table in ("calls", "attempts")
                }
            # The app server has closed its ledger; verify durable rows using a new connection.
            connection = sqlite3.connect(directory / "gateway.db")
            result["persistence"] = {
                **counts,
                "reopened_calls": connection.execute(
                    "SELECT count(*) FROM calls"
                ).fetchone()[0],
                "duplicate_attempt_pairs": connection.execute(
                    "SELECT count(*) FROM (SELECT call_id,attempt FROM attempts GROUP BY call_id,attempt HAVING count(*)>1)"
                ).fetchone()[0],
            }
            connection.close()
        result["database_write_benchmark"] = database_benchmark(
            directory, templates, args.db_pairs
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--delay-ms", type=int, default=20)
    parser.add_argument("--db-pairs", type=int, default=500)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/benchmark-results.json")
    )
    args = parser.parse_args()
    if args.requests < 20 or args.db_pairs < 1 or args.delay_ms < 0:
        parser.error("requests >= 20, db-pairs >= 1, delay-ms >= 0 required")
    result = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": [
                    {
                        "concurrency": case["concurrency"],
                        "stream": case["stream"],
                        "upstream_delay_ms": case["upstream_delay_ms"],
                        "success": case["success"],
                        "errors": case["errors"],
                        "success_rps": case["success_rps"],
                        "latency": case["latency"],
                        "ttft": case["ttft"],
                    }
                    for case in result["cases"]
                ],
                "admission_burst": result["admission_burst"],
                "persistence": result["persistence"],
                "database_write_benchmark": result["database_write_benchmark"],
            },
            indent=2,
        )
    )
