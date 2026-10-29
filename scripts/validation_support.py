"""Local HTTP server and measurement helpers used by validation commands."""

from __future__ import annotations

import asyncio
import math
import socket
import time
from collections import Counter
from contextlib import asynccontextmanager

import uvicorn
from openai import APIStatusError


@asynccontextmanager
async def serve(app):
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    listener.setblocking(False)
    server = uvicorn.Server(
        uvicorn.Config(app, log_level="error", access_log=False, lifespan="on")
    )
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        async with asyncio.timeout(10):
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("server failed to start")
                await asyncio.sleep(0.005)
        yield f"http://127.0.0.1:{listener.getsockname()[1]}"
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, 10)
        finally:
            listener.close()


def distribution(values):
    ordered = sorted(values)

    def percentile(p):
        return (
            round(
                ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * p) - 1))],
                3,
            )
            if ordered
            else None
        )

    return {
        "count": len(values),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "max_ms": round(max(values), 3) if values else None,
    }


def safe_error(exc):
    # Provider error bodies and credentials never go into validation artifacts.
    result = {"type": type(exc).__name__}
    if isinstance(exc, APIStatusError):
        result["status"] = exc.status_code
        if exc.request_id:
            result["request_id"] = exc.request_id
    return result


async def load(sdk, model, count, concurrency, *, stream=False):
    gate = asyncio.Semaphore(concurrency)
    latencies, ttft, failures, lag = [], [], [], []
    stop = asyncio.Event()

    async def heartbeat():
        while not stop.is_set():
            start = time.perf_counter()
            await asyncio.sleep(0.005)
            lag.append(max(0, (time.perf_counter() - start - 0.005) * 1000))

    async def request():
        async with gate:
            started = time.perf_counter()
            try:
                if stream:
                    first = False
                    async for event in await sdk.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": "Reply with OK."}],
                        max_tokens=32,
                        stream=True,
                        stream_options={"include_usage": True},
                    ):
                        if (
                            event.choices
                            and event.choices[0].delta.content
                            and not first
                        ):
                            ttft.append((time.perf_counter() - started) * 1000)
                            first = True
                    if not first:
                        raise ValueError("stream returned no business content")
                else:
                    result = await sdk.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": "Reply with OK."}],
                        max_tokens=32,
                    )
                    if (
                        not result.choices
                        or not result.choices[0].message.content
                        or result.choices[0].message.refusal
                    ):
                        raise ValueError("completion returned no answer")
                latencies.append((time.perf_counter() - started) * 1000)
            except Exception as exc:
                failures.append(safe_error(exc))

    probe = asyncio.create_task(heartbeat())
    started = time.perf_counter()
    try:
        await asyncio.gather(*(request() for _ in range(count)))
    finally:
        elapsed = time.perf_counter() - started
        stop.set()
        await probe
    return {
        "requests": count,
        "concurrency": concurrency,
        "stream": stream,
        "success": len(latencies),
        "errors": dict(Counter(str(e.get("status", e["type"])) for e in failures)),
        "elapsed_s": round(elapsed, 4),
        "success_rps": round(len(latencies) / elapsed, 2),
        "latency": distribution(latencies),
        "ttft": distribution(ttft),
        "event_loop_lag": distribution(lag),
    }
