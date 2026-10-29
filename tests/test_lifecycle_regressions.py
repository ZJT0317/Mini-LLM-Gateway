"""Cancellation, deadline, and circuit recovery contracts."""

import asyncio
import json
from dataclasses import replace
from datetime import timedelta

import httpx
import pytest

from app.core.models import Capability, Invocation, Message
from app.services.router import HealthBoard
from tests.conftest import build_app
from tests.test_gateway_api import _ok, _post
from tests.test_protocol_regressions import BASE, _call
from tests.test_router_and_adapter import _spec


def _invocation(app, *, stream=False):
    key = app.state.settings.keys[0]
    return Invocation(
        "call",
        "trace",
        "demo",
        (Message("user", "hi"),),
        frozenset({Capability.STREAMING}) if stream else frozenset(),
        None,
        None,
        None,
        (),
        None,
        app.state.settings.server.gateway_timeout,
        key.tenant_id,
        key.id,
    ), key


def test_circuit_cooldown_single_probe_success_and_failure():
    now = [0.0]
    health = HealthBoard(threshold=1, cooldown=10, clock=lambda: now[0])
    spec = _spec("primary", 1, {Capability.STREAMING})
    health.record(spec, success=False)
    assert not health.allow(spec)
    now[0] = 10
    assert health.allow(spec)
    assert health.acquire(spec)
    assert not health.acquire(spec)
    health.record(spec, success=False)
    assert not health.allow(spec)
    now[0] = 20
    assert health.acquire(spec)
    health.record(spec, success=True)
    assert health.acquire(spec)


def test_cancelled_probe_releases_reservation():
    now = [0.0]
    health = HealthBoard(threshold=1, cooldown=10, clock=lambda: now[0])
    spec = _spec("primary", 1, set())
    health.record(spec, success=False)
    now[0] = 10
    assert health.acquire(spec)
    health.release(spec)
    now[0] = 20
    assert health.acquire(spec)


async def test_unavailable_model_is_503_then_recovers_without_restart(tmp_path):
    healthy = False
    count = 0
    now = [0.0]

    def handler(request):
        nonlocal count
        count += 1
        return _ok() if healthy else httpx.Response(500, text="down")

    app = build_app(tmp_path, httpx.MockTransport(handler), retries=1)
    app.state.gateway._health._clock = lambda: now[0]
    async with app.router.lifespan_context(app):
        for _ in range(3):
            assert (await _post(app, BASE)).status_code == 502
        unavailable = await _post(app, BASE)
        assert unavailable.status_code == 503 and count == 3
        assert unavailable.json()["error"]["code"] == "model_unavailable"
        assert (await _post(app, {**BASE, "model": "missing"})).status_code == 404
        now[0] = 30
        healthy = True
        assert (await _post(app, BASE)).status_code == 200
        assert count == 4


async def test_cancelled_completion_is_recorded_and_releases_slot(tmp_path):
    entered = asyncio.Event()
    closed = asyncio.Event()

    async def handler(request):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    app = build_app(tmp_path, httpx.MockTransport(handler), concurrency=1)
    async with app.router.lifespan_context(app):
        invocation, key = _invocation(app)
        task = asyncio.create_task(app.state.gateway.complete(invocation, key))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set()
        call = app.state.ledger.calls_by_trace("trace")[0]
        attempts = app.state.ledger.attempts_for("call")
        assert call["terminal"] == "canceled" and call["http_status"] == 499
        assert len(attempts) == 1 and attempts[0]["terminal"] == "canceled"
        status, release = await app.state.gateway._admission.acquire(
            key.id, key.rpm, 1, app.state.gateway._clock()
        )
        assert status == "ok"
        await release()


class HeldStream(httpx.AsyncByteStream):
    def __init__(self):
        self.closed = False

    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
        await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


async def test_closing_stream_across_tasks_records_partial_cost_and_closes_upstream(
    tmp_path,
):
    source = HeldStream()
    app = build_app(
        tmp_path,
        httpx.MockTransport(lambda request: httpx.Response(200, stream=source)),
    )
    async with app.router.lifespan_context(app):
        invocation, key = _invocation(app, stream=True)
        events = app.state.gateway.stream(invocation, key)
        assert (await asyncio.create_task(anext(events))).text == "partial"
        await events.aclose()
        assert source.closed
        call = app.state.ledger.calls_by_trace("trace")[0]
        attempts = app.state.ledger.attempts_for("call")
        assert call["terminal"] == "canceled"
        assert len(attempts) == 1 and attempts[0]["output_tokens"] > 0
        assert call["cost_total"] == attempts[0]["cost_total"]
        assert app.state.gateway._admission._inflight[key.id] == 0


async def _disconnected_request(app, payload, trigger, *, after_frame=False):
    body_sent = False
    messages = []

    async def receive():
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {
                "type": "http.request",
                "body": json.dumps(payload).encode(),
                "more_body": False,
            }
        await trigger.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)
        if (
            after_frame
            and message["type"] == "http.response.body"
            and b"partial" in message.get("body", b"")
        ):
            trigger.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"authorization", b"Bearer gateway-demo-secret"),
            (b"content-type", b"application/json"),
            (b"x-request-id", b"disconnect-trace"),
        ],
        "client": ("127.0.0.1", 1),
        "server": ("gateway.test", 80),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=1)
    return messages


@pytest.mark.parametrize("stream", [False, True])
async def test_asgi_disconnect_cancels_work_and_records_once(tmp_path, stream):
    trigger = asyncio.Event()
    source = HeldStream()
    closed = asyncio.Event()

    async def handler(request):
        if stream:
            return httpx.Response(200, stream=source)
        trigger.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    app = build_app(tmp_path, httpx.MockTransport(handler), concurrency=1)
    async with app.router.lifespan_context(app):
        await _disconnected_request(
            app, {**BASE, "stream": stream}, trigger, after_frame=stream
        )
        assert source.closed if stream else closed.is_set()
        calls = app.state.ledger.calls_by_trace("disconnect-trace")
        assert len(calls) == 1 and calls[0]["terminal"] == "canceled"
        attempts = app.state.ledger.attempts_for(calls[0]["call_id"])
        assert len(attempts) == 1 and attempts[0]["terminal"] == "canceled"
        assert app.state.gateway._admission._inflight["demo"] == 0


@pytest.mark.parametrize("stream", [False, True])
async def test_deadline_before_and_after_first_token(tmp_path, monkeypatch, stream):
    monkeypatch.setattr("app.services.gateway.budget_is_usable", lambda budget: True)
    source = HeldStream()

    async def handler(request):
        if stream:
            return httpx.Response(200, stream=source)
        await asyncio.Event().wait()

    app = build_app(tmp_path, httpx.MockTransport(handler))
    app.state.settings = replace(
        app.state.settings,
        server=replace(
            app.state.settings.server, gateway_timeout=timedelta(milliseconds=20)
        ),
    )
    async with app.router.lifespan_context(app):
        response = await asyncio.wait_for(_post(app, {**BASE, "stream": stream}), 1)
        assert response.status_code == (200 if stream else 504)
        assert "gateway timeout" in response.text
        if stream:
            assert "[DONE]" not in response.text and source.closed
        call = _call(app, response)
        assert call["terminal"] == "error"
        attempts = app.state.ledger.attempts_for(call["call_id"])
        assert len(attempts) == 1 and attempts[0]["terminal"] == "error"


async def test_disconnect_before_first_stream_event_cancels_prefetch(tmp_path):
    trigger = asyncio.Event()
    closed = asyncio.Event()

    async def handler(request):
        trigger.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    app = build_app(tmp_path, httpx.MockTransport(handler), concurrency=1)
    async with app.router.lifespan_context(app):
        await _disconnected_request(app, {**BASE, "stream": True}, trigger)
        assert closed.is_set()
        calls = app.state.ledger.calls_by_trace("disconnect-trace")
        assert len(calls) == 1 and calls[0]["terminal"] == "canceled"
        assert app.state.gateway._admission._inflight["demo"] == 0


async def test_deadline_survives_prefetch_and_slow_consumption(tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.gateway.budget_is_usable", lambda budget: True)
    source = HeldStream()
    app = build_app(
        tmp_path,
        httpx.MockTransport(lambda request: httpx.Response(200, stream=source)),
    )
    async with app.router.lifespan_context(app):
        invocation, key = _invocation(app, stream=True)
        invocation = replace(invocation, timeout=timedelta(milliseconds=20))
        events = app.state.gateway.stream(invocation, key)
        prefetch = asyncio.create_task(anext(events))
        assert (await prefetch).text == "partial"
        # The consumer resumes in another task after spending the shared budget.
        await asyncio.sleep(0.03)
        error = await asyncio.wait_for(anext(events), 1)
        assert error.error.message == "gateway timeout"
        await events.aclose()
        assert source.closed and not prefetch.cancelled()
        assert app.state.ledger.calls_by_trace("trace")[0]["terminal"] == "error"
