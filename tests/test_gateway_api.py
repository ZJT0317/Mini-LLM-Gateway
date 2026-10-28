import asyncio
import json
from datetime import timedelta

import httpx
import pytest

from app.core.errors import Code, GatewayError, Stage
from app.core.models import Capability, EventKind, GatewayEvent, ModelSpec
from app.core.retry import backoff
from app.services.gateway import GatewayService
from tests.conftest import build_app


def _ok(content: str = "hello", usage: bool = True) -> httpx.Response:
    body = {
        "id": "prov",
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
    }
    if usage:
        body["usage"] = {"prompt_tokens": 3, "completion_tokens": 1}
    return httpx.Response(200, json=body)


def _sse(text: str) -> httpx.Response:
    payload = json.dumps({"choices": [{"delta": {"content": text}}]})
    done = json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    raw = f"data: {payload}\n\ndata: {done}\n\ndata: [DONE]\n\n"
    return httpx.Response(200, text=raw)


async def _post(app, payload: dict, key: str = "gateway-demo-secret"):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway.test") as client:
        return await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
        )


async def test_chat_and_models(tmp_path):
    app = build_app(tmp_path, httpx.MockTransport(lambda request: _ok()))
    response = await _post(app, {"model": "demo", "messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "demo"
    assert body["choices"][0]["message"]["content"] == "hello"
    assert "usage" not in body
    assert response.headers["x-request-id"]
    rows = app.state.ledger.calls_by_trace(response.headers["x-request-id"])
    assert rows[0]["requested_model"] == "demo"
    attempts = app.state.ledger.attempts_for(rows[0]["call_id"])
    assert attempts[0]["resolved_model"] == "upstream-demo"
    assert "gateway-demo-secret" not in response.text

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway.test") as client:
        models = await client.get("/v1/models", headers={"Authorization": "Bearer gateway-demo-secret"})
    assert models.json()["data"][0]["id"] == "demo"


async def test_auth_and_unknown_model(tmp_path):
    app = build_app(tmp_path, httpx.MockTransport(lambda request: _ok()))
    missing = await _post(app, {"model": "demo", "messages": [{"role": "user", "content": "hi"}]}, key="nope")
    assert missing.status_code == 401
    assert missing.json()["error"]["message"] == "invalid api key"
    disabled = await _post(
        app,
        {"model": "demo", "messages": [{"role": "user", "content": "hi"}]},
        key="gateway-off-secret",
    )
    assert disabled.status_code == 403
    unknown = await _post(app, {"model": "missing", "messages": [{"role": "user", "content": "hi"}]})
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "no_compatible_model"


async def test_upstream_secret_is_not_returned(tmp_path, caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="failure sk-upstream-secret")

    app = build_app(tmp_path, httpx.MockTransport(handler))
    with caplog.at_level("INFO"):
        response = await _post(app, {"model": "demo", "messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 502
    assert "sk-upstream-secret" not in response.text
    assert "sk-upstream-secret" not in caplog.text
    assert "gateway-demo-secret" not in caplog.text
    assert response.headers["x-request-id"] in caplog.text


async def test_retry_records_wait_without_sleeping(tmp_path):
    calls = {"n": 0}
    waits = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="slow down")
        return _ok()

    async def sleeper(delay: timedelta) -> None:
        waits.append(delay)

    app = build_app(
        tmp_path,
        httpx.MockTransport(handler),
        sleeper=sleeper,
        rng=lambda: 0.0,
    )
    response = await _post(app, {"model": "demo", "messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 200
    assert calls["n"] == 2
    assert waits == [backoff(1, 0.0)]
    rows = app.state.ledger.calls_by_trace(response.headers["x-request-id"])
    attempts = app.state.ledger.attempts_for(rows[0]["call_id"])
    assert len(attempts) == 2
    assert attempts[0]["call_id"] == attempts[1]["call_id"]


async def test_fallback_shares_call_id_and_records_rejections(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["model"] == "primary":
            return httpx.Response(500, text="down")
        return _ok("from-backup")

    app = build_app(
        tmp_path,
        httpx.MockTransport(handler),
        retries=1,
        models=[
            {
                "logical": "demo",
                "resolved": "primary",
                "priority": 1,
                "capabilities": "streaming, structured_output, tool_calling",
            },
            {
                "logical": "demo",
                "resolved": "backup",
                "priority": 2,
                "capabilities": "streaming, structured_output, tool_calling",
            },
        ],
    )
    response = await _post(app, {"model": "demo", "messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "from-backup"
    call = app.state.ledger.calls_by_trace(response.headers["x-request-id"])[0]
    attempts = app.state.ledger.attempts_for(call["call_id"])
    assert [row["resolved_model"] for row in attempts] == ["primary", "backup"]
    assert attempts[1]["is_fallback"] == 1
    reasons = [row["reason"] for row in app.state.ledger.rejections_for(call["call_id"])]
    assert "priority_lost" in reasons
    assert float(call["cost_total"]) > 0


async def test_stream_emits_one_terminal_and_records_ttft(tmp_path):
    app = build_app(tmp_path, httpx.MockTransport(lambda request: _sse("hi")))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway.test") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer gateway-demo-secret"},
            json={"model": "demo", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
    text = response.text
    assert text.count("data: [DONE]") == 1
    assert "hi" in text
    call = app.state.ledger.calls_by_trace(response.headers["x-request-id"])[0]
    attempt = app.state.ledger.attempts_for(call["call_id"])[0]
    assert attempt["ttft_ms"] is not None
    assert call["ttft_ms"] is not None


async def test_stream_does_not_switch_models_after_the_first_delta(tmp_path):
    from datetime import datetime, timezone

    from app.core.admission import AdmissionGate
    from app.core.models import APIKey, Invocation, Message
    from app.services.metrics import Metrics
    from app.services.prompts import PromptLibrary
    from app.services.router import AdapterRegistry, HealthBoard, Router
    from app.services.usage import Ledger

    class Scripted:
        name = "openai"
        capabilities = frozenset(Capability)

        def __init__(self):
            self.models: list[str] = []

        async def invoke(self, call):
            raise AssertionError("stream path")

        async def stream(self, call):
            self.models.append(call.resolved_model)
            if call.resolved_model == "primary":
                yield GatewayEvent(kind=EventKind.DELTA, text="partial")
                raise GatewayError(
                    Code.UPSTREAM_UNAVAILABLE,
                    "upstream request failed",
                    retryable=True,
                    stage=Stage.STREAMING,
                )
            yield GatewayEvent(kind=EventKind.DELTA, text="other")
            yield GatewayEvent(kind=EventKind.DONE)

    scripted = Scripted()
    specs = (
        ModelSpec("demo", "openai", "primary", "http://127.0.0.1:9/v1", scripted.capabilities, 1, __import__("decimal").Decimal("1"), __import__("decimal").Decimal("1"), timedelta(seconds=5)),
        ModelSpec("demo", "openai", "backup", "http://127.0.0.1:9/v1", scripted.capabilities, 2, __import__("decimal").Decimal("1"), __import__("decimal").Decimal("1"), timedelta(seconds=5)),
    )
    ledger = Ledger(tmp_path / "ledger.db")
    service = GatewayService(
        prompts=PromptLibrary(()),
        admission=AdmissionGate(),
        router=Router(specs, HealthBoard()),
        registry=AdapterRegistry(specs, {"openai": scripted}),
        health=HealthBoard(),
        ledger=ledger,
        metrics=Metrics(),
        clock=lambda: datetime.now(timezone.utc),
        sleeper=lambda delay: _noop(),
        rng=lambda: 0.0,
        max_attempts=1,
        repair_max_attempts=0,
        max_tokens_cap=128,
    )
    key = APIKey("demo", b"", "demo", "active", 10, 2, ())
    invocation = Invocation(
        call_id="call",
        trace_id="trace",
        requested_model="demo",
        messages=(Message("user", "hi"),),
        required=frozenset({Capability.STREAMING}),
        max_tokens=None,
        temperature=None,
        json_schema=None,
        tools=(),
        prompt_ref=None,
        timeout=timedelta(seconds=30),
        tenant_id="demo",
        key_id="demo",
    )
    events = [event async for event in service.stream(invocation, key)]
    assert scripted.models == ["primary"]
    assert events[0].kind is EventKind.DELTA
    assert events[-1].kind is EventKind.ERROR


async def _noop():
    return None


async def test_structured_output_validates_and_stops_repairing(tmp_path):
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "schema": {
                "type": "object",
                "properties": {"a": {"type": "integer"}},
                "required": ["a"],
            },
        },
    }
    calls = {"n": 0}

    def good(request: httpx.Request) -> httpx.Response:
        return _ok('note {"a": 1}', usage=False)

    app = build_app(tmp_path, httpx.MockTransport(good), repairs=1)
    response = await _post(
        app,
        {
            "model": "demo",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": schema,
        },
    )
    assert response.status_code == 200
    assert json.loads(response.json()["choices"][0]["message"]["content"]) == {"a": 1}

    def always_bad(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _ok("nope", usage=False)

    app = build_app(tmp_path, httpx.MockTransport(always_bad), repairs=1)
    failed = await _post(
        app,
        {
            "model": "demo",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": schema,
        },
    )
    assert failed.status_code == 502
    assert failed.json()["error"]["code"] == "output_invalid"
    assert calls["n"] == 2


async def test_prompt_render_rejects_missing_variables_before_upstream(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        body = json.loads(request.content)
        calls["system"] = body["messages"][0]["content"]
        return _ok()

    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "demo.yaml").write_text(
        """
name: demo
version: "1"
status: published
system: "You are {{ role }}."
variables:
  type: object
  required: [role]
  properties:
    role:
      type: string
      maxLength: 20
few_shot: []
default_model: demo
context_budget: 100
schema_version: "1"
""",
        encoding="utf-8",
    )
    from app.services.prompts import load_prompts

    library = load_prompts(prompts)
    app = build_app(tmp_path, httpx.MockTransport(handler), prompts=library)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway.test") as client:
        rejected = await client.post(
            "/v1/prompts/render",
            headers={"Authorization": "Bearer gateway-demo-secret"},
            json={"name": "demo", "version": "1", "variables": {}},
        )
        accepted = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer gateway-demo-secret"},
            json={
                "model": "demo",
                "prompt": {"name": "demo", "version": "1", "variables": {"role": "guide"}},
                "messages": [{"role": "user", "content": "Go"}],
            },
        )
    assert rejected.status_code == 400
    assert calls["n"] == 1
    assert calls["system"] == "You are guide."
    assert accepted.status_code == 200
    call = app.state.ledger.calls_by_trace(accepted.headers["x-request-id"])[0]
    assert call["prompt_name"] == "demo"
    assert call["prompt_hash"]


class HoldTransport(httpx.AsyncBaseTransport):
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.started.set()
        await self.release.wait()
        return _ok()


async def test_concurrency_limit_beats_a_free_rate_bucket(tmp_path):
    transport = HoldTransport()
    app = build_app(tmp_path, transport, concurrency=1)
    client_transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=client_transport, base_url="http://gateway.test") as client:
        headers = {"Authorization": "Bearer gateway-demo-secret"}
        payload = {"model": "demo", "messages": [{"role": "user", "content": "hi"}]}
        first = asyncio.create_task(client.post("/v1/chat/completions", headers=headers, json=payload))
        await transport.started.wait()
        second = await client.post("/v1/chat/completions", headers=headers, json=payload)
        assert second.status_code == 429
        assert second.json()["error"]["message"] == "concurrency limit exceeded"
        transport.release.set()
        assert (await first).status_code == 200


async def test_failed_prompt_call_keeps_prompt_and_resolved_model(tmp_path):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "demo.yaml").write_text(
        """
name: demo
version: "1"
status: published
system: "You are {{ role }}."
variables:
  type: object
  required: [role]
  properties:
    role:
      type: string
      maxLength: 20
few_shot: []
default_model: demo
context_budget: 100
schema_version: "1"
""",
        encoding="utf-8",
    )
    from app.services.prompts import load_prompts

    app = build_app(
        tmp_path,
        httpx.MockTransport(lambda request: httpx.Response(500, text="upstream down")),
        retries=1,
        prompts=load_prompts(prompts),
    )
    response = await _post(
        app,
        {
            "model": "demo",
            "prompt": {"name": "demo", "version": "1", "variables": {"role": "guide"}},
            "messages": [{"role": "user", "content": "Go"}],
        },
    )
    assert response.status_code == 502
    call = app.state.ledger.calls_by_trace(response.headers["x-request-id"])[0]
    assert call["prompt_name"] == "demo"
    assert call["prompt_version"] == "1"
    assert call["prompt_hash"]
    assert call["schema_version"] == "1"
    assert call["provider"] == "openai"
    assert call["resolved_model"] == "upstream-demo"
    assert call["requested_model"] == "demo"
