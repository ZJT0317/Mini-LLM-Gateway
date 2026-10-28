import json
from decimal import Decimal
from datetime import timedelta

import httpx

from app.core.models import Capability, Message, ModelSpec, ToolSpec, UpstreamCall
from app.services.router import HealthBoard, Router
from app.services.upstream import OpenAIAdapter, redact


def _spec(name: str, priority: int, capabilities: set[Capability], cost: str = "1") -> ModelSpec:
    return ModelSpec(
        logical_model="demo",
        provider="openai",
        resolved_model=name,
        base_url="http://127.0.0.1:9/v1",
        capabilities=frozenset(capabilities),
        priority=priority,
        input_cost=Decimal(cost),
        output_cost=Decimal(cost),
        timeout=timedelta(seconds=5),
    )


def test_router_rejects_missing_capabilities_and_keeps_fallback_order():
    specs = (
        _spec("small", 2, {Capability.STREAMING}, "1"),
        _spec("tools", 1, {Capability.STREAMING, Capability.TOOL_CALLING}, "9"),
        _spec("cheap-tools", 1, {Capability.STREAMING, Capability.TOOL_CALLING}, "1"),
    )
    decision = Router(specs, HealthBoard()).decide("demo", frozenset({Capability.TOOL_CALLING}))
    assert decision.selected is not None
    assert decision.selected.resolved_model == "cheap-tools"
    assert [item.resolved_model for item in decision.candidates] == ["cheap-tools", "tools"]
    reasons = {(item.resolved_model, item.reason) for item in decision.rejections}
    assert ("small", "capability_mismatch") in reasons
    assert ("tools", "priority_lost") in reasons


def test_unhealthy_candidates_are_recorded():
    health = HealthBoard(threshold=1)
    spec = _spec("small", 1, {Capability.STREAMING})
    health.record(spec, success=False)
    decision = Router((spec,), health).decide("demo", frozenset())
    assert decision.selected is None
    assert decision.rejections[0].reason == "unhealthy"


def test_health_opens_on_the_next_decision_after_three_failures():
    health = HealthBoard()
    spec = _spec("small", 1, {Capability.STREAMING})
    health.record(spec, success=False)
    health.record(spec, success=False)
    assert Router((spec,), health).decide("demo", frozenset()).selected == spec
    health.record(spec, success=False)
    decision = Router((spec,), health).decide("demo", frozenset())
    assert decision.selected is None
    assert decision.rejections[0].reason == "unhealthy"
    health.record(spec, success=True)
    assert Router((spec,), health).decide("demo", frozenset()).selected == spec


def test_redact_removes_bearer_and_provider_keys():
    assert "sk-live-secret" not in redact("boom sk-live-secret tail")
    assert "gateway-secret" not in redact("Bearer gateway-secret")


async def test_adapter_translates_chat_and_hides_the_upstream_key():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        if seen["body"].get("stream"):
            lines = [
                'data: {"choices":[{"delta":{"content":"hi"}}]}',
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1}}',
                "data: [DONE]",
            ]
            return httpx.Response(200, text="\n".join(lines) + "\n")
        return httpx.Response(
            200,
            headers={"x-request-id": "prov-1"},
            json={
                "id": "prov-1",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            },
        )

    adapter = OpenAIAdapter(
        "openai",
        "sk-upstream-secret",
        frozenset(Capability),
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    call = UpstreamCall(
        requested_model="demo",
        resolved_model="upstream-demo",
        base_url="http://127.0.0.1:9/v1",
        messages=(Message("user", "hi"),),
        max_tokens=8,
        temperature=0,
        json_schema=None,
        tools=(ToolSpec("lookup", "find", {"type": "object"}),),
        timeout=timedelta(seconds=5),
    )
    outcome = await adapter.invoke(call)
    assert outcome.content == "hello"
    assert outcome.usage is not None and outcome.usage.input_tokens == 4
    assert seen["body"]["model"] == "upstream-demo"
    assert seen["body"]["tools"][0]["function"]["name"] == "lookup"
    assert "sk-upstream-secret" not in str(outcome)

    events = [event async for event in adapter.stream(call)]
    kinds = [event.kind.value for event in events]
    assert kinds[0] == "delta"
    assert "usage" in kinds
    assert kinds[-1] == "done"
    assert seen["auth"] == "Bearer sk-upstream-secret"
    assert "sk-upstream-secret" not in outcome.content


async def test_upstream_failure_body_is_not_part_of_the_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="provider blew up sk-upstream-secret")

    adapter = OpenAIAdapter(
        "openai",
        "sk-upstream-secret",
        frozenset({Capability.STREAMING}),
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    call = UpstreamCall(
        "demo",
        "upstream-demo",
        "http://127.0.0.1:9/v1",
        (Message("user", "hi"),),
        None,
        None,
        None,
        (),
        timedelta(seconds=5),
    )
    try:
        await adapter.invoke(call)
    except Exception as exc:
        text = str(exc)
    else:
        raise AssertionError("expected an error")
    assert "sk-upstream-secret" not in text
    assert text == "upstream request failed"
