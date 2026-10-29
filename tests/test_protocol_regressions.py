"""Exercise complete northbound/upstream protocol round trips."""

import json
from decimal import Decimal

import httpx
import pytest

from tests.conftest import build_app
from tests.test_gateway_api import _post, _ok

BASE = {"model": "demo", "messages": [{"role": "user", "content": "hi"}]}
TOOLS = [
    {
        "type": "function",
        "function": {"name": "lookup", "parameters": {"type": "object"}},
    }
]
SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
FORMAT = {"type": "json_schema", "json_schema": {"name": "answer", "schema": SCHEMA}}


def _stream_response(chunks, done=True):
    raw = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
    return httpx.Response(
        200,
        text=raw + ("data: [DONE]\n\n" if done else ""),
        headers={"x-request-id": "stream-provider"},
    )


def _call(app, response):
    return app.state.ledger.calls_by_trace(response.headers["x-request-id"])[0]


def _chunks(response):
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


@pytest.mark.parametrize(
    "choice",
    ["none", "auto", "required", {"type": "function", "function": {"name": "lookup"}}],
)
async def test_tool_choice_and_tool_history_round_trip(tmp_path, choice):
    seen = []
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"q":"hi"}'},
    }

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if body["messages"][-1]["role"] == "tool":
            return _ok("found")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": None, "tool_calls": [call]},
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    app = build_app(tmp_path, httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        response = await _post(app, {**BASE, "tools": TOOLS, "tool_choice": choice})
        assert response.status_code == 200
        message = response.json()["choices"][0]["message"]
        assert message == {"role": "assistant", "content": None, "tool_calls": [call]}
        assert seen[0]["tool_choice"] == choice
        response = await _post(
            app,
            {
                **BASE,
                "messages": BASE["messages"]
                + [
                    message,
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": "result",
                    },
                ],
            },
        )
        assert response.status_code == 200
        assert seen[1]["messages"][-2]["tool_calls"] == [call]
        assert seen[1]["messages"][-1]["tool_call_id"] == "call_1"


@pytest.mark.parametrize(
    "updates",
    [
        {"tool_choice": "invalid", "tools": TOOLS},
        {"tool_choice": "required"},
        {
            "tool_choice": {"type": "function", "function": {"name": "unknown"}},
            "tools": TOOLS,
        },
        {"messages": [{"role": "tool", "content": "result"}]},
        {"messages": [{"role": "user", "content": None}]},
    ],
)
async def test_invalid_tool_requests_do_not_reach_provider(tmp_path, updates):
    def handler(request):
        raise AssertionError("invalid request reached upstream")

    app = build_app(tmp_path, httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        assert (await _post(app, {**BASE, **updates})).status_code == 400


async def test_stream_keeps_usage_and_finish_reason(tmp_path):
    upstream = [
        {"choices": [{"delta": {"content": "hello"}}]},
        {"choices": [{"delta": {}, "finish_reason": "length"}]},
        {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 50}},
    ]
    app = build_app(
        tmp_path, httpx.MockTransport(lambda request: _stream_response(upstream))
    )
    async with app.router.lifespan_context(app):
        response = await _post(app, {**BASE, "stream": True})
        assert response.status_code == 200
        assert _chunks(response)[-1]["choices"][0]["finish_reason"] == "length"
        assert response.text.count("data: [DONE]") == 1
        call = _call(app, response)
        assert (call["input_tokens"], call["output_tokens"], call["estimated"]) == (
            100,
            50,
            0,
        )
        attempt = app.state.ledger.attempts_for(call["call_id"])[0]
        assert attempt["finish_reason"] == "length"
        assert attempt["provider_request_id"] == "stream-provider"


async def test_parallel_tool_deltas_keep_all_indexes_and_fragments(tmp_path):
    upstream = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c0",
                                "function": {"name": "lookup", "arguments": '{"q":'},
                            },
                            {
                                "index": 1,
                                "id": "c1",
                                "function": {"name": "lookup", "arguments": '{"q":'},
                            },
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '"a"}'}},
                            {"index": 1, "function": {"arguments": '"b"}'}},
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]
    app = build_app(
        tmp_path, httpx.MockTransport(lambda request: _stream_response(upstream))
    )
    async with app.router.lifespan_context(app):
        response = await _post(app, {**BASE, "stream": True, "tools": TOOLS})
        calls = [
            c["choices"][0]["delta"]["tool_calls"][0]
            for c in _chunks(response)
            if c["choices"][0]["delta"].get("tool_calls")
        ]
        assert [c["index"] for c in calls] == [0, 1, 0, 1]
        assert "id" not in calls[2]
        assert json.loads(
            calls[0]["function"]["arguments"] + calls[2]["function"]["arguments"]
        ) == {"q": "a"}
        assert _chunks(response)[-1]["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.parametrize(
    "ending",
    [
        [],
        [{"choices": [{"delta": {}, "finish_reason": "stop"}]}],
        [{"error": {"message": "failure"}}],
        ["malformed"],
    ],
)
async def test_partial_stream_failure_has_one_error_and_no_success(tmp_path, ending):
    prefix = 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
    suffix = "".join(
        "data: " + (item if isinstance(item, str) else json.dumps(item)) + "\n\n"
        for item in ending
    )
    app = build_app(
        tmp_path,
        httpx.MockTransport(lambda request: httpx.Response(200, text=prefix + suffix)),
    )
    async with app.router.lifespan_context(app):
        response = await _post(app, {**BASE, "stream": True})
        assert response.status_code == 200
        assert response.text.count('"error":') == 1
        assert "[DONE]" not in response.text
        call = _call(app, response)
        assert call["terminal"] == "error"
        attempts = app.state.ledger.attempts_for(call["call_id"])
        assert len(attempts) == 1 and attempts[0]["terminal"] == "error"
        assert attempts[0]["estimated"]
        assert call["cost_total"] == attempts[0]["cost_total"]


async def test_empty_incomplete_stream_is_an_http_error(tmp_path):
    app = build_app(
        tmp_path,
        httpx.MockTransport(lambda request: httpx.Response(200, text="")),
        retries=1,
    )
    async with app.router.lifespan_context(app):
        response = await _post(app, {**BASE, "stream": True})
        assert response.status_code == 502
        assert _call(app, response)["terminal"] == "error"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("repaired", [True, False])
async def test_repairs_have_unique_attempts_and_aggregate_all_costs(
    tmp_path, stream, repaired
):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if body.get("stream"):
            return _stream_response(
                [
                    {"choices": [{"delta": {"content": "invalid"}}]},
                    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                    {
                        "choices": [],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                    },
                ]
            )
        content = '{"a":1}' if len(requests) == 2 and repaired else "invalid"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 3 if len(requests) == 1 else 10,
                    "completion_tokens": 1 if len(requests) == 1 else 20,
                },
            },
        )

    app = build_app(tmp_path, httpx.MockTransport(handler), repairs=1)
    async with app.router.lifespan_context(app):
        response = await _post(
            app, {**BASE, "stream": stream, "response_format": FORMAT}
        )
        assert response.status_code == (200 if repaired else 502)
        call = _call(app, response)
        attempts = app.state.ledger.attempts_for(call["call_id"])
        assert len(requests) == 2
        assert [a["attempt"] for a in attempts] == [1, 2]
        assert [a["kind"] for a in attempts] == ["generation", "repair"]
        assert attempts[0]["terminal"] == "error" and attempts[0]["output_hash"]
        assert Decimal(call["cost_total"]) == sum(
            Decimal(a["cost_total"]) for a in attempts
        )
        assert (call["input_tokens"], call["output_tokens"]) == (13, 21)
        assert call["terminal"] == ("completed" if repaired else "error")


async def test_zero_repairs_does_not_call_provider_again(tmp_path):
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        return _ok("invalid")

    app = build_app(tmp_path, httpx.MockTransport(handler), repairs=0)
    async with app.router.lifespan_context(app):
        response = await _post(app, {**BASE, "response_format": FORMAT})
        assert response.status_code == 502 and count == 1
        call = _call(app, response)
        assert len(app.state.ledger.attempts_for(call["call_id"])) == 1
        assert Decimal(call["cost_total"]) > 0


async def test_repair_transport_failure_does_not_restart_generation(tmp_path):
    seen = 0

    def handler(request):
        nonlocal seen
        seen += 1
        return (
            _ok("invalid") if seen == 1 else httpx.Response(500, text="repair failed")
        )

    app = build_app(tmp_path, httpx.MockTransport(handler), retries=2, repairs=1)
    async with app.router.lifespan_context(app):
        response = await _post(app, {**BASE, "response_format": FORMAT})
        assert response.status_code == 502 and seen == 2
        call = _call(app, response)
        attempts = app.state.ledger.attempts_for(call["call_id"])
        assert [a["attempt"] for a in attempts] == [1, 2]
        assert [a["terminal"] for a in attempts] == ["error", "error"]
        assert attempts[1]["estimated"]
        assert Decimal(call["cost_total"]) == sum(
            Decimal(a["cost_total"]) for a in attempts
        )


@pytest.mark.parametrize(
    "payload", [[], {"choices": [None]}, {"choices": [{"message": {"content": {}}}]}]
)
async def test_invalid_provider_payload_is_a_sanitized_502(tmp_path, payload):
    app = build_app(
        tmp_path, httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    )
    async with app.router.lifespan_context(app):
        response = await _post(app, BASE)
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "upstream_error"


async def test_ledger_additive_migration_preserves_old_attempts(tmp_path):
    import sqlite3
    from dataclasses import fields
    from app.services.usage import AttemptRow, Ledger

    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    columns = ", ".join(
        f"{field.name} TEXT"
        for field in fields(AttemptRow)
        if field.name not in {"kind", "output_hash"}
    )
    connection.execute(f"CREATE TABLE attempts ({columns})")
    connection.execute(
        "INSERT INTO attempts(call_id, attempt) VALUES ('old-call', '1')"
    )
    connection.commit()
    connection.close()
    ledger = Ledger(path)
    try:
        old = ledger.attempts_for("old-call")[0]
        assert old["kind"] == "generation" and old["output_hash"] == ""
        assert old["call_id"] == "old-call"
    finally:
        ledger.close()
