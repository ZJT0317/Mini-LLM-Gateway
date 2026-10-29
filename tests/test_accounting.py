"""Accounting can be verified without running gateway orchestration or HTTP."""

from collections.abc import Iterator
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
import sqlite3

import pytest

from app.core.errors import Code, GatewayError, Stage
from app.core.models import (
    Invocation,
    Message,
    ModelSpec,
    PromptRef,
    Terminal,
    TokenUsage,
    ToolCall,
    UnifiedResult,
)
from app.services.accounting import CallRecorder, attempt_usage
from app.services.execution import AttemptState, CallState
from app.services.metrics import Metrics
from app.services.usage import Ledger


@pytest.fixture
def accounting(tmp_path: Path) -> Iterator[tuple[CallRecorder, Ledger, Metrics]]:
    ledger = Ledger(tmp_path / "accounting.db")
    metrics = Metrics()
    try:
        yield CallRecorder(ledger, metrics), ledger, metrics
    finally:
        ledger.close()


def _run(call_id: str = "call") -> CallState:
    return CallState(
        Invocation(
            call_id=call_id,
            trace_id="trace",
            requested_model="demo",
            messages=(Message("user", "hello"),),
            required=frozenset(),
            max_tokens=128,
            temperature=None,
            json_schema=None,
            tools=(),
            prompt_ref=None,
            timeout=timedelta(seconds=30),
            tenant_id="tenant",
            key_id="key",
        )
    )


def _attempt(run: CallState, model: str = "primary") -> AttemptState:
    spec = ModelSpec(
        logical_model="demo",
        provider="provider",
        resolved_model=model,
        base_url="http://127.0.0.1:9/v1",
        capabilities=frozenset(),
        priority=1,
        input_cost=Decimal("1"),
        output_cost=Decimal("2"),
        timeout=timedelta(seconds=10),
    )
    run.spec = spec
    run.active = AttemptState(
        spec=spec,
        number=len(run.attempts) + 1,
        started=run.started,
        messages=run.invocation.messages,
        kind="generation",
        is_retry=False,
        is_fallback=False,
        budget_ms=10000,
    )
    return run.active


def test_all_attempt_costs_do_not_change_the_final_response_usage(accounting):
    recorder, ledger, metrics = accounting
    run = _run()
    first = _attempt(run)
    first.usage = TokenUsage(100, 10, cached_tokens=20)
    recorder.record_attempt(run, Terminal.ERROR, Code.UPSTREAM_ERROR.value)

    second = _attempt(run, "fallback")
    second.kind = "repair"
    second.is_fallback = True
    second.usage = TokenUsage(50, 5, reasoning_tokens=2)
    result = UnifiedResult(
        "demo",
        "fallback",
        "provider",
        "ok",
        (),
        "stop",
        Terminal.COMPLETED,
        second.usage,
        None,
        None,
    )
    recorder.record_attempt(run, Terminal.COMPLETED, "")
    recorder.record_call(run, Terminal.COMPLETED, result=result)
    # Repeated cleanup must not bill an attempt or count a call twice.
    recorder.record_attempt(run, Terminal.COMPLETED, "")
    recorder.record_call(run, Terminal.COMPLETED, result=result)

    rows = ledger.calls_by_trace("trace")
    assert len(rows) == 1
    call = rows[0]
    assert (call["input_tokens"], call["output_tokens"]) == (150, 15)
    assert (call["cached_tokens"], call["reasoning_tokens"]) == (20, 2)
    assert Decimal(call["cost_total"]) == Decimal("0.00018")
    assert call["resolved_model"] == "fallback"
    assert call["estimated"] == 0
    assert [row["attempt"] for row in ledger.attempts_for("call")] == [1, 2]
    assert result.usage == TokenUsage(50, 5, reasoning_tokens=2)
    assert metrics.requests == {("demo", "provider", "completed"): 1}


def test_shared_recorder_keeps_interleaved_call_costs_separate(accounting):
    recorder, ledger, _ = accounting
    first, second = _run("first"), _run("second")
    _attempt(first).usage = TokenUsage(10, 1)
    _attempt(second).usage = TokenUsage(20, 2)
    recorder.record_attempt(second, Terminal.CANCELED, Code.CANCELED.value)
    recorder.record_attempt(first, Terminal.CANCELED, Code.CANCELED.value)
    recorder.record_call(first, Terminal.CANCELED)
    recorder.record_call(second, Terminal.CANCELED)
    calls = {row["call_id"]: row for row in ledger.calls_by_trace("trace")}
    assert Decimal(calls["first"]["cost_total"]) == Decimal("0.000012")
    assert Decimal(calls["second"]["cost_total"]) == Decimal("0.000024")
    assert all(row["http_status"] == 499 for row in calls.values())


def test_partial_refusal_and_tool_arguments_are_included_in_estimated_usage(accounting):
    recorder, ledger, _ = accounting
    run = _run()
    attempt = _attempt(run)
    assert attempt_usage(attempt) == TokenUsage(0, 0, estimated=True)
    attempt.received = True
    attempt.content = "abcd"
    attempt.refusal = "deny"
    attempt.tools = {0: ToolCall("id", "echo", '{"x":1}')}
    usage = attempt_usage(attempt)
    assert usage.estimated and usage.input_tokens > 0
    assert usage.output_tokens == 5
    recorder.record_attempt(run, Terminal.CANCELED, Code.CANCELED.value)
    recorder.record_call(run, Terminal.CANCELED)
    call = ledger.calls_by_trace("trace")[0]
    assert call["estimated"] == 1 and call["output_tokens"] == 5
    assert Decimal(call["cost_total"]) > 0


def test_failure_before_preparation_still_records_prompt_reference(accounting):
    recorder, ledger, _ = accounting
    run = _run()
    run.original = replace(run.original, prompt_ref=PromptRef("missing", "v1", {}))
    error = GatewayError(Code.PROMPT_INVALID, "invalid prompt", stage=Stage.PROMPT)
    recorder.record_call(run, Terminal.ERROR, error=error)
    call = ledger.calls_by_trace("trace")[0]
    assert (call["prompt_name"], call["prompt_version"]) == ("missing", "v1")
    assert call["http_status"] == 400 and call["cost_total"] == "0"
    assert ledger.attempts_for("call") == []


def test_failed_persistence_does_not_mark_records_as_saved(accounting, monkeypatch):
    recorder, ledger, metrics = accounting
    run = _run()
    attempt = _attempt(run)

    def fail(_row):
        raise sqlite3.OperationalError("test write failure")

    with monkeypatch.context() as patch:
        patch.setattr(ledger, "save_attempt", fail)
        with pytest.raises(sqlite3.OperationalError):
            recorder.record_attempt(run, Terminal.CANCELED, Code.CANCELED.value)
    assert run.active is attempt and run.attempts == []
    recorder.record_attempt(run, Terminal.CANCELED, Code.CANCELED.value)
    with monkeypatch.context() as patch:
        patch.setattr(ledger, "save_call", fail)
        with pytest.raises(sqlite3.OperationalError):
            recorder.record_call(run, Terminal.CANCELED)
    assert not run.saved and metrics.requests == {}
    recorder.record_call(run, Terminal.CANCELED)
    assert len(ledger.calls_by_trace("trace")) == 1
    assert metrics.requests == {("demo", "provider", "canceled"): 1}
