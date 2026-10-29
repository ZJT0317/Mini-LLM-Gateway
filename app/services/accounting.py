"""Build ledger records, estimate usage, and sum costs across all attempts."""

from __future__ import annotations

import time
from decimal import Decimal

from app.core.errors import Code, GatewayError, http_status
from app.core.models import (
    Rejection,
    Terminal,
    TokenUsage,
    UnifiedResult,
    estimate_tokens,
    price_usage,
)
from app.services.execution import AttemptState, CallState
from app.services.metrics import Metrics
from app.services.usage import AttemptRow, CallRow, Ledger


def attempt_usage(attempt: AttemptState) -> TokenUsage:
    """Use provider usage, or mark a local approximation explicitly."""
    if attempt.usage is not None:
        return attempt.usage
    if not attempt.received:
        return TokenUsage(0, 0, estimated=True)
    return estimate_tokens(
        attempt.messages,
        attempt.content
        + attempt.refusal
        + "".join(tool.name + tool.arguments for tool in attempt.tools.values()),
    )


class CallRecorder:
    """Persist attempt details and one call summary; execution owns the lifecycle."""

    def __init__(self, ledger: Ledger, metrics: Metrics) -> None:
        self._ledger = ledger
        self._metrics = metrics

    def record_rejections(
        self, call_id: str, rejections: tuple[Rejection, ...]
    ) -> None:
        for rejected in rejections:
            self._ledger.save_rejection(
                call_id, rejected.provider, rejected.resolved_model, rejected.reason
            )

    def record_attempt(
        self, run: CallState, terminal: Terminal, error_code: str
    ) -> None:
        attempt = run.active
        if attempt is None:
            return
        usage = attempt_usage(attempt)
        cost = price_usage(usage, attempt.spec)
        elapsed = (time.monotonic() - attempt.started) * 1000
        row = AttemptRow(
            call_id=run.invocation.call_id,
            attempt=attempt.number,
            provider=attempt.spec.provider,
            resolved_model=attempt.spec.resolved_model,
            is_retry=attempt.is_retry,
            is_fallback=attempt.is_fallback,
            queue_ms=run.queue_ms,
            route_ms=run.route_ms,
            ttft_ms=attempt.ttft_ms,
            generation_ms=None
            if attempt.ttft_ms is None
            else max(0, elapsed - attempt.ttft_ms),
            total_ms=elapsed,
            timeout_budget_ms=attempt.budget_ms,
            finish_reason=attempt.finish_reason,
            terminal=terminal.value,
            error_code=error_code,
            provider_request_id=attempt.provider_request_id,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_tokens=usage.cached_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            estimated=usage.estimated,
            cost_input=str(cost.input),
            cost_output=str(cost.output),
            cost_cache=str(cost.cache),
            cost_total=str(cost.total),
            kind=attempt.kind,
            output_hash=""
            if attempt.validation is None
            else attempt.validation.text_hash,
        )
        self._ledger.save_attempt(row)
        run.attempts.append(row)
        run.active = None

    def record_call(
        self,
        run: CallState,
        terminal: Terminal,
        *,
        result: UnifiedResult | None = None,
        error: GatewayError | None = None,
    ) -> None:
        if run.saved:
            return
        invocation = run.invocation
        prompt = None if run.prepared is None else run.prepared.prompt
        prompt_ref = invocation.prompt_ref
        spec = run.spec
        rows = run.attempts
        if terminal is Terminal.CANCELED:
            status = 499
        elif result is not None or run.sent:
            status = 200
        elif error is not None:
            status = http_status(error)[0]
        else:
            raise ValueError("an unsuccessful call requires an error")
        self._ledger.save_call(
            CallRow(
                call_id=invocation.call_id,
                trace_id=invocation.trace_id,
                run_id=invocation.run_id,
                step_id=invocation.step_id,
                requested_model=invocation.requested_model,
                provider="" if spec is None else spec.provider,
                resolved_model="" if spec is None else spec.resolved_model,
                tenant_id=invocation.tenant_id,
                key_id=invocation.key_id,
                prompt_name=prompt.name
                if prompt
                else prompt_ref.name
                if prompt_ref
                else "",
                prompt_version=prompt.version
                if prompt
                else prompt_ref.version
                if prompt_ref
                else "",
                prompt_hash="" if prompt is None else prompt.content_hash,
                schema_version="" if prompt is None else prompt.schema_version,
                terminal=terminal.value,
                http_status=status,
                error_code=Code.CANCELED.value
                if terminal is Terminal.CANCELED
                else Code.CONTENT_REJECTED.value
                if terminal is Terminal.REFUSED
                else ""
                if error is None
                else error.code.value,
                finish_reason="" if result is None else result.finish_reason,
                validation_ok=(
                    int(result.validation.ok)
                    if result is not None and result.validation is not None
                    else 0
                    if error is not None and error.code is Code.OUTPUT_INVALID
                    else None
                ),
                ttft_ms=run.ttft_ms,
                total_ms=(time.monotonic() - run.started) * 1000,
                input_tokens=sum(row.input_tokens for row in rows),
                output_tokens=sum(row.output_tokens for row in rows),
                cached_tokens=sum(row.cached_tokens for row in rows),
                reasoning_tokens=sum(row.reasoning_tokens for row in rows),
                estimated=any(row.estimated for row in rows),
                cost_total=str(
                    sum((Decimal(row.cost_total) for row in rows), Decimal(0))
                ),
            )
        )
        run.saved = True
        self._metrics.observe(
            invocation.requested_model,
            "" if spec is None else spec.provider,
            terminal.value,
            (time.monotonic() - run.started) * 1000,
        )
