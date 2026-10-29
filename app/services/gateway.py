"""Execute calls with one lifecycle, deadline, and ledger entry per attempt."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Literal

import anyio

from app.core.admission import AdmissionGate, Admit, Release, budget_is_usable
from app.core.auth import ensure_model_visible
from app.core.errors import Code, GatewayError, Stage
from app.core.models import (
    APIKey,
    AdapterOutcome,
    EventKind,
    GatewayEvent,
    Invocation,
    Message,
    ModelSpec,
    Terminal,
    ToolCall,
    UnifiedResult,
    UpstreamCall,
    ValidationResult,
)
from app.core.retry import backoff, should_fallback, should_retry
from app.services.accounting import CallRecorder, attempt_usage
from app.services.execution import AttemptKind, AttemptState, CallState
from app.services.metrics import Metrics
from app.services.prompts import PromptLibrary
from app.services.router import AdapterRegistry, HealthBoard, Router
from app.services.structured import check_schema, validate_json
from app.services.usage import Ledger

Sleeper = Callable[[timedelta], Awaitable[None]]
Rng = Callable[[], float]
RetryAction = Literal["retry", "fallback"]


@asynccontextmanager
async def _closing(
    iterator: AsyncGenerator[GatewayEvent, None],
) -> AsyncIterator[AsyncGenerator[GatewayEvent, None]]:
    try:
        yield iterator
    finally:
        # Starlette disconnects cancel an AnyIO scope. Finish resource cleanup in it.
        with anyio.CancelScope(shield=True):
            await iterator.aclose()


def _timeout() -> GatewayError:
    return GatewayError(
        Code.UPSTREAM_OVERLOADED, "gateway timeout", stage=Stage.BEFORE_FIRST_TOKEN
    )


class GatewayService:
    def __init__(
        self,
        *,
        prompts: PromptLibrary,
        admission: AdmissionGate,
        router: Router,
        registry: AdapterRegistry,
        health: HealthBoard,
        ledger: Ledger,
        metrics: Metrics,
        clock: Callable[[], datetime],
        sleeper: Sleeper,
        rng: Rng,
        max_attempts: int,
        repair_max_attempts: int,
        max_tokens_cap: int,
    ) -> None:
        self._prompts = prompts
        self._admission = admission
        self._router = router
        self._registry = registry
        self._health = health
        self._recorder = CallRecorder(ledger, metrics)
        self._clock = clock
        self._sleeper = sleeper
        self._rng = rng
        self._max_attempts = max_attempts
        self._repair_max_attempts = repair_max_attempts
        self._max_tokens_cap = max_tokens_cap

    async def complete(self, invocation: Invocation, key: APIKey) -> UnifiedResult:
        run = CallState(invocation)
        release = None
        try:
            async with asyncio.timeout_at(run.deadline):
                release = await self._open(run, key)
            result = await self._complete(run)
            self._recorder.record_call(run, result.terminal, result=result)
            return result
        except TimeoutError as exc:
            error = _timeout()
            self._recorder.record_call(run, Terminal.ERROR, error=error)
            raise error from exc
        except GatewayError as exc:
            self._recorder.record_call(run, Terminal.ERROR, error=exc)
            raise
        except (asyncio.CancelledError, GeneratorExit):
            self._recorder.record_call(run, Terminal.CANCELED)
            raise
        except Exception as exc:
            error = GatewayError(
                Code.INTERNAL, "internal error", stage=Stage.OUTPUT, cause=exc
            )
            self._recorder.record_call(run, Terminal.ERROR, error=error)
            raise error from exc
        finally:
            if release is not None:
                with anyio.CancelScope(shield=True):
                    await release()

    async def stream(
        self, invocation: Invocation, key: APIKey
    ) -> AsyncGenerator[GatewayEvent, None]:
        run = CallState(invocation)
        release = None
        try:
            async with asyncio.timeout_at(run.deadline):
                release = await self._open(run, key)
            async with _closing(self._stream(run)) as events:
                async for event in events:
                    yield replace(event, requested_model=run.invocation.requested_model)
        except TimeoutError:
            error = _timeout()
            self._recorder.record_call(run, Terminal.ERROR, error=error)
            yield GatewayEvent(kind=EventKind.ERROR, error=error)
        except GatewayError as exc:
            self._recorder.record_call(run, Terminal.ERROR, error=exc)
            yield GatewayEvent(kind=EventKind.ERROR, error=exc)
        except (asyncio.CancelledError, GeneratorExit):
            self._recorder.record_call(run, Terminal.CANCELED)
            raise
        except Exception as exc:
            error = GatewayError(
                Code.INTERNAL, "internal error", stage=Stage.OUTPUT, cause=exc
            )
            self._recorder.record_call(run, Terminal.ERROR, error=error)
            yield GatewayEvent(kind=EventKind.ERROR, error=error)
        finally:
            if release is not None:
                with anyio.CancelScope(shield=True):
                    await release()

    async def _open(self, run: CallState, key: APIKey) -> Release | None:
        run.prepared = self._prompts.prepare(run.original)
        invocation = run.invocation
        ensure_model_visible(key, invocation.requested_model)
        if invocation.max_tokens is None:
            invocation = replace(invocation, max_tokens=self._max_tokens_cap)
            run.prepared = replace(run.prepared, invocation=invocation)
        if (
            isinstance(invocation.max_tokens, bool)
            or not isinstance(invocation.max_tokens, int)
            or not 1 <= invocation.max_tokens <= self._max_tokens_cap
        ):
            raise GatewayError(
                Code.INVALID_REQUEST, "max_tokens exceeds cap", stage=Stage.VALIDATION
            )
        if invocation.temperature is not None and (
            isinstance(invocation.temperature, bool)
            or not isinstance(invocation.temperature, (int, float))
            or not 0 <= invocation.temperature <= 2
        ):
            raise GatewayError(
                Code.INVALID_REQUEST, "invalid temperature", stage=Stage.VALIDATION
            )
        if invocation.json_schema is not None:
            check_schema(invocation.json_schema)
        for tool in invocation.tools:
            check_schema(tool.parameters)
        choice = invocation.tool_choice
        if choice is not None:
            if not invocation.tools:
                raise GatewayError(
                    Code.INVALID_REQUEST,
                    "tool_choice requires tools",
                    stage=Stage.VALIDATION,
                )
            if isinstance(choice, dict) and choice["function"]["name"] not in {
                t.name for t in invocation.tools
            }:
                raise GatewayError(
                    Code.INVALID_REQUEST,
                    "tool_choice names an unknown tool",
                    stage=Stage.VALIDATION,
                )
        if not budget_is_usable(invocation.timeout):
            raise GatewayError(
                Code.DEADLINE_TOO_SHORT, "deadline too short", stage=Stage.ADMISSION
            )
        queued = time.monotonic()
        status, release = await self._admission.acquire(
            key.id, key.rpm, key.max_concurrency, self._clock()
        )
        run.queue_ms = (time.monotonic() - queued) * 1000
        if status != Admit.OK:
            message = (
                "rate limit exceeded"
                if status == Admit.RATE
                else "concurrency limit exceeded"
            )
            raise GatewayError(Code.ADMISSION_REJECTED, message, stage=Stage.ADMISSION)
        return release

    def _route(self, run: CallState) -> tuple[ModelSpec, ...]:
        started = time.monotonic()
        decision = self._router.decide(
            run.invocation.requested_model, run.invocation.required
        )
        run.route_ms = (time.monotonic() - started) * 1000
        self._recorder.record_rejections(run.invocation.call_id, decision.rejections)
        if not decision.candidates:
            if any(item.reason == "unhealthy" for item in decision.rejections):
                raise GatewayError(
                    Code.MODEL_UNAVAILABLE,
                    "model temporarily unavailable",
                    stage=Stage.ROUTING,
                )
            raise GatewayError(
                Code.NO_COMPATIBLE_MODEL, "model not found", stage=Stage.ROUTING
            )
        return decision.candidates

    @asynccontextmanager
    async def _attempt(
        self,
        run: CallState,
        spec: ModelSpec,
        *,
        kind: AttemptKind = "generation",
        is_retry: bool = False,
        is_fallback: bool = False,
        messages: tuple[Message, ...] | None = None,
    ) -> AsyncIterator[AttemptState]:
        run.spec = spec
        attempt = AttemptState(
            spec,
            len(run.attempts) + 1,
            time.monotonic(),
            run.invocation.messages if messages is None else messages,
            kind,
            is_retry,
            is_fallback,
            max(0, min(run.deadline - time.monotonic(), spec.timeout.total_seconds()))
            * 1000,
        )
        run.active = attempt
        try:
            yield attempt
        except (asyncio.CancelledError, GeneratorExit):
            self._health.release(spec)
            self._recorder.record_attempt(run, Terminal.CANCELED, Code.CANCELED.value)
            raise
        except TimeoutError as exc:
            self._health.record(spec, success=False)
            self._recorder.record_attempt(
                run, Terminal.ERROR, Code.UPSTREAM_OVERLOADED.value
            )
            raise _timeout() from exc
        except GatewayError as exc:
            attempt.provider_request_id = (
                exc.provider_request_id or attempt.provider_request_id
            )
            if exc.code in {
                Code.UPSTREAM_UNAVAILABLE,
                Code.UPSTREAM_OVERLOADED,
                Code.UPSTREAM_ERROR,
            }:
                self._health.record(spec, success=False)
            else:
                self._health.release(spec)
            self._recorder.record_attempt(
                run,
                Terminal.REFUSED if attempt.refusal else Terminal.ERROR,
                Code.CONTENT_REJECTED.value if attempt.refusal else exc.code.value,
            )
            raise
        except Exception as exc:
            self._health.release(spec)
            self._recorder.record_attempt(run, Terminal.ERROR, Code.INTERNAL.value)
            raise GatewayError(
                Code.INTERNAL, "internal error", stage=Stage.OUTPUT, cause=exc
            ) from exc
        else:
            self._health.record(spec, success=True)
            invalid = attempt.validation is not None and not attempt.validation.ok
            self._recorder.record_attempt(
                run,
                Terminal.REFUSED
                if attempt.refusal
                else Terminal.ERROR
                if invalid
                else Terminal.COMPLETED,
                Code.CONTENT_REJECTED.value
                if attempt.refusal
                else Code.OUTPUT_INVALID.value
                if invalid
                else "",
            )

    async def _invoke(
        self,
        run: CallState,
        spec: ModelSpec,
        *,
        kind: AttemptKind = "generation",
        is_retry: bool = False,
        is_fallback: bool = False,
        messages: tuple[Message, ...] | None = None,
    ) -> tuple[AdapterOutcome, ValidationResult | None, str]:
        async with self._attempt(
            run,
            spec,
            kind=kind,
            is_retry=is_retry,
            is_fallback=is_fallback,
            messages=messages,
        ) as attempt:
            async with asyncio.timeout_at(run.deadline):
                outcome = await self._registry.adapter_for(spec.provider).invoke(
                    self._upstream_call(run, spec, attempt.messages)
                )
            attempt.received = True
            attempt.content = outcome.content
            attempt.tools = {tool.index: tool for tool in outcome.tool_calls}
            attempt.usage = outcome.usage
            attempt.finish_reason = outcome.finish_reason
            attempt.provider_request_id = outcome.provider_request_id or ""
            attempt.refusal = outcome.refusal or ""
            canonical = outcome.content
            if (
                run.invocation.json_schema is not None
                and not outcome.tool_calls
                and not outcome.refusal
            ):
                attempt.validation, canonical = validate_json(
                    outcome.content, run.invocation.json_schema
                )
            validation = attempt.validation
            usage = attempt_usage(attempt)
            outcome = replace(outcome, usage=usage)
        return outcome, validation, canonical

    async def _repair(
        self, run: CallState, spec: ModelSpec, content: str
    ) -> tuple[AdapterOutcome, ValidationResult | None, str]:
        for _ in range(self._repair_max_attempts):
            if not self._health.acquire(spec):
                raise GatewayError(
                    Code.MODEL_UNAVAILABLE,
                    "model temporarily unavailable",
                    stage=Stage.ROUTING,
                )
            messages = run.invocation.messages + (
                Message("assistant", content),
                Message("user", "Return only JSON that matches the requested schema."),
            )
            outcome, validation, canonical = await self._invoke(
                run, spec, kind="repair", messages=messages
            )
            if outcome.refusal or (validation is not None and validation.ok):
                return outcome, validation, canonical
            content = outcome.content
        raise GatewayError(
            Code.OUTPUT_INVALID, "output validation failed", stage=Stage.OUTPUT
        )

    async def _retry(
        self, run: CallState, error: GatewayError, tries: int, remaining: int
    ) -> RetryAction:
        if time.monotonic() >= run.deadline or error.message == "gateway timeout":
            raise _timeout()
        # A failed repair must not restart generation and reset the repair budget.
        if run.attempts and run.attempts[-1].kind == "repair":
            raise error
        if run.attempts and run.attempts[-1].terminal == Terminal.REFUSED.value:
            raise error
        if should_retry(error.stage, error.code, tries, self._max_attempts, run.sent):
            async with asyncio.timeout_at(run.deadline):
                await self._sleeper(backoff(tries, self._rng()))
            return "retry"
        if should_fallback(error.code, run.sent, remaining):
            return "fallback"
        raise error

    async def _complete(self, run: CallState) -> UnifiedResult:
        candidates = self._route(run)
        for index, spec in enumerate(candidates):
            for tries in range(1, self._max_attempts + 1):
                if not self._health.acquire(spec):
                    break
                try:
                    outcome, validation, content = await self._invoke(
                        run,
                        spec,
                        is_retry=tries > 1,
                        is_fallback=index > 0,
                    )
                    if validation is not None and not validation.ok:
                        outcome, validation, content = await self._repair(
                            run, spec, outcome.content
                        )
                    return self._result(run, spec, outcome, validation, content)
                except GatewayError as error:
                    if (
                        await self._retry(
                            run, error, tries, len(candidates) - index - 1
                        )
                        == "fallback"
                    ):
                        break
        raise GatewayError(
            Code.MODEL_UNAVAILABLE, "model temporarily unavailable", stage=Stage.ROUTING
        )

    async def _stream(self, run: CallState) -> AsyncGenerator[GatewayEvent, None]:
        candidates = self._route(run)
        for index, spec in enumerate(candidates):
            for tries in range(1, self._max_attempts + 1):
                if not self._health.acquire(spec):
                    break
                try:
                    async with self._attempt(
                        run, spec, is_retry=tries > 1, is_fallback=index > 0
                    ) as attempt:
                        source = self._registry.adapter_for(spec.provider).stream(
                            self._upstream_call(run, spec, attempt.messages)
                        )
                        async with _closing(source):
                            while True:
                                # Never keep a task-owned timeout context open across yield.
                                async with asyncio.timeout_at(run.deadline):
                                    try:
                                        event = await anext(source)
                                    except StopAsyncIteration as exc:
                                        raise GatewayError(
                                            Code.UPSTREAM_UNAVAILABLE,
                                            "upstream request failed",
                                            stage=Stage.STREAMING,
                                            retryable=True,
                                        ) from exc
                                attempt.provider_request_id = (
                                    event.provider_request_id
                                    or attempt.provider_request_id
                                )
                                if event.kind is EventKind.ERROR:
                                    if isinstance(event.error, GatewayError):
                                        raise event.error
                                    raise GatewayError(
                                        Code.UPSTREAM_ERROR,
                                        "upstream request failed",
                                        stage=Stage.STREAMING,
                                    )
                                if event.kind is EventKind.USAGE:
                                    attempt.usage = event.usage
                                elif event.kind is EventKind.FINISH:
                                    attempt.finish_reason = event.finish_reason or ""
                                elif event.kind is EventKind.DONE:
                                    attempt.received = True
                                    attempt.finish_reason = (
                                        event.finish_reason
                                        or attempt.finish_reason
                                        or "stop"
                                    )
                                    if (
                                        attempt.finish_reason == "content_filter"
                                        and not attempt.refusal
                                    ):
                                        attempt.refusal = "content rejected"
                                    break
                                elif event.kind in {
                                    EventKind.DELTA,
                                    EventKind.TOOL_CALL,
                                    EventKind.REFUSAL,
                                }:
                                    attempt.received = True
                                    if attempt.ttft_ms is None:
                                        attempt.ttft_ms = (
                                            time.monotonic() - attempt.started
                                        ) * 1000
                                    if event.kind is EventKind.DELTA:
                                        attempt.content += event.text
                                    elif event.kind is EventKind.REFUSAL:
                                        attempt.refusal += event.text
                                    elif event.tool_call is not None:
                                        _merge_tool(attempt.tools, event.tool_call)
                                    if run.invocation.json_schema is None:
                                        self._mark_sent(run)
                                        if event.kind is EventKind.REFUSAL:
                                            attempt.refusal_emitted = True
                                        yield event
                        content = attempt.content
                        if (
                            run.invocation.json_schema is not None
                            and not attempt.tools
                            and not attempt.refusal
                        ):
                            attempt.validation, content = validate_json(
                                content, run.invocation.json_schema
                            )
                        validation = attempt.validation
                        outcome = AdapterOutcome(
                            attempt.content,
                            tuple(attempt.tools.values()),
                            attempt.finish_reason,
                            attempt_usage(attempt),
                            attempt.provider_request_id or None,
                            attempt.refusal or None,
                        )
                    if validation is not None and not validation.ok:
                        outcome, validation, content = await self._repair(
                            run, spec, outcome.content
                        )
                    if run.invocation.json_schema is not None:
                        self._mark_sent(run)
                        if outcome.refusal:
                            yield GatewayEvent(
                                kind=EventKind.REFUSAL, text=outcome.refusal
                            )
                        elif outcome.tool_calls:
                            for tool in outcome.tool_calls:
                                yield GatewayEvent(
                                    kind=EventKind.TOOL_CALL, tool_call=tool
                                )
                        else:
                            yield GatewayEvent(kind=EventKind.DELTA, text=content)
                    elif outcome.refusal and not attempt.refusal_emitted:
                        self._mark_sent(run)
                        yield GatewayEvent(kind=EventKind.REFUSAL, text=outcome.refusal)
                    result = self._result(run, spec, outcome, validation, content)
                    self._recorder.record_call(run, result.terminal, result=result)
                    yield GatewayEvent(
                        kind=EventKind.DONE,
                        finish_reason=result.finish_reason,
                        usage=result.usage,
                    )
                    return
                except GatewayError as error:
                    if (
                        await self._retry(
                            run, error, tries, len(candidates) - index - 1
                        )
                        == "fallback"
                    ):
                        break
        raise GatewayError(
            Code.MODEL_UNAVAILABLE, "model temporarily unavailable", stage=Stage.ROUTING
        )

    @staticmethod
    def _mark_sent(run: CallState) -> None:
        run.sent = True
        if run.ttft_ms is None:
            run.ttft_ms = (time.monotonic() - run.started) * 1000

    @staticmethod
    def _result(
        run: CallState,
        spec: ModelSpec,
        outcome: AdapterOutcome,
        validation: ValidationResult | None,
        content: str,
    ) -> UnifiedResult:
        # Both invoke and stream normalize usage before constructing a result.
        assert outcome.usage is not None
        return UnifiedResult(
            run.invocation.requested_model,
            spec.resolved_model,
            spec.provider,
            content,
            outcome.tool_calls,
            outcome.finish_reason,
            Terminal.REFUSED if outcome.refusal else Terminal.COMPLETED,
            outcome.usage,
            validation,
            outcome.provider_request_id,
            outcome.refusal,
        )

    @staticmethod
    def _upstream_call(
        run: CallState, spec: ModelSpec, messages: tuple[Message, ...]
    ) -> UpstreamCall:
        invocation = run.invocation
        return UpstreamCall(
            invocation.requested_model,
            spec.resolved_model,
            spec.base_url,
            messages,
            invocation.max_tokens,
            invocation.temperature,
            invocation.json_schema,
            invocation.tools,
            timedelta(
                seconds=max(
                    0.001,
                    min(run.deadline - time.monotonic(), spec.timeout.total_seconds()),
                )
            ),
            invocation.tool_choice,
            invocation.schema_name,
            invocation.schema_description,
            invocation.schema_strict,
        )


def _merge_tool(tools: dict[int, ToolCall], delta: ToolCall) -> None:
    previous = tools.get(delta.index)
    tools[delta.index] = (
        delta
        if previous is None
        else ToolCall(
            previous.id or delta.id,
            previous.name + delta.name,
            previous.arguments + delta.arguments,
            delta.index,
        )
    )
