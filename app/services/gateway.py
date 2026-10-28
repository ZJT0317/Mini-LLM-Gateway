"""Orchestrate prompt rendering, admission, routing, execution, and the ledger."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime, timedelta

from app.core.admission import AdmissionGate, Admit, budget_is_usable
from app.core.auth import ensure_model_visible
from app.core.errors import Code, GatewayError, Stage, http_status
from app.core.models import (
    APIKey,
    EventKind,
    GatewayEvent,
    Invocation,
    Message,
    ModelSpec,
    PromptUse,
    Terminal,
    TokenUsage,
    UnifiedResult,
    UpstreamCall,
    estimate_tokens,
    price_usage,
)
from app.core.retry import backoff, should_fallback, should_retry
from app.services.metrics import Metrics
from app.services.prompts import PreparedCall, PromptLibrary
from app.services.router import AdapterRegistry, HealthBoard, Router
from app.services.structured import validate_json
from app.services.usage import AttemptRow, CallRow, Ledger

Sleeper = Callable[[timedelta], Awaitable[None]]
Rng = Callable[[], float]


class _Run:
    def __init__(self, started: float) -> None:
        self.started = started
        self.prepared: PreparedCall | None = None
        self.spec: ModelSpec | None = None
        self.saved = False


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
        self._ledger = ledger
        self._metrics = metrics
        self._clock = clock
        self._sleeper = sleeper
        self._rng = rng
        self._max_attempts = max_attempts
        self._repair_max_attempts = repair_max_attempts
        self._max_tokens_cap = max_tokens_cap

    async def complete(self, invocation: Invocation, key: APIKey) -> UnifiedResult:
        started = time.monotonic()
        run = _Run(started)
        release = None
        try:
            prepared, release, queue_ms = await self._open(invocation, key)
            run.prepared = prepared
            async with asyncio_timeout(prepared.invocation.timeout):
                return await self._complete(run, queue_ms)
        except TimeoutError as exc:
            error = GatewayError(
                Code.UPSTREAM_OVERLOADED,
                "gateway timeout",
                stage=Stage.BEFORE_FIRST_TOKEN,
                cause=exc,
            )
            self._save_failure(run, invocation, error, 504, None)
            raise error from exc
        except GatewayError as exc:
            self._save_failure(run, invocation, exc, http_status(exc)[0], None)
            raise
        finally:
            if release is not None:
                await release()

    async def stream(self, invocation: Invocation, key: APIKey) -> AsyncIterator[GatewayEvent]:
        started = time.monotonic()
        run = _Run(started)
        release = None
        try:
            prepared, release, queue_ms = await self._open(invocation, key)
            run.prepared = prepared
            async with asyncio_timeout(prepared.invocation.timeout):
                async for event in self._stream(run, queue_ms):
                    yield event
        except TimeoutError as exc:
            error = GatewayError(
                Code.UPSTREAM_OVERLOADED,
                "gateway timeout",
                stage=Stage.BEFORE_FIRST_TOKEN,
                cause=exc,
            )
            self._save_failure(run, invocation, error, 504, None)
            raise error from exc
        except GatewayError as exc:
            self._save_failure(run, invocation, exc, http_status(exc)[0], None)
            raise
        finally:
            if release is not None:
                await release()

    async def _open(
        self,
        invocation: Invocation,
        key: APIKey,
    ) -> tuple[PreparedCall, Callable[[], Awaitable[None]], float]:
        self._check_limits(invocation)
        prepared = self._prompts.prepare(invocation)
        ensure_model_visible(key, prepared.invocation.requested_model)
        if not budget_is_usable(prepared.invocation.timeout):
            raise GatewayError(
                Code.DEADLINE_TOO_SHORT,
                "deadline too short",
                stage=Stage.ADMISSION,
            )
        queue_started = time.monotonic()
        status, release = await self._admission.acquire(
            key.id,
            key.rpm,
            key.max_concurrency,
            self._clock(),
        )
        queue_ms = (time.monotonic() - queue_started) * 1000
        if status == Admit.CONCURRENCY or release is None and status != Admit.OK:
            if status == Admit.RATE:
                raise GatewayError(
                    Code.ADMISSION_REJECTED,
                    "rate limit exceeded",
                    stage=Stage.ADMISSION,
                )
            raise GatewayError(
                Code.ADMISSION_REJECTED,
                "concurrency limit exceeded",
                stage=Stage.ADMISSION,
            )
        assert release is not None
        return prepared, release, queue_ms

    def _check_limits(self, invocation: Invocation) -> None:
        if invocation.max_tokens is not None and invocation.max_tokens > self._max_tokens_cap:
            raise GatewayError(
                Code.INVALID_REQUEST,
                "max_tokens exceeds cap",
                stage=Stage.VALIDATION,
            )

    def _route(self, invocation: Invocation):
        route_started = time.monotonic()
        decision = self._router.decide(invocation.requested_model, invocation.required)
        route_ms = (time.monotonic() - route_started) * 1000
        for rejection in decision.rejections:
            self._ledger.save_rejection(
                invocation.call_id,
                rejection.provider,
                rejection.resolved_model,
                rejection.reason,
            )
        if not decision.candidates:
            raise GatewayError(Code.NO_COMPATIBLE_MODEL, "model not found", stage=Stage.ROUTING)
        return decision, route_ms

    async def _classify_failure(
        self,
        *,
        error: GatewayError,
        invocation: Invocation,
        spec: ModelSpec,
        attempt: int,
        tries: int,
        index: int,
        business: bool,
        remaining: int,
        queue_ms: float,
        route_ms: float,
        started: float,
    ) -> str:
        if error.stage == Stage.OUTPUT:
            return "raise"
        self._health.record(spec, success=False)
        if business:
            return "stop"
        self._save_attempt(
            invocation,
            spec,
            attempt,
            is_retry=tries > 1,
            is_fallback=index > 0,
            queue_ms=queue_ms,
            route_ms=route_ms,
            ttft_ms=None,
            total_ms=(time.monotonic() - started) * 1000,
            usage=TokenUsage(0, 0),
            finish_reason="",
            terminal=Terminal.ERROR,
            error_code=error.code.value,
            provider_request_id=error.provider_request_id or "",
        )
        if should_retry(error.stage, error.code, tries, self._max_attempts, False):
            await self._sleeper(backoff(tries, self._rng()))
            return "retry"
        if should_fallback(error.code, False, remaining):
            return "fallback"
        return "raise"

    def _save_failure(
        self,
        run: _Run,
        invocation: Invocation,
        error: GatewayError,
        status: int,
        ttft_ms: float | None,
    ) -> None:
        if run.saved:
            return
        source = invocation if run.prepared is None else run.prepared.invocation
        prompt = None if run.prepared is None else run.prepared.prompt
        if prompt is None and source.prompt_ref is not None:
            prompt = PromptUse(source.prompt_ref.name, source.prompt_ref.version, "", "")
        self._save_call(run, source, prompt, error, ttft_ms, status)

    async def _complete(self, run: _Run, queue_ms: float) -> UnifiedResult:
        assert run.prepared is not None
        prepared = run.prepared
        invocation = prepared.invocation
        started = run.started
        decision, route_ms = self._route(invocation)
        attempt = 0
        for index, spec in enumerate(decision.candidates):
            tries = 0
            while True:
                tries += 1
                attempt += 1
                run.spec = spec
                call = _upstream_call(invocation, spec)
                try:
                    outcome = await self._registry.adapter_for(spec.provider).invoke(call)
                    self._health.record(spec, success=True)
                    usage = outcome.usage or estimate_tokens(invocation.messages, outcome.content)
                    content = outcome.content
                    validation = None
                    if invocation.json_schema is not None:
                        content, validation, attempt = await self._repair(
                            invocation,
                            spec,
                            content,
                            attempt,
                            queue_ms,
                            route_ms,
                            started,
                        )
                    result = UnifiedResult(
                        requested_model=invocation.requested_model,
                        resolved_model=spec.resolved_model,
                        provider=spec.provider,
                        content=content,
                        tool_calls=outcome.tool_calls,
                        finish_reason=outcome.finish_reason,
                        terminal=Terminal.COMPLETED,
                        usage=usage,
                        validation=validation,
                        provider_request_id=outcome.provider_request_id,
                    )
                    total_ms = (time.monotonic() - started) * 1000
                    self._save_attempt(
                        invocation,
                        spec,
                        attempt,
                        is_retry=tries > 1,
                        is_fallback=index > 0,
                        queue_ms=queue_ms,
                        route_ms=route_ms,
                        ttft_ms=None,
                        total_ms=total_ms,
                        usage=usage,
                        finish_reason=result.finish_reason,
                        terminal=Terminal.COMPLETED,
                        error_code="",
                        provider_request_id=outcome.provider_request_id or "",
                    )
                    self._save_success(run, result, None)
                    return result
                except GatewayError as error:
                    action = await self._classify_failure(
                        error=error,
                        invocation=invocation,
                        spec=spec,
                        attempt=attempt,
                        tries=tries,
                        index=index,
                        business=False,
                        remaining=len(decision.candidates) - index - 1,
                        queue_ms=queue_ms,
                        route_ms=route_ms,
                        started=started,
                    )
                    if action == "retry":
                        continue
                    if action == "fallback":
                        break
                    raise
        raise GatewayError(Code.UPSTREAM_ERROR, "upstream request failed", stage=Stage.BEFORE_FIRST_TOKEN)

    async def _repair(
        self,
        invocation: Invocation,
        spec: ModelSpec,
        content: str,
        attempt: int,
        queue_ms: float,
        route_ms: float,
        started: float,
    ) -> tuple[str, object, int]:
        assert invocation.json_schema is not None
        validation, canonical = validate_json(content, invocation.json_schema)
        if validation.ok:
            return canonical, validation, attempt
        current = content
        repairs = 0
        while repairs < self._repair_max_attempts:
            repairs += 1
            attempt += 1
            messages = invocation.messages + (
                Message("assistant", current),
                Message("user", "Return only JSON that matches the requested schema."),
            )
            repaired = await self._registry.adapter_for(spec.provider).invoke(
                _upstream_call(invocation, spec, messages=messages)
            )
            current = repaired.content
            validation, canonical = validate_json(current, invocation.json_schema)
            usage = repaired.usage or estimate_tokens(messages, current)
            self._save_attempt(
                invocation,
                spec,
                attempt,
                is_retry=True,
                is_fallback=False,
                queue_ms=queue_ms,
                route_ms=route_ms,
                ttft_ms=None,
                total_ms=(time.monotonic() - started) * 1000,
                usage=usage,
                finish_reason=repaired.finish_reason,
                terminal=Terminal.COMPLETED if validation.ok else Terminal.ERROR,
                error_code="" if validation.ok else Code.OUTPUT_INVALID.value,
                provider_request_id=repaired.provider_request_id or "",
            )
            if validation.ok:
                return canonical, validation, attempt
        raise GatewayError(
            Code.OUTPUT_INVALID,
            "output validation failed",
            stage=Stage.OUTPUT,
        )

    async def _stream(self, run: _Run, queue_ms: float) -> AsyncIterator[GatewayEvent]:
        assert run.prepared is not None
        prepared = run.prepared
        invocation = prepared.invocation
        started = run.started
        decision, route_ms = self._route(invocation)
        attempt = 0
        for index, spec in enumerate(decision.candidates):
            tries = 0
            while True:
                tries += 1
                attempt += 1
                run.spec = spec
                call = _upstream_call(invocation, spec)
                business = False
                ttft_ms: float | None = None
                parts: list[str] = []
                usage: TokenUsage | None = None
                try:
                    async for event in self._registry.adapter_for(spec.provider).stream(call):
                        if event.kind is EventKind.ERROR and not business:
                            error = event.error
                            if isinstance(error, GatewayError):
                                raise error
                            raise GatewayError(
                                Code.UPSTREAM_ERROR,
                                "upstream request failed",
                                stage=Stage.STREAMING,
                            )
                        if event.kind in {EventKind.DELTA, EventKind.TOOL_CALL} and ttft_ms is None:
                            ttft_ms = (time.monotonic() - started) * 1000
                            business = True
                        if event.kind is EventKind.DELTA:
                            parts.append(event.text)
                        if event.kind is EventKind.USAGE and event.usage is not None:
                            usage = event.usage
                            continue
                        if invocation.json_schema is None and business:
                            yield event
                        if event.kind is EventKind.DONE:
                            content = "".join(parts)
                            validation = None
                            if invocation.json_schema is not None:
                                content, validation, attempt = await self._repair(
                                    invocation,
                                    spec,
                                    content,
                                    attempt,
                                    queue_ms,
                                    route_ms,
                                    started,
                                )
                                yield GatewayEvent(kind=EventKind.DELTA, text=content)
                            elif not business:
                                yield event
                            final_usage = usage or estimate_tokens(invocation.messages, content)
                            result = UnifiedResult(
                                requested_model=invocation.requested_model,
                                resolved_model=spec.resolved_model,
                                provider=spec.provider,
                                content=content,
                                tool_calls=(),
                                finish_reason="stop",
                                terminal=Terminal.COMPLETED,
                                usage=final_usage,
                                validation=validation,
                                provider_request_id=None,
                            )
                            self._health.record(spec, success=True)
                            self._save_attempt(
                                invocation,
                                spec,
                                attempt,
                                is_retry=tries > 1,
                                is_fallback=index > 0,
                                queue_ms=queue_ms,
                                route_ms=route_ms,
                                ttft_ms=ttft_ms,
                                total_ms=(time.monotonic() - started) * 1000,
                                usage=final_usage,
                                finish_reason="stop",
                                terminal=Terminal.COMPLETED,
                                error_code="",
                                provider_request_id="",
                            )
                            self._save_success(run, result, ttft_ms)
                            if invocation.json_schema is not None:
                                yield GatewayEvent(kind=EventKind.DONE)
                            return
                    raise GatewayError(
                        Code.UPSTREAM_ERROR,
                        "upstream request failed",
                        stage=Stage.STREAMING,
                    )
                except GatewayError as error:
                    action = await self._classify_failure(
                        error=error,
                        invocation=invocation,
                        spec=spec,
                        attempt=attempt,
                        tries=tries,
                        index=index,
                        business=business,
                        remaining=len(decision.candidates) - index - 1,
                        queue_ms=queue_ms,
                        route_ms=route_ms,
                        started=started,
                    )
                    if action == "stop":
                        yield GatewayEvent(kind=EventKind.ERROR, error=error)
                        self._save_call(run, invocation, prepared.prompt, error, ttft_ms, 200)
                        return
                    if action == "retry":
                        continue
                    if action == "fallback":
                        break
                    raise

    def _save_success(self, run: _Run, result: UnifiedResult, ttft_ms: float | None) -> None:
        assert run.prepared is not None
        prepared = run.prepared
        started = run.started
        invocation = prepared.invocation
        prompt = prepared.prompt
        cost = price_usage(result.usage, _spec_for(self._registry, result))
        validation_ok = None if result.validation is None else int(result.validation.ok)
        self._ledger.save_call(
            CallRow(
                call_id=invocation.call_id,
                trace_id=invocation.trace_id,
                run_id=invocation.run_id,
                step_id=invocation.step_id,
                requested_model=invocation.requested_model,
                provider=result.provider,
                resolved_model=result.resolved_model,
                tenant_id=invocation.tenant_id,
                key_id=invocation.key_id,
                prompt_name="" if prompt is None else prompt.name,
                prompt_version="" if prompt is None else prompt.version,
                prompt_hash="" if prompt is None else prompt.content_hash,
                schema_version="" if prompt is None else prompt.schema_version,
                terminal=result.terminal.value,
                http_status=200,
                error_code="",
                finish_reason=result.finish_reason,
                validation_ok=validation_ok,
                ttft_ms=ttft_ms,
                total_ms=(time.monotonic() - started) * 1000,
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
                cached_tokens=result.usage.cached_tokens,
                reasoning_tokens=result.usage.reasoning_tokens,
                estimated=result.usage.estimated,
                cost_total=str(cost.total),
            )
        )
        run.saved = True
        self._metrics.observe(
            invocation.requested_model,
            result.provider,
            result.terminal.value,
            (time.monotonic() - started) * 1000,
        )

    def _save_call(
        self,
        run: _Run,
        invocation: Invocation,
        prompt: PromptUse | None,
        error: GatewayError,
        ttft_ms: float | None,
        status: int,
    ) -> None:
        spec = run.spec
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
                prompt_name="" if prompt is None else prompt.name,
                prompt_version="" if prompt is None else prompt.version,
                prompt_hash="" if prompt is None else prompt.content_hash,
                schema_version="" if prompt is None else prompt.schema_version,
                terminal=Terminal.ERROR.value,
                http_status=status,
                error_code=error.code.value,
                finish_reason="",
                validation_ok=None,
                ttft_ms=ttft_ms,
                total_ms=(time.monotonic() - run.started) * 1000,
                input_tokens=0,
                output_tokens=0,
                cached_tokens=0,
                reasoning_tokens=0,
                estimated=False,
                cost_total="0",
            )
        )
        run.saved = True
        provider = "" if spec is None else spec.provider
        self._metrics.observe(
            invocation.requested_model,
            provider,
            Terminal.ERROR.value,
            (time.monotonic() - run.started) * 1000,
        )

    def _save_attempt(
        self,
        invocation: Invocation,
        spec: ModelSpec,
        attempt: int,
        *,
        is_retry: bool,
        is_fallback: bool,
        queue_ms: float,
        route_ms: float,
        ttft_ms: float | None,
        total_ms: float,
        usage: TokenUsage,
        finish_reason: str,
        terminal: Terminal,
        error_code: str,
        provider_request_id: str,
    ) -> None:
        cost = price_usage(usage, spec)
        generation = None if ttft_ms is None else max(0.0, total_ms - ttft_ms)
        self._ledger.save_attempt(
            AttemptRow(
                call_id=invocation.call_id,
                attempt=attempt,
                provider=spec.provider,
                resolved_model=spec.resolved_model,
                is_retry=is_retry,
                is_fallback=is_fallback,
                queue_ms=queue_ms,
                route_ms=route_ms,
                ttft_ms=ttft_ms,
                generation_ms=generation,
                total_ms=total_ms,
                timeout_budget_ms=invocation.timeout.total_seconds() * 1000,
                finish_reason=finish_reason,
                terminal=terminal.value,
                error_code=error_code,
                provider_request_id=provider_request_id,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cached_tokens=usage.cached_tokens,
                reasoning_tokens=usage.reasoning_tokens,
                estimated=usage.estimated,
                cost_input=str(cost.input),
                cost_output=str(cost.output),
                cost_cache=str(cost.cache),
                cost_total=str(cost.total),
            )
        )


def asyncio_timeout(delay: timedelta):
    import asyncio

    return asyncio.timeout(delay.total_seconds())


def _upstream_call(
    invocation: Invocation,
    spec: ModelSpec,
    messages: tuple[Message, ...] | None = None,
) -> UpstreamCall:
    return UpstreamCall(
        requested_model=invocation.requested_model,
        resolved_model=spec.resolved_model,
        base_url=spec.base_url,
        messages=invocation.messages if messages is None else messages,
        max_tokens=invocation.max_tokens,
        temperature=invocation.temperature,
        json_schema=invocation.json_schema,
        tools=invocation.tools,
        timeout=min(invocation.timeout, spec.timeout),
    )


def _spec_for(registry: AdapterRegistry, result: UnifiedResult) -> ModelSpec:
    for spec in registry.specs:
        if spec.provider == result.provider and spec.resolved_model == result.resolved_model:
            return spec
    return registry.specs[0]
