"""Internal call, result, and adapter protocols."""

from __future__ import annotations

import math
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Protocol


class Capability(StrEnum):
    STREAMING = "streaming"
    STRUCTURED_OUTPUT = "structured_output"
    TOOL_CALLING = "tool_calling"


class Terminal(StrEnum):
    COMPLETED = "completed"
    REFUSED = "refused"
    ERROR = "error"
    CANCELED = "canceled"


class EventKind(StrEnum):
    DELTA = "delta"
    REFUSAL = "refusal"
    TOOL_CALL = "tool_call"
    USAGE = "usage"
    FINISH = "finish"
    DONE = "done"
    ERROR = "error"


@dataclass(frozen=True)
class Message:
    role: str
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str
    index: int = 0


@dataclass(frozen=True)
class PromptRef:
    name: str
    version: str
    variables: dict


@dataclass(frozen=True)
class PromptUse:
    name: str
    version: str
    content_hash: str
    schema_version: str


@dataclass(frozen=True)
class Invocation:
    call_id: str
    trace_id: str
    requested_model: str
    messages: tuple[Message, ...]
    required: frozenset[Capability]
    max_tokens: int | None
    temperature: float | None
    json_schema: dict | bool | None
    tools: tuple[ToolSpec, ...]
    prompt_ref: PromptRef | None
    timeout: timedelta
    tenant_id: str
    key_id: str
    run_id: str = ""
    step_id: str = ""
    tool_choice: str | dict | None = None
    schema_name: str = "response"
    schema_description: str | None = None
    schema_strict: bool | None = None
    include_usage: bool = False
    created: int = 0


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    text_hash: str


@dataclass(frozen=True)
class UnifiedResult:
    requested_model: str
    resolved_model: str
    provider: str
    content: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str
    terminal: Terminal
    usage: TokenUsage
    validation: ValidationResult | None
    provider_request_id: str | None
    refusal: str | None = None


@dataclass(frozen=True)
class UpstreamCall:
    requested_model: str
    resolved_model: str
    base_url: str
    messages: tuple[Message, ...]
    max_tokens: int | None
    temperature: float | None
    json_schema: dict | bool | None
    tools: tuple[ToolSpec, ...]
    timeout: timedelta
    tool_choice: str | dict | None = None
    schema_name: str = "response"
    schema_description: str | None = None
    schema_strict: bool | None = None


@dataclass(frozen=True)
class AdapterOutcome:
    content: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str
    usage: TokenUsage | None
    provider_request_id: str | None
    refusal: str | None = None


@dataclass(frozen=True)
class GatewayEvent:
    kind: EventKind
    text: str = ""
    tool_call: ToolCall | None = None
    usage: TokenUsage | None = None
    error: object | None = None
    finish_reason: str | None = None
    provider_request_id: str | None = None
    requested_model: str | None = None


@dataclass(frozen=True)
class ModelSpec:
    logical_model: str
    provider: str
    resolved_model: str
    base_url: str
    capabilities: frozenset[Capability]
    priority: int
    input_cost: Decimal
    output_cost: Decimal
    timeout: timedelta


@dataclass(frozen=True)
class Rejection:
    provider: str
    resolved_model: str
    reason: str


@dataclass(frozen=True)
class RoutingDecision:
    requested_model: str
    selected: ModelSpec | None
    candidates: tuple[ModelSpec, ...]
    rejections: tuple[Rejection, ...]


@dataclass(frozen=True)
class Cost:
    input: Decimal
    output: Decimal
    cache: Decimal
    total: Decimal


@dataclass(frozen=True)
class APIKey:
    id: str
    secret_hash: bytes
    tenant_id: str
    status: str
    rpm: int
    max_concurrency: int
    allowed_models: tuple[str, ...]


class ModelAdapter(Protocol):
    name: str
    capabilities: frozenset[Capability]

    async def invoke(self, call: UpstreamCall) -> AdapterOutcome: ...

    def stream(self, call: UpstreamCall) -> AsyncGenerator[GatewayEvent, None]: ...


def estimate_tokens(messages: tuple[Message, ...], completion: str) -> TokenUsage:
    """Approximate usage when the upstream omits it. Not a model tokenizer."""
    prompt = sum(
        4
        + math.ceil(
            (
                len(message.content or "")
                + sum(
                    len(call.name) + len(call.arguments) for call in message.tool_calls
                )
            )
            / 4
        )
        for message in messages
    )
    completion_tokens = math.ceil(len(completion) / 4) if completion else 0
    return TokenUsage(
        input_tokens=prompt,
        output_tokens=completion_tokens,
        estimated=True,
    )


def price_usage(usage: TokenUsage, spec: ModelSpec) -> Cost:
    million = Decimal(1_000_000)
    cached = min(usage.cached_tokens, usage.input_tokens)
    fresh = usage.input_tokens - cached
    cost_input = Decimal(fresh) / million * spec.input_cost
    cost_output = Decimal(usage.output_tokens) / million * spec.output_cost
    cost_cache = Decimal(cached) / million * spec.input_cost
    return Cost(
        input=cost_input,
        output=cost_output,
        cache=cost_cache,
        total=cost_input + cost_output + cost_cache,
    )
