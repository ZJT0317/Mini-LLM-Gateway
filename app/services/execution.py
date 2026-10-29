"""Mutable state shared by execution and accounting for a single call."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from app.core.models import (
    Invocation,
    Message,
    ModelSpec,
    TokenUsage,
    ToolCall,
    ValidationResult,
)
from app.services.prompts import PreparedCall
from app.services.usage import AttemptRow

AttemptKind = Literal["generation", "repair"]


@dataclass
class AttemptState:
    spec: ModelSpec
    number: int
    started: float
    messages: tuple[Message, ...]
    kind: AttemptKind
    is_retry: bool
    is_fallback: bool
    budget_ms: float
    content: str = ""
    tools: dict[int, ToolCall] = field(default_factory=dict)
    usage: TokenUsage | None = None
    finish_reason: str = ""
    provider_request_id: str = ""
    ttft_ms: float | None = None
    validation: ValidationResult | None = None
    received: bool = False
    refusal: str = ""
    refusal_emitted: bool = False


@dataclass
class CallState:
    original: Invocation
    started: float = field(default_factory=time.monotonic)
    prepared: PreparedCall | None = None
    spec: ModelSpec | None = None
    active: AttemptState | None = None
    attempts: list[AttemptRow] = field(default_factory=list)
    queue_ms: float = 0
    route_ms: float = 0
    ttft_ms: float | None = None
    sent: bool = False
    saved: bool = False

    @property
    def deadline(self) -> float:
        return self.started + self.original.timeout.total_seconds()

    @property
    def invocation(self) -> Invocation:
        return self.original if self.prepared is None else self.prepared.invocation
