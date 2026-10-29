"""Capability matching, priority, and in-process health."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from app.core.models import (
    Capability,
    ModelAdapter,
    ModelSpec,
    Rejection,
    RoutingDecision,
)


@dataclass
class _Circuit:
    failures: int = 0
    opened_at: float = 0
    probing: bool = False


class HealthBoard:
    """Closed/open/half-open circuit with one in-flight recovery probe."""

    def __init__(
        self,
        threshold: int = 3,
        cooldown: float = 30,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if threshold < 1 or cooldown <= 0:
            raise ValueError("circuit threshold and cooldown must be positive")
        self._threshold = threshold
        self._cooldown = cooldown
        self._clock = clock
        self._circuits: dict[tuple[str, str], _Circuit] = {}

    def _state(self, spec: ModelSpec) -> _Circuit:
        return self._circuits.setdefault(
            (spec.provider, spec.resolved_model), _Circuit()
        )

    def allow(self, spec: ModelSpec) -> bool:
        state = self._state(spec)
        return state.failures < self._threshold or (
            not state.probing and self._clock() - state.opened_at >= self._cooldown
        )

    def acquire(self, spec: ModelSpec) -> bool:
        # No awaits: selection and reservation are atomic in this single event loop.
        if not self.allow(spec):
            return False
        state = self._state(spec)
        if state.failures >= self._threshold:
            state.probing = True
        return True

    def release(self, spec: ModelSpec) -> None:
        state = self._state(spec)
        if state.probing:
            state.probing = False
            state.opened_at = self._clock()

    def record(self, spec: ModelSpec, *, success: bool) -> None:
        state = self._state(spec)
        state.probing = False
        if success:
            state.failures = 0
            return
        state.failures += 1
        if state.failures >= self._threshold:
            state.opened_at = self._clock()


class AdapterRegistry:
    def __init__(
        self, specs: tuple[ModelSpec, ...], adapters: Mapping[str, ModelAdapter]
    ) -> None:
        for spec in specs:
            adapter = adapters[spec.provider]
            if not spec.capabilities <= adapter.capabilities:
                raise ValueError(
                    f"{spec.logical_model} asks for capabilities the {spec.provider} adapter does not provide"
                )
        self.specs = specs
        self._adapters = adapters

    def adapter_for(self, provider: str) -> ModelAdapter:
        return self._adapters[provider]

    def logical_models(self) -> tuple[str, ...]:
        return tuple(sorted({spec.logical_model for spec in self.specs}))


class Router:
    def __init__(self, specs: tuple[ModelSpec, ...], health: HealthBoard) -> None:
        self._specs = specs
        self._health = health

    def decide(
        self,
        logical_model: str,
        required: frozenset[Capability],
    ) -> RoutingDecision:
        rejections: list[Rejection] = []
        eligible: list[ModelSpec] = []
        for spec in self._specs:
            if spec.logical_model != logical_model:
                continue
            if not required <= spec.capabilities:
                rejections.append(
                    Rejection(spec.provider, spec.resolved_model, "capability_mismatch")
                )
                continue
            if not self._health.allow(spec):
                rejections.append(
                    Rejection(spec.provider, spec.resolved_model, "unhealthy")
                )
                continue
            eligible.append(spec)
        eligible.sort(
            key=lambda spec: (spec.priority, spec.input_cost + spec.output_cost)
        )
        for spec in eligible[1:]:
            rejections.append(
                Rejection(spec.provider, spec.resolved_model, "priority_lost")
            )
        selected = eligible[0] if eligible else None
        return RoutingDecision(
            requested_model=logical_model,
            selected=selected,
            candidates=tuple(eligible),
            rejections=tuple(rejections),
        )
