"""Capability matching, priority, and in-process health."""

from __future__ import annotations

from app.core.models import Capability, ModelSpec, Rejection, RoutingDecision


class HealthBoard:
    """Opens after consecutive failures and closes on the next success."""

    def __init__(self, threshold: int = 3) -> None:
        self._threshold = threshold
        self._failures: dict[tuple[str, str], int] = {}

    def allow(self, spec: ModelSpec) -> bool:
        return self._failures.get((spec.provider, spec.resolved_model), 0) < self._threshold

    def record(self, spec: ModelSpec, *, success: bool) -> None:
        key = (spec.provider, spec.resolved_model)
        if success:
            self._failures[key] = 0
            return
        self._failures[key] = self._failures.get(key, 0) + 1


class AdapterRegistry:
    def __init__(self, specs: tuple[ModelSpec, ...], adapters: dict[str, object]) -> None:
        for spec in specs:
            adapter = adapters[spec.provider]
            if not spec.capabilities <= adapter.capabilities:
                raise ValueError(
                    f"{spec.logical_model} asks for capabilities the {spec.provider} adapter does not provide"
                )
        self.specs = specs
        self._adapters = adapters

    def adapter_for(self, provider: str):
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
                rejections.append(Rejection(spec.provider, spec.resolved_model, "unhealthy"))
                continue
            eligible.append(spec)
        eligible.sort(key=lambda spec: (spec.priority, spec.input_cost + spec.output_cost))
        for spec in eligible[1:]:
            rejections.append(Rejection(spec.provider, spec.resolved_model, "priority_lost"))
        selected = eligible[0] if eligible else None
        return RoutingDecision(
            requested_model=logical_model,
            selected=selected,
            candidates=tuple(eligible),
            rejections=tuple(rejections),
        )
