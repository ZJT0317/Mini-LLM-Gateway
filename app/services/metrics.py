"""In-process counters and latency samples. Tests can read them directly."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Metrics:
    requests: dict[tuple[str, str, str], int] = field(default_factory=dict)
    latency_ms: dict[tuple[str, str, str], float] = field(default_factory=dict)
    samples: dict[tuple[str, str, str], list[float]] = field(default_factory=dict)

    def observe(self, model: str, provider: str, terminal: str, latency_ms: float) -> None:
        key = (model, provider, terminal)
        self.requests[key] = self.requests.get(key, 0) + 1
        self.latency_ms[key] = self.latency_ms.get(key, 0.0) + latency_ms
        bucket = self.samples.setdefault(key, [])
        bucket.append(latency_ms)
        if len(bucket) > 512:
            del bucket[:-512]

    def percentile(self, model: str, provider: str, terminal: str, pct: float) -> float | None:
        bucket = self.samples.get((model, provider, terminal))
        if not bucket:
            return None
        ordered = sorted(bucket)
        index = min(len(ordered) - 1, max(0, math.ceil(pct / 100 * len(ordered)) - 1))
        return ordered[index]
