"""Concurrency slots first, then a per-key token bucket."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta


MIN_BUDGET = timedelta(seconds=1)
Release = Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class TokenBucket:
    tokens: float
    updated: datetime
    rpm: int


class Admit:
    OK = "ok"
    CONCURRENCY = "concurrency"
    RATE = "rate"


def budget_is_usable(timeout: timedelta, minimum: timedelta = MIN_BUDGET) -> bool:
    return timeout >= minimum


def refill(bucket: TokenBucket, now: datetime) -> TokenBucket:
    elapsed = max(0.0, (now - bucket.updated).total_seconds())
    tokens = min(float(bucket.rpm), bucket.tokens + elapsed * (bucket.rpm / 60.0))
    return TokenBucket(tokens=tokens, updated=now, rpm=bucket.rpm)


def take_token(bucket: TokenBucket, now: datetime) -> tuple[TokenBucket, bool]:
    current = refill(bucket, now)
    if current.tokens < 1:
        return current, False
    return TokenBucket(current.tokens - 1, current.updated, current.rpm), True


class AdmissionGate:
    """In-memory admission. RPM is not refunded; the concurrency slot is."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._inflight: dict[str, int] = {}
        self._buckets: dict[str, TokenBucket] = {}

    async def acquire(
        self,
        key_id: str,
        rpm: int,
        limit: int,
        now: datetime,
    ) -> tuple[str, Release | None]:
        async with self._lock:
            inflight = self._inflight.get(key_id, 0)
            if inflight >= limit:
                return Admit.CONCURRENCY, None
            bucket = self._buckets.get(key_id) or TokenBucket(float(rpm), now, rpm)
            if bucket.rpm != rpm:
                bucket = TokenBucket(float(rpm), now, rpm)
            bucket, ok = take_token(bucket, now)
            self._buckets[key_id] = bucket
            if not ok:
                return Admit.RATE, None
            self._inflight[key_id] = inflight + 1

        released = False

        async def release() -> None:
            nonlocal released
            async with self._lock:
                if released:
                    return
                released = True
                self._inflight[key_id] = max(0, self._inflight.get(key_id, 1) - 1)

        return Admit.OK, release
