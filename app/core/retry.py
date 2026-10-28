"""Retry and fallback decisions. Callers inject sleep; these functions do not wait."""

from __future__ import annotations

from datetime import timedelta

from app.core.errors import Code

_FALLBACK_CODES = {Code.UPSTREAM_UNAVAILABLE, Code.UPSTREAM_OVERLOADED}


def should_retry(
    stage: str,
    code: Code,
    attempt: int,
    max_attempts: int,
    business_event_sent: bool,
) -> bool:
    if business_event_sent or attempt >= max_attempts or code is Code.CONTENT_REJECTED:
        return False
    if code is Code.UPSTREAM_UNAVAILABLE and stage in {"connection", "before_first_token"}:
        return True
    if code is Code.UPSTREAM_OVERLOADED and stage == "before_first_token":
        return True
    return False


def should_fallback(
    code: Code,
    business_event_sent: bool,
    remaining_candidates: int,
) -> bool:
    if business_event_sent or remaining_candidates <= 0 or code is Code.CONTENT_REJECTED:
        return False
    return code in _FALLBACK_CODES


def backoff(retry_index: int, random01: float) -> timedelta:
    """Bounded exponential wait with jitter in ``[0.5, 1.5]``."""
    if retry_index < 1:
        raise ValueError("retry_index starts at 1")
    if not 0 <= random01 <= 1:
        raise ValueError("random01 must be between 0 and 1")
    base = min(0.2 * (2 ** (retry_index - 1)), 2.0)
    return timedelta(seconds=base * (0.5 + random01))
