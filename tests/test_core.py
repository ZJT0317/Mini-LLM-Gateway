import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.core.admission import AdmissionGate, Admit, budget_is_usable, refill, take_token, TokenBucket
from app.core.auth import authorize, hash_secret
from app.core.errors import Code, GatewayError, Stage
from app.core.models import APIKey, Message, estimate_tokens
from app.core.retry import backoff, should_fallback, should_retry


def test_backoff_is_bounded_and_jittered():
    assert backoff(1, 0) == timedelta(seconds=0.1)
    assert backoff(1, 1) == timedelta(seconds=0.3)
    assert backoff(20, 1) == timedelta(seconds=3.0)


def test_retry_and_fallback_follow_the_stage_table():
    assert should_retry("before_first_token", Code.UPSTREAM_OVERLOADED, 1, 2, False)
    assert not should_retry("before_first_token", Code.UPSTREAM_OVERLOADED, 2, 2, False)
    assert not should_retry("before_first_token", Code.UPSTREAM_OVERLOADED, 1, 2, True)
    assert not should_retry("auth", Code.UNAUTHENTICATED, 1, 2, False)
    assert not should_retry("content", Code.CONTENT_REJECTED, 1, 2, False)
    assert should_fallback(Code.UPSTREAM_UNAVAILABLE, False, 1)
    assert not should_fallback(Code.UPSTREAM_UNAVAILABLE, True, 1)
    assert not should_fallback(Code.CONTENT_REJECTED, False, 1)
    assert not should_fallback(Code.NO_COMPATIBLE_MODEL, False, 1)


def test_estimate_tokens_marks_the_result_estimated():
    usage = estimate_tokens((Message("user", "abcd"),), "abcd")
    assert usage.estimated
    assert usage.input_tokens == 5
    assert usage.output_tokens == 1


def test_short_budget_is_rejected():
    assert budget_is_usable(timedelta(seconds=1))
    assert not budget_is_usable(timedelta(milliseconds=100))


def test_disabled_key_is_not_retried():
    key = APIKey("off", hash_secret("x"), "t", "disabled", 1, 1, ())
    with pytest.raises(GatewayError) as caught:
        authorize(key)
    assert caught.value.code is Code.KEY_DISABLED
    assert caught.value.stage == Stage.AUTH
    assert "x" not in str(caught.value)


@pytest.mark.asyncio
async def test_concurrency_is_checked_before_rpm_is_refunded():
    gate = AdmissionGate()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    status, release = await gate.acquire("demo", rpm=10, limit=1, now=now)
    assert status == Admit.OK
    blocked, nothing = await gate.acquire("demo", rpm=10, limit=1, now=now)
    assert blocked == Admit.CONCURRENCY
    assert nothing is None
    assert release is not None
    await release()
    again, second = await gate.acquire("demo", rpm=10, limit=1, now=now)
    assert again == Admit.OK
    assert second is not None


def test_token_bucket_refills_after_a_minute():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bucket = TokenBucket(tokens=0, updated=start, rpm=60)
    denied, ok = take_token(bucket, start)
    assert not ok
    restored = refill(denied, start + timedelta(seconds=60))
    taken, ok = take_token(restored, start + timedelta(seconds=60))
    assert ok
    assert taken.tokens < 60


def test_core_does_not_import_frameworks():
    root = Path("app/core")
    forbidden = ("fastapi", "httpx", "sqlite3", "yaml", "jinja2", "app.services", "app.api")
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                assert not any(module == item or module.startswith(item + ".") for item in forbidden)
