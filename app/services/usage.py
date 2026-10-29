"""SQLite ledger for calls, attempts, and routing rejections."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AttemptRow:
    call_id: str
    attempt: int
    provider: str
    resolved_model: str
    is_retry: bool
    is_fallback: bool
    queue_ms: float | None
    route_ms: float | None
    ttft_ms: float | None
    generation_ms: float | None
    total_ms: float | None
    timeout_budget_ms: float
    finish_reason: str
    terminal: str
    error_code: str
    provider_request_id: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    estimated: bool
    cost_input: str
    cost_output: str
    cost_cache: str
    cost_total: str
    kind: str = "generation"
    output_hash: str = ""


@dataclass(frozen=True)
class CallRow:
    call_id: str
    trace_id: str
    run_id: str
    step_id: str
    requested_model: str
    provider: str
    resolved_model: str
    tenant_id: str
    key_id: str
    prompt_name: str
    prompt_version: str
    prompt_hash: str
    schema_version: str
    terminal: str
    http_status: int
    error_code: str
    finish_reason: str
    validation_ok: int | None
    ttft_ms: float | None
    total_ms: float
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    estimated: bool
    cost_total: str


class Ledger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._create()

    def _create(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS calls (
                    call_id TEXT PRIMARY KEY,
                    trace_id TEXT,
                    run_id TEXT,
                    step_id TEXT,
                    requested_model TEXT,
                    tenant_id TEXT,
                    key_id TEXT,
                    prompt_name TEXT,
                    prompt_version TEXT,
                    prompt_hash TEXT,
                    schema_version TEXT,
                    terminal TEXT,
                    http_status INTEGER,
                    error_code TEXT,
                    finish_reason TEXT,
                    validation_ok INTEGER,
                    ttft_ms REAL,
                    total_ms REAL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cached_tokens INTEGER,
                    reasoning_tokens INTEGER,
                    estimated INTEGER,
                    cost_total TEXT
                );
                """
            )
            self._add_column("calls", "provider", "TEXT")
            self._add_column("calls", "resolved_model", "TEXT")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    call_id TEXT,
                    attempt INTEGER,
                    provider TEXT,
                    resolved_model TEXT,
                    is_retry INTEGER,
                    is_fallback INTEGER,
                    queue_ms REAL,
                    route_ms REAL,
                    ttft_ms REAL,
                    generation_ms REAL,
                    total_ms REAL,
                    timeout_budget_ms REAL,
                    finish_reason TEXT,
                    terminal TEXT,
                    error_code TEXT,
                    provider_request_id TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cached_tokens INTEGER,
                    reasoning_tokens INTEGER,
                    estimated INTEGER,
                    cost_input TEXT,
                    cost_output TEXT,
                    cost_cache TEXT,
                    cost_total TEXT
                );
                CREATE TABLE IF NOT EXISTS rejections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    call_id TEXT,
                    provider TEXT,
                    resolved_model TEXT,
                    reason TEXT
                );
                """
            )
            self._add_column("attempts", "kind", "TEXT DEFAULT 'generation'")
            self._add_column("attempts", "output_hash", "TEXT DEFAULT ''")
            self._conn.executescript(
                """
                CREATE INDEX IF NOT EXISTS calls_trace_idx ON calls(trace_id);
                CREATE INDEX IF NOT EXISTS attempts_call_idx ON attempts(call_id, attempt);
                CREATE INDEX IF NOT EXISTS rejections_call_idx ON rejections(call_id);
                """
            )
            self._conn.commit()

    def save_call(self, row: CallRow) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO calls (
                    call_id, trace_id, run_id, step_id, requested_model, provider, resolved_model,
                    tenant_id, key_id, prompt_name, prompt_version, prompt_hash, schema_version,
                    terminal, http_status, error_code, finish_reason, validation_ok, ttft_ms, total_ms,
                    input_tokens, output_tokens, cached_tokens, reasoning_tokens, estimated, cost_total
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    row.call_id,
                    row.trace_id,
                    row.run_id,
                    row.step_id,
                    row.requested_model,
                    row.provider,
                    row.resolved_model,
                    row.tenant_id,
                    row.key_id,
                    row.prompt_name,
                    row.prompt_version,
                    row.prompt_hash,
                    row.schema_version,
                    row.terminal,
                    row.http_status,
                    row.error_code,
                    row.finish_reason,
                    row.validation_ok,
                    row.ttft_ms,
                    row.total_ms,
                    row.input_tokens,
                    row.output_tokens,
                    row.cached_tokens,
                    row.reasoning_tokens,
                    int(row.estimated),
                    row.cost_total,
                ),
            )
            self._conn.commit()

    def save_attempt(self, row: AttemptRow) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO attempts (
                    call_id, attempt, provider, resolved_model, is_retry, is_fallback,
                    queue_ms, route_ms, ttft_ms, generation_ms, total_ms, timeout_budget_ms,
                    finish_reason, terminal, error_code, provider_request_id,
                    input_tokens, output_tokens, cached_tokens, reasoning_tokens, estimated,
                    cost_input, cost_output, cost_cache, cost_total, kind, output_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.call_id,
                    row.attempt,
                    row.provider,
                    row.resolved_model,
                    int(row.is_retry),
                    int(row.is_fallback),
                    row.queue_ms,
                    row.route_ms,
                    row.ttft_ms,
                    row.generation_ms,
                    row.total_ms,
                    row.timeout_budget_ms,
                    row.finish_reason,
                    row.terminal,
                    row.error_code,
                    row.provider_request_id,
                    row.input_tokens,
                    row.output_tokens,
                    row.cached_tokens,
                    row.reasoning_tokens,
                    int(row.estimated),
                    row.cost_input,
                    row.cost_output,
                    row.cost_cache,
                    row.cost_total,
                    row.kind,
                    row.output_hash,
                ),
            )
            self._conn.commit()

    def save_rejection(
        self, call_id: str, provider: str, resolved_model: str, reason: str
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO rejections (call_id, provider, resolved_model, reason) VALUES (?, ?, ?, ?)",
                (call_id, provider, resolved_model, reason),
            )
            self._conn.commit()

    def calls_by_trace(self, trace_id: str) -> list[sqlite3.Row]:
        with self._lock:
            return list(
                self._conn.execute(
                    "SELECT * FROM calls WHERE trace_id = ?", (trace_id,)
                )
            )

    def attempts_for(self, call_id: str) -> list[sqlite3.Row]:
        with self._lock:
            return list(
                self._conn.execute(
                    "SELECT * FROM attempts WHERE call_id = ? ORDER BY attempt",
                    (call_id,),
                )
            )

    def rejections_for(self, call_id: str) -> list[sqlite3.Row]:
        with self._lock:
            return list(
                self._conn.execute(
                    "SELECT * FROM rejections WHERE call_id = ?", (call_id,)
                )
            )

    def _add_column(self, table: str, name: str, declaration: str) -> None:
        found = {row[1] for row in self._conn.execute(f"PRAGMA table_info({table})")}
        if name not in found:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")

    def close(self) -> None:
        with self._lock:
            self._conn.close()
