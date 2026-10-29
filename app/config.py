"""Load YAML, read secrets from the environment, and refuse unsafe upstream URLs."""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlparse

import yaml

from app.core.auth import hash_secret
from app.core.models import APIKey, Capability, ModelSpec


@dataclass(frozen=True)
class ServerSettings:
    host: str
    port: int
    gateway_timeout: timedelta
    max_body_bytes: int
    shutdown_grace: timedelta
    database: Path


@dataclass(frozen=True)
class LimitSettings:
    max_tokens_cap: int
    retry_max_attempts: int
    repair_max_attempts: int
    circuit_failure_threshold: int = 3
    circuit_cooldown: timedelta = timedelta(seconds=30)


@dataclass(frozen=True)
class LogSettings:
    level: str
    debug_payload: bool
    payload_max_runes: int


@dataclass(frozen=True)
class ProviderSettings:
    name: str
    base_url: str
    api_key: str
    upstream_timeout: timedelta
    allowed_hosts: tuple[str, ...]
    capabilities: frozenset[Capability]


@dataclass(frozen=True)
class Settings:
    server: ServerSettings
    limits: LimitSettings
    log: LogSettings
    providers: tuple[ProviderSettings, ...]
    models: tuple[ModelSpec, ...]
    keys: tuple[APIKey, ...]
    prompts_directory: Path


def load_settings(path: Path) -> Settings:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    server_raw = raw.get("server") or {}
    limits_raw = raw.get("limits") or {}
    log_raw = raw.get("log") or {}
    host, port = _split_addr(str(server_raw.get("addr") or "0.0.0.0:8080"))
    gateway_timeout = _duration(server_raw.get("gateway_timeout") or "60s")
    providers = tuple(
        _provider(item, gateway_timeout) for item in raw.get("providers") or []
    )
    provider_by_name = {item.name: item for item in providers}
    if len(provider_by_name) != len(providers):
        raise ValueError("provider names must be unique")
    models = tuple(_model(item, provider_by_name) for item in raw.get("models") or [])
    route_keys = [
        (item.logical_model, item.provider, item.resolved_model) for item in models
    ]
    if len(route_keys) != len(set(route_keys)):
        raise ValueError("model routes must be unique")
    keys = tuple(_key(item) for item in raw.get("keys") or [])
    key_ids = [item.id for item in keys]
    if len(key_ids) != len(set(key_ids)):
        raise ValueError("key ids must be unique")
    prompts = raw.get("prompts") or {}
    directory = Path(prompts.get("directory") or path.parent / "prompts")
    if "directory" in prompts and not directory.exists():
        raise ValueError(f"prompt directory does not exist: {directory}")
    database = Path(
        os.environ.get("GATEWAY_DATABASE")
        or server_raw.get("database")
        or path.parent / "gateway.db"
    )
    return Settings(
        server=ServerSettings(
            host=host,
            port=port,
            gateway_timeout=gateway_timeout,
            max_body_bytes=_integer(server_raw, "max_body_bytes", 1_048_576),
            shutdown_grace=_duration(server_raw.get("shutdown_grace") or "15s"),
            database=database,
        ),
        limits=LimitSettings(
            max_tokens_cap=_integer(limits_raw, "max_tokens_cap", 4096),
            retry_max_attempts=_integer(limits_raw, "retry_max_attempts", 2),
            repair_max_attempts=_integer(
                limits_raw, "repair_max_attempts", 1, minimum=0
            ),
            circuit_failure_threshold=_integer(
                limits_raw, "circuit_failure_threshold", 3
            ),
            circuit_cooldown=_duration(limits_raw.get("circuit_cooldown", "30s")),
        ),
        log=LogSettings(
            level=str(log_raw.get("level") or "info"),
            debug_payload=bool(log_raw.get("debug_payload") or False),
            payload_max_runes=int(log_raw.get("payload_max_runes") or 200),
        ),
        providers=providers,
        models=models,
        keys=keys,
        prompts_directory=directory,
    )


def _split_addr(addr: str) -> tuple[str, int]:
    host, _, port = addr.rpartition(":")
    if not host:
        raise ValueError("server.addr must look like host:port")
    return host, int(port)


def _duration(value: object) -> timedelta:
    text = str(value).strip()
    if text.endswith("ms"):
        seconds = float(text[:-2]) / 1000
    elif text.endswith("s"):
        seconds = float(text[:-1])
    else:
        raise ValueError(f"unsupported duration: {value}")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("durations must be finite and positive")
    return timedelta(seconds=seconds)


def _integer(raw: dict, name: str, default: int, *, minimum: int = 1) -> int:
    value = raw.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _provider(raw: dict, gateway_timeout: timedelta) -> ProviderSettings:
    name = str(raw["name"])
    base_url = str(raw["base_url"])
    allowed = tuple(str(host) for host in raw.get("allowed_hosts") or [])
    _check_url(base_url, allowed)
    timeout = _duration(raw.get("upstream_timeout") or "90s")
    if gateway_timeout >= timeout:
        raise ValueError(
            f"{name}: gateway_timeout must be shorter than upstream_timeout"
        )
    api_key = os.environ.get(str(raw["api_key_env"]), "")
    if not api_key:
        raise ValueError(f"missing environment variable {raw['api_key_env']}")
    capabilities = frozenset(Capability(item) for item in raw.get("capabilities") or [])
    return ProviderSettings(
        name=name,
        base_url=base_url,
        api_key=api_key,
        upstream_timeout=timeout,
        allowed_hosts=allowed,
        capabilities=capabilities,
    )


def _check_url(base_url: str, allowed_hosts: tuple[str, ...]) -> None:
    parsed = urlparse(base_url)
    if parsed.username or parsed.password:
        raise ValueError("upstream URL must not include userinfo")
    host = parsed.hostname or ""
    if host not in allowed_hosts:
        raise ValueError(f"upstream host {host} is not in allowed_hosts")
    if parsed.scheme == "https":
        return
    if parsed.scheme == "http" and host in {"localhost", "127.0.0.1"}:
        return
    raise ValueError("upstream URL must use https")


def _model(raw: dict, providers: dict[str, ProviderSettings]) -> ModelSpec:
    provider_name = str(raw["provider"])
    provider = providers.get(provider_name)
    if provider is None:
        raise ValueError(f"unknown provider {provider_name}")
    capabilities = frozenset(Capability(item) for item in raw.get("capabilities") or [])
    if not capabilities <= provider.capabilities:
        raise ValueError(f"{raw.get('logical_model')} capabilities exceed the provider")
    return ModelSpec(
        logical_model=str(raw["logical_model"]),
        provider=provider_name,
        resolved_model=str(raw["resolved_model"]),
        base_url=provider.base_url,
        capabilities=capabilities,
        priority=int(raw.get("priority") or 1),
        input_cost=Decimal(str(raw.get("input_cost_per_million") or "0")),
        output_cost=Decimal(str(raw.get("output_cost_per_million") or "0")),
        timeout=provider.upstream_timeout,
    )


def _key(raw: dict) -> APIKey:
    secret = os.environ.get(str(raw["secret_env"]), "")
    if not secret:
        raise ValueError(f"missing environment variable {raw['secret_env']}")
    rpm = int(raw["rpm"])
    limit = int(raw["max_concurrency"])
    if rpm <= 0 or limit <= 0:
        raise ValueError("rpm and max_concurrency must be positive")
    allowed = tuple(str(item) for item in raw.get("allowed_models") or [])
    return APIKey(
        id=str(raw["id"]),
        secret_hash=hash_secret(secret),
        tenant_id=str(raw.get("tenant_id") or ""),
        status=str(raw.get("status") or "active"),
        rpm=rpm,
        max_concurrency=limit,
        allowed_models=allowed,
    )
