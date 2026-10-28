"""API key hashing and authorization rules."""

from __future__ import annotations

import hashlib

from app.core.errors import Code, GatewayError, Stage
from app.core.models import APIKey


def hash_secret(secret: str) -> bytes:
    return hashlib.sha256(secret.encode("utf-8")).digest()


def authorize(key: APIKey | None) -> APIKey:
    if key is None:
        raise GatewayError(
            Code.UNAUTHENTICATED,
            "invalid api key",
            stage=Stage.AUTH,
        )
    if key.status != "active":
        raise GatewayError(
            Code.KEY_DISABLED,
            "api key disabled",
            stage=Stage.AUTH,
        )
    return key


def ensure_model_visible(key: APIKey, logical_model: str) -> None:
    if key.allowed_models and logical_model not in key.allowed_models:
        raise GatewayError(
            Code.NO_COMPATIBLE_MODEL,
            "model not found",
            stage=Stage.ROUTING,
        )
