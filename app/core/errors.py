"""Stable gateway errors and their public HTTP mapping."""

from __future__ import annotations

from enum import StrEnum


class Code(StrEnum):
    UNAUTHENTICATED = "unauthenticated"
    KEY_DISABLED = "key_disabled"
    INVALID_REQUEST = "invalid_request"
    PROMPT_INVALID = "prompt_invalid"
    NO_COMPATIBLE_MODEL = "no_compatible_model"
    ADMISSION_REJECTED = "admission_rejected"
    DEADLINE_TOO_SHORT = "deadline_too_short"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    UPSTREAM_OVERLOADED = "upstream_overloaded"
    UPSTREAM_ERROR = "upstream_error"
    OUTPUT_INVALID = "output_invalid"
    CONTENT_REJECTED = "content_rejected"
    CANCELED = "canceled"
    INTERNAL = "internal"


class Stage:
    AUTH = "auth"
    VALIDATION = "validation"
    PROMPT = "prompt"
    ROUTING = "routing"
    ADMISSION = "admission"
    CONNECTION = "connection"
    BEFORE_FIRST_TOKEN = "before_first_token"
    STREAMING = "streaming"
    OUTPUT = "output"
    CONTENT = "content"


class GatewayError(Exception):
    """Public message only. ``cause`` stays off the response and ``str()``."""

    def __init__(
        self,
        code: Code,
        message: str,
        *,
        retryable: bool = False,
        stage: str,
        provider_request_id: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.stage = stage
        self.provider_request_id = provider_request_id
        self.cause = cause

    def __str__(self) -> str:
        return self.message


def http_status(error: GatewayError) -> tuple[int, str, str]:
    """Return HTTP status, OpenAI error type, and the public message."""
    if error.message == "gateway timeout":
        return 504, "timeout_error", "gateway timeout"
    table: dict[Code, tuple[int, str, str]] = {
        Code.INVALID_REQUEST: (400, "invalid_request_error", error.message),
        Code.PROMPT_INVALID: (400, "invalid_request_error", "prompt variable rejected"),
        Code.UNAUTHENTICATED: (401, "authentication_error", "invalid api key"),
        Code.KEY_DISABLED: (403, "permission_error", "api key disabled"),
        Code.NO_COMPATIBLE_MODEL: (404, "invalid_request_error", "model not found"),
        Code.ADMISSION_REJECTED: (429, "rate_limit_error", error.message),
        Code.DEADLINE_TOO_SHORT: (429, "rate_limit_error", "deadline too short"),
        Code.UPSTREAM_UNAVAILABLE: (502, "upstream_error", "upstream request failed"),
        Code.UPSTREAM_OVERLOADED: (502, "upstream_error", "upstream request failed"),
        Code.UPSTREAM_ERROR: (502, "upstream_error", "upstream request failed"),
        Code.OUTPUT_INVALID: (502, "upstream_error", "output validation failed"),
        Code.CONTENT_REJECTED: (400, "invalid_request_error", "content rejected"),
        Code.INTERNAL: (500, "internal_error", "internal error"),
    }
    return table.get(error.code, (500, "internal_error", "internal error"))
