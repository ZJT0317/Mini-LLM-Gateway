"""Assemble the process. Business decisions stay in services and core."""

from __future__ import annotations

import argparse
import logging
import re
import secrets
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import build_router
from app.config import Settings, load_settings
from app.core.admission import AdmissionGate
from app.core.errors import Code, GatewayError, http_status
from app.core.models import APIKey, PromptRef
from app.services.gateway import GatewayService
from app.services.metrics import Metrics
from app.services.prompts import PromptLibrary, load_prompts
from app.services.router import AdapterRegistry, HealthBoard, Router
from app.services.upstream import OpenAIAdapter
from app.services.usage import Ledger

_TRACE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def create_app(
    settings: Settings,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    sleeper: Callable | None = None,
    rng: Callable[[], float] | None = None,
    clock: Callable[[], datetime] | None = None,
    prompts: PromptLibrary | None = None,
) -> FastAPI:
    ledger = Ledger(settings.server.database)
    health = HealthBoard()
    clients: list[httpx.AsyncClient] = []
    adapters = {}
    for provider in settings.providers:
        client = httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(provider.upstream_timeout.total_seconds()),
        )
        clients.append(client)
        adapters[provider.name] = OpenAIAdapter(
            provider.name,
            provider.api_key,
            provider.capabilities,
            client,
        )
    registry = AdapterRegistry(settings.models, adapters)
    metrics = Metrics()
    library = prompts if prompts is not None else load_prompts(settings.prompts_directory)
    gateway = GatewayService(
        prompts=library,
        admission=AdmissionGate(),
        router=Router(settings.models, health),
        registry=registry,
        health=health,
        ledger=ledger,
        metrics=metrics,
        clock=clock or (lambda: datetime.now(timezone.utc)),
        sleeper=sleeper or _sleep,
        rng=rng or _random,
        max_attempts=settings.limits.retry_max_attempts,
        repair_max_attempts=settings.limits.repair_max_attempts,
        max_tokens_cap=settings.limits.max_tokens_cap,
    )
    keys = {item.secret_hash: item for item in settings.keys}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        for client in clients:
            await client.aclose()
        ledger.close()

    app = FastAPI(lifespan=lifespan)
    app.state.settings = settings
    app.state.ledger = ledger
    app.state.metrics = metrics
    app.state.gateway = gateway

    @app.middleware("http")
    async def observe(request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        request.state.trace_id = incoming if _TRACE.fullmatch(incoming or "") else secrets.token_hex(8)
        request.state.run_id = request.headers.get("x-run-id", "")
        request.state.step_id = request.headers.get("x-step-id", "")
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > settings.server.max_body_bytes:
            return _error_response(
                request.state.trace_id,
                400,
                "invalid_request_error",
                "invalid_request",
                "request body is too large",
            )
        response = await call_next(request)
        response.headers["X-Request-Id"] = request.state.trace_id
        logging.getLogger("gateway.access").info(
            "request_id=%s method=%s path=%s status=%s",
            request.state.trace_id,
            request.method,
            request.url.path,
            response.status_code,
        )
        return response

    @app.exception_handler(GatewayError)
    async def on_gateway_error(request: Request, exc: GatewayError):
        if exc.code is Code.CANCELED:
            return JSONResponse(status_code=204, content=None, headers={"X-Request-Id": _trace(request)})
        status, error_type, message = http_status(exc)
        return _error_response(_trace(request), status, error_type, exc.code.value, message)

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError):
        detail = exc.errors()[0]["msg"] if exc.errors() else "invalid request"
        return _error_response(
            _trace(request),
            400,
            "invalid_request_error",
            Code.INVALID_REQUEST.value,
            detail,
        )

    def visible(key: APIKey) -> tuple[str, ...]:
        names = registry.logical_models()
        if key.allowed_models:
            allowed = set(key.allowed_models)
            names = tuple(name for name in names if name in allowed)
        return names

    def render(ref: PromptRef):
        return library.render_only(ref)

    app.include_router(
        build_router(gateway, keys, visible, render),
    )
    return app


def _error_response(trace_id: str, status: int, error_type: str, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {"message": message, "type": error_type, "code": code},
            "request_id": trace_id,
        },
        headers={"X-Request-Id": trace_id},
    )


def _trace(request: Request) -> str:
    return getattr(request.state, "trace_id", "")


async def _sleep(delay) -> None:
    import asyncio

    await asyncio.sleep(delay.total_seconds())


def _random() -> float:
    import random

    return random.random()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="gateway.yaml")
    args = parser.parse_args(argv)
    settings = load_settings(Path(args.config))
    logging.basicConfig(level=getattr(logging, settings.log.level.upper(), logging.INFO))
    uvicorn.run(
        create_app(settings),
        host=settings.server.host,
        port=settings.server.port,
        timeout_graceful_shutdown=settings.server.shutdown_grace.total_seconds(),
    )


if __name__ == "__main__":
    main()
