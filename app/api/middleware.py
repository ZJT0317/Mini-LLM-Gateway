"""ASGI request context and bounded body reads, including chunked uploads."""

import logging
import re
import secrets

from starlette.datastructures import Headers
from starlette.responses import JSONResponse

_TRACE = re.compile(r"[A-Za-z0-9_-]{1,64}")


class RequestContextMiddleware:
    def __init__(self, app, max_body_bytes: int):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        incoming = headers.get("x-request-id", "")
        trace_id = incoming if _TRACE.fullmatch(incoming) else secrets.token_hex(8)
        scope.setdefault("state", {}).update(
            trace_id=trace_id,
            run_id=headers.get("x-run-id", ""),
            step_id=headers.get("x-step-id", ""),
        )
        length = headers.get("content-length", "")
        if length.isdigit() and int(length) > self.max_body_bytes:
            await self._reject(trace_id, scope, receive, send)
            return
        chunks = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self.max_body_bytes:
                await self._reject(trace_id, scope, receive, send)
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        chunks.clear()
        delivered = False
        status = 500

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        async def observed_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message = dict(message)
                message["headers"] = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() != b"x-request-id"
                ] + [(b"x-request-id", trace_id.encode())]
            await send(message)

        try:
            await self.app(scope, replay, observed_send)
        finally:
            logging.getLogger("gateway.access").info(
                "request_id=%s method=%s path=%s status=%s",
                trace_id,
                scope["method"],
                scope["path"],
                status,
            )

    @staticmethod
    async def _reject(trace_id, scope, receive, send):
        response = JSONResponse(
            {
                "error": {
                    "message": "request body is too large",
                    "type": "invalid_request_error",
                    "code": "invalid_request",
                },
                "request_id": trace_id,
            },
            status_code=400,
            headers={"X-Request-Id": trace_id},
        )
        await response(scope, receive, send)
