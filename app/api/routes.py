"""HTTP routes. Provider URLs and secrets stay out of this module."""

from __future__ import annotations

import secrets
from collections.abc import Callable

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.auth import authorize
from app.core.errors import Code, GatewayError, Stage
from app.core.models import (
    APIKey,
    Capability,
    EventKind,
    GatewayEvent,
    Invocation,
    Message,
    PromptRef,
    ToolSpec,
)
from app.schemas import (
    ChatRequest,
    ChatResponse,
    Choice,
    ChoiceMessage,
    ModelCard,
    ModelList,
    PromptRenderRequest,
)
from app.services.gateway import GatewayService


def build_router(
    gateway: GatewayService,
    keys_by_hash: dict[bytes, APIKey],
    list_models: Callable[[APIKey], tuple[str, ...]],
    render_prompt: Callable[[PromptRef], tuple[tuple[Message, ...], object]],
) -> APIRouter:
    router = APIRouter()

    def current_key(request: Request) -> APIKey:
        header = request.headers.get("authorization", "")
        if not header.startswith("Bearer ") or header == "Bearer ":
            raise GatewayError(Code.UNAUTHENTICATED, "invalid api key", stage=Stage.AUTH)
        from app.core.auth import hash_secret

        return authorize(keys_by_hash.get(hash_secret(header.removeprefix("Bearer "))))

    @router.get("/v1/models")
    async def models(key: APIKey = Depends(current_key)) -> ModelList:
        return ModelList(data=[ModelCard(id=name) for name in list_models(key)])

    @router.post("/v1/chat/completions")
    async def chat(body: ChatRequest, request: Request, key: APIKey = Depends(current_key)):
        invocation = _invocation(body, request, key)
        if body.stream:
            return await _stream_response(gateway, invocation, key, request)
        result = await gateway.complete(invocation, key)
        payload = ChatResponse(
            id=invocation.call_id,
            model=result.requested_model,
            choices=[
                Choice(
                    message=ChoiceMessage(content=result.content),
                    finish_reason=result.finish_reason,
                )
            ],
        )
        return JSONResponse(
            payload.model_dump(),
            headers={"X-Request-Id": invocation.trace_id},
        )

    @router.post("/v1/prompts/render")
    async def render(body: PromptRenderRequest, key: APIKey = Depends(current_key)):
        del key
        messages, use = render_prompt(
            PromptRef(name=body.name, version=body.version, variables=body.variables)
        )
        return {
            "messages": [{"role": item.role, "content": item.content} for item in messages],
            "prompt": {
                "name": use.name,
                "version": use.version,
                "hash": use.content_hash,
                "schema_version": use.schema_version,
            },
        }

    return router


async def _stream_response(gateway: GatewayService, invocation: Invocation, key: APIKey, request: Request):
    iterator = gateway.stream(invocation, key)
    try:
        first = await anext(iterator)
    except StopAsyncIteration:
        first = GatewayEvent(kind=EventKind.DONE)
    if first.kind is EventKind.ERROR and isinstance(first.error, GatewayError):
        raise first.error

    async def frames():
        yield _sse(invocation, first)
        async for event in iterator:
            if event.kind is EventKind.ERROR and isinstance(event.error, GatewayError):
                yield _sse_error(invocation.trace_id, event.error)
                return
            yield _sse(invocation, event)
            if await request.is_disconnected():
                return

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers={"X-Request-Id": invocation.trace_id},
    )


def _invocation(body: ChatRequest, request: Request, key: APIKey) -> Invocation:
    required: set[Capability] = set()
    if body.stream:
        required.add(Capability.STREAMING)
    schema = None
    if body.response_format is not None:
        schema = body.response_format.json_schema.get("schema") or body.response_format.json_schema
        required.add(Capability.STRUCTURED_OUTPUT)
    tools: tuple[ToolSpec, ...] = ()
    if body.tools:
        required.add(Capability.TOOL_CALLING)
        tools = tuple(
            ToolSpec(item.function.name, item.function.description, item.function.parameters)
            for item in body.tools
        )
    prompt_ref = None
    if body.prompt is not None:
        prompt_ref = PromptRef(body.prompt.name, body.prompt.version, body.prompt.variables)
    if not body.messages and prompt_ref is None:
        raise GatewayError(Code.INVALID_REQUEST, "messages are required", stage=Stage.VALIDATION)
    return Invocation(
        call_id=secrets.token_hex(8),
        trace_id=request.state.trace_id,
        requested_model=body.model,
        messages=tuple(Message(item.role, item.content) for item in body.messages),
        required=frozenset(required),
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        json_schema=schema,
        tools=tools,
        prompt_ref=prompt_ref,
        timeout=request.app.state.settings.server.gateway_timeout,
        tenant_id=key.tenant_id,
        key_id=key.id,
        run_id=getattr(request.state, "run_id", ""),
        step_id=getattr(request.state, "step_id", ""),
    )


def _sse(invocation: Invocation, event: GatewayEvent) -> str:
    if event.kind is EventKind.DONE:
        chunk = {
            "id": invocation.call_id,
            "object": "chat.completion.chunk",
            "model": invocation.requested_model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        return _dump(chunk) + "data: [DONE]\n\n"
    if event.kind is EventKind.TOOL_CALL and event.tool_call is not None:
        chunk = {
            "id": invocation.call_id,
            "object": "chat.completion.chunk",
            "model": invocation.requested_model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "id": event.tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": event.tool_call.name,
                                    "arguments": event.tool_call.arguments,
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        return _dump(chunk)
    chunk = {
        "id": invocation.call_id,
        "object": "chat.completion.chunk",
        "model": invocation.requested_model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": event.text},
                "finish_reason": None,
            }
        ],
    }
    return _dump(chunk)


def _sse_error(trace_id: str, error: GatewayError) -> str:
    import json

    body = {
        "error": {"message": error.message, "type": "upstream_error", "code": error.code.value},
        "request_id": trace_id,
    }
    return f"data: {json.dumps(body)}\n\n"


def _dump(payload: dict) -> str:
    import json

    return f"data: {json.dumps(payload)}\n\n"
