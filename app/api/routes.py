"""HTTP routes. Provider URLs and secrets stay out of this module."""

from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import Callable

import anyio
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
    ToolCall,
    ToolSpec,
    TokenUsage,
)
from app.schemas import (
    ChatRequest,
    ChatResponse,
    Choice,
    ChoiceMessage,
    ModelCard,
    ModelList,
    PromptRenderRequest,
    ToolCallIn,
    ToolCallFunctionIn,
    ToolChoiceIn,
    JsonSchemaEnvelope,
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
            raise GatewayError(
                Code.UNAUTHENTICATED, "invalid api key", stage=Stage.AUTH
            )
        from app.core.auth import hash_secret

        return authorize(keys_by_hash.get(hash_secret(header.removeprefix("Bearer "))))

    @router.get("/v1/models")
    async def models(key: APIKey = Depends(current_key)) -> ModelList:
        return ModelList(data=[ModelCard(id=name) for name in list_models(key)])

    @router.post("/v1/chat/completions")
    async def chat(
        body: ChatRequest, request: Request, key: APIKey = Depends(current_key)
    ):
        invocation = _invocation(body, request, key)
        if body.stream:
            return await _stream_response(gateway, invocation, key, request)
        result = await _while_connected(gateway.complete(invocation, key), request)
        payload = ChatResponse(
            id=invocation.call_id,
            model=result.requested_model,
            choices=[
                Choice(
                    message=ChoiceMessage(
                        content=result.content
                        or (None if result.tool_calls or result.refusal else ""),
                        tool_calls=[
                            ToolCallIn(
                                id=call.id,
                                function=ToolCallFunctionIn(
                                    name=call.name,
                                    arguments=call.arguments,
                                ),
                            )
                            for call in result.tool_calls
                        ]
                        or None,
                        refusal=result.refusal,
                    ),
                    finish_reason=result.finish_reason,
                )
            ],
            created=invocation.created,
            usage=_usage_payload(result.usage),
        )
        data = payload.model_dump()
        if not result.tool_calls:
            data["choices"][0]["message"].pop("tool_calls")
        if result.refusal is None:
            data["choices"][0]["message"].pop("refusal")
        return JSONResponse(
            data,
            headers={"X-Request-Id": invocation.trace_id},
        )

    @router.post("/v1/prompts/render")
    async def render(body: PromptRenderRequest, key: APIKey = Depends(current_key)):
        del key
        messages, use = render_prompt(
            PromptRef(name=body.name, version=body.version, variables=body.variables)
        )
        return {
            "messages": [
                {"role": item.role, "content": item.content} for item in messages
            ],
            "prompt": {
                "name": use.name,
                "version": use.version,
                "hash": use.content_hash,
                "schema_version": use.schema_version,
            },
        }

    return router


async def _stream_response(
    gateway: GatewayService, invocation: Invocation, key: APIKey, request: Request
):
    iterator = gateway.stream(invocation, key)
    try:
        first = await _while_connected(anext(iterator), request)
    except StopAsyncIteration:
        first = GatewayEvent(kind=EventKind.DONE)
    except BaseException:
        with anyio.CancelScope(shield=True):
            await iterator.aclose()
        raise
    if first.kind is EventKind.ERROR and isinstance(first.error, GatewayError):
        await iterator.aclose()
        raise first.error

    async def frames():
        try:
            yield _sse(invocation, first)
            async for event in iterator:
                if event.kind is EventKind.ERROR and isinstance(
                    event.error, GatewayError
                ):
                    yield _sse_error(invocation.trace_id, event.error)
                    return
                yield _sse(invocation, event)
        finally:
            with anyio.CancelScope(shield=True):
                await iterator.aclose()

    return _ClosingStreamResponse(
        frames(),
        source=iterator,
        media_type="text/event-stream",
        headers={"X-Request-Id": invocation.trace_id},
    )


async def _while_connected(operation, request: Request):
    """Cancel upstream work if the peer disconnects before the response starts."""

    async def disconnected():
        while (await request.receive())["type"] != "http.disconnect":
            pass

    work = asyncio.ensure_future(operation)
    monitor = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait(
            (work, monitor), return_when=asyncio.FIRST_COMPLETED
        )
        if work in done:
            return await work
        raise GatewayError(Code.CANCELED, "client disconnected", stage=Stage.STREAMING)
    finally:
        for task in (work, monitor):
            if not task.done():
                task.cancel()
        with anyio.CancelScope(shield=True):
            await asyncio.gather(work, monitor, return_exceptions=True)


class _ClosingStreamResponse(StreamingResponse):
    """Also close a prefetched stream when sending headers/body fails."""

    def __init__(self, *args, source, **kwargs):
        super().__init__(*args, **kwargs)
        self._source = source

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await self.body_iterator.aclose()
                await self._source.aclose()


def _invocation(body: ChatRequest, request: Request, key: APIKey) -> Invocation:
    required: set[Capability] = set()
    if body.stream:
        required.add(Capability.STREAMING)
    schema = None
    schema_options = None
    if body.response_format is not None:
        raw_schema = body.response_format.json_schema
        schema = raw_schema["schema"] if "schema" in raw_schema else raw_schema
        if "schema" in raw_schema:
            schema_options = JsonSchemaEnvelope.model_validate(raw_schema)
        if not isinstance(schema, (dict, bool)):
            raise GatewayError(
                Code.INVALID_REQUEST, "invalid JSON schema", stage=Stage.VALIDATION
            )
        required.add(Capability.STRUCTURED_OUTPUT)
    tools: tuple[ToolSpec, ...] = ()
    if body.tools:
        required.add(Capability.TOOL_CALLING)
        tools = tuple(
            ToolSpec(
                item.function.name, item.function.description, item.function.parameters
            )
            for item in body.tools
        )
    if any(message.role == "tool" or message.tool_calls for message in body.messages):
        required.add(Capability.TOOL_CALLING)
    prompt_ref = None
    if body.prompt is not None:
        prompt_ref = PromptRef(
            body.prompt.name, body.prompt.version, body.prompt.variables
        )
    if not body.messages and prompt_ref is None:
        raise GatewayError(
            Code.INVALID_REQUEST, "messages are required", stage=Stage.VALIDATION
        )
    return Invocation(
        call_id=secrets.token_hex(8),
        trace_id=request.state.trace_id,
        requested_model=body.model,
        messages=tuple(
            Message(
                item.role,
                item.content,
                tuple(
                    ToolCall(
                        call.id, call.function.name, call.function.arguments, index
                    )
                    for index, call in enumerate(item.tool_calls or [])
                ),
                item.tool_call_id,
            )
            for item in body.messages
        ),
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
        tool_choice=(
            body.tool_choice.model_dump()
            if isinstance(body.tool_choice, ToolChoiceIn)
            else body.tool_choice
        ),
        schema_name="response" if schema_options is None else schema_options.name,
        schema_description=None
        if schema_options is None
        else schema_options.description,
        schema_strict=None if schema_options is None else schema_options.strict,
        include_usage=body.stream_options is not None
        and body.stream_options.include_usage,
        created=int(time.time()),
    )


def _sse(invocation: Invocation, event: GatewayEvent) -> str:
    if event.kind is EventKind.DONE:
        chunk = {
            "id": invocation.call_id,
            "object": "chat.completion.chunk",
            "model": event.requested_model or invocation.requested_model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": event.finish_reason or "stop",
                }
            ],
        }
        chunk["created"] = invocation.created
        if invocation.include_usage:
            chunk["usage"] = None
        frames = _dump(chunk)
        if invocation.include_usage and event.usage is not None:
            frames += _dump(
                {
                    "id": invocation.call_id,
                    "object": "chat.completion.chunk",
                    "created": invocation.created,
                    "model": event.requested_model or invocation.requested_model,
                    "choices": [],
                    "usage": _usage_payload(event.usage),
                }
            )
        return frames + "data: [DONE]\n\n"
    if event.kind is EventKind.TOOL_CALL and event.tool_call is not None:
        chunk = {
            "id": invocation.call_id,
            "object": "chat.completion.chunk",
            "model": event.requested_model or invocation.requested_model,
            "created": invocation.created,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": event.tool_call.index,
                                **(
                                    {"id": event.tool_call.id, "type": "function"}
                                    if event.tool_call.id
                                    else {}
                                ),
                                "function": {
                                    **(
                                        {"name": event.tool_call.name}
                                        if event.tool_call.name
                                        else {}
                                    ),
                                    "arguments": event.tool_call.arguments,
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        if invocation.include_usage:
            chunk["usage"] = None
        return _dump(chunk)
    chunk = {
        "id": invocation.call_id,
        "object": "chat.completion.chunk",
        "model": event.requested_model or invocation.requested_model,
        "created": invocation.created,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "refusal"
                    if event.kind is EventKind.REFUSAL
                    else "content": event.text
                },
                "finish_reason": None,
            }
        ],
    }
    if invocation.include_usage:
        chunk["usage"] = None
    return _dump(chunk)


def _usage_payload(usage: TokenUsage) -> dict:
    return {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "prompt_tokens_details": {"cached_tokens": usage.cached_tokens},
        "completion_tokens_details": {"reasoning_tokens": usage.reasoning_tokens},
    }


def _sse_error(trace_id: str, error: GatewayError) -> str:
    import json

    body = {
        "error": {
            "message": error.message,
            "type": "upstream_error",
            "code": error.code.value,
        },
        "request_id": trace_id,
    }
    return f"data: {json.dumps(body)}\n\n"


def _dump(payload: dict) -> str:
    import json

    return f"data: {json.dumps(payload)}\n\n"
