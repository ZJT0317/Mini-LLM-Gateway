"""OpenAI-compatible adapter. Provider JSON does not leave this module."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncGenerator
from dataclasses import replace

import httpx

from app.core.errors import Code, GatewayError, Stage
from app.core.models import (
    AdapterOutcome,
    Capability,
    EventKind,
    GatewayEvent,
    Message,
    TokenUsage,
    ToolCall,
    UpstreamCall,
)

logger = logging.getLogger("gateway.upstream")
_SECRET = re.compile(r"(Bearer\s+)\S+|(sk-[A-Za-z0-9_\-]+)")


def redact(text: str) -> str:
    return _SECRET.sub(lambda match: (match.group(1) or "") + "***", text)


class OpenAIAdapter:
    def __init__(
        self,
        name: str,
        api_key: str,
        capabilities: frozenset[Capability],
        client: httpx.AsyncClient,
    ) -> None:
        self.name = name
        self.capabilities = capabilities
        self._api_key = api_key
        self._client = client

    async def invoke(self, call: UpstreamCall) -> AdapterOutcome:
        response = await self._send(call, stream=False)
        self._raise_for_status(response)
        try:
            payload = response.json()
            return _outcome_from_payload(payload, response)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError) as exc:
            raise self._error(
                Code.UPSTREAM_ERROR, Stage.BEFORE_FIRST_TOKEN, response
            ) from exc

    async def stream(self, call: UpstreamCall) -> AsyncGenerator[GatewayEvent, None]:
        business = False
        try:
            async with self._client.stream(
                "POST",
                _url(call.base_url),
                json=_body(call, stream=True),
                headers=self._headers(),
                timeout=call.timeout.total_seconds(),
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    self._log_failure(
                        response.status_code, body.decode("utf-8", "replace")
                    )
                    raise self._error(
                        *_status_code(response.status_code),
                        response,
                    )
                finish_reason = None
                async for line in response.aiter_lines():
                    for event in _parse_sse_line(line):
                        if event.kind is EventKind.DONE:
                            if finish_reason is None:
                                raise _stream_error()
                            yield replace(
                                event,
                                finish_reason=finish_reason,
                                provider_request_id=response.headers.get(
                                    "x-request-id"
                                ),
                            )
                            return
                        if event.kind is EventKind.FINISH:
                            finish_reason = event.finish_reason
                        elif event.kind in {
                            EventKind.DELTA,
                            EventKind.TOOL_CALL,
                            EventKind.REFUSAL,
                        }:
                            if finish_reason is not None:
                                raise _stream_error()
                            business = True
                        yield replace(
                            event,
                            provider_request_id=(
                                response.headers.get("x-request-id")
                                or event.provider_request_id
                            ),
                        )
                raise GatewayError(
                    Code.UPSTREAM_UNAVAILABLE,
                    "upstream request failed",
                    retryable=True,
                    stage=Stage.STREAMING if business else Stage.BEFORE_FIRST_TOKEN,
                    provider_request_id=response.headers.get("x-request-id"),
                )
        except GatewayError:
            raise
        except (ValueError, TypeError, AttributeError, KeyError, IndexError) as exc:
            raise _stream_error() from exc
        except httpx.TimeoutException as exc:
            raise GatewayError(
                Code.UPSTREAM_OVERLOADED,
                "upstream request failed",
                retryable=True,
                stage=Stage.STREAMING if business else Stage.BEFORE_FIRST_TOKEN,
                cause=exc,
            ) from exc
        except httpx.TransportError as exc:
            raise GatewayError(
                Code.UPSTREAM_UNAVAILABLE,
                "upstream request failed",
                retryable=True,
                stage=Stage.STREAMING if business else Stage.CONNECTION,
                cause=exc,
            ) from exc

    async def _send(self, call: UpstreamCall, *, stream: bool) -> httpx.Response:
        try:
            return await self._client.post(
                _url(call.base_url),
                json=_body(call, stream=stream),
                headers=self._headers(),
                timeout=call.timeout.total_seconds(),
            )
        except httpx.TimeoutException as exc:
            raise GatewayError(
                Code.UPSTREAM_OVERLOADED,
                "upstream request failed",
                retryable=True,
                stage=Stage.BEFORE_FIRST_TOKEN,
                cause=exc,
            ) from exc
        except httpx.TransportError as exc:
            raise GatewayError(
                Code.UPSTREAM_UNAVAILABLE,
                "upstream request failed",
                retryable=True,
                stage=Stage.CONNECTION,
                cause=exc,
            ) from exc

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code == 200:
            return
        self._log_failure(response.status_code, response.text)
        code, stage = _status_code(response.status_code)
        raise self._error(code, stage, response)

    def _error(self, code: Code, stage: str, response: httpx.Response) -> GatewayError:
        return GatewayError(
            code,
            "upstream request failed",
            retryable=code in {Code.UPSTREAM_UNAVAILABLE, Code.UPSTREAM_OVERLOADED},
            stage=stage,
            provider_request_id=response.headers.get("x-request-id"),
        )

    def _log_failure(self, status: int, body: str) -> None:
        snippet = redact(body)[:256]
        logger.warning("upstream status=%s body=%s", status, snippet)


def _url(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def _status_code(status: int) -> tuple[Code, str]:
    if status == 429:
        return Code.UPSTREAM_OVERLOADED, Stage.BEFORE_FIRST_TOKEN
    if status >= 500:
        return Code.UPSTREAM_UNAVAILABLE, Stage.CONNECTION
    return Code.UPSTREAM_ERROR, Stage.BEFORE_FIRST_TOKEN


def _body(call: UpstreamCall, *, stream: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": call.resolved_model,
        "messages": [_message_body(message) for message in call.messages],
    }
    if call.max_tokens is not None:
        payload["max_tokens"] = call.max_tokens
    if call.temperature is not None:
        payload["temperature"] = call.temperature
    if call.json_schema is not None:
        schema_options = {"name": call.schema_name, "schema": call.json_schema}
        if call.schema_description is not None:
            schema_options["description"] = call.schema_description
        if call.schema_strict is not None:
            schema_options["strict"] = call.schema_strict
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": schema_options,
        }
    if call.tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in call.tools
        ]
    if call.tool_choice is not None:
        payload["tool_choice"] = call.tool_choice
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    return payload


def _message_body(message: Message) -> dict[str, object]:
    body: dict[str, object] = {"role": message.role, "content": message.content}
    if message.tool_call_id is not None:
        body["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        body["tool_calls"] = [
            {
                "id": tool.id,
                "type": "function",
                "function": {
                    "name": tool.name,
                    "arguments": tool.arguments,
                },
            }
            for tool in message.tool_calls
        ]
    return body


def _outcome_from_payload(payload: dict, response: httpx.Response) -> AdapterOutcome:
    choices = payload.get("choices") or []
    if not choices:
        raise GatewayError(
            Code.UPSTREAM_ERROR,
            "upstream request failed",
            stage=Stage.BEFORE_FIRST_TOKEN,
            provider_request_id=response.headers.get("x-request-id"),
        )
    message = choices[0].get("message") or {}
    if message.get("content") is not None and not isinstance(message["content"], str):
        raise ValueError("invalid content")
    tool_calls = tuple(
        _tool_call(item, index)
        for index, item in enumerate(message.get("tool_calls") or [])
    )
    if any(not tool.id or not tool.name for tool in tool_calls):
        raise ValueError("incomplete tool call")
    refusal = message.get("refusal")
    if refusal is not None and not isinstance(refusal, str):
        raise ValueError("invalid refusal")
    if choices[0].get("finish_reason") == "content_filter":
        refusal = refusal or "content rejected"
    return AdapterOutcome(
        content=message.get("content") or "",
        tool_calls=tool_calls,
        finish_reason=choices[0].get("finish_reason") or "stop",
        usage=_usage(payload.get("usage")),
        provider_request_id=response.headers.get("x-request-id") or payload.get("id"),
        refusal=refusal or None,
    )


def _tool_call(item: dict, index: int = 0) -> ToolCall:
    function = item.get("function") or {}
    index = int(item.get("index", index))
    if index < 0:
        raise ValueError("invalid tool index")
    return ToolCall(
        id=str(item.get("id") or ""),
        name=str(function.get("name") or ""),
        arguments=str(function.get("arguments") or ""),
        index=index,
    )


def _usage(raw: dict | None) -> TokenUsage | None:
    if not raw:
        return None
    details = raw.get("prompt_tokens_details") or {}
    completion = raw.get("completion_tokens_details") or {}
    return TokenUsage(
        input_tokens=int(raw.get("prompt_tokens") or 0),
        output_tokens=int(raw.get("completion_tokens") or 0),
        cached_tokens=int(details.get("cached_tokens") or 0),
        reasoning_tokens=int(completion.get("reasoning_tokens") or 0),
        estimated=False,
    )


def _stream_error() -> GatewayError:
    return GatewayError(
        Code.UPSTREAM_ERROR, "upstream request failed", stage=Stage.STREAMING
    )


def _parse_sse_line(line: str) -> tuple[GatewayEvent, ...]:
    text = line.strip()
    if not text or text.startswith(":"):
        return ()
    if not text.startswith("data:"):
        return ()
    data = text[5:].strip()
    if data == "[DONE]":
        return (GatewayEvent(kind=EventKind.DONE),)
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise _stream_error() from exc
    if not isinstance(payload, dict) or payload.get("error") is not None:
        raise _stream_error()
    usage = _usage(payload.get("usage"))
    choices = payload.get("choices") or []
    events: list[GatewayEvent] = []
    if choices:
        choice = choices[0]
        delta = choice.get("delta") or {}
        if delta.get("content"):
            events.append(GatewayEvent(kind=EventKind.DELTA, text=delta["content"]))
        if delta.get("refusal"):
            if not isinstance(delta["refusal"], str):
                raise _stream_error()
            events.append(GatewayEvent(kind=EventKind.REFUSAL, text=delta["refusal"]))
        for item in delta.get("tool_calls") or []:
            events.append(
                GatewayEvent(kind=EventKind.TOOL_CALL, tool_call=_tool_call(item))
            )
        if choice.get("finish_reason"):
            events.append(
                GatewayEvent(
                    kind=EventKind.FINISH, finish_reason=choice["finish_reason"]
                )
            )
    if usage is not None:
        events.append(GatewayEvent(kind=EventKind.USAGE, usage=usage))
    return tuple(
        replace(event, provider_request_id=payload.get("id")) for event in events
    )
