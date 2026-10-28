"""OpenAI-compatible adapter. Provider JSON does not leave this module."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator

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
    ToolSpec,
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
        except json.JSONDecodeError as exc:
            raise self._error(Code.UPSTREAM_ERROR, Stage.BEFORE_FIRST_TOKEN, response) from exc
        return _outcome_from_payload(payload, response)

    async def stream(self, call: UpstreamCall) -> AsyncIterator[GatewayEvent]:
        try:
            async with self._client.stream(
                "POST",
                _url(call.base_url),
                json=_body(call, stream=True),
                headers=self._headers(),
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    self._log_failure(response.status_code, body.decode("utf-8", "replace"))
                    raise self._error(
                        *_status_code(response.status_code),
                        response,
                    )
                done = False
                async for line in response.aiter_lines():
                    event = _parse_sse_line(line)
                    if event is None:
                        continue
                    if event.kind is EventKind.DONE:
                        done = True
                    yield event
                if not done:
                    yield GatewayEvent(kind=EventKind.DONE)
        except GatewayError:
            raise
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

    async def _send(self, call: UpstreamCall, *, stream: bool) -> httpx.Response:
        try:
            return await self._client.post(
                _url(call.base_url),
                json=_body(call, stream=stream),
                headers=self._headers(),
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


def _body(call: UpstreamCall, *, stream: bool) -> dict:
    payload: dict = {
        "model": call.resolved_model,
        "messages": [{"role": message.role, "content": message.content} for message in call.messages],
    }
    if call.max_tokens is not None:
        payload["max_tokens"] = call.max_tokens
    if call.temperature is not None:
        payload["temperature"] = call.temperature
    if call.json_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": call.json_schema},
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
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    return payload


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
    tool_calls = tuple(_tool_call(item) for item in message.get("tool_calls") or [])
    return AdapterOutcome(
        content=message.get("content") or "",
        tool_calls=tool_calls,
        finish_reason=choices[0].get("finish_reason") or "stop",
        usage=_usage(payload.get("usage")),
        provider_request_id=response.headers.get("x-request-id") or payload.get("id"),
    )


def _tool_call(item: dict) -> ToolCall:
    function = item.get("function") or {}
    return ToolCall(
        id=str(item.get("id") or ""),
        name=str(function.get("name") or ""),
        arguments=str(function.get("arguments") or ""),
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


def _parse_sse_line(line: str) -> GatewayEvent | None:
    text = line.strip()
    if not text or text.startswith(":"):
        return None
    if not text.startswith("data:"):
        return None
    data = text[5:].strip()
    if data == "[DONE]":
        return GatewayEvent(kind=EventKind.DONE)
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return GatewayEvent(
            kind=EventKind.ERROR,
            error=GatewayError(
                Code.UPSTREAM_ERROR,
                "upstream request failed",
                stage=Stage.STREAMING,
            ),
        )
    usage = _usage(payload.get("usage"))
    choices = payload.get("choices") or []
    if not choices:
        if usage is not None:
            return GatewayEvent(kind=EventKind.USAGE, usage=usage)
        return None
    choice = choices[0]
    delta = choice.get("delta") or {}
    finish = choice.get("finish_reason")
    if delta.get("tool_calls"):
        call = _tool_call(delta["tool_calls"][0])
        return GatewayEvent(kind=EventKind.TOOL_CALL, tool_call=call)
    content = delta.get("content") or ""
    if content:
        return GatewayEvent(kind=EventKind.DELTA, text=content)
    if usage is not None:
        return GatewayEvent(kind=EventKind.USAGE, usage=usage)
    if finish:
        return GatewayEvent(kind=EventKind.DONE)
    return None
