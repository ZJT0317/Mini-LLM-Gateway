"""Prompt versions, variable checks, and sandboxed rendering."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import yaml
from jinja2 import StrictUndefined
from jinja2.sandbox import SandboxedEnvironment
from jsonschema import ValidationError

from app.core.errors import Code, GatewayError, Stage
from app.core.models import (
    Capability,
    Invocation,
    Message,
    PromptRef,
    PromptUse,
    ToolSpec,
    estimate_tokens,
)
from app.services.structured import check_schema


@dataclass(frozen=True)
class PromptBundle:
    name: str
    version: str
    status: str
    system_template: str
    variable_schema: dict
    few_shot: tuple[Message, ...]
    output_schema: dict | None
    tools: tuple[ToolSpec, ...]
    default_model: str | None
    temperature: float | None
    max_tokens: int | None
    context_budget: int
    schema_version: str
    content_hash: str
    eval_set: str | None
    eval_threshold: float | None
    changelog: str


@dataclass(frozen=True)
class PreparedCall:
    invocation: Invocation
    prompt: PromptUse | None
    messages_text: tuple[Message, ...]


class PromptLibrary:
    def __init__(self, bundles: tuple[PromptBundle, ...]) -> None:
        self._bundles = {(item.name, item.version): item for item in bundles}
        if len(self._bundles) != len(bundles):
            raise ValueError("prompt names and versions must be unique")
        for bundle in bundles:
            check_schema(bundle.variable_schema)

    def get(self, name: str, version: str) -> PromptBundle | None:
        return self._bundles.get((name, version))

    def prepare(self, invocation: Invocation) -> PreparedCall:
        if invocation.prompt_ref is None:
            return PreparedCall(invocation, None, invocation.messages)
        bundle = self._require(invocation.prompt_ref)
        variables = invocation.prompt_ref.variables
        _check_variables(bundle.variable_schema, variables)
        system = _render(bundle.system_template, variables)
        rendered = (Message("system", system),) + bundle.few_shot + invocation.messages
        usage = estimate_tokens(rendered, "")
        if usage.input_tokens > bundle.context_budget:
            raise GatewayError(
                Code.PROMPT_INVALID,
                "prompt variable rejected",
                stage=Stage.PROMPT,
            )
        required = set(invocation.required)
        schema = invocation.json_schema
        if bundle.output_schema is not None and schema is None:
            schema = bundle.output_schema
            required.add(Capability.STRUCTURED_OUTPUT)
        tools = invocation.tools or bundle.tools
        if bundle.tools:
            required.add(Capability.TOOL_CALLING)
        model = invocation.requested_model or bundle.default_model or ""
        if not model:
            raise GatewayError(
                Code.PROMPT_INVALID,
                "prompt variable rejected",
                stage=Stage.PROMPT,
            )
        updated = replace(
            invocation,
            requested_model=model,
            messages=rendered,
            required=frozenset(required),
            max_tokens=invocation.max_tokens
            if invocation.max_tokens is not None
            else bundle.max_tokens,
            temperature=invocation.temperature
            if invocation.temperature is not None
            else bundle.temperature,
            json_schema=schema,
            tools=tools,
        )
        use = PromptUse(
            name=bundle.name,
            version=bundle.version,
            content_hash=bundle.content_hash,
            schema_version=bundle.schema_version,
        )
        return PreparedCall(updated, use, rendered)

    def render_only(self, ref: PromptRef) -> tuple[tuple[Message, ...], PromptUse]:
        bundle = self._require(ref)
        _check_variables(bundle.variable_schema, ref.variables)
        system = _render(bundle.system_template, ref.variables)
        messages = (Message("system", system),) + bundle.few_shot
        use = PromptUse(
            bundle.name, bundle.version, bundle.content_hash, bundle.schema_version
        )
        return messages, use

    def _require(self, ref: PromptRef) -> PromptBundle:
        bundle = self.get(ref.name, ref.version)
        if bundle is None or bundle.status != "published":
            raise GatewayError(
                Code.PROMPT_INVALID,
                "prompt variable rejected",
                stage=Stage.PROMPT,
            )
        return bundle


def load_prompts(directory: Path) -> PromptLibrary:
    bundles: list[PromptBundle] = []
    if not directory.exists():
        return PromptLibrary(())
    for path in sorted(directory.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        bundles.append(_bundle(raw))
    return PromptLibrary(tuple(bundles))


def _bundle(raw: dict) -> PromptBundle:
    few_shot = tuple(
        Message(str(item["role"]), str(item["content"]))
        for item in raw.get("few_shot") or []
    )
    tools = tuple(
        ToolSpec(
            name=str(item["name"]),
            description=str(item.get("description") or ""),
            parameters=dict(item.get("parameters") or {}),
        )
        for item in raw.get("tools") or []
    )
    system = str(raw.get("system") or "")
    fingerprint = hashlib.sha256()
    fingerprint.update(system.encode("utf-8"))
    fingerprint.update(
        json.dumps([(item.role, item.content) for item in few_shot]).encode("utf-8")
    )
    threshold = raw.get("eval_threshold")
    return PromptBundle(
        name=str(raw["name"]),
        version=str(raw["version"]),
        status=str(raw.get("status") or "draft"),
        system_template=system,
        variable_schema=dict(
            raw.get("variables") or {"type": "object", "properties": {}}
        ),
        few_shot=few_shot,
        output_schema=raw.get("output_schema"),
        tools=tools,
        default_model=raw.get("default_model"),
        temperature=raw.get("temperature"),
        max_tokens=raw.get("max_tokens"),
        context_budget=int(raw.get("context_budget") or 8192),
        schema_version=str(raw.get("schema_version") or "1"),
        content_hash=fingerprint.hexdigest(),
        eval_set=raw.get("eval_set"),
        eval_threshold=float(threshold) if threshold is not None else None,
        changelog=str(raw.get("changelog") or ""),
    )


def _check_variables(schema: dict, variables: dict) -> None:
    try:
        check_schema(schema).validate(variables)
    except (ValidationError, GatewayError) as exc:
        raise GatewayError(
            Code.PROMPT_INVALID,
            "prompt variable rejected",
            stage=Stage.PROMPT,
            cause=exc,
        ) from exc


def _render(template: str, variables: dict) -> str:
    environment = SandboxedEnvironment(autoescape=False, undefined=StrictUndefined)
    environment.filters.clear()
    environment.globals.clear()
    try:
        return environment.from_string(template).render(**variables)
    except Exception as exc:
        raise GatewayError(
            Code.PROMPT_INVALID,
            "prompt variable rejected",
            stage=Stage.PROMPT,
            cause=exc,
        ) from exc
