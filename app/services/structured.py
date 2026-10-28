"""Extract one JSON value and validate it with a Pydantic model."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    create_model,
)

from app.core.models import ValidationResult


def extract_json(text: str) -> Any:
    start_object = text.find("{")
    start_array = text.find("[")
    starts = [index for index in (start_object, start_array) if index >= 0]
    if not starts:
        raise ValueError("no json")
    decoder = json.JSONDecoder()
    value, _ = decoder.raw_decode(text[min(starts) :])
    return value


def validate_json(text: str, schema: dict) -> tuple[ValidationResult, str]:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    try:
        value = extract_json(text)
        checked = _validate(value, schema)
        dumped = checked.model_dump() if isinstance(checked, BaseModel) else checked
        canonical = json.dumps(dumped, sort_keys=True)
    except (ValueError, ValidationError, json.JSONDecodeError, TypeError):
        return ValidationResult(ok=False, text_hash=digest), text
    return ValidationResult(ok=True, text_hash=digest), canonical


def _validate(value: Any, schema: dict) -> Any:
    if _is_object(schema):
        return model_from_schema(schema).model_validate(value)
    return TypeAdapter(_annotation(schema)).validate_python(value)


def model_from_schema(schema: dict) -> type[BaseModel]:
    if not _is_object(schema):
        raise ValueError("object schema required")
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    fields: dict[str, tuple] = {}
    for name, spec in properties.items():
        annotation = _annotation(spec)
        default = ... if name in required else None
        fields[name] = (annotation, default)
    extra = "forbid" if schema.get("additionalProperties") is False else "ignore"
    return create_model(
        "StructuredOutput",
        __config__=ConfigDict(extra=extra),
        **fields,
    )


def _is_object(schema: dict) -> bool:
    kind = schema.get("type")
    return kind == "object" or (kind is None and "properties" in schema)


def _annotation(spec: dict) -> Any:
    if any(key in spec for key in ("oneOf", "anyOf", "allOf")):
        raise ValueError("unsupported schema composition")
    if "enum" in spec:
        values = tuple(spec["enum"])
        if not values:
            raise ValueError("empty enum")
        return Literal[*values]
    kind = spec.get("type", "object" if "properties" in spec else "string")
    if kind == "string":
        return _string_type(spec)
    if kind == "integer":
        return Annotated[int, Field(**_bounds(spec))]
    if kind == "number":
        return Annotated[float, BeforeValidator(_as_number), Field(**_bounds(spec, strict=False))]
    if kind == "boolean":
        return Annotated[bool, Field(strict=True)]
    if kind == "object":
        return model_from_schema(spec)
    if kind == "array":
        item_schema = spec.get("items") or {"type": "string"}
        return list[_annotation(item_schema)]  # type: ignore[valid-type]
    return str


def _as_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("number required")
    return float(value)


def _bounds(spec: dict, *, strict: bool = True) -> dict[str, Any]:
    constraints: dict[str, Any] = {}
    if strict:
        constraints["strict"] = True
    if "minimum" in spec:
        constraints["ge"] = spec["minimum"]
    if "maximum" in spec:
        constraints["le"] = spec["maximum"]
    return constraints


def _string_type(spec: dict) -> Any:
    constraints: dict[str, Any] = {}
    if "minLength" in spec:
        constraints["min_length"] = int(spec["minLength"])
    if "maxLength" in spec:
        constraints["max_length"] = int(spec["maxLength"])
    if "pattern" in spec:
        constraints["pattern"] = str(spec["pattern"])
    metadata: list[Any] = []
    if constraints:
        metadata.append(Field(**constraints))
    fmt = spec.get("format")
    if fmt in {"date", "date-time", "email"}:
        metadata.append(AfterValidator(lambda value, expected=fmt: _check_format(value, expected)))
    if not metadata:
        return str
    return Annotated[str, *metadata]


def _check_format(value: str, fmt: str) -> str:
    if fmt == "date":
        date.fromisoformat(value)
    elif fmt == "date-time":
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif fmt == "email" and re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value) is None:
        raise ValueError("invalid email")
    return value
