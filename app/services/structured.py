"""JSON Schema 2020-12 validation without coercion or remote references."""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker, SchemaError, ValidationError
from referencing import Registry, Resource
from referencing.exceptions import Unresolvable
from referencing.jsonschema import DRAFT202012

from app.core.errors import Code, GatewayError, Stage
from app.core.models import ValidationResult

DIALECT = "https://json-schema.org/draft/2020-12/schema"


def _deny_remote(uri: str):
    raise Unresolvable(ref=uri)


def _subschemas(schema):
    """Walk schema positions, not arbitrary data in examples/const/default."""
    if not isinstance(schema, dict):
        return
    yield schema
    for key in ("properties", "patternProperties", "$defs", "dependentSchemas"):
        for child in schema.get(key, {}).values():
            yield from _subschemas(child)
    for key in (
        "items",
        "contains",
        "additionalProperties",
        "propertyNames",
        "not",
        "if",
        "then",
        "else",
        "unevaluatedProperties",
        "unevaluatedItems",
        "contentSchema",
    ):
        if key in schema:
            yield from _subschemas(schema[key])
    for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
        for child in schema.get(key, []):
            yield from _subschemas(child)


@lru_cache(maxsize=128)
def _compile(serialized: str) -> Draft202012Validator:
    schema = json.loads(serialized)
    Draft202012Validator.check_schema(schema)
    registry = Registry(retrieve=_deny_remote)
    resource = Resource.from_contents(schema, default_specification=DRAFT202012)
    resolver = registry.resolver_with_root(resource)
    for node in _subschemas(schema):
        if "$schema" in node and node["$schema"].rstrip("#") != DIALECT:
            raise ValueError("unsupported JSON Schema dialect")
        for keyword in ("$ref", "$dynamicRef"):
            if keyword in node:
                ref = node[keyword]
                if not ref.startswith("#"):
                    raise ValueError("only local JSON Schema references are supported")
                resolver.lookup(ref)
    return Draft202012Validator(
        schema, registry=registry, format_checker=FormatChecker()
    )


def check_schema(schema: dict | bool) -> Draft202012Validator:
    try:
        return _compile(json.dumps(schema, sort_keys=True, allow_nan=False))
    except (
        SchemaError,
        ValueError,
        TypeError,
        Unresolvable,
        re.error,
        RecursionError,
    ) as exc:
        raise GatewayError(
            Code.INVALID_REQUEST,
            "invalid or unsupported JSON schema",
            stage=Stage.VALIDATION,
            cause=exc,
        ) from exc


def _reject_constant(value: str):
    raise ValueError(f"invalid JSON constant: {value}")


def extract_json(text: str) -> Any:
    decoder = json.JSONDecoder(parse_constant=_reject_constant)
    try:
        return decoder.decode(text.strip())
    except ValueError:
        starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
        if not starts:
            raise ValueError("no json")
        value, end = decoder.raw_decode(text[min(starts) :])
        remaining = text[min(starts) + end :].strip()
        if remaining.startswith(("{", "[")):
            raise ValueError("multiple JSON values")
        return value


def validate_json(text: str, schema: dict | bool) -> tuple[ValidationResult, str]:
    validator = check_schema(schema)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    try:
        value = extract_json(text)
        validator.validate(value)
        canonical = json.dumps(value, sort_keys=True, allow_nan=False)
    except (ValueError, TypeError, ValidationError, Unresolvable, RecursionError):
        return ValidationResult(ok=False, text_hash=digest), text
    return ValidationResult(ok=True, text_hash=digest), canonical
