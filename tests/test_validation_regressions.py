"""Effective limits and standard JSON Schema semantics."""

import json
from dataclasses import replace

import httpx
import pytest

from app.config import load_settings
from app.main import create_app
from app.services.prompts import load_prompts
from app.services.structured import validate_json
from tests.conftest import build_app, write_config
from tests.test_gateway_api import _ok, _post
from tests.test_protocol_regressions import BASE


@pytest.mark.parametrize("headers", [{}, {"Content-Length": "1"}])
async def test_chunked_body_limit_uses_actual_bytes(tmp_path, headers):
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        return _ok()

    settings = load_settings(write_config(tmp_path))
    settings = replace(settings, server=replace(settings.server, max_body_bytes=32))
    app = create_app(settings, transport=httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):

        async def content():
            yield b'{"model":"demo",'
            yield b'"messages":[{"role":"user","content":"a long message"}]}'

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    **headers,
                    "Authorization": "Bearer gateway-demo-secret",
                    "Content-Type": "application/json",
                },
                content=content(),
            )
        assert response.status_code == 400 and count == 0
        assert response.json()["error"]["code"] == "invalid_request"
        assert response.headers["x-request-id"] == response.json()["request_id"]


async def test_body_exactly_at_limit_is_accepted(tmp_path):
    content = json.dumps(BASE).encode()
    settings = load_settings(write_config(tmp_path))
    settings = replace(
        settings, server=replace(settings.server, max_body_bytes=len(content))
    )
    app = create_app(settings, transport=httpx.MockTransport(lambda request: _ok()))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer gateway-demo-secret",
                    "Content-Type": "application/json",
                },
                content=content,
            )
        assert response.status_code == 200


async def test_omitted_output_limit_is_applied_to_upstream(tmp_path):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return _ok()

    app = build_app(tmp_path, httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        assert (await _post(app, BASE)).status_code == 200
        assert seen["max_tokens"] == 128
        assert (await _post(app, {**BASE, "max_tokens": 129})).status_code == 400


@pytest.mark.parametrize("max_tokens, status", [(129, 400), (64, 200)])
async def test_prompt_defaults_checked_after_merge(tmp_path, max_tokens, status):
    directory = tmp_path / "prompts"
    directory.mkdir()
    (directory / "demo.yaml").write_text(f"""name: demo
version: "1"
status: published
system: hello
variables: {{type: object, properties: {{}}}}
default_model: demo
max_tokens: {max_tokens}
""")
    app = build_app(
        tmp_path,
        httpx.MockTransport(lambda request: _ok()),
        prompts=load_prompts(directory),
    )
    async with app.router.lifespan_context(app):
        response = await _post(app, {"prompt": {"name": "demo", "version": "1"}})
        assert response.status_code == status


@pytest.mark.parametrize(
    "name,value",
    [
        ("max_tokens_cap", 0),
        ("max_tokens_cap", -1),
        ("retry_max_attempts", 0),
        ("repair_max_attempts", -1),
        ("circuit_failure_threshold", 0),
        ("circuit_cooldown", "0s"),
    ],
)
def test_invalid_limit_configuration_fails_at_startup(tmp_path, name, value):
    import yaml

    path = write_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["limits"][name] = value
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_settings(path)


def test_zero_repairs_and_circuit_settings_are_preserved(tmp_path):
    settings = load_settings(write_config(tmp_path, repairs=0))
    assert settings.limits.repair_max_attempts == 0
    assert settings.limits.circuit_cooldown.total_seconds() == 30


@pytest.mark.parametrize(
    "schema,valid,invalid",
    [
        (
            {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 3,
            },
            "[1,2]",
            "[]",
        ),
        ({"type": "array", "uniqueItems": True}, "[1,2]", "[1,1]"),
        (
            {"type": "object", "properties": {"x": {"const": "a"}}},
            '{"x":"a"}',
            '{"x":"b"}',
        ),
        ({"anyOf": [{"type": "null"}, {"type": "integer"}]}, "null", '"1"'),
        ({"oneOf": [{"type": "number"}, {"type": "integer"}]}, "1.5", "1"),
        ({"allOf": [{"type": "integer"}, {"minimum": 2}]}, "2", "1"),
        (
            {
                "$defs": {"item": {"type": "integer"}},
                "type": "array",
                "items": {"$ref": "#/$defs/item"},
            },
            "[1]",
            '["1"]',
        ),
        ({"type": "string", "format": "date"}, '"2024-01-02"', '"yesterday"'),
        ({"type": "string", "format": "email"}, '"a@example.com"', '"invalid"'),
    ],
)
def test_standard_schema_keywords_are_enforced(schema, valid, invalid):
    assert validate_json(valid, schema)[0].ok
    assert not validate_json(invalid, schema)[0].ok


def test_validation_preserves_data_and_omitted_optional_properties():
    schema = {"type": "object", "properties": {"optional": {"type": "string"}}}
    assert json.loads(validate_json("{}", schema)[1]) == {}
    assert json.loads(validate_json('{"extra":1}', schema)[1]) == {"extra": 1}
    assert not validate_json('{"optional":null}', schema)[0].ok
    assert not validate_json('{"optional":42}', schema)[0].ok


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "not-a-type"},
        {"type": "array", "minItems": -1},
        {"$schema": "http://json-schema.org/draft-07/schema#"},
        {"$ref": "https://example.com/schema.json"},
        {"$ref": "#/missing"},
        {"type": "string", "pattern": "["},
    ],
)
async def test_invalid_schema_is_rejected_before_upstream(tmp_path, schema):
    def handler(request):
        raise AssertionError("invalid schema reached upstream")

    app = build_app(tmp_path, httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        response = await _post(
            app,
            {
                **BASE,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "a", "schema": schema},
                },
            },
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request"


@pytest.mark.parametrize("text", ['{"a":NaN}', '{"a":Infinity}', '{"a":1} {"b":2}'])
def test_non_json_constants_and_multiple_values_are_rejected(text):
    assert not validate_json(text, {"type": "object"})[0].ok


def test_date_time_format_is_checked_with_declared_format_dependencies():
    schema = {"type": "string", "format": "date-time"}
    assert validate_json('"2024-01-02T03:04:05Z"', schema)[0].ok
    assert not validate_json('"yesterday"', schema)[0].ok
    assert not validate_json('"2024-01-02"', schema)[0].ok


@pytest.mark.parametrize("schema", [None, [], "not a schema"])
async def test_schema_wrapper_cannot_bypass_validation(tmp_path, schema):
    def handler(request):
        raise AssertionError("invalid schema reached upstream")

    app = build_app(tmp_path, httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        response = await _post(
            app,
            {
                **BASE,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "a", "schema": schema},
                },
            },
        )
        assert response.status_code == 400
