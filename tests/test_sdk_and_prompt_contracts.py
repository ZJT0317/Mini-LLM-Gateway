"""Official SDK contract tests and prompt publication validation."""

import json
from dataclasses import replace

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict

from app.core.errors import GatewayError
from app.services.prompts import (
    PromptLibrary,
    _bundle,
    _check_variables,
    _render,
    load_prompts,
)
from tests.conftest import build_app, write_config
from tests.test_gateway_api import _post, _ok
from tests.test_protocol_regressions import BASE, FORMAT, _call, _stream_response


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool


def sdk_for(app):
    return AsyncOpenAI(
        api_key="gateway-demo-secret",
        base_url="http://gateway.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=app)),
    )


async def test_official_sdk_parse_keeps_schema_metadata(tmp_path):
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return _ok('{"ok":true}')

    app = build_app(tmp_path, httpx.MockTransport(handler))
    async with app.router.lifespan_context(app), sdk_for(app) as sdk:
        result = await sdk.chat.completions.parse(
            model="demo", messages=BASE["messages"], response_format=Answer
        )
        assert result.choices[0].message.parsed == Answer(ok=True)
        assert result.created > 0 and result.model == "demo"
        assert result.usage.prompt_tokens == 3
        schema = seen[0]["response_format"]["json_schema"]
        assert schema["name"] == "Answer" and schema["strict"] is True
        assert schema["schema"]["additionalProperties"] is False


@pytest.mark.parametrize("strict", [True, False])
async def test_schema_name_description_and_strict_roundtrip(tmp_path, strict):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return _ok('{"a":1}')

    app = build_app(tmp_path, httpx.MockTransport(handler))
    expected = {
        **FORMAT["json_schema"],
        "strict": strict,
        "description": "Business answer",
    }
    async with app.router.lifespan_context(app):
        response = await _post(
            app,
            {
                **BASE,
                "response_format": {"type": "json_schema", "json_schema": expected},
            },
        )
        assert response.status_code == 200
        assert seen["response_format"]["json_schema"] == expected


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("structured", [False, True])
async def test_sdk_refusal_is_preserved_without_repair_or_fallback(
    tmp_path, stream, structured
):
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        if stream:
            return _stream_response(
                [
                    {"choices": [{"delta": {"refusal": "Cannot "}}]},
                    {
                        "choices": [
                            {"delta": {"refusal": "answer."}, "finish_reason": "stop"}
                        ]
                    },
                    {
                        "choices": [],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                    },
                ]
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": None, "refusal": "Cannot answer."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )

    app = build_app(tmp_path, httpx.MockTransport(handler))
    async with app.router.lifespan_context(app), sdk_for(app) as sdk:
        options = (
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "strict": True,
                        "schema": Answer.model_json_schema(),
                    },
                }
            }
            if structured
            else {}
        )
        if stream:
            events = [
                chunk
                async for chunk in await sdk.chat.completions.create(
                    model="demo",
                    messages=BASE["messages"],
                    stream=True,
                    stream_options={"include_usage": True},
                    **options,
                )
            ]
            assert (
                "".join(c.choices[0].delta.refusal or "" for c in events if c.choices)
                == "Cannot answer."
            )
            assert events[-1].choices == [] and events[-1].usage.total_tokens == 7
        else:
            result = (
                await sdk.chat.completions.parse(
                    model="demo", messages=BASE["messages"], response_format=Answer
                )
                if structured
                else await sdk.chat.completions.create(
                    model="demo", messages=BASE["messages"]
                )
            )
            assert result.choices[0].message.refusal == "Cannot answer."
            assert result.choices[0].message.content is None
        assert count == 1
        row = app.state.ledger._conn.execute("SELECT * FROM calls").fetchone()
        assert row["terminal"] == "refused" and row["error_code"] == "content_rejected"
        attempt = app.state.ledger.attempts_for(row["call_id"])[0]
        assert attempt["terminal"] == "refused" and attempt["input_tokens"] == 5


async def test_repair_that_refuses_stops_and_is_accounted(tmp_path):
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        if count == 1:
            return _ok("invalid")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"refusal": "Cannot answer."}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )

    app = build_app(tmp_path, httpx.MockTransport(handler), repairs=3)
    async with app.router.lifespan_context(app):
        response = await _post(app, {**BASE, "response_format": FORMAT})
        assert response.status_code == 200 and count == 2
        assert response.json()["choices"][0]["message"]["refusal"]
        call = _call(app, response)
        assert call["terminal"] == "refused" and call["input_tokens"] == 6
        assert [
            r["attempt"] for r in app.state.ledger.attempts_for(call["call_id"])
        ] == [1, 2]


async def test_refusal_followed_by_disconnect_is_never_retried(tmp_path):
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        return _stream_response(
            [{"choices": [{"delta": {"refusal": "Cannot answer."}}]}], done=False
        )

    app = build_app(tmp_path, httpx.MockTransport(handler), retries=2)
    async with app.router.lifespan_context(app):
        response = await _post(app, {**BASE, "response_format": FORMAT, "stream": True})
        assert response.status_code == 502 and count == 1


@pytest.mark.parametrize("include_usage", [True, False])
async def test_sdk_stream_usage_opt_in(tmp_path, include_usage):
    upstream = [
        {"choices": [{"delta": {"content": "hi"}}]},
        {"choices": [{"delta": {}, "finish_reason": "length"}]},
        {"choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": 4}},
    ]
    app = build_app(
        tmp_path, httpx.MockTransport(lambda request: _stream_response(upstream))
    )
    async with app.router.lifespan_context(app), sdk_for(app) as sdk:
        events = [
            chunk
            async for chunk in await sdk.chat.completions.create(
                model="demo",
                messages=BASE["messages"],
                stream=True,
                stream_options={"include_usage": include_usage},
            )
        ]
        assert (
            "".join(c.choices[0].delta.content or "" for c in events if c.choices)
            == "hi"
        )
        assert all(c.created > 0 for c in events)
        if include_usage:
            assert events[-1].choices == [] and events[-1].usage.total_tokens == 12
            assert events[-2].choices[0].finish_reason == "length"
        else:
            assert all(c.usage is None for c in events)


@pytest.mark.parametrize(
    "values",
    [
        {"count": -1, "values": []},
        {"count": 1, "values": ["bad"]},
        {"count": True, "values": ["ok"]},
    ],
)
def test_prompt_variables_use_full_schema(values):
    schema = {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "minimum": 1},
            "values": {"type": "array", "minItems": 1, "items": {"enum": ["ok"]}},
        },
        "required": ["count", "values"],
    }
    with pytest.raises(GatewayError) as caught:
        _check_variables(schema, values)
    assert caught.value.code.value == "prompt_invalid"


def test_prompt_extra_properties_follow_schema_and_undefined_is_rejected():
    _check_variables({"type": "object"}, {"extra": 1})
    with pytest.raises(GatewayError):
        _check_variables(
            {"type": "object", "additionalProperties": False}, {"extra": 1}
        )
    with pytest.raises(GatewayError):
        _render("Hello {{ typo }}", {})


def test_duplicate_prompt_publication_is_rejected(tmp_path):
    bundle = _bundle({"name": "p", "version": "1"})
    with pytest.raises(ValueError, match="unique"):
        PromptLibrary((bundle, replace(bundle, system_template="new")))
    (tmp_path / "a.yaml").write_text('name: p\nversion: "1"\n')
    (tmp_path / "b.yaml").write_text('name: p\nversion: "1"\n')
    with pytest.raises(ValueError, match="unique"):
        load_prompts(tmp_path)


def test_docker_volume_database_override(tmp_path, monkeypatch):
    from app.config import load_settings

    path = write_config(tmp_path)
    monkeypatch.setenv("GATEWAY_DATABASE", str(tmp_path / "data" / "persistent.db"))
    settings = load_settings(path)
    assert settings.server.database == tmp_path / "data" / "persistent.db"


async def test_official_sdk_stream_helper_parses_structured_output(tmp_path):
    chunks = [
        {"choices": [{"delta": {"content": '{"ok":true}'}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 6, "completion_tokens": 4}},
    ]
    app = build_app(
        tmp_path, httpx.MockTransport(lambda request: _stream_response(chunks))
    )
    async with app.router.lifespan_context(app), sdk_for(app) as sdk:
        async with sdk.chat.completions.stream(
            model="demo",
            messages=BASE["messages"],
            response_format=Answer,
            stream_options={"include_usage": True},
        ) as stream:
            result = await stream.get_final_completion()
        assert result.choices[0].message.parsed == Answer(ok=True)
        assert result.usage.total_tokens == 10


@pytest.mark.parametrize("explicit_refusal", [False, True])
async def test_content_filter_stream_is_refused_without_duplicate_text(
    tmp_path, explicit_refusal
):
    chunks = [
        {
            "choices": [
                {
                    "delta": {"refusal": "content rejected"}
                    if explicit_refusal
                    else {},
                    "finish_reason": "content_filter",
                }
            ]
        }
    ]
    app = build_app(
        tmp_path, httpx.MockTransport(lambda request: _stream_response(chunks))
    )
    async with app.router.lifespan_context(app):
        response = await _post(app, {**BASE, "stream": True})
        assert response.text.count('"refusal": "content rejected"') == 1
        assert _call(app, response)["terminal"] == "refused"


@pytest.mark.parametrize(
    "override",
    [{"strict": "yes"}, {"name": ""}, {"description": []}, {"unsupported": 1}],
)
async def test_invalid_schema_options_are_rejected(tmp_path, override):
    def handler(request):
        raise AssertionError("invalid schema metadata reached provider")

    app = build_app(tmp_path, httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        response = await _post(
            app,
            {
                **BASE,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        **FORMAT["json_schema"],
                        **override,
                    },
                },
            },
        )
        assert response.status_code == 400


async def test_persisted_ledger_can_be_reopened_after_app_shutdown(tmp_path):
    from app.services.usage import Ledger

    app = build_app(tmp_path, httpx.MockTransport(lambda request: _ok()))
    async with app.router.lifespan_context(app):
        response = await _post(app, BASE)
        trace = response.headers["x-request-id"]
        call_id = _call(app, response)["call_id"]
    ledger = Ledger(app.state.settings.server.database)
    try:
        assert ledger.calls_by_trace(trace)[0]["terminal"] == "completed"
        assert len(ledger.attempts_for(call_id)) == 1
    finally:
        ledger.close()


def test_compose_database_path_is_inside_a_named_volume():
    from pathlib import Path
    import yaml

    compose = yaml.safe_load(Path("docker-compose.yml").read_text())
    service = compose["services"]["gateway"]
    database = Path(service["environment"]["GATEWAY_DATABASE"])
    assert "gateway-data:/srv/data" in service["volumes"]
    assert "gateway-data" in compose["volumes"]
    assert database.parent == Path("/srv/data")
