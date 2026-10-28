from app.services.metrics import Metrics
from app.services.structured import validate_json

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["user", "score", "tags"],
    "properties": {
        "user": {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string", "minLength": 2},
                "joined": {"type": "string", "format": "date"},
            },
        },
        "score": {"type": "integer", "minimum": 0, "maximum": 10},
        "ratio": {"type": "number"},
        "tags": {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}},
    },
}


def test_nested_schema_accepts_a_matching_object():
    result, canonical = validate_json(
        'prefix {"user": {"name": "ada", "joined": "2024-01-02"}, "score": 3, "ratio": 1, "tags": ["a"]}',
        SCHEMA,
    )
    assert result.ok
    assert '"name": "ada"' in canonical
    assert '"ratio": 1.0' in canonical


def test_nested_schema_rejects_shape_type_and_format_errors():
    samples = [
        '{"user": {"name": "a"}, "score": 3, "tags": ["a"]}',
        '{"user": {"name": "ada"}, "score": true, "tags": ["a"]}',
        '{"user": {"name": "ada"}, "score": 3, "tags": ["a"], "extra": 1}',
        '{"user": {"name": "ada", "joined": "yesterday"}, "score": 3, "tags": ["a"]}',
        '{"user": {"name": "ada"}, "score": 3, "tags": ["c"]}',
    ]
    for sample in samples:
        result, _ = validate_json(sample, SCHEMA)
        assert not result.ok, sample


def test_composition_schemas_do_not_pass_silently():
    result, _ = validate_json('{"name": "ada"}', {"oneOf": [{"type": "object"}]})
    assert not result.ok


def test_metrics_percentile_uses_recorded_samples():
    metrics = Metrics()
    for value in (10, 20, 30, 40, 100):
        metrics.observe("demo", "openai", "completed", value)
    assert metrics.percentile("demo", "openai", "completed", 50) == 30
    assert metrics.percentile("demo", "openai", "completed", 99) == 100
    assert metrics.percentile("missing", "openai", "completed", 99) is None
