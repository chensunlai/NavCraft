"""Score a JSON value using the private judgment label."""
from math import isfinite


def _matches_type(value, schema):
    kind = schema["type"]
    if kind == "number":
        return type(value) in (int, float) and abs(value) <= 1e12 and isfinite(value)
    if kind == "integer":
        return type(value) is int and abs(value) <= 10**12
    if kind == "boolean":
        return type(value) is bool
    if kind == "string":
        return isinstance(value, str) and ("enum" not in schema or value in schema["enum"])
    if kind == "array":
        return (isinstance(value, list) and schema.get("minItems", 0) <= len(value) <=
                schema.get("maxItems", 512) and all(
                    _matches_type(item, schema["items"]) for item in value))
    if kind == "object":
        fields = schema["properties"]
        return (isinstance(value, dict) and value.keys() == fields.keys() and
                all(_matches_type(value[key], shape) for key, shape in fields.items()))
    raise ValueError(f"Unsupported judgment answer type: {kind}")


def _equal(value, expected, schema):
    kind = schema["type"]
    if kind == "number":
        error = value - expected
        if schema.get("unit") == "signed_degrees":
            error = (error + 180) % 360 - 180
        return abs(error) <= schema.get("tolerance", 0)
    if kind == "array":
        if len(value) != len(expected):
            return False
        if schema.get("unordered"):
            from collections import Counter
            from json import dumps
            counts = lambda items: Counter(dumps(x, sort_keys=True, allow_nan=False) for x in items)
            return counts(value) == counts(expected)
        return all(_equal(a, b, schema["items"]) for a, b in zip(value, expected))
    if kind == "object":
        return all(_equal(value[key], expected[key], sub) for key, sub in schema["properties"].items())
    return value == expected


def score_answer(answer, label):
    """Distinguish malformed JSON values from well-typed but incorrect answers."""
    schema = label["answer_schema"]
    if not _matches_type(answer, schema):
        return {"success": False, "status": "invalid_answer", "details": {}}
    success = any(_equal(answer, expected, schema) for expected in label["accepted"])
    return {"success": success, "status": "success" if success else "incorrect", "details": {}}
