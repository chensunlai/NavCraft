"""Shared schema constructors for model-facing navigation tools."""

def definition(name, description, properties, required=()):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": list(required), "additionalProperties": False}}


def reference():
    pair = {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}}
    return {"oneOf": [
        {"type": "object", "properties": {"entity": {"type": "string"}}, "required": ["entity"], "additionalProperties": False},
        {"type": "object", "properties": {"image": {"type": "string"}, "pixel": pair},
         "required": ["image", "pixel"], "additionalProperties": False},
        {"type": "object", "properties": {"uv": pair}, "required": ["uv"], "additionalProperties": False},
    ]}


