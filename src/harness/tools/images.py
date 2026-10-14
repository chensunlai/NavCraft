"""Read existing public images without requesting new observations."""

from .. import memory

from .base import BOX_SCHEMA, Tool, object_schema


class ViewImagesTool(Tool):
    name = "view_images"
    description = ("Read existing workspace images or stable image IDs; optional crops use "
                   "normalized original-image coordinates. Does not observe the environment.")
    parameters = object_schema({
        "refs": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 10},
        "crops": {"type": "array", "items": {"anyOf": [BOX_SCHEMA, {"type": "null"}]}},
        "layout": {"type": "string", "enum": ["separate", "grid"]},
    }, ["refs"])

    def execute(self, call_id, args):
        value = memory.view(workspace=self.context.workspace, **args)
        return self.context.result(value["data"], value["images"])
