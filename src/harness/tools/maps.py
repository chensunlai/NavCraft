"""Render current public memory; never infer a true pose or route."""

from .. import memory

from .base import BOX_SCHEMA, Tool, object_schema


class RenderMapTool(Tool):
    name = "render_map"
    description = ("Render the public wall layer and CURRENT memory geometry/state. Never "
                   "locates you or plans routes. Region coordinates refer to the full map. "
                   "Invalid records return errors.")
    parameters = object_schema({
        "map_id": {"type": "string"}, "region": BOX_SCHEMA,
        "highlight_ids": {"type": "array", "items": {"type": "string"}},
        "grid": {"type": "boolean"},
        "layers": {"type": "array", "items": {"type": "string", "enum": [
            "walls", "region", "portal", "object", "pose"]}},
    }, ["map_id"])

    def execute(self, call_id, args):
        value = memory.render(workspace=self.context.workspace, **args)
        self.context.state.update(latest_map=value["images"][0], latest_map_metadata=value["data"])
        self.context.state["revision"] += 1
        return self.context.result(value["data"], value["images"])
