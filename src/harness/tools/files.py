"""Read approved visual memory and maintain an actor-authored location marker."""

from src.common.register import tool_registry
from src.harness.agent.base import ToolSpec

from .navigation import ToolContext, ToolInputError, fields, schema


@tool_registry.register("read_text")
class ReadTextTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "read_text",
            "Read scene_description.md or a saved session summary. Evidence links are valid image refs; raw files and arbitrary paths are unavailable. Lines start at 1.",
            schema(
                {
                    "ref": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "max_lines": {"type": "integer", "minimum": 1, "maximum": 300},
                },
                ["ref"],
            ),
        )

    def execute(self, arguments: dict, context: ToolContext):
        fields(arguments, {"ref"}, {"start_line", "max_lines"})
        if context.read_text is None:
            raise ToolInputError("File memory is unavailable")
        if not isinstance(arguments["ref"], str):
            raise ToolInputError("ref must be a resource string")
        for key, default, maximum in [("start_line", 1, None), ("max_lines", 200, 300)]:
            value = arguments.get(key, default)
            if type(value) is not int or value < 1 or (maximum and value > maximum):
                raise ToolInputError(f"Invalid {key}")
        return context.read_text(
            arguments["ref"],
            arguments.get("start_line", 1),
            arguments.get("max_lines", 200),
        )


@tool_registry.register("view_images")
class ViewImagesTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "view_images",
            "Open 1-10 approved map, evidence, goal or already captured RGB refs. layout=grid returns a labelled contact sheet; separate preserves detail. Optional crops are [left,top,right,bottom] in normalized coordinates, one per ref (null means full image). Returned crop refs preserve mapping to the original map for mark_pose.",
            schema(
                {
                    "refs": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 10,
                        "items": {"type": "string"},
                    },
                    "layout": {"type": "string", "enum": ["separate", "grid"]},
                    "crops": {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {"type": "null"},
                                {
                                    "type": "array",
                                    "minItems": 4,
                                    "maxItems": 4,
                                    "items": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1,
                                    },
                                },
                            ]
                        },
                    },
                },
                ["refs"],
            ),
        )

    def execute(self, arguments: dict, context: ToolContext):
        fields(arguments, {"refs"}, {"layout", "crops"})
        refs = arguments["refs"]
        if (
            not isinstance(refs, list)
            or not 1 <= len(refs) <= 10
            or any(not isinstance(ref, str) for ref in refs)
        ):
            raise ToolInputError("refs must contain 1-10 resource strings")
        if arguments.get("layout", "separate") not in {"separate", "grid"}:
            raise ToolInputError("layout must be separate or grid")
        if context.view_images is None:
            raise ToolInputError("Image memory is unavailable")
        return context.view_images(
            refs, arguments.get("layout", "separate"), arguments.get("crops")
        )


@tool_registry.register("mark_pose")
class MarkPoseTool:
    def spec(self) -> ToolSpec:
        point = {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {"type": "number", "minimum": 0, "maximum": 1},
        }
        return ToolSpec(
            "mark_pose",
            "Draw YOUR estimated location and heading on a map or map-crop ref. position and heading are normalized [x,y], origin top-left; heading is a second point toward which the arrow points. This draws only your estimate, never corrects it from true pose and does not move. Give short evidence-based note.",
            schema(
                {
                    "map_ref": {"type": "string"},
                    "position": point,
                    "heading": point,
                    "note": {"type": "string"},
                },
                ["map_ref", "position", "heading", "note"],
            ),
        )

    def execute(self, arguments: dict, context: ToolContext):
        fields(arguments, {"map_ref", "position", "heading", "note"})
        if context.mark_pose is None:
            raise ToolInputError("Map markers are unavailable")
        if not isinstance(arguments["map_ref"], str) or not isinstance(
            arguments["note"], str
        ):
            raise ToolInputError("map_ref and note must be strings")
        if len(arguments["note"]) > 2000:
            raise ToolInputError("note must be at most 2000 characters")
        return context.mark_pose(
            arguments["map_ref"],
            arguments["position"],
            arguments["heading"],
            arguments["note"],
        )
