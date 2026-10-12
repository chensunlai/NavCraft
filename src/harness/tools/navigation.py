"""The baseline exposes observations, measured depth, primitive steps and STOP."""

from dataclasses import dataclass
from typing import Callable

from src.common.register import tool_registry
from src.harness.agent.base import ToolResult, ToolSpec


class ToolInputError(ValueError):
    """A rejected call that has not changed environment state."""


@dataclass(frozen=True)
class ToolContext:
    observe: Callable[..., ToolResult]
    depth: Callable[[str, list[list[int]]], ToolResult]
    step: Callable[[list[str]], ToolResult]
    finish: Callable[[], ToolResult]
    movement_actions: tuple[str, ...]
    read_text: Callable[..., ToolResult] | None = None
    view_images: Callable[..., ToolResult] | None = None
    mark_pose: Callable[..., ToolResult] | None = None


def fields(
    arguments: dict, expected: set[str], optional: set[str] = frozenset()
) -> None:
    if (
        not isinstance(arguments, dict)
        or not expected <= set(arguments)
        or set(arguments) - expected - optional
    ):
        raise ToolInputError(f"Expected exactly these arguments: {sorted(expected)}")


def schema(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


@tool_registry.register("nav_observe")
class ObserveTool:
    visual = False

    def spec(self) -> ToolSpec:
        if self.visual:
            return ToolSpec(
                "nav_observe",
                "Observe current RGB or six horizontal views at the current position. Panorama directions are relative to current heading; capture restores camera state and does not move.",
                schema(
                    {
                        "mode": {"type": "string", "enum": ["single", "panorama"]},
                    },
                    [],
                ),
            )
        return ToolSpec(
            "nav_observe",
            "Inspect the current observed RGB, world pose and camera parameters. Does not move.",
            schema({}, []),
        )

    def execute(self, arguments: dict, context: ToolContext) -> ToolResult:
        if self.visual:
            fields(arguments, set(), {"mode"})
            mode = arguments.get("mode", "single")
            if mode not in {"single", "panorama"}:
                raise ToolInputError("mode must be single or panorama")
            return context.observe(mode)
        fields(arguments, set())
        return context.observe()


@tool_registry.register("nav_depth")
class DepthTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "nav_depth",
            "Query measured depth in meters at [u,v] pixels of an already observed frame. Coordinates are integer pixels, top-left origin.",
            schema(
                {
                    "frame_id": {"type": "string"},
                    "points": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 32,
                        "items": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": {"type": "integer"},
                        },
                    },
                },
                ["frame_id", "points"],
            ),
        )

    def execute(self, arguments: dict, context: ToolContext) -> ToolResult:
        fields(arguments, {"frame_id", "points"})
        points = arguments["points"]
        if (
            not isinstance(arguments["frame_id"], str)
            or not isinstance(points, list)
            or not 1 <= len(points) <= 32
        ):
            raise ToolInputError(
                "frame_id must be a string; points must contain 1-32 [u,v] pairs"
            )
        if any(
            not isinstance(p, list) or len(p) != 2 or any(type(v) is not int for v in p)
            for p in points
        ):
            raise ToolInputError("Each point must be two integer pixel coordinates")
        return context.depth(arguments["frame_id"], points)


@tool_registry.register("nav_step")
class StepTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "nav_step",
            "Execute 1-6 atomic actions sequentially. Returns actual actions, collision feedback and latest RGB. STOP must be submitted through nav_finish.",
            schema(
                {
                    "actions": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 6,
                        "items": {"type": "string"},
                    },
                },
                ["actions"],
            ),
        )

    def execute(self, arguments: dict, context: ToolContext) -> ToolResult:
        fields(arguments, {"actions"})
        actions = arguments["actions"]
        if not isinstance(actions, list) or not 1 <= len(actions) <= 6:
            raise ToolInputError("actions must contain 1-6 atomic movement actions")
        if any(
            not isinstance(a, str) or a not in context.movement_actions for a in actions
        ):
            raise ToolInputError(
                f"Allowed movement actions: {list(context.movement_actions)}"
            )
        return context.step(actions)


@tool_registry.register("nav_finish")
class FinishTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "nav_finish",
            "Declare the current goal complete (task STOP). A multi-goal task can return a next goal; this does not imply success.",
            schema({}, []),
        )

    def execute(self, arguments: dict, context: ToolContext) -> ToolResult:
        fields(arguments, set())
        return context.finish()
