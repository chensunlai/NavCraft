"""Synchronous navigation callbacks into the Python-owned environment."""

from .. import memory

from .base import Tool, object_schema


class NavigationTool(Tool):
    def execute(self, call_id, args):
        state = self.context.state
        if state["done"]:
            raise ValueError("Task already ended")
        if state["goal_changed"]:
            raise ValueError("Target changed during this turn; read the new goal and decide again next turn")
        response = self.context.bridge.dispatch({"id": call_id, "op": self.operation, "args": args})
        if not response["ok"]:
            raise ValueError(response["error"])
        data = response["data"]
        current = {**state["current"], **data}
        if self.operation != "observe":
            for key in ("mode", "views", "restored_camera", "panorama_views"):
                current.pop(key, None)
        elif args.get("mode", "single") == "panorama":
            # The six captures replace the visible observation, while body/camera
            # state is restored by the environment before this method returns.
            current.pop("observation", None)
        else:
            current.pop("restored_camera", None)
            current.pop("panorama_views", None)
        state.update(current=current, done=bool(data.get("terminated") or data.get("truncated")),
                     goal_changed=bool(data.get("goal_changed")))
        state["revision"] += 1
        if self.operation == "observe" and args.get("mode") == "panorama":
            montage = memory.view(self.context.workspace, data["images"], layout="grid")
            data = {**data, "images": montage["images"], "panorama_views": data["views"]}
            current["images"] = montage["images"]
        return self.context.result(data, data.get("images", []))


class ObserveTool(NavigationTool):
    name, operation = "nav_observe", "observe"
    description = ("Capture NEW RGB at the current position. panorama gives six horizontal views "
                   "relative to body heading and restores body/camera state; it does not turn the body.")
    parameters = object_schema({"mode": {"type": "string", "enum": ["single", "panorama"]}}, ["mode"])


class StepTool(NavigationTool):
    name, operation = "nav_step", "step"
    description = ("Execute 1–6 allowed atomic actions in sequence. move_forward, move_left, move_right "
                   "and move_backward translate relative to body heading by translation_metres "
                   "(same as forward_metres). Translations may slide and automatically "
                   "turn the body by at most 15 degrees per action; auto_yaw_delta_degrees reports "
                   "this relative turn (positive left). A progressing collision can continue; blocked "
                   "interrupts the batch. Goal change or task completion always interrupts. "
                   "Returns actual feedback, last RGB and references "
                   "to intermediate frames; no true pose.")
    parameters = object_schema({"actions": {"type": "array", "items": {"type": "string"},
                                           "minItems": 1, "maxItems": 6}}, ["actions"])


class FinishTool(NavigationTool):
    name, operation = "nav_finish", "finish"
    description = ("Submit completion of the CURRENT goal by its ID. It may end the episode or switch "
                   "goals, without certifying success.")
    parameters = object_schema({"goal_id": {"type": "integer"}}, ["goal_id"])
