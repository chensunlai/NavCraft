"""Python-owned tool catalogue and exactly-once execution for one episode."""

from copy import deepcopy
import json

from .base import ToolContext
from .images import ViewImagesTool
from .maps import RenderMapTool
from .navigation import FinishTool, ObserveTool, StepTool


class ToolSet:
    def __init__(self, repo, workspace, bridge=None, state=None, allow_finish=True, cancel_event=None):
        self.state = {} if state is None else state
        self.context = ToolContext(repo, workspace, bridge, self.state, cancel_event)
        definitions = [ViewImagesTool, RenderMapTool]
        if bridge is not None:
            definitions.extend([ObserveTool, StepTool])
            if allow_finish:
                definitions.append(FinishTool)
        self.tools = {tool.name: tool(self.context) for tool in definitions}
        self.cache = {}

    def specs(self):
        return [tool.spec() for tool in self.tools.values()]

    def turn_start(self):
        self.state["goal_changed"] = False

    def checkpoint(self):
        return self.context.checkpoint()

    def _control(self):
        cancelled = bool(self.context.cancel_event and self.context.cancel_event.is_set())
        return {"stop_remaining_tools": bool(self.state["done"] or self.state["goal_changed"] or cancelled),
                "end_run": bool(self.state["done"] or cancelled)}

    def _error(self, message):
        result = self.context.result({"error": message})
        result["isError"] = True
        return {"result": result, "control": self._control()}

    def call(self, call_id, name, args):
        if not isinstance(call_id, str) or not call_id:
            return self._error("A nonempty string call id is required")
        try:
            fingerprint = json.dumps({"name": name, "args": args}, sort_keys=True)
        except (TypeError, ValueError):
            return self._error("Tool arguments must be JSON serializable")
        if call_id in self.cache:
            prior, response = self.cache[call_id]
            if prior != fingerprint:
                return self._error("Call id reused with different arguments")
            return deepcopy(response)
        try:
            if name not in self.tools:
                raise ValueError(f"Unknown tool: {name}")
            if not isinstance(args, dict):
                raise ValueError("Tool arguments must be an object")
            if self.context.cancel_event and self.context.cancel_event.is_set():
                raise ValueError("Run cancelled; no new tool execution is allowed")
            result = self.tools[name].execute(call_id, args)
            response = {"result": result, "checkpoint": self.checkpoint(), "control": self._control()}
        except Exception as error:
            response = self._error(f"{type(error).__name__}: {error}")
        self.cache[call_id] = fingerprint, deepcopy(response)
        return response


__all__ = ["ToolSet"]
