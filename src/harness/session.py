"""One episode, one environment; tools only receive public operation callbacks."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.common.register import tool_registry
from src.dataset.episodes.base import EpisodeSpec
from src.environment.base import Environment, Observation, Transition
from src.harness.agent.base import ToolCall, ToolResult
from src.harness.map import MapAsset
from src.harness.prompts import SYSTEM_PROMPT, observation_message
from src.harness.tools import ToolSet, load_builtin_tools
from src.harness.tools.navigation import ToolContext, ToolInputError
from src.harness.trace import TraceWriter


class NavigationSession:
    def __init__(self, environment: Environment, episode: EpisodeSpec, trace: TraceWriter,
                 tool_names: list[str], map_asset: MapAsset | None = None,
                 action_parameters: dict | None = None):
        load_builtin_tools()
        if len(tool_names) != len(set(tool_names)) or not tool_names:
            raise ValueError("Harness tools must be a nonempty unique list")
        self.environment, self.episode, self.trace = environment, episode, trace
        self.tools = {name: tool_registry.create(name) for name in tool_names}
        self.toolset = ToolSet(self)
        self.map_asset = map_asset  # Reserved input, not exposed without a map tool.
        self.action_parameters = dict(action_parameters or {})
        self.transition: Transition | None = None
        self.frames: dict[str, Observation] = {}
        self._calls: dict[str, tuple[str, ToolResult]] = {}
        self._call_id = ""

    def start(self) -> list[dict]:
        if self.transition is not None:
            raise RuntimeError("Session already started")
        self._accept(self.environment.reset(self.episode))
        self.trace.emit("episode_started", public=self.transition.public_dict())
        data = self.transition.public_dict()
        data["movement_actions"] = list(self.movement_actions)
        data["action_parameters"] = self.action_parameters
        return [{"role": "system", "content": SYSTEM_PROMPT}, observation_message(data, self._images(include_goal=True))]

    @property
    def movement_actions(self) -> tuple[str, ...]:
        return ("move_forward", "turn_left", "turn_right") + (() if self.episode.benchmark == "r2r" else ("look_up", "look_down"))

    def _accept(self, transition: Transition) -> None:
        self.transition = transition
        self.frames[transition.observation.frame_id] = transition.observation
        self.trace.goal_id = transition.task.goal_id if transition.task is not None else None

    def _images(self, include_goal: bool = False) -> list[Path]:
        images = [Path(self.transition.observation.rgb)]
        task = self.transition.task
        if include_goal and task is not None and task.image is not None:
            images.append(Path(task.image))
        return images

    def _result(self, data: dict | None = None, images: bool = True) -> ToolResult:
        transition = self.transition
        return ToolResult(self._call_id, True, transition.public_dict() if data is None else data,
                          self._images(include_goal=transition.goal_changed) if images else [],
                          transition.terminated, transition.truncated)

    def execute(self, call: ToolCall) -> ToolResult:
        if self.transition is None:
            raise RuntimeError("Session has not started")
        self._call_id = call.call_id
        try:
            arguments = json.loads(call.arguments) if isinstance(call.arguments, str) else call.arguments
        except json.JSONDecodeError:
            arguments = call.arguments
        signature = json.dumps({"name": call.name, "arguments": arguments}, sort_keys=True, ensure_ascii=False)
        if call.call_id in self._calls:
            previous, result = self._calls[call.call_id]
            if previous != signature:
                result = ToolResult(call.call_id, False, {"error": "call_id reused with different arguments"})
            self.trace.emit("tool_replayed", call_id=call.call_id, result=result)
            return result
        self.trace.emit("tool_call", call_id=call.call_id, name=call.name, arguments=arguments)
        try:
            if not call.call_id:
                raise ToolInputError("call_id must not be empty")
            if self.transition.terminated or self.transition.truncated:
                raise ToolInputError("Episode is already finished")
            if call.name not in self.tools:
                raise ToolInputError(f"Unavailable tool: {call.name}")
            context = ToolContext(self._observe, self._depth, self._step, self._finish, self.movement_actions)
            result = self.tools[call.name].execute(arguments, context)
        except ToolInputError as error:
            result = ToolResult(call.call_id, False, {"error": str(error)}, [], self.transition.terminated, self.transition.truncated)
        except Exception:
            # The action may already have happened before capture/transport failed.
            result = ToolResult(call.call_id, False, {"error": "Execution failed; this call will not be executed again"}, [], False, True)
            self._calls[call.call_id] = (signature, result)
            self.trace.emit("tool_execution_failed", call_id=call.call_id, result=result)
            raise
        self._calls[call.call_id] = (signature, result)
        self.trace.emit("tool_result", call_id=call.call_id, result=result)
        return result

    def _observe(self) -> ToolResult:
        observation = self.environment.observe()
        self.frames[observation.frame_id] = observation
        self.transition = replace(self.transition, observation=observation, goal_changed=False,
                                  feedback={"event": "observe"})
        return self._result()

    def _depth(self, frame_id: str, points: list[list[int]]) -> ToolResult:
        if frame_id not in self.frames:
            raise ToolInputError("Unknown frame; only already observed frames can be queried")
        frame = self.frames[frame_id]
        if frame.depth is None:
            raise ToolInputError("Depth is not enabled")
        depth = np.load(frame.depth, allow_pickle=False)
        height, width = depth.shape
        if any(not (0 <= u < width and 0 <= v < height) for u, v in points):
            raise ToolInputError(f"Pixels must be within width={width}, height={height}")
        values = []
        for u, v in points:
            value = float(depth[v, u])
            valid = bool(np.isfinite(value) and value > 0)
            values.append({"pixel": [u, v], "depth_m": value if valid else None, "valid": valid})
        return self._result({"frame_id": frame_id, "unit": "meters", "points": values}, images=False)

    def _step(self, actions: list[str]) -> ToolResult:
        executed = []
        for action in actions:
            transition = self.environment.step(action)
            self._accept(transition)
            executed.append({"action": action, "frame_id": transition.observation.frame_id, **transition.feedback})
            self.trace.emit("action_executed", call_id=self._call_id, action=action, public=transition.public_dict())
            if transition.terminated or transition.truncated or transition.goal_changed:
                break
        data = self.transition.public_dict()
        data["executed_actions"] = executed
        return self._result(data)

    def _finish(self) -> ToolResult:
        self._accept(self.environment.finish_goal())
        self.trace.emit("goal_finished", call_id=self._call_id, public=self.transition.public_dict())
        return self._result()

    def close(self) -> None:
        self.environment.close()
