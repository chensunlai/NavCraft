"""Navigation tools usable by an Agent, scripted sampling, or replay."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading

from src.common.io import write_json
from ..errors import ToolInputError
from ..memory.entity import MemoryStore
from ..memory.node import NodeMemory
from ..spatial.maps import MapRenderer
from ..spatial.observations import Observations
from .request import RequestState
from ..spatial.assistance import SpatialTools
from ..tools import Toolsets, bind_tools
from .trace import TraceWriter
from .workspace import prepare_workspace, copy_snapshot_assets


class NavigationExecutor:
    def __init__(self, directory, environment, scene_map=None, *,
                 toolsets=("navigation", "entity_memory"), goto_planner="local", initial_state=None):
        self.directory = Path(directory).resolve()
        self.workspace, self.private = self.directory / "workspace", self.directory / "private"
        self.environment = environment
        self.selection = toolsets if isinstance(toolsets, Toolsets) else Toolsets.resolve(toolsets)
        self.memory_type = memory_type = self.selection.memory_type
        self.goto_planner = goto_planner
        self.broken = False
        self.cancel = threading.Event()
        self.request = RequestState()
        self._calls = {}
        self.submitted_requests = []
        self.observation_count = 0
        self.trace = None
        self._executing = False
        state = initial_state or {}
        try:
            self.transform = prepare_workspace(self.workspace, self.private, environment,
                                               scene_map, memory_type, state)
            self.observations = Observations(self.workspace, self.transform)
            self.memory = (MemoryStore(self.workspace, self.private) if memory_type == "entity" else
                           NodeMemory(self.workspace, self.observations, self.transform, self.private))
            self.renderer = MapRenderer(self.workspace, self.memory, memory_type=memory_type)
            self.spatial = SpatialTools(self)
            self.current = (self.observations.get(state["current_frame"]) if state.get("current_frame")
                            else self.capture(environment.observe()))
            self._calls = deepcopy(state.get("tool_receipts", {}))
            self.request = RequestState(text=state.get("current_request", ""),
                                        target_frames=list(state.get("target_frames", [])))
            self.submitted_requests = list(state.get("submitted_requests", []))
            self.observation_count = state.get("observations_count", 0)
            self.publish_state()
            self.trace = TraceWriter(self.private)
            self.tools = bind_tools(self, self.selection)
            self.tool_specs = [tool.schema for tool in self.tools.values()]
            self.trace.emit("navigation_started", {
                "map_sha256": hashlib.sha256((self.workspace / "scene/map.json").read_bytes()).hexdigest(),
                "memory_source": state.get("memory_source", "empty"), "memory_type": memory_type,
                "toolsets": self.selection.names, "goto_planner": goto_planner if self.spatial.enabled else None})
        except BaseException:
            self.close()
            raise

    def begin_request(self, text, target_frames=(), limits=None):
        if self.broken:
            raise RuntimeError("Navigation execution is uncertain")
        self.cancel.clear()
        self.request = RequestState(text=text, target_frames=list(target_frames))
        if limits is not None:
            self.request.action_end = self.environment.steps + limits.max_actions
            self.request.path_end = (self.environment.path_length + limits.max_path_length
                                     if limits.max_path_length is not None else None)
            self.request.observation_end = (self.observation_count + limits.max_observations
                                            if limits.max_observations is not None else None)

    def end_request(self):
        if self.request.submitted:
            self.submitted_requests.append(self.request.text)
        self.request.action_end = self.request.path_end = self.request.observation_end = None

    def snapshot(self, destination):
        """Call only at an idle tool boundary; no capture or model history is added."""
        if self._executing or self.broken:
            raise RuntimeError("Cannot snapshot active or uncertain execution")
        destination = Path(destination)
        copy_snapshot_assets(self.workspace, self.private, destination)
        state = {"memory_type": self.memory_type,
                 "pose": self.environment.pose(), "current_frame": self.current["frame"],
                 "workspace": "workspace", "private_dir": "private",
                 "current_request": self.request.text, "target_frames": self.request.target_frames,
                 "submitted_requests": self.submitted_requests,
                 "observations_count": self.observation_count, "tool_receipts": deepcopy(self._calls),
                 "environment": self.environment.state()}
        write_json(destination / "state.json", state)
        return state

    def refresh_observation(self):
        self.current = self.capture(self.environment.observe())
        self.publish_state()
        self.trace.emit("observation_refreshed", self.current)
        return self.current.copy()

    def publish_state(self):
        write_json(self.workspace / "scene/state.json", {"memory_type": self.memory_type, "pose": self.current["pose"],
                "frame": self.current["frame"], "step": self.environment.steps})

    def register_target(self, image):
        return self.observations.add(image, kind="target")["frame"]

    def capture(self, observation, body_pose=None, **metadata):
        row = self.observations.capture(observation, self.environment.steps, **metadata)
        if observation.depth is not None and self.spatial.enabled:
            self.spatial.record_frame(observation, row, body_pose)
        return row

    def perform_motion(self, action, degrees=None):
        motion_limit = ((self.request.action_end is not None and self.environment.steps >= self.request.action_end)
                        or (self.request.path_end is not None and self.environment.path_length >= self.request.path_end))
        if self.cancel.is_set() or motion_limit:
            self.request.stop_reason = "aborted" if self.cancel.is_set() else (
                "path_budget" if self.request.path_end is not None and self.environment.path_length >= self.request.path_end else "action_budget")
            return None
        try:
            motion, observation = self.environment.step(action) if degrees is None else self.environment.turn(degrees)
            self.current = self.capture(observation)
        except BaseException:
            self.broken = True
            self.request.stop_reason = "execution_uncertain"
            raise
        self.trace.emit("motion", {"action": action, "samples": motion.samples,
                        "blocked": motion.blocked, "frame": self.current["frame"],
                        **({"turn_degrees": degrees} if degrees is not None else {})})
        if ((self.request.action_end is not None and self.environment.steps >= self.request.action_end)
                or (self.request.path_end is not None and self.environment.path_length >= self.request.path_end)):
            self.request.stop_reason = "path_budget" if self.request.path_end is not None and self.environment.path_length >= self.request.path_end else "action_budget"
        return motion

    def reply(self, value, refs=(), labels=(), *, error=False, end=False):
        result = {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False, separators=(",", ":"))}],
                  "images": list(refs), "image_labels": list(labels), "isError": error}
        reply = {"result": result}
        if end:
            reply["control"] = {"stop_remaining_tools": True, "end_run": True}
        return reply

    def _execute(self, name, args):
        tool = self.tools.get(name)
        if tool is None:
            raise ToolInputError(f"Unknown tool: {name}")
        return tool.call(args)

    def execute(self, call_id, name, args):
        serialized = json.dumps([name, args], sort_keys=True)
        if call_id in self._calls:
            previous, reply = self._calls[call_id]
            if serialized != previous:
                self.broken = True
                return self.reply({"error": "Call ID reused with different arguments"}, error=True, end=True)
            return reply
        if self.request.submitted or self.request.stop_reason or self.broken:
            return self.reply({"error": "Not executed: this request has ended."}, error=True, end=True)
        self._executing = True
        try:
            reply = self._execute(name, args)
        except ToolInputError as error:
            reply = self.reply({"error": str(error)}, error=True, end=self.broken)
        except Exception as error:
            self.broken = True
            reply = self.reply({"error": f"{type(error).__name__}: {error}"}, error=True, end=True)
        finally:
            self._executing = False
        self._calls[call_id] = serialized, reply
        try:
            self.trace.emit("public_tool_reply", {"call_id": call_id, "name": name, "args": args, **reply})
        except Exception:
            self.broken = True
            raise
        return reply

    def close(self):
        if self.trace:
            self.trace.close()
