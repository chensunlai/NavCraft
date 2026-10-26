"""Persistent navigation application; Pi owns the only model history."""

from dataclasses import asdict, dataclass
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import threading
import time
import uuid

from PIL import Image

from src.common.actions import ACTIONS
from src.common.io import write_json

from .config import core_options
from .core_client import CoreClient
from .memory import MemoryStore
from .maps import MapRenderer
from .observations import Observations, map_pose
from .service_retry import charged_requests, wait_seconds
from .tools.specs import specifications
from .trace import TraceWriter


@dataclass(frozen=True)
class PromptLimits:
    max_requests: int = 150
    max_actions: int = 500
    seconds: float = 3600
    max_path_length: float | None = None
    max_turns: int | None = None

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in (self.max_requests, self.max_actions)):
            raise ValueError("Request and action limits must be positive integers")
        if not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError("seconds must be finite and positive")
        if self.max_path_length is not None and (not math.isfinite(self.max_path_length) or self.max_path_length <= 0):
            raise ValueError("max_path_length must be finite and positive")
        if self.max_turns is not None and (type(self.max_turns) is not int or self.max_turns < 1):
            raise ValueError("max_turns must be a positive integer")


@dataclass(frozen=True)
class PromptResult:
    prompt_id: str
    submitted: bool
    answer: str
    reason: str
    requests: int
    actions: int
    record: str
    elapsed_seconds: float
    error: str | None = None

    def to_dict(self):
        return asdict(self)


class HarnessSession:
    def __init__(self, repo, config, directory, environment, scene_map, initial_state=None,
                 *, client_factory=CoreClient, start_core=True):
        self.repo, self.directory = Path(repo).resolve(), Path(directory).resolve()
        self.workspace = self.directory / "workspace"
        self.private = self.directory / "private"
        self.environment = environment
        self.config = config
        self.tool_tier = config.get("tool_tier", "basic")
        self.tool_specs = specifications(self.tool_tier)
        self.super_tools = None
        self.client = None
        self.session_id = None
        self._client_factory = client_factory
        self._history = []
        self._session_file = None
        self._qa = False
        self._path_end = None
        self._decision_window = False
        self._movement_exhausted = False
        self._run_turns = 0
        self._closing_scheduled = False
        self.trace = None
        self._busy = self._closed = self._broken = False
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._calls = {}
        self._submitted = False
        self._answer = ""
        self._stop_reason = None
        self._action_end = None
        self._requests = 0
        self._prompts = []
        self._target_frames = []
        self._image_counter = 0
        self._revision = 0
        for name in ("scene", "observations", "memory", "scratch"):
            (self.workspace / name).mkdir(parents=True, exist_ok=True)
        self.private.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(scene_map, self.workspace / "scene/map.json")
            self.transform = json.loads((self.workspace / "scene/map.json").read_text())
            shutil.copyfile(Path(__file__).with_name("navlib.py"), self.workspace / "scene/navlib.py")
            state = initial_state or {}
            if state.get("environment"):
                environment.restore_state(state["environment"])
            elif state.get("pose"):
                environment.restore_pose(state["pose"])
            if state.get("workspace"):
                for name in ("scene", "observations", "memory", "scratch"):
                    source = Path(state["workspace"]) / name
                    if source.exists():
                        shutil.copytree(source, self.workspace / name, dirs_exist_ok=True)
            if state.get("private_dir"):
                for name in ("depth", "semantics", "location_sources.json", "memory_used_ids.json"):
                    source = Path(state["private_dir"]) / name
                    if source.is_dir():
                        shutil.copytree(source, self.private / name, dirs_exist_ok=True)
                    elif source.is_file():
                        shutil.copyfile(source, self.private / name)
            if state.get("session_file"):
                self._session_file = self.private / "restored-session.jsonl"
                shutil.copyfile(state["session_file"], self._session_file)
            if state.get("observations"):
                shutil.copytree(state["observations"], self.workspace / "observations", dirs_exist_ok=True)
            if "memory" in state or not (self.workspace / "memory/entities.json").exists():
                write_json(self.workspace / "memory/entities.json", state.get("memory", []))
            task_notes = state.get("task_memory", "")
            if not isinstance(task_notes, str):
                raise ValueError("task_memory must be text")
            self.task_memory_path = self.workspace / "memory/task.json"
            if "task_memory" in state or not self.task_memory_path.exists():
                write_json(self.task_memory_path, {"content": task_notes})
            self.observations = Observations(self.workspace, self.transform)
            self.memory = MemoryStore(self.workspace, self.private)
            self.renderer = MapRenderer(self.workspace, self.memory)
            if self.tool_tier == "super":
                from .super_tools import SuperTools
                self.super_tools = SuperTools(self)
            self.current = (self.observations.get(state["current_frame"]) if state.get("current_frame")
                            else self._capture(environment.observe()))
            self._history = deepcopy(state.get("history", []))
            self._current_request = state.get("current_request", "")
            self._target_frames = list(state.get("target_frames", []))
            self._prompts = list(state.get("submitted_requests", []))
            self._image_counter = state.get("image_counter", 0)
            self._publish_state()
            self.trace = TraceWriter(self.private)
            source_files = []
            for folder in ("harness", "environment", "bench", "common", "dataset/episodes"):
                source_files.extend((self.repo / "src" / folder).rglob("*.py"))
            source_files.extend((self.repo / "src/agent-core").glob("*.ts"))
            source_files.append(self.repo / "src/agent-core/pi/packages/coding-agent/src/core/compaction/compaction.ts")
            source_hash = hashlib.sha256()
            for file in sorted(source_files):
                source_hash.update(file.relative_to(self.repo).as_posix().encode() + b"\0" + file.read_bytes())
            self.trace.emit("application_started", {
                "model": config["model"], "reasoning_effort": config.get("reasoning_effort", "high"),
                "source_sha256": source_hash.hexdigest(),
                "prompt_sha256": hashlib.sha256((self.repo / "config/prompts/navigation.md").read_bytes()).hexdigest(),
                "map_sha256": hashlib.sha256(Path(scene_map).read_bytes()).hexdigest(),
                "memory_source": state.get("memory_source", "empty"),
                "tool_tier": self.tool_tier,
                "goto_planner": self.super_tools.planner if self.super_tools else None,
            })
            if start_core:
                self._start_core()
        except BaseException:
            self.close()
            raise

    def _start_core(self):
        if self.client is not None:
            return
        options = core_options(self.config, self.repo, self.directory, self.workspace,
                               self.tool_specs, self.checkpoint())
        if self._session_file:
            options["sessionFile"] = str(self._session_file)
        elif self._history:
            options["initialMessages"] = deepcopy(self._history)
        self.trace._secrets = (options["auth"]["apiKey"],)
        self.client = self._client_factory(self.repo, self.private)
        receipt = self.client.request("init", options, timeout=120)
        self.session_id = receipt["session_id"]

    def begin_script(self, text, images=()):
        """Begin an explicitly scripted prefix without sending any model request."""
        if self.client or self._busy or self._closed:
            raise RuntimeError("Script collection requires an idle session without a Core")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Expected a navigation request")
        self._current_request = text
        self._target_frames = [self.register_target(image) for image in images]
        content = [{"type": "text", "text": text},
                   {"type": "text", "text": f"[CURRENT {self.current['frame']}]"},
                   {"type": "image", "path": self.current["path"]}]
        for frame in self._target_frames:
            content.extend([{"type": "text", "text": f"[TARGET {frame}]"},
                            {"type": "image", "path": self.observations.get(frame)["path"]}])
        self._history.append({"role": "user", "timestamp": int(time.time() * 1000), "content": content})

    def script_tool(self, name, args):
        """Execute a real tool and retain its provenance as non-model prefix."""
        if self.client or self._busy or self._closed:
            raise RuntimeError("Script collection requires an idle session without a Core")
        if not self._history:
            raise RuntimeError("Call begin_script before collecting tool receipts")
        call_id = "script_" + uuid.uuid4().hex
        reply = self._tool(call_id, name, args)
        stamp = int(time.time() * 1000)
        self._history.append({"role": "assistant", "api": "openai-responses", "provider": "script",
                              "model": "script", "timestamp": stamp, "stopReason": "toolUse",
                              "content": [{"type": "toolCall", "id": call_id, "name": name, "arguments": args}],
                              "usage": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
                                        "totalTokens": 0, "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}}})
        result = reply["result"]
        content = deepcopy(result["content"])
        for ref, label in zip(result.get("images", []), result.get("image_labels", [])):
            content.extend([{"type": "text", "text": label}, {"type": "image", "path": ref}])
        self._history.append({"role": "toolResult", "toolCallId": call_id, "toolName": name,
                              "content": content, "isError": result.get("isError", False), "timestamp": stamp})
        self.trace.emit("script_tool", {"name": name, "args": args, "call_id": call_id, "trainable": False})
        return reply

    def snapshot(self, destination):
        """Save an idle state; source images and geometric indices stay immutable."""
        if self._busy or self._closed or self._broken:
            raise RuntimeError("Cannot snapshot a busy, closed or uncertain session")
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError(destination)
        exported = self.client.request("export_state") if self.client else {"messages": deepcopy(self._history)}
        history = exported["messages"]
        destination.mkdir(parents=True)
        shutil.copytree(self.workspace, destination / "workspace")
        for name in ("depth", "semantics", "location_sources.json", "memory_used_ids.json"):
            source = self.private / name
            if source.is_dir():
                shutil.copytree(source, destination / "private" / name)
            elif source.is_file():
                (destination / "private").mkdir(exist_ok=True)
                shutil.copyfile(source, destination / "private" / name)
        session_file = exported.get("session_file")
        if session_file:
            (destination / "private").mkdir(exist_ok=True)
            shutil.copyfile(session_file, destination / "private/pi-session.jsonl")
        state = {"pose": self.environment.pose(), "current_frame": self.current["frame"],
                 "workspace": "workspace", "private_dir": "private", "history": history,
                 "current_request": getattr(self, "_current_request", ""),
                 "target_frames": self._target_frames, "submitted_requests": self._prompts,
                 "image_counter": self._image_counter,
                 "environment": self.environment.state() if hasattr(self.environment, "state") else None}
        if session_file:
            state["session_file"] = "private/pi-session.jsonl"
        write_json(destination / "state.json", state)
        return state

    @property
    def requests(self):
        path = self.private / "requests.json"
        return len(json.loads(path.read_text())["attempts"]) if path.exists() else self._requests

    def _publish_state(self):
        self._revision += 1
        write_json(self.workspace / "scene/state.json", {"pose": self.current["pose"],
                "frame": self.current["frame"], "step": self.environment.steps})

    def checkpoint(self):
        entities = [{key: entity.get(key) for key in ("id", "type", "name", "room")}
                    for entity in self.memory.entities]
        targets = [self.observations.get(frame) for frame in self._target_frames]
        facts = {"current_request": getattr(self, "_current_request", ""),
                 "submitted_requests": self._prompts, "pose": self.current["pose"],
                 "frame": self.current["frame"], "target_frames": self._target_frames,
                 "entities": entities,
                 "task_memory": json.loads(self.task_memory_path.read_text())["content"]}
        return {"content": "Public facts (no evaluation outcome):\n" + json.dumps(facts, ensure_ascii=False),
                "images": [self.current["path"], *[row["path"] for row in targets]],
                "image_labels": [f"[CURRENT {self.current['frame']}]", *[f"[TARGET {row['frame']}]" for row in targets]],
                "revision": self._revision}

    def register_target(self, image):
        return self.observations.add(image, kind="target")["frame"]

    def refresh_observation(self):
        """Publish a fresh view after an explicit host-side state change."""
        with self._lock:
            if self._busy or self._closed or self._broken:
                raise RuntimeError("Session is busy, closed or has uncertain execution")
            self.current = self._capture(self.environment.observe())
            self._publish_state()
            self.trace.emit("observation_refreshed", self.current)
            return self.current.copy()

    def _capture(self, observation, body_pose=None, **metadata):
        row = self.observations.capture(observation, self.environment.steps, **metadata)
        if self.super_tools:
            self.super_tools.record_frame(observation, row, body_pose)
        semantic = getattr(observation, "semantic", None)
        if semantic is not None:
            directory = self.private / "semantics"
            directory.mkdir(exist_ok=True)
            shutil.copyfile(semantic, directory / (row["frame"] + ".npy"))
        return row

    def _perform_motion(self, action, degrees=None):
        motion_limit = ((self._action_end is not None and self.environment.steps >= self._action_end)
                        or (self._path_end is not None and self.environment.path_length >= self._path_end))
        if self._cancel.is_set() or motion_limit:
            if self._decision_window and not self._cancel.is_set():
                self._movement_exhausted = True
            else:
                self._stop_reason = "aborted" if self._cancel.is_set() else "action_budget"
            return None
        try:
            motion, observation = self.environment.step(action) if degrees is None else self.environment.turn(degrees)
            self.current = self._capture(observation)
        except BaseException:
            self._broken = True
            self._stop_reason = "execution_uncertain"
            raise
        self.trace.emit("motion", {"action": action, "samples": motion.samples,
                        "blocked": motion.blocked, "frame": self.current["frame"],
                        **({"turn_degrees": degrees} if degrees is not None else {})})
        if ((self._action_end is not None and self.environment.steps >= self._action_end)
                or (self._path_end is not None and self.environment.path_length >= self._path_end)):
            if self._decision_window:
                self._movement_exhausted = True
            else:
                self._stop_reason = "path_budget" if self._path_end is not None and self.environment.path_length >= self._path_end else "action_budget"
        return motion

    def _reply(self, value, refs=(), labels=(), *, error=False, end=False):
        result = {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False, separators=(",", ":"))}],
                  "images": list(refs), "image_labels": list(labels), "isError": error}
        reply = {"result": result, "checkpoint": self.checkpoint()}
        if end:
            reply["control"] = {"stop_remaining_tools": True, "end_run": True}
        return reply

    def _observe(self, mode="current", project=False):
        if type(project) is not bool:
            raise ValueError("project must be boolean")
        if mode not in {"current", "panorama"}:
            raise ValueError("mode must be current or panorama")
        if mode == "current":
            self.current = self._capture(self.environment.observe())
            self._publish_state()
            value = {key: self.current[key] for key in ("frame", "step", "pose")}
            refs, labels = [self.current["path"]], [f"[CURRENT {self.current['frame']}]"]
            views = [self.current]
        else:
            captured_pose = self.environment.pose()
            body_pose = map_pose(captured_pose, self.transform)
            group = "p_" + uuid.uuid4().hex[:8]
            views = [self._capture(obs, body_pose=captured_pose, group=group, yaw=i * 60)
                     for i, obs in enumerate(self.environment.panorama())]
            self.current = views[0]
            self._publish_state()
            image = self.observations.collage([v["path"] for v in views], f"observations/{group}.jpg",
                        [f"{v['frame']} | right {v['yaw']} deg" for v in views])
            value = {"group": group, "views": [{"frame": v["frame"], "yaw": v["yaw"]} for v in views],
                     "step": self.environment.steps, "pose": body_pose}
            refs, labels = [image], [f"[CURRENT {group}]"]
        if project:
            mapped = self._render_map(view={"ego": 6}, overlay=[v["frame"] for v in views])
            value["projection"] = mapped
            refs.append(mapped["map"])
            labels.append(f"[MAP {mapped['view_id']}]")
        return self._reply(value, refs, labels)

    def _step(self, actions):
        if (not isinstance(actions, list) or not 1 <= len(actions) <= 6
                or any(not isinstance(a, str) or a not in ACTIONS for a in actions)):
            raise ValueError("actions must contain 1–6 valid atomic actions")
        done, events, stop = [], [], "completed"
        for i, action in enumerate(actions, 1):
            motion = self._perform_motion(action)
            if motion is None:
                stop = "interrupted"
                break
            if done and done[-1][0] == action:
                done[-1][1] += 1
            else:
                done.append([action, 1])
            if motion.collision or motion.blocked:
                event = {"i": i, "collision": bool(motion.collision)}
                if motion.auto_yaw_delta_degrees:
                    event["auto_yaw"] = round(-motion.auto_yaw_delta_degrees, 2)
                events.append(event)
            if motion.blocked:
                stop = "blocked"
                break
        self._publish_state()
        value = {"done": done, "stop": stop, "frame": self.current["frame"],
                 "step": self.environment.steps, "pose": self.current["pose"]}
        if events:
            value["events"] = events
        if not self._decision_window and self._action_end is not None and self.environment.steps >= self._action_end:
            self._stop_reason = "action_budget"
        return self._reply(value, [self.current["path"]], [f"[CURRENT {self.current['frame']}]"], end=bool(self._stop_reason))

    def _images(self, frames, layout="separate"):
        if not isinstance(frames, list) or not 1 <= len(frames) <= 6 or layout not in {"separate", "grid"}:
            raise ValueError("Expected 1–6 frames and separate/grid layout")
        rows, refs, labels = [], [], []
        # Validate every source/crop before producing any derived image.
        for value in frames:
            if not isinstance(value, dict) or set(value) - {"frame", "crop"}:
                raise ValueError("Expected frame and optional crop")
            row = self.observations.get(value.get("frame"))
            crop = value.get("crop")
            if crop is not None and (not isinstance(crop, list) or len(crop) != 4 or
                    any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in crop)
                    or crop[0] >= crop[2] or crop[1] >= crop[3]):
                raise ValueError("crop must be normalized [left,top,right,bottom]")
            rows.append((row, crop))
        public = []
        for row, crop in rows:
            ref = row["path"]
            metadata = {k: row[k] for k in ("frame", "step", "pose", "group", "yaw") if k in row}
            if crop is not None:
                self._image_counter += 1
                ref = f"observations/derived/c_{self._image_counter:04d}.jpg"
                destination = self.workspace / ref
                destination.parent.mkdir(exist_ok=True)
                with Image.open(self.workspace / row["path"]) as source:
                    w, h = source.size
                    bounds = [int(crop[0] * w), int(crop[1] * h), max(int(crop[2] * w), int(crop[0] * w) + 1),
                              max(int(crop[3] * h), int(crop[1] * h) + 1)]
                    source.crop(bounds).save(destination, quality=85)
                metadata["crop"] = crop
            refs.append(ref)
            labels.append(f"[{'TARGET' if row['kind'] == 'target' else 'HISTORY'} {row['frame']}]")
            public.append(metadata)
        if layout == "grid":
            self._image_counter += 1
            refs = [self.observations.collage(refs, f"observations/derived/g_{self._image_counter:04d}.jpg", labels)]
            labels = ["[HISTORY grid: " + ", ".join(row[0]["frame"] for row in rows) + "]"]
        return self._reply({"frames": public}, refs, labels)

    def _render_map(self, overlay=None, **args):
        if not self.super_tools and isinstance(args.get("view"), dict) and "ego" in args["view"]:
            raise ValueError("ego maps require tool_tier=super")
        if overlay is not None and not isinstance(overlay, list):
            raise ValueError("overlay must be an array of frame IDs")
        points = self.super_tools.geometry.overlay(overlay) if overlay else None
        origin, size = self.transform["origin"], self.transform["size_m"]
        trajectory = [[(p[0] - origin[0]) / size, (p[2] - origin[1]) / size] for p in self.environment.trajectory]
        value = self.renderer.render(self.current["pose"], trajectory, overlay_points=points,
                                     trajectory_breaks=getattr(self.environment, "trajectory_breaks", ()), **args)
        return value if self.super_tools else {key: value[key] for key in ("map", "bounds", "warnings")}

    def _locate(self, source, points):
        resolved = self.super_tools.geometry.locate(source, points)
        value = {"source": source, "points": []}
        located = []
        for index, point in enumerate(resolved["points"], 1):
            label = f"P{index}"
            if point.get("valid", True) and "uv" in point:
                value["points"].append({"label": label, "uv": point["uv"]})
                located.append({"label": label, "uv": point["uv"]})
            else:
                value["points"].append({"label": label, "error": point["reason"]})
        if not located:
            return self._reply(value)
        mapped = self._render_map(view="explored", located_points=located)
        value["view_id"] = mapped["view_id"]
        return self._reply(value, [mapped["map"]], [f"[MAP {mapped['view_id']} | locate points]"])

    def _task_memory(self, action, content=None):
        if action not in {"read", "edit"}:
            raise ValueError("action must be read or edit")
        if action == "read":
            if content is not None:
                raise ValueError("read does not accept content")
            return self._reply(json.loads(self.task_memory_path.read_text()))
        if not isinstance(content, str):
            raise ValueError("edit requires text content")
        before = self.trace.store(json.loads(self.task_memory_path.read_text()))
        write_json(self.task_memory_path, {"content": content})
        self._revision += 1
        self.trace.emit("task_memory_changed", {"before": before, "after": self.trace.store({"content": content})})
        return self._reply({"updated": True})

    def _execute(self, name, args):
        definition = next((tool for tool in self.tool_specs if tool["name"] == name), None)
        if definition is None or not isinstance(args, dict) or set(args) - definition["parameters"]["properties"].keys():
            raise ValueError("Unknown tool or unsupported arguments for this tool tier")
        if self._qa and (name not in {"view_images", "memory_read", "map_view", "locate", "submit", "task_memory"}
                         or name == "task_memory" and args.get("action") != "read"):
            raise ValueError("This question permits only existing-evidence queries and submission")
        if self._movement_exhausted and name in {"nav_step", "nav_face", "nav_goto", "nav_observe"}:
            raise ValueError("Movement window ended; finish using existing evidence")
        if name == "locate":
            return self._locate(**args)
        if name in {"nav_face", "nav_goto"}:
            return self.super_tools.move(**args, face=name == "nav_face")
        if name == "nav_observe":
            return self._observe(**args)
        if name == "nav_step":
            return self._step(**args)
        if name == "view_images":
            return self._images(**args)
        if name == "task_memory":
            return self._task_memory(**args)
        if name == "memory_read":
            result = self.super_tools.near(**args) if "near" in args else self.memory.read(**args)
            return self._reply(result, error="error" in result)
        if name == "memory_edit":
            before = self.trace.store(self.memory.entities)
            operations, sources = self.super_tools.prepare_edits(args.get("operations")) if self.super_tools else (args.get("operations"), {})
            result = self.memory.edit(operations)
            if "error" not in result:
                if self.super_tools:
                    self.super_tools.remember_sources(operations, sources)
                self._revision += 1
                self.trace.emit("memory_changed", {"before": before, "after": self.trace.store(self.memory.entities)})
            return self._reply(result, error="error" in result)
        if name == "map_view":
            value = self._render_map(**args)
            return self._reply(value, [value["map"]], ["[MAP " + Path(value["map"]).stem + "]"])
        if name == "submit":
            if set(args) - {"answer"} or ("answer" in args and not isinstance(args["answer"], str)):
                raise ValueError("submit accepts only an optional answer string")
            self._submitted, self._answer = True, args.get("answer", "")
            return self._reply({"submitted": True}, end=True)
        raise ValueError(f"Unknown tool: {name}")

    def _tool(self, call_id, name, args):
        serialized = json.dumps([name, args], sort_keys=True)
        if call_id in self._calls:
            previous, reply = self._calls[call_id]
            if serialized != previous:
                return self._reply({"error": "Call ID reused with different arguments"}, error=True, end=True)
            return reply
        if self._submitted or self._stop_reason or self._broken:
            return self._reply({"error": "Not executed: this request has ended."}, error=True, end=True)
        try:
            reply = self._execute(name, args)
        except (ValueError, TypeError, KeyError) as error:
            reply = self._reply({"error": str(error)}, error=True, end=self._broken)
        except Exception as error:
            self._broken = True
            reply = self._reply({"error": f"{type(error).__name__}: {error}"}, error=True, end=True)
        self._calls[call_id] = serialized, reply
        self.trace.emit("public_tool_reply", {"call_id": call_id, "name": name, "args": args, **reply})
        if self._decision_window and self._movement_exhausted and not self._closing_scheduled and self.client:
            self._closing_scheduled = True
            self.client.request("configure", {"maxTurns": min(self._run_turns + 1, self._turn_limit)})
        return reply

    def _event(self, value, callback=None):
        self.trace.accept(value)
        if value.get("type") == "event":
            event = value.get("event", {})
            if event.get("type") == "message_end" and event.get("data", {}).get("message", {}).get("role") == "assistant":
                self._run_turns += 1
        if callback:
            callback(value)

    def prompt(self, text, images=(), limits=None, on_event=None):
        return self._run(text, images, limits, on_event, mode="prompt")

    def ask(self, question, images=(), limits=None, on_event=None):
        """Ask about the frozen current state without changing its navigation goal."""
        return self._run(question, images, limits, on_event, mode="ask")

    def continue_run(self, limits=None, on_event=None):
        """Resume after existing receipts without appending a user message."""
        return self._run(None, (), limits, on_event, mode="continue")

    def _run(self, text, images=(), limits=None, on_event=None, *, mode):
        limits = limits or PromptLimits()
        if mode != "continue" and (not isinstance(text, str) or not text.strip()):
            raise ValueError("Expected nonempty user request")
        if not isinstance(limits, PromptLimits):
            raise TypeError("limits must be PromptLimits")
        if any(not isinstance(ref, (str, Path)) for ref in images):
            raise ValueError("images must contain frame IDs or image paths")
        with self._lock:
            if self._busy or self._closed or self._broken:
                raise RuntimeError("Session is busy, closed or has uncertain execution")
            # Resolve input images before accepting the prompt. A missing input
            # must not leave the session busy or create an incomplete prompt record.
            refs = [] if mode == "continue" else [self.current["path"]]
            captions = [f"Image 1: [CURRENT {self.current['frame']}]"]
            target_frames = []
            for ref in images:
                if ref in self.observations.items:
                    row = self.observations.get(ref)
                else:
                    row = self.observations.add(ref, kind="question" if mode == "ask" else "target")
                refs.append(row["path"])
                target_frames.append(row["frame"])
                captions.append(f"Image {len(refs)}: [{'QUESTION' if mode == 'ask' else 'TARGET'} {row['frame']}]")
            if mode == "prompt":
                self._target_frames = target_frames
            self._revision += 1
            self._busy = True
        started, prompt_id = time.monotonic(), uuid.uuid4().hex
        before, action_start = charged_requests(self.private), self.environment.steps
        wait_before = wait_seconds(self.private)
        self._submitted, self._answer, self._stop_reason = False, "", None
        self._cancel.clear()
        self._qa = mode == "ask"
        self._decision_window = mode == "continue" and limits.max_turns is not None
        self._movement_exhausted = False
        self._run_turns = 0
        self._closing_scheduled = False
        self._turn_limit = limits.max_turns
        if mode == "prompt":
            self._current_request = text
        self._action_end = action_start + limits.max_actions
        self._path_end = self.environment.path_length + limits.max_path_length if limits.max_path_length is not None else None
        model_text = (text or "") + "\nCurrent map pose: " + json.dumps(self.current["pose"]) + "\n" + "\n".join(captions)
        if mode == "ask":
            model_text = ("Question about the current state. Keep the active navigation request unchanged. "
                          "Use only existing evidence and call submit with your answer.\n" + model_text)
        settled = {"reason": "runtime_error"}
        reason = "runtime_error"
        error_text = None
        try:
            self._start_core()
            self.trace.emit("prompt_start", {"prompt_id": prompt_id, "text": text, "mode": mode,
                            "images": refs, "limits": asdict(limits)})
            active_tools = ["read", "write", "edit", "bash", *[t["name"] for t in self.tool_specs]]
            if self._qa:
                active_tools = [name for name in active_tools if name in
                                {"read", "write", "edit", "bash", "view_images", "memory_read", "map_view", "locate", "submit"}]
            self.client.request("configure", {"maxRequests": before + limits.max_requests, "allowedTools": active_tools})
            self.client.request("set_checkpoint", {"checkpoint": self.checkpoint()})
            for attempt in range(2):
                active_seconds = time.monotonic() - started - (wait_seconds(self.private) - wait_before)
                extra = {}
                if mode == "continue" and not attempt:
                    extra["mode"] = "continue"
                if limits.max_turns is not None:
                    extra["max_turns"] = limits.max_turns
                settled = self.client.run_prompt(None if mode == "continue" and not attempt else model_text, refs, self._tool,
                            max(.01, limits.seconds - active_seconds),
                            on_event=lambda value: self._event(value, on_event),
                            cancel_event=self._cancel, prompt_id=prompt_id if not attempt else prompt_id + "-continue", **extra)
                self._requests = settled.get("requests", self._requests)
                if (settled.get("reason") != "no_action_output" or self._submitted or self._broken or attempt
                        or self._stop_reason or charged_requests(self.private) - before >= limits.max_requests):
                    break
                model_text, refs = "Your last response contained no visible answer or tool call. Continue the current request using the available tools, and call submit when complete.", []
                self.trace.emit("empty_output_continuation", {"prompt_id": prompt_id})
            reason = "submitted" if self._submitted else self._stop_reason or settled.get("reason", "runtime_error")
            if reason == "model_finished":
                reason = "unsubmitted"
            error_text = settled.get("error")
            if self._broken:
                reason = self._stop_reason or "runtime_error"
        except Exception as error:
            reason, error_text = "runtime_error", f"{type(error).__name__}: {error}"
            self._broken = True
        finally:
            self._action_end = None
            self._path_end = None
            self._qa = self._decision_window = False
            self._movement_exhausted = False
            if self._submitted and mode != "ask":
                self._prompts.append(getattr(self, "_current_request", ""))
            elapsed = time.monotonic() - started - (wait_seconds(self.private) - wait_before)
            result = PromptResult(prompt_id, self._submitted, self._answer, reason,
                                  charged_requests(self.private) - before, self.environment.steps - action_start,
                                  f"private/prompts/{prompt_id}.json", max(0, elapsed), self.trace.sanitize(error_text))
            self.trace.emit("prompt_end", result.to_dict())
            write_json(self.directory / result.record, result.to_dict())
            with self._lock:
                self._busy = False
        return result

    def abort(self):
        self._cancel.set()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.client:
                self.client.close()
                if self.trace:
                    self.client.drain_events(self.trace.accept)
        finally:
            if self.trace:
                self.trace.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def open_session(repo, config, directory, environment, scene_map, initial_state=None, **kwargs):
    return HarnessSession(repo, config, directory, environment, scene_map, initial_state, **kwargs)


def read_snapshot(path):
    """Resolve a saved snapshot without exposing its private files to the model."""
    path = Path(path)
    path = path / "state.json" if path.is_dir() else path
    state = json.loads(path.read_text())
    for name in ("workspace", "private_dir"):
        state[name] = str((path.parent / state[name]).resolve())
    if state.get("session_file"):
        state["session_file"] = str((path.parent / state["session_file"]).resolve())
    return state
