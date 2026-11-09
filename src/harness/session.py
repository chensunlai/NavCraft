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

from jsonschema import Draft7Validator

from src.common.coordinates import validate_map

from src.common.io import write_json

from .config import core_options
from .core_client import CoreClient
from .memory import MemoryStore
from .node_memory import NodeMemory
from .maps import MapRenderer
from .observations import Observations
from .service_retry import charged_requests, wait_seconds
from .tools.specs import specifications
from .tools.runtime import NavigationTools
from .spatial import SpatialTools
from .trace import TraceWriter


@dataclass(frozen=True)
class PromptLimits:
    max_requests: int = 150
    max_actions: int = 500
    seconds: float = 3600
    max_path_length: float | None = None
    max_turns: int | None = None
    max_observations: int | None = None

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in (self.max_requests, self.max_actions)):
            raise ValueError("Request and action limits must be positive integers")
        if not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError("seconds must be finite and positive")
        if self.max_path_length is not None and (not math.isfinite(self.max_path_length) or self.max_path_length <= 0):
            raise ValueError("max_path_length must be finite and positive")
        if self.max_observations is not None and (type(self.max_observations) is not int or self.max_observations < 0):
            raise ValueError("max_observations must be a nonnegative integer")
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
        self.memory_type = config.get("memory_type", "entity")
        if self.memory_type not in {"entity", "node"}:
            raise ValueError("memory_type must be entity or node")
        self.tool_tier = config.get("tool_tier", "basic")
        self.tool_specs = specifications(self.tool_tier, self.memory_type)
        self._validators = {tool["name"]: Draft7Validator(tool["parameters"]) for tool in self.tool_specs}
        self.spatial = None
        self.tools = NavigationTools(self)
        self._observations = 0
        self._observation_end = None
        self._current_question = None
        self._question_frames = []
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
        self._prompts = []
        self._target_frames = []
        for name in ("scene", "observations", "memory", "scratch"):
            (self.workspace / name).mkdir(parents=True, exist_ok=True)
        self.private.mkdir(parents=True, exist_ok=True)
        try:
            state = initial_state or {}
            restoring = any(key in state for key in ("workspace", "session_file", "history", "memory"))
            if ("memory_type" in state or restoring) and state.get("memory_type", "entity") != self.memory_type:
                raise ValueError("Snapshot memory_type does not match the requested mode; start a new session")
            if state.get("environment"):
                environment.restore_state(state["environment"])
            elif state.get("pose"):
                environment.restore_pose(state["pose"])
            if self.memory_type == "entity":
                shutil.copyfile(scene_map, self.workspace / "scene/map.json")
            elif state.get("workspace"):
                shutil.copyfile(Path(state["workspace"]) / "scene/map.json", self.workspace / "scene/map.json")
            else:
                x, height, z = environment.pose()["position"]
                write_json(self.workspace / "scene/map.json", {
                    "origin": [x - 10, z - 10], "size_m": 20,
                    "floor_height": height, "walls": [],
                })
            self.transform = json.loads((self.workspace / "scene/map.json").read_text())
            validate_map(self.transform)
            if self.memory_type == "node" and self.transform["walls"]:
                raise ValueError("Node memory requires a public map without walls")
            shutil.copyfile(Path(__file__).with_name("navlib.py"), self.workspace / "scene/navlib.py")
            shutil.copyfile(Path(__file__).parents[1] / "common/coordinates.py", self.workspace / "scene/coordinates.py")
            if state.get("workspace"):
                for name in ("scene", "observations", "memory", "scratch"):
                    source = Path(state["workspace"]) / name
                    if source.exists():
                        shutil.copytree(source, self.workspace / name, dirs_exist_ok=True, symlinks=True)
            if state.get("private_dir"):
                for name in ("depth", "location_sources.json", "memory_used_ids.json"):
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
            memory_name = "entities.json" if self.memory_type == "entity" else "nodes.json"
            other_name = "nodes.json" if self.memory_type == "entity" else "entities.json"
            if (self.workspace / "memory" / other_name).exists():
                raise ValueError("Workspace contains the other memory representation")
            if "memory" in state or not (self.workspace / "memory" / memory_name).exists():
                empty_memory = [] if self.memory_type == "entity" else {"nodes": [], "rooms": {}}
                write_json(self.workspace / "memory" / memory_name, state.get("memory", empty_memory))
            restored_map = json.loads((self.workspace / "scene/map.json").read_text())
            if restored_map != self.transform:
                raise ValueError("Snapshot map does not match the requested scene map")
            self.observations = Observations(self.workspace, self.transform)
            self.memory = (MemoryStore(self.workspace, self.private) if self.memory_type == "entity" else
                           NodeMemory(self.workspace, self.observations, self.transform, self.private))
            self.renderer = MapRenderer(self.workspace, self.memory, memory_type=self.memory_type)
            self.spatial = SpatialTools(self)
            self.current = (self.observations.get(state["current_frame"]) if state.get("current_frame")
                            else self._capture(environment.observe()))
            self._history = deepcopy(state.get("history", []))
            self._calls = deepcopy(state.get("tool_receipts", {}))
            self._current_request = state.get("current_request", "")
            self._target_frames = list(state.get("target_frames", []))
            self._prompts = list(state.get("submitted_requests", []))
            self._observations = state.get("observations_count", 0)
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
                "prompt_sha256": hashlib.sha256((self.repo / "config/prompts" /
                    ("navigation_node.md" if self.memory_type == "node" else "navigation.md")).read_bytes()).hexdigest(),
                "map_sha256": hashlib.sha256((self.workspace / "scene/map.json").read_bytes()).hexdigest(),
                "memory_source": state.get("memory_source", "empty"),
                "memory_type": self.memory_type,
                "tool_tier": self.tool_tier,
                "goto_planner": self.spatial.planner if self.spatial.enabled else None,
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
                               self.tool_specs)
        if self._session_file:
            options["sessionFile"] = str(self._session_file)
        elif self._history:
            options["initialMessages"] = deepcopy(self._history)
        self.trace._secrets = (options["auth"]["apiKey"],)
        self.client = self._client_factory(self.repo, self.private)
        receipt = self.client.request("init", options, timeout=120)
        self.session_id = receipt["session_id"]

    def snapshot(self, destination):
        """Save an idle state; source images and geometric indices stay immutable."""
        with self._lock:
            if self._busy or self._closed or self._broken:
                raise RuntimeError("Cannot snapshot a busy, closed or uncertain session")
            destination = Path(destination)
            if destination.exists():
                raise FileExistsError(destination)
            exported = self.client.request("export_state") if self.client else {"messages": deepcopy(self._history)}
            history = exported["messages"]
            destination.mkdir(parents=True)
            shutil.copytree(self.workspace, destination / "workspace", symlinks=True)
            for name in ("depth", "location_sources.json", "memory_used_ids.json"):
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
            state = {"memory_type": self.memory_type,
                     "pose": self.environment.pose(), "current_frame": self.current["frame"],
                     "workspace": "workspace", "private_dir": "private", "history": history,
                     "current_request": self._current_request,
                     "target_frames": self._target_frames, "submitted_requests": self._prompts,
                     "observations_count": self._observations,
                     "tool_receipts": deepcopy(self._calls),
                     "environment": self.environment.state()}
            if session_file:
                state["session_file"] = "private/pi-session.jsonl"
            write_json(destination / "state.json", state)
            return state

    def _publish_state(self):
        write_json(self.workspace / "scene/state.json", {"memory_type": self.memory_type, "pose": self.current["pose"],
                "frame": self.current["frame"], "step": self.environment.steps})

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
        if observation.depth is not None and self.spatial.enabled:
            self.spatial.record_frame(observation, row, body_pose)
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
        reply = {"result": result}
        if end:
            reply["control"] = {"stop_remaining_tools": True, "end_run": True}
        return reply

    def _execute(self, name, args):
        validator = self._validators.get(name)
        if validator is None:
            raise ValueError(f"Unknown tool: {name}")
        error = next(validator.iter_errors(args), None)
        if error:
            field = ".".join(map(str, error.absolute_path)) or "arguments"
            raise ValueError(f"{field}: {error.message}")
        return self.tools.execute(name, args)

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
            self._busy = True
        started, prompt_id = time.monotonic(), uuid.uuid4().hex
        before, action_start = charged_requests(self.private), self.environment.steps
        wait_before = wait_seconds(self.private)
        self._submitted, self._answer, self._stop_reason = False, "", None
        self._cancel.clear()
        self._qa = mode == "ask"
        self._current_question = text if self._qa else None
        self._question_frames = target_frames if self._qa else []
        self._decision_window = mode == "continue" and limits.max_turns is not None
        self._movement_exhausted = False
        self._run_turns = 0
        self._closing_scheduled = False
        self._turn_limit = limits.max_turns
        if mode == "prompt":
            self._current_request = text
        self._observation_end = self._observations + limits.max_observations if limits.max_observations is not None else None
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
            self.client.request("configure", {"maxRequests": before + limits.max_requests, "allowedTools": active_tools})
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
            self._observation_end = None
            self._current_question = None
            self._question_frames = []
            self._path_end = None
            self._qa = self._decision_window = False
            self._movement_exhausted = False
            if self._submitted and mode != "ask":
                self._prompts.append(self._current_request)
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
        with self._lock:
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
