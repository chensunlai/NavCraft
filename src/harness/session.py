"""Persistent navigation application; Pi owns the only model history."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import threading
import time
import uuid

from PIL import Image

from .config import core_options
from .core_client import CoreClient
from .memory import MemoryStore
from .maps import MapRenderer
from .observations import Observations, map_pose
from .service_retry import charged_requests, wait_seconds
from .tools.specs import ACTIONS, specifications
from .trace import TraceWriter


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class PromptLimits:
    max_requests: int = 150
    max_actions: int = 500
    seconds: float = 3600

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in (self.max_requests, self.max_actions)):
            raise ValueError("Request and action limits must be positive integers")
        if not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError("seconds must be finite and positive")


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
                 *, client_factory=CoreClient):
        self.repo, self.directory = Path(repo).resolve(), Path(directory).resolve()
        self.workspace = self.directory / "workspace"
        self.private = self.directory / "private"
        self.environment = environment
        self.config = config
        self.client = None
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
            if state.get("pose"):
                environment.restore_pose(state["pose"])
            if state.get("observations"):
                shutil.copytree(state["observations"], self.workspace / "observations", dirs_exist_ok=True)
            _write(self.workspace / "memory/entities.json", state.get("memory", []))
            self.observations = Observations(self.workspace, self.transform)
            self.memory = MemoryStore(self.workspace, self.private)
            self.renderer = MapRenderer(self.workspace)
            self.current = self.observations.capture(environment.observe(), environment.steps)
            self._publish_state()
            options = core_options(config, self.repo, self.directory, self.workspace,
                                   specifications(), self.checkpoint())
            if state.get("history"):
                options["initialMessages"] = state["history"]
            self.trace = TraceWriter(self.private, [options["auth"]["apiKey"]])
            self.trace.emit("application_started", {
                "model": config["model"], "reasoning_effort": config.get("reasoning_effort", "high"),
                "prompt_sha256": hashlib.sha256(options["systemPrompt"].encode()).hexdigest(),
                "map_sha256": hashlib.sha256(Path(scene_map).read_bytes()).hexdigest(),
                "memory_source": state.get("memory_source", "empty"),
            })
            self.client = client_factory(self.repo, self.private)
            receipt = self.client.request("init", options, timeout=120)
            self.session_id = receipt["session_id"]
        except BaseException:
            self.close()
            raise

    @property
    def requests(self):
        path = self.private / "requests.json"
        return len(json.loads(path.read_text())["attempts"]) if path.exists() else self._requests

    def _publish_state(self):
        self._revision += 1
        _write(self.workspace / "scene/state.json", {"pose": self.current["pose"],
                "frame": self.current["frame"], "step": self.environment.steps})

    def checkpoint(self):
        entities = [{key: entity.get(key) for key in ("id", "type", "name", "room")}
                    for entity in self.memory.entities]
        facts = {"current_request": getattr(self, "_current_request", ""),
                 "submitted_requests": self._prompts, "pose": self.current["pose"],
                 "frame": self.current["frame"], "entities": entities}
        return {"content": "Public facts (no evaluation outcome):\n" + json.dumps(facts, ensure_ascii=False),
                "images": [self.current["path"]],
                "image_labels": [f"[CURRENT {self.current['frame']}]"], "revision": self._revision}

    def register_target(self, image):
        return self.observations.add(image, kind="target")["frame"]

    def _reply(self, value, refs=(), labels=(), *, error=False, end=False):
        result = {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False, separators=(",", ":"))}],
                  "images": list(refs), "image_labels": list(labels), "isError": error}
        reply = {"result": result, "checkpoint": self.checkpoint()}
        if end:
            reply["control"] = {"stop_remaining_tools": True, "end_run": True}
        return reply

    def _observe(self, mode="current"):
        if mode not in {"current", "panorama"}:
            raise ValueError("mode must be current or panorama")
        if mode == "current":
            self.current = self.observations.capture(self.environment.observe(), self.environment.steps)
            self._publish_state()
            value = {key: self.current[key] for key in ("frame", "step", "pose")}
            return self._reply(value, [self.current["path"]], [f"[CURRENT {self.current['frame']}]"])
        body_pose = map_pose(self.environment.pose(), self.transform)
        group = "p_" + uuid.uuid4().hex[:8]
        views = [self.observations.capture(obs, self.environment.steps, group=group, yaw=i * 60)
                 for i, obs in enumerate(self.environment.panorama())]
        self.current = views[0]
        self._publish_state()
        image = self.observations.collage([v["path"] for v in views], f"observations/{group}.jpg",
                    [f"{v['frame']} | right {v['yaw']} deg" for v in views])
        return self._reply({"group": group, "views": [{"frame": v["frame"], "yaw": v["yaw"]} for v in views],
                            "step": self.environment.steps, "pose": body_pose}, [image], [f"[CURRENT {group}]"])

    def _step(self, actions):
        if (not isinstance(actions, list) or not 1 <= len(actions) <= 6
                or any(not isinstance(a, str) or a not in ACTIONS for a in actions)):
            raise ValueError("actions must contain 1–6 valid atomic actions")
        done, events, stop = [], [], "completed"
        for i, action in enumerate(actions, 1):
            if self._cancel.is_set() or (self._action_end is not None and self.environment.steps >= self._action_end):
                self._stop_reason = "aborted" if self._cancel.is_set() else "action_budget"
                stop = "interrupted"
                break
            try:
                motion, observation = self.environment.step(action)
            except BaseException:
                self._broken = True
                self._stop_reason = "execution_uncertain"
                raise
            self.current = self.observations.capture(observation, self.environment.steps)
            if done and done[-1][0] == action:
                done[-1][1] += 1
            else:
                done.append([action, 1])
            if motion.collision or motion.blocked:
                event = {"i": i, "collision": bool(motion.collision)}
                if motion.auto_yaw_delta_degrees:
                    event["auto_yaw"] = round(-motion.auto_yaw_delta_degrees, 2)
                events.append(event)
            self.trace.emit("motion", {"action": action, "samples": motion.samples,
                            "blocked": motion.blocked, "frame": self.current["frame"]})
            if motion.blocked:
                stop = "blocked"
                break
        self._publish_state()
        value = {"done": done, "stop": stop, "frame": self.current["frame"],
                 "step": self.environment.steps, "pose": self.current["pose"]}
        if events:
            value["events"] = events
        if self._action_end is not None and self.environment.steps >= self._action_end:
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
                    source.crop((int(crop[0] * w), int(crop[1] * h), max(int(crop[2] * w), int(crop[0] * w) + 1),
                                 max(int(crop[3] * h), int(crop[1] * h) + 1))).save(destination, quality=85)
                metadata["crop"] = crop
            refs.append(ref)
            labels.append(f"[{'TARGET' if row['kind'] == 'target' else 'HISTORY'} {row['frame']}]")
            public.append(metadata)
        if layout == "grid":
            self._image_counter += 1
            refs = [self.observations.collage(refs, f"observations/derived/g_{self._image_counter:04d}.jpg", labels)]
            labels = ["[HISTORY grid: " + ", ".join(row[0]["frame"] for row in rows) + "]"]
        return self._reply({"frames": public}, refs, labels)

    def _execute(self, name, args):
        if name == "nav_observe":
            return self._observe(**args)
        if name == "nav_step":
            return self._step(**args)
        if name == "view_images":
            return self._images(**args)
        if name == "memory_read":
            result = self.memory.read(**args)
            return self._reply(result, error="error" in result)
        if name == "memory_edit":
            before = self.trace.store(self.memory.entities)
            result = self.memory.edit(**args)
            self._revision += 1
            self.trace.emit("memory_changed", {"before": before, "after": self.trace.store(self.memory.entities)})
            return self._reply(result, error="error" in result)
        if name == "map_view":
            origin, size = self.transform["origin"], self.transform["size_m"]
            trajectory = [[(p[0] - origin[0]) / size, (p[2] - origin[1]) / size]
                          for p in self.environment.trajectory]
            value = self.renderer.render(self.current["pose"], trajectory, **args)
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
        return reply

    def _event(self, value, callback=None):
        self.trace.accept(value)
        if callback:
            callback(value)

    def prompt(self, text, images=(), limits=None, on_event=None):
        limits = limits or PromptLimits()
        if not isinstance(text, str) or not text.strip():
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
            refs = [self.current["path"]]
            captions = [f"Image 1: [CURRENT {self.current['frame']}]"]
            for ref in images:
                if ref in self.observations.items:
                    row = self.observations.get(ref)
                else:
                    row = self.observations.get(self.register_target(ref))
                refs.append(row["path"])
                captions.append(f"Image {len(refs)}: [TARGET {row['frame']}]")
            self._busy = True
        started, prompt_id = time.monotonic(), uuid.uuid4().hex
        before, action_start = charged_requests(self.private), self.environment.steps
        wait_before = wait_seconds(self.private)
        self._submitted, self._answer, self._stop_reason = False, "", None
        self._cancel.clear()
        self._current_request = text
        self._action_end = action_start + limits.max_actions
        model_text = text + "\nCurrent map pose: " + json.dumps(self.current["pose"]) + "\n" + "\n".join(captions)
        settled = {"reason": "runtime_error"}
        reason = "runtime_error"
        error_text = None
        try:
            self.trace.emit("prompt_start", {"prompt_id": prompt_id, "text": text,
                            "images": refs, "limits": asdict(limits)})
            self.client.request("configure", {"maxRequests": before + limits.max_requests})
            self.client.request("set_checkpoint", {"checkpoint": self.checkpoint()})
            for attempt in range(2):
                active_seconds = time.monotonic() - started - (wait_seconds(self.private) - wait_before)
                settled = self.client.run_prompt(model_text, refs, self._tool,
                            max(.01, limits.seconds - active_seconds),
                            on_event=lambda value: self._event(value, on_event),
                            cancel_event=self._cancel, prompt_id=prompt_id if not attempt else prompt_id + "-continue")
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
            if self._submitted:
                self._prompts.append(text)
            elapsed = time.monotonic() - started - (wait_seconds(self.private) - wait_before)
            result = PromptResult(prompt_id, self._submitted, self._answer, reason,
                                  charged_requests(self.private) - before, self.environment.steps - action_start,
                                  f"private/prompts/{prompt_id}.json", max(0, elapsed), self.trace.sanitize(error_text))
            self.trace.emit("prompt_end", result.to_dict())
            _write(self.directory / result.record, result.to_dict())
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
