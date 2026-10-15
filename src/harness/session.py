"""Python-owned Habitat session. Only explicit RGB and public feedback enter tool results.

Initialize, observe, move and close in the same episode worker thread.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.dataset.episodes.base import EpisodeSpec
from src.environment.base import NavigationInputError, TRANSLATION_ACTIONS
from src.environment.tasks.base import TaskAdapter, TaskUpdate


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


class AtomicTaskAdapter(TaskAdapter):
    """Use existing local-event geometry without exposing labels to the agent."""

    family = "atomic"

    def __init__(self, labels, public, config, mapping=False):
        super().__init__(config)
        self.labels, self.public, self.mapping = labels, public, mapping

    @property
    def movement_actions(self):
        return TRANSLATION_ACTIONS + ("turn_left", "turn_right")

    def event(self):
        from src.dataset.atomic.geometry import local_event

        return local_event(self.labels, [{"pose": {"position": p}} for p in self.trajectory])

    def on_step(self, point, distance):
        self._count(point, distance)
        if not self.mapping and self.event()["passed"]:
            self.terminated, self.reason = True, "local_event_complete"
        elif self.steps >= self.max_steps:
            self.truncated, self.reason = True, "max_steps"
        elif self.path_length >= 4.0001:
            self.truncated, self.reason = True, "travel_budget"
        return self.update()

    def finish(self, point, distance):
        raise NavigationInputError("This atomic task ends on its local event; nav_finish is unavailable")

    def finalize(self, reason):
        if not self.done:
            self.truncated, self.reason = True, reason

    def metrics(self):
        return {"steps": self.steps, "path_length": self.path_length,
                "reason": self.reason, "local_event": self.event(), "mapping": self.mapping}

    def private_state(self):
        return {"metrics": self.metrics(), "trajectory": self.trajectory,
                "terminated": self.terminated, "truncated": self.truncated}


class NavigationSession:
    def __init__(self, cancel_event=None):
        self.env = None
        self.cache = {}
        self.closed = False
        self.workspace = self.private_dir = None
        self.published = {}
        self.config = {}
        self.mode = "navigation"
        self.cancel_event = cancel_event
        # Private lifecycle guard: no further prompt may trust state after an
        # action ran but its public/private receipt could not be completed.
        self.execution_uncertain = False

    def dispatch(self, request):
        """Idempotent requests: even failed calls are retained; changed retries fail."""
        request_id = request.get("id") if isinstance(request, dict) else None
        if not isinstance(request_id, str) or not request_id:
            return {"id": request_id, "ok": False, "error": "A nonempty string request id is required"}
        fingerprint = json.dumps(request, sort_keys=True)
        if request_id in self.cache:
            prior, response = self.cache[request_id]
            if prior != fingerprint:
                return {"id": request_id, "ok": False, "error": "Request id reused with different arguments"}
            return deepcopy(response)
        try:
            op, args = request.get("op"), request.get("args", {})
            if not isinstance(args, dict):
                raise ValueError("args must be an object")
            operations = {"init": self.init, "observe": self.observe, "step": self.step,
                          "finish": self.finish, "finalize": self.finalize, "close": self.close}
            if op not in operations:
                raise ValueError(f"Unknown operation: {op}")
            if self.closed and op != "close":
                raise RuntimeError("Navigation session is closed")
            data = operations[op](**args)
            response = {"id": request_id, "ok": True, "data": data}
        except Exception as error:
            response = {"id": request_id, "ok": False,
                        "error": f"{type(error).__name__}: {error}"}
        self.cache[request_id] = fingerprint, deepcopy(response)
        return response

    def _load(self, value):
        import yaml

        if isinstance(value, dict):
            return deepcopy(value)
        path = Path(value)
        if not path.is_absolute():
            path = self.repo / path
        return yaml.safe_load(path.read_text())

    def init(self, *, repo, workspace, private_dir, env_config, task_config=None,
             episode_id=None, atomic_sample=None, goal_limit=None, scene_archive=None,
             purpose="navigation", max_actions=None, scene_id=None, instruction_override=None):
        if self.env is not None:
            raise RuntimeError("One bridge owns one initialized episode")
        if purpose not in {"navigation", "mapping"}:
            raise ValueError("purpose must be navigation or mapping")
        if instruction_override is not None:
            if purpose != "mapping" or not atomic_sample:
                raise ValueError("instruction_override is only allowed for mapping with an atomic seed")
            if not isinstance(instruction_override, str) or not instruction_override.strip():
                raise ValueError("instruction_override must be a nonempty string")
        if max_actions is not None and (type(max_actions) is not int or max_actions < 1):
            raise ValueError("max_actions must be a positive integer")
        if not atomic_sample and (not isinstance(episode_id, str) or not episode_id):
            raise ValueError("A canonical episode_id is required for navigation")
        self.repo = Path(repo).resolve()
        self.workspace, self.private_dir = Path(workspace).resolve(), Path(private_dir).resolve()
        if (self.private_dir.is_relative_to(self.workspace)
                or self.workspace.is_relative_to(self.private_dir)):
            raise ValueError("Public workspace and private directory must be separate siblings")
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.private_dir.mkdir(parents=True, exist_ok=True)
        config = self._load(env_config)
        self.config = config.get("environment", config)
        if self.config.get("scenes"):
            self.config["scenes"] = str((self.repo / self.config["scenes"]).resolve())
        self.config["gpu"] = int(self.config.get("gpu", self.config.get("gpus", [2])[0]))
        self.config.pop("gpus", None)
        self.config["depth"] = False
        if self.config.get("egl_vendor_file"):
            os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = self.config["egl_vendor_file"]
        self.mode = "mapping" if purpose == "mapping" else "navigation"
        if atomic_sample:
            sample = (self.repo / atomic_sample).resolve()
            public = json.loads((sample / "task.json").read_text())
            labels = json.loads((sample / "private/labels.json").read_text())
            if public["form"] != "interactive_ia" and self.mode != "mapping":
                raise ValueError("Environment atomic mode requires an interactive_ia sample")
            if scene_id is None:
                if self.config.get("name") == "fake":
                    scene_id = "fake.glb"
                elif scene_archive:
                    manifest = json.loads(((self.repo / scene_archive) / "raw/manifest.json").read_text())
                    scene_id = manifest["scene_dataset_asset"]
                else:
                    matches = list(Path(self.config["scenes"]).glob(
                        f"hm3d/*/*-{public['scene']}/{public['scene']}.basis.glb"))
                    if len(matches) != 1:
                        raise ValueError("Set scene_id or scene_archive for this atomic scene")
                    scene_id = str(matches[0])
            pose, dest = labels["start_pose"], labels["destination"]
            instruction = public["question"] if instruction_override is None else instruction_override
            goal = {"modality": "instruction", "instruction": instruction,
                    "positions": [[dest[0], pose["position"][1], -dest[1]]]}
            episode = EpisodeSpec(public["id"], public["id"], "atomic", scene_id,
                                  pose["position"], pose["rotation_xyzw"], [goal], str(sample))
            if instruction_override is not None:
                episode.provenance = "derived_mapping_smoke;atomic_seed_only"
            task = AtomicTaskAdapter(labels, public, {"max_steps": max_actions or 24}, self.mode == "mapping")
            if self.mode != "mapping":
                self.mode = "atomic_ia"
        else:
            from src.dataset.episodes.manifest import ManifestEpisodeSource
            from src.environment.tasks.base import GOATTask, ObjectNavTask, VLNCETask

            document = self._load(task_config)
            source = deepcopy(document["episodes"])
            for key in ("root", "manifest"):
                if key in source:
                    source[key] = str((self.repo / source[key]).resolve())
            episodes = ManifestEpisodeSource(source).load([episode_id])
            if len(episodes) != 1:
                raise ValueError("Exactly one canonical episode_id is required")
            episode = episodes[0]
            if goal_limit is not None:
                if (episode.benchmark != "goat" or isinstance(goal_limit, bool)
                        or not isinstance(goal_limit, int) or not 1 <= goal_limit <= len(episode.goals)):
                    raise ValueError("goal_limit must select a nonempty GOAT prefix")
                episode.goals = episode.goals[:goal_limit]
                episode.provenance = f"{episode.provenance};truncated_smoke_prefix={goal_limit}"
            task_options = deepcopy(document["task"])
            if max_actions is not None:
                task_options["max_steps"] = max_actions
            types = {"objectnav": ObjectNavTask, "vlnce": VLNCETask, "goat": GOATTask}
            task = types[task_options["name"]](task_options)
        _write(self.private_dir / "episode.json", episode.to_dict())
        _write(self.private_dir / "config.json", {"environment": self.config, "mode": self.mode})
        from src.environment.fake import FakeEnvironment
        from src.environment.habitat import HabitatEnvironment

        factory = FakeEnvironment if self.config.get("name") == "fake" else HabitatEnvironment
        env = factory(self.config, task, self.private_dir / "environment")
        try:
            transition = env.reset(episode)
            self.env = env
            self._save_private_frame(transition.observation)
            result = self._transition(transition)
            result["actions"] = {"allowed": list(env.movement_actions),
                                 "forward_metres": self.config.get("forward_step", .25),
                                 "translation_metres": self.config.get("forward_step", .25),
                                 "turn_degrees": self.config.get("turn_angle", 15),
                                 "look_degrees": self.config.get("turn_angle", 15)}
            result["mode"] = self.mode
            result["episode_id"] = episode.id
            if scene_archive:
                from src.common.map import MapCalibration

                calibration = MapCalibration((self.repo / scene_archive).resolve())
                level, position, heading = calibration.initial_arrow(transition.observation.pose)
                dx, dy = heading[0] - position[0], heading[1] - position[1]
                scale = min(1.0, .04 / max(math.hypot(dx, dy), 1e-9))
                while not all(0 <= coordinate <= 1 for coordinate in
                              (position[0] + dx * scale, position[1] + dy * scale)):
                    scale *= .5
                heading = [position[0] + dx * scale, position[1] + dy * scale]
                result["initial_hint"] = {"map_id": f"map_{level}", "position": position,
                                          "heading_point": heading, "at_frame": result["observation"]["id"]}
            self._save_result()
            return result
        except BaseException:
            env.close()
            self.env = None
            raise

    def _running(self):
        if self.env is None:
            raise RuntimeError("Initialize the environment first")
        self.env._require_running()

    def _save_private_frame(self, observation):
        _append(self.private_dir / "frames.jsonl", {"id": observation.frame_id,
                "step": self.env.task.steps, "pose": observation.pose, "camera": observation.camera})

    def _publish(self, observation, *, right=0, pitch=None, panorama_id=None):
        if observation.frame_id in self.published:
            return self.published[observation.frame_id]
        destination = self.workspace / "observations" / f"{observation.frame_id}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ValueError(f"Observation destination already exists: {destination.name}")
        shutil.copyfile(observation.rgb, destination)
        if pitch is None:
            matrix = observation.pose.get("world_from_optical")
            pitch = math.degrees(math.asin(max(-1.0, min(1.0, matrix[1][2])))) if matrix else 0
        record = {"id": observation.frame_id, "path": destination.relative_to(self.workspace).as_posix(),
                  "source": "observation", "step": self.env.task.steps,
                  "view": {"right_degrees": right, "pitch_degrees": round(pitch, 6),
                           "panorama_id": panorama_id}, "derived_from": []}
        _append(self.workspace / "observations/index.jsonl", record)
        self.published[observation.frame_id] = record
        return record

    def _task(self, task):
        if task is None:
            return None
        result = {"goal_id": task.goal_id, "modality": task.modality,
                  "instruction": task.instruction, "image": None}
        if task.image:
            destination = self.workspace / "observations" / f"target_{task.goal_id:03d}.png"
            if not destination.exists():
                shutil.copyfile(task.image, destination)
                _append(self.workspace / "observations/index.jsonl", {
                    "id": f"target_{task.goal_id:03d}", "path": destination.relative_to(self.workspace).as_posix(),
                    "source": "target", "step": None, "view": None, "derived_from": []})
            result["image"] = destination.relative_to(self.workspace).as_posix()
        return result

    def _transition(self, transition):
        observation = self._publish(transition.observation)
        task = self._task(transition.task)
        result = {"task": task, "observation": observation, "images": [observation["path"]],
                  "goal_changed": transition.goal_changed, "terminated": transition.terminated,
                  "truncated": transition.truncated, "reason": transition.reason,
                  "steps": self.env.task.steps,
                  "remaining_total_actions": max(0, self.env.task.max_steps - self.env.task.steps)}
        if task and task["image"] and (transition.goal_changed or self.env.task.steps == 0):
            result["images"].append(task["image"])
        if self.env.task.family == "goat":
            result["remaining_goal_actions"] = max(0, self.env.task.max_subtask_steps - self.env.task.goal_steps)
        return result

    def observe(self, mode="single"):
        self._running()
        if mode not in {"single", "panorama"}:
            raise ValueError("mode must be single or panorama")
        if mode == "single":
            observation = self.env.capture_current()
            self._save_private_frame(observation)
            record = self._publish(observation)
            return {"mode": mode, "observation": record, "views": [record],
                    "steps": self.env.task.steps, "images": [record["path"]]}
        observations = self.env.panorama()
        group = f"panorama_{observations[0].frame_id}"
        records = []
        for index, observation in enumerate(observations):
            self._save_private_frame(observation)
            records.append(self._publish(observation, right=index * 60, pitch=0, panorama_id=group))
        return {"mode": mode, "views": records, "steps": self.env.task.steps,
                "restored_camera": True, "images": [record["path"] for record in records]}

    def step(self, actions):
        self._running()
        if not isinstance(actions, list) or not 1 <= len(actions) <= 6:
            raise ValueError("actions requires 1–6 atomic actions")
        if any(not isinstance(action, str) or action not in self.env.movement_actions for action in actions):
            raise ValueError(f"Allowed actions: {self.env.movement_actions}")
        executed = []
        interruption = None
        for action in actions:
            if self.cancel_event is not None and self.cancel_event.is_set():
                interruption = "cancelled"
                break
            if self.mode in {"atomic_ia", "mapping"} and action in TRANSLATION_ACTIONS:
                if self.env.task.path_length + self.config.get("forward_step", .25) > 4.0001:
                    self.env.finalize("travel_budget")
                    interruption = "travel_budget"
                    break
            before = self.env.observe().frame_id
            self.execution_uncertain = True
            transition = self.env.step(action)
            self._save_private_frame(transition.observation)
            result = self._transition(transition)
            event = {"step": self.env.task.steps, "action": action,
                     "collision": bool(transition.feedback.get("collision", False)),
                     "before_frame": before, "after_frame": result["observation"]["id"],
                     "goal_changed": transition.goal_changed}
            _append(self.workspace / "observations/actions.jsonl", event)
            executed.append(event)
            if transition.terminated or transition.truncated:
                interruption = transition.reason or "episode_finished"
                break
            if transition.goal_changed:
                interruption = "goal_changed"
                break
            if event["collision"]:
                interruption = "collision"
                break
        if not executed:
            from src.environment.base import Transition

            result = self._transition(Transition(
                self.env.observe(), None if self.env.task.done else self.env.public_task, {},
                terminated=self.env.task.terminated, truncated=self.env.task.truncated,
                reason=self.env.task.reason))
        result.update(requested_actions=actions, executed_actions=executed, interruption=interruption,
                      terminated=self.env.task.terminated, truncated=self.env.task.truncated,
                      reason=self.env.task.reason)
        if self.mode == "atomic_ia":
            result["local_event_complete"] = self.env.task.reason == "local_event_complete"
        self._save_result()
        self.execution_uncertain = False
        return result

    def finish(self, goal_id):
        self._running()
        if self.mode != "navigation":
            raise NavigationInputError("nav_finish is only available for full navigation")
        if type(goal_id) is not int or goal_id != self.env.task.goal_index:
            raise NavigationInputError("goal_id does not match the current target")
        before = self.env.observe().frame_id
        self.execution_uncertain = True
        transition = self.env.finish_goal()
        self._save_private_frame(transition.observation)
        result = self._transition(transition)
        _append(self.workspace / "observations/actions.jsonl", {
            "step": self.env.task.steps, "action": "stop", "collision": False,
            "before_frame": before, "after_frame": result["observation"]["id"],
            "goal_changed": transition.goal_changed})
        self._save_result()
        self.execution_uncertain = False
        return result

    def _save_result(self):
        if self.env is not None:
            _write(self.private_dir / "result.json", self.env.task.private_state())

    def finalize(self, reason="agent_stopped"):
        if self.env is None:
            raise RuntimeError("Initialize the environment first")
        self.env.finalize(reason)
        self._save_result()
        return {"terminated": self.env.task.terminated, "truncated": self.env.task.truncated,
                "reason": self.env.task.reason, "steps": self.env.task.steps}

    def close(self):
        if self.closed:
            return {"closed": True}
        self.closed = True
        if self.env is not None:
            try:
                if not self.env.task.done:
                    self.env.finalize("bridge_closed")
                self._save_result()
            finally:
                self.env.close()
        return {"closed": True}
