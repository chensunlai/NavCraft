"""Bounded collection using the production model protocol and navigation tools."""

from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

from src.legacy.harness.agent.base import ModelRequest, ToolCall, ToolResult, ToolSpec
from src.legacy.harness.agent.context import AgentContext
from src.legacy.harness.agent.model import ResponsesModel
from src.legacy.harness.assets import compose_images
from src.legacy.harness.tools.files import ReadTextTool, ViewImagesTool
from src.legacy.harness.tools.navigation import ObserveTool, StepTool, ToolContext

from .geometry import distance, horizontal, local_event
from .schema import ALLOWED_TOOLS, checked_resource, score_answer, write_json


def text_message(text, *, role="user", images=()):
    return {"role": role, "content": [{"type": "input_text", "text": text},
            *[{"type": "input_image", "image_path": str(path)} for path in images]]}


def response_text(items):
    return "\n".join(part.get("text", "") for item in items
                     if item.get("type") != "reasoning"
                     for part in item.get("content", []) if part.get("type") == "output_text")


def parse_answer(items):
    text = response_text(items).strip()
    start, end = text.find("{"), text.rfind("}")
    try:
        value = json.loads(text[start:end + 1])
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


class RequestBudget:
    """One persisted debit per actual HTTP request, including failures; no retries."""

    def __init__(self, config, root, limit=200):
        if config.get("reasoning_effort") != "high":
            raise ValueError("Collection requires reasoning_effort=high")
        self.root, self.limit = Path(root), min(limit, 200)
        self.ledger = self.root / "requests.jsonl"
        self.count = sum(1 for line in self.ledger.read_text().splitlines() if line) if self.ledger.exists() else 0
        self.model = ResponsesModel({**config, "max_retries": 0})

    def complete(self, items, tools, purpose):
        if self.count >= self.limit:
            raise RuntimeError("Global 200-request budget exhausted")
        self.count += 1
        request_id = f"request-{self.count:04d}"
        record = {"id": request_id, "purpose": purpose, "time": time.time(),
                  "options": self.model.request_options(), "timeout_seconds": 300}
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        directory = self.root / "private" / "requests"
        write_json(directory / f"{request_id}-input.json", {**record, "items": items, "tools": tools})
        try:
            turn = self.model.complete(ModelRequest(request_id, items, tools, 300))
        except Exception as exc:
            write_json(directory / f"{request_id}-error.json", {"error": str(exc)})
            raise
        write_json(directory / f"{request_id}-output.json", asdict(turn))
        return turn


class AtomicWorld:
    """One renderer, reusable across seeds; simulation details remain private."""

    def __init__(self, scene_id, config):
        from src.dataset.episodes.base import EpisodeSpec
        from src.environment.simulator import HabitatSimulator
        episode = EpisodeSpec("atomic", "atomic", "atomic", scene_id,
                              [0, 0, 0], [0, 0, 0, 1], [], "rule_sampler")
        self.sim = HabitatSimulator(episode, config)
        self.counter = 0

    def reset(self, pose):
        import habitat_sim
        from habitat_sim.utils.common import quat_from_coeffs
        state = habitat_sim.AgentState()
        state.position = pose["position"]
        state.rotation = quat_from_coeffs(pose["rotation_xyzw"])
        self.sim.agent.set_state(state)

    def capture(self, directory, mode="single"):
        if mode == "single":
            self.counter += 1
            return self.sim.capture(directory, self.counter).rgb
        ids = list(range(self.counter + 1, self.counter + 7))
        self.counter += 6
        views = self.sim.panorama(directory, ids)
        return compose_images([(f"view {i}: right {i * 60} deg", view.rgb)
                               for i, view in enumerate(views)],
                              directory / "observations" / f"panorama_{ids[0]:06d}.jpg")

    def semantic(self):
        """Optional private instance render at exactly the RGB camera pose."""
        import habitat_sim
        import magnum as mn
        import numpy as np
        if "atomic_semantic" not in self.sim.sim._sensors:
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid = "atomic_semantic"
            spec.sensor_type = habitat_sim.SensorType.SEMANTIC
            spec.resolution = [self.sim.config.height, self.sim.config.width]
            spec.hfov = self.sim.config.hfov
            spec.position = mn.Vector3(0, self.sim.config.sensor_height, 0)
            self.sim.sim.add_sensor(spec)
        state = deepcopy(self.sim.agent.get_state())
        state.sensor_states["atomic_semantic"] = deepcopy(state.sensor_states["rgb"])
        self.sim.agent.set_state(state, reset_sensors=False, infer_sensor_states=False)
        sensor = self.sim.sim._sensors["atomic_semantic"]
        sensor.draw_observation()
        return np.asarray(sensor.get_observation()).copy()

    def close(self):
        self.sim.close()


class AtomicTools:
    """Form gating plus genuine executions of existing tool argument validators."""

    def __init__(self, task, sample_root, attempt_root, world):
        self.task, self.root, self.directory, self.world = task, sample_root, attempt_root, world
        self.bank = dict(task.resources)
        self.generated = {}
        self.history = [{"pose": world.sim.pose(), "action": None}]
        observe = ObserveTool()
        observe.visual = True
        available = {"read_text": ReadTextTool(), "view_images": ViewImagesTool(),
                     "nav_observe": observe, "nav_step": StepTool()}
        self.tools = {name: available[name] for name in ALLOWED_TOOLS[task.form]}
        self.context = ToolContext(self.observe, None, self.step, None,
                                  ("move_forward", "turn_left", "turn_right"),
                                  self.read_text, self.view_images, None)
        self.call_cache = {}
        self.observation_checks = []

    def specs(self):
        specs = [tool.spec().to_dict() for tool in self.tools.values()]
        for spec in specs:
            if spec["name"] == "read_text":
                spec["description"] = "Read a text ref listed in this sample's fixed resource bank; no arbitrary paths."
            if spec["name"] == "nav_step":
                spec["description"] = "Execute 1-6 turn_left/turn_right/move_forward actions, 30 degrees / 0.25m each. Returns latest RGB and collision. Local event ends IA automatically; no STOP tool."
        return specs

    def resolve(self, ref):
        if ref in self.generated:
            return self.generated[ref]
        return checked_resource(ref, self.bank, self.root)

    def read_text(self, ref, start_line=1, max_lines=200):
        path = self.resolve(ref)
        if path.suffix not in {".md", ".txt"}:
            raise ValueError("Not a text resource")
        lines = path.read_text().splitlines()
        return ToolResult("", True, {"ref": ref, "start_line": start_line,
                                     "text": "\n".join(lines[start_line - 1:start_line - 1 + max_lines])})

    def view_images(self, refs, layout="separate", crops=None):
        from PIL import Image
        paths = [self.resolve(ref) for ref in refs]
        if any(path.suffix.lower() not in {".png", ".jpg", ".jpeg"} for path in paths):
            raise ValueError("Not an image resource")
        if crops is not None:
            if len(crops) != len(refs):
                raise ValueError("One crop per ref required")
            for i, crop in enumerate(crops):
                if crop is None:
                    continue
                if len(crop) != 4 or not all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1 for x in crop) or crop[0] >= crop[2] or crop[1] >= crop[3]:
                    raise ValueError("Invalid normalized crop")
                with Image.open(paths[i]) as source:
                    box = [round(crop[j] * (source.width if j % 2 == 0 else source.height)) for j in range(4)]
                    dest = self.directory / "assets" / f"crop_{len(self.generated)}.png"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    source.crop(box).save(dest)
                paths[i] = dest
                self.generated[f"crop_{len(self.generated)}"] = dest
        if layout == "grid":
            paths = [compose_images(list(zip(refs, paths)), self.directory / "assets" / f"grid_{len(self.generated)}.jpg")]
        return ToolResult("", True, {"refs": refs, "layout": layout}, paths)

    def observe(self, mode="single"):
        if self.task.private.get("annotate_observation"):
            path = self.annotated_observation(mode)
        else:
            path = self.world.capture(self.directory, mode)
        ref = path.stem
        self.generated[ref] = path
        return ToolResult("", True, {"frame_ref": ref, "mode": mode,
                                     "observation_condition_change": "new capture; panorama is six current-position views" if mode == "panorama" else "new current RGB"}, [path])

    def annotated_observation(self, mode):
        from .build import draw_labels, semantic_boxes
        import quaternion
        import numpy as np
        original = deepcopy(self.world.sim.agent.get_state())
        forward = quaternion.as_rotation_matrix(original.rotation) @ np.array([0, 0, -1])
        yaw = math.atan2(-float(forward[0]), -float(forward[2]))
        entries = []
        try:
            for index in range(6 if mode == "panorama" else 1):
                state = deepcopy(original)
                for sensor in state.sensor_states.values():
                    sensor.rotation = quaternion.from_rotation_vector([0, yaw - index * math.pi / 3, 0])
                self.world.sim.agent.set_state(state, reset_sensors=False, infer_sensor_states=False)
                raw = self.world.capture(self.directory)
                mask = self.world.semantic()
                boxes = semantic_boxes(mask, self.task.private["observation_labels"])
                private = self.directory / "private/visibility"
                private.mkdir(parents=True, exist_ok=True)
                np.save(private / f"{raw.stem}.npy", mask)
                self.observation_checks.append({"raw_image": str(raw), "pose": self.world.sim.pose(),
                                                "visible_candidates": [name for name, _ in boxes],
                                                "target_pixels": int((mask == self.task.private["reference_instance"]).sum())})
                annotated = draw_labels(raw, self.directory / "assets" / f"candidates_{raw.stem}.png", boxes)
                entries.append((f"view {index}: right {index * 60} deg; candidate regions", annotated))
        finally:
            self.world.sim.agent.set_state(original, reset_sensors=False, infer_sensor_states=False)
        if mode == "single":
            return entries[0][1]
        return compose_images(entries, self.directory / "assets" / f"candidates_pano_{self.world.counter}.jpg")

    def step(self, actions):
        executed = []
        for action in actions:
            travelled = sum(distance(a["pose"]["position"], b["pose"]["position"])
                            for a, b in zip(self.history, self.history[1:]))
            if len(self.history) - 1 >= 24 or (action == "move_forward" and travelled + 0.25 > 4.0001):
                break
            collision = self.world.sim.step(action)
            path = self.world.capture(self.directory)
            self.history.append({"pose": self.world.sim.pose(), "action": action,
                                 "collision": collision, "image_path": str(path)})
            executed.append({"action": action, "collision": collision, "frame_ref": path.stem})
            self.generated[path.stem] = path
            if self.task.form == "interactive_ia" and local_event(self.task.private, self.history)["passed"]:
                break
            if collision:
                break
        if not executed:
            raise ValueError("Atomic motion budget exhausted")
        event = self.task.form == "interactive_ia" and local_event(self.task.private, self.history)["passed"]
        return ToolResult("", True, {"executed_actions": executed, "local_event_complete": bool(event),
                                     "actions_used": len(self.history) - 1}, [path], terminated=bool(event))

    def call(self, call):
        if call.call_id in self.call_cache:
            return deepcopy(self.call_cache[call.call_id])
        try:
            if call.name not in self.tools:
                raise ValueError("Tool is not allowed for this task form")
            arguments = json.loads(call.arguments) if isinstance(call.arguments, str) else call.arguments
            result = self.tools[call.name].execute(arguments, self.context)
            result.call_id = call.call_id
        except (ValueError, OSError, TypeError) as error:
            result = ToolResult(call.call_id, False, {"error": str(error)})
        self.call_cache[call.call_id] = deepcopy(result)
        return result


def initial_context(task, root):
    system = ("You solve SHORT grounded spatial tasks. Use only supplied visual evidence and approved tools. "
              "The map is an authored schematic: lines are walls, labels identify objects/rooms. "
              "Map top is +(-Z), right is +X. Panorama cells show clockwise/right offsets from body heading. "
              "Each turn is 30 degrees; forward is 0.25m; no automatic path follower. "
              "For QA, answer exactly a JSON object with choice, anchor, and a short evidence-based explanation. "
              "For IA, execute the spatial instruction until the tool reports local_event_complete; do not merely propose actions. "
              "No tool result gives your current ground-truth pose. Resource refs are sample-local.")
    resource_list = "\n".join(f"- {ref}" for ref in task.resources)
    prompt = f"{task.question}\n\nAnswer anchor: {task.anchor}\nPose condition: {task.pose_condition}\nResource bank (fixed at start):\n{resource_list}\nInitial images in order: {task.initial_images}"
    return [text_message(system, role="system"), text_message(prompt, images=[root / p for p in task.initial_images])]


def collect_attempt(task, root, world, budget=None, *, suffix="teacher", assisted=None):
    """Assisted turns are explicitly curator-authored; all tools still execute live."""
    directory = root / "attempts" / suffix
    directory.mkdir(parents=True, exist_ok=True)
    world.reset(task.private["start_pose"])
    runtime = AtomicTools(task, root, directory, world)
    context = AgentContext(initial_context(task, root))
    decisions, outputs, calls = [], [], []
    error = None
    limit = {"simple_qa": 1, "retrieval_qa": 4}.get(task.form, 12)
    try:
        for index in range(limit):
            before = context.snapshot()
            if assisted is None:
                turn = budget.complete(before, runtime.specs(), f"{task.id}/{suffix}/actor/{index}")
                items, tool_calls = turn.items, turn.calls
            else:
                items = assisted[index] if index < len(assisted) else []
                tool_calls = [ToolCall(i["call_id"], i["name"], i["arguments"]) for i in items if i.get("type") == "function_call"]
            decisions.append({"index": index, "input": before, "tools": runtime.specs(), "output": items})
            outputs = items
            context.append_output(items)
            finished = False
            for call in tool_calls:
                result = (ToolResult(call.call_id, False, {"skipped": "local event already complete"})
                          if finished else runtime.call(call))
                calls.append({"call": asdict(call), "result": {**asdict(result), "images": [str(p) for p in result.images]}})
                context.append_tool_result(result)
                if result.terminated:
                    finished = True
            if finished or not tool_calls:
                break
    except Exception as exc:
        error = str(exc)
    if task.form == "interactive_ia":
        validation = local_event(task.private, runtime.history)
    else:
        validation = score_answer(parse_answer(outputs), task.private)
    if task.form == "retrieval_qa":
        validation["real_query"] = any(c["call"]["name"] in {"read_text", "view_images"} and c["result"]["ok"] for c in calls)
        validation["passed"] &= validation["real_query"]
    if task.form == "interactive_qa":
        validation["real_interaction"] = any(c["call"]["name"] in {"nav_observe", "nav_step"} and c["result"]["ok"] for c in calls)
        validation["passed"] &= validation["real_interaction"]
        if task.private.get("require_motion"):
            validation["motion_present"] = len(runtime.history) > 1
            validation["passed"] &= validation["motion_present"]
        if task.private.get("require_translation"):
            validation["translation_present"] = any(distance(row["pose"]["position"], runtime.history[0]["pose"]["position"]) > .10 for row in runtime.history)
            validation["passed"] &= validation["translation_present"]
        if task.private.get("annotate_observation"):
            validation["target_visible_in_new_observation"] = any(row["target_pixels"] >= 300 for row in runtime.observation_checks)
            validation["passed"] &= validation["target_visible_in_new_observation"]
    if error:
        validation["passed"] = False
    record = {"sample_id": task.id, "attempt": suffix, "source": "agent_assisted" if assisted is not None else "deepseek_demonstration",
              "task_snapshot": task.public(),
              "decisions": decisions, "messages": context.snapshot(), "tools": runtime.specs(),
              "calls": calls, "validation": validation, "error": error}
    write_json(directory / "record.json", record)
    write_json(directory / "private" / "motion.json", runtime.history)
    write_json(directory / "private" / "observations.json", runtime.observation_checks)
    return record
