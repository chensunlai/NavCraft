"""Run one frozen short task; dataset construction never happens during scoring."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import traceback

import numpy as np

from src.common.config import read_config
from src.common.coordinates import uv_to_world
from src.common.io import read_json, redact, write_json
from src.dataset.episodes import EpisodeSpec
from src.environment import Environment
from src.harness.tools import Toolsets
from src.harness.runtime.trace import read_trace
from src.harness import PromptLimits, open_session
from src.harness.runtime.workspace import read_snapshot

from .runner import INFRASTRUCTURE_REASONS


REPO = Path(__file__).resolve().parents[2]


def _category(value):
    value = str(value or "").strip().lower().replace("_", " ")
    return {"couch": "sofa", "television": "tv", "potted plant": "plant"}.get(value, value)


def _footprint_distance(point, instance):
    return math.hypot(*(max(abs(point[i] - instance["center"][i]) - instance["size"][i] / 2, 0)
                        for i in (0, 2)))


def _last_image_selections(session):
    """Only accepted edits count, and later explicit coordinate writes supersede pixels."""
    selected = {}
    for event in read_trace(session.navigation.private / "trace.jsonl"):
        if event["type"] != "public_tool_reply":
            continue
        data = event["data"]
        if data["name"] != "memory_edit" or data["result"].get("isError"):
            continue
        args = data["args"]
        for operation in args["operations"]:
            identifier = operation.get("id", operation.get("keep"))
            fields = operation.get("fields", {})
            if any(key in fields for key in ("point", "anchor", "location")):
                selected[identifier] = fields.get("location")
            if operation["op"] == "delete":
                selected.pop(identifier, None)
            for removed in operation.get("remove", []):
                selected.pop(removed, None)
    return selected


def score_memory(session, label, case):
    """Match public records against private instance geometry and observed pixels.

    Free-form descriptions are retained for review, not claimed semantically verified.
    New observations without frozen semantic masks require review rather than receiving
    invented instance labels or being automatically called hallucinations.
    """
    masks = {frame: np.load(Path(case) / path) for frame, path in label["evidence_masks"].items()}
    instances = {item["id"]: item for item in label["instances"]}
    target = label["target"]["id"]
    selections = _last_image_selections(session)
    baseline = {}
    if label.get("baseline_bindings"):
        task = read_json(Path(case) / "task.json")
        baseline = {entity["id"]: entity for entity in read_json(
            Path(case) / task["snapshot"] / "workspace/memory/entities.json")}
    bindings, failures, review = {}, [], []
    rows = []
    for entity in session.navigation.memory.entities:
        if entity["type"] != "object":
            if entity["id"] not in label.get("baseline_bindings", {}):
                review.append(entity["id"])
            continue
        identifier, point = entity["id"], entity.get("point")
        category = _category(entity.get("category") or entity.get("name"))
        candidates = []
        if point is not None:
            world = uv_to_world(point, session.navigation.transform)
            candidates = [instance for instance in instances.values()
                          if _category(instance["category"]) == category
                          and _footprint_distance(world, instance) <= label.get("position_tolerance_m", .3)]
        pixel_instance = None
        selection = selections.get(identifier)
        unknown_pixel = False
        if isinstance(selection, dict) and "image" in selection:
            frame, pixel = session.navigation.observations.resolve_pixel(selection["image"], selection["pixel"])
            if frame.startswith("f_"):
                if frame in masks:
                    mask = masks[frame]
                    x, y = min(round(pixel[0]), mask.shape[1] - 1), min(round(pixel[1]), mask.shape[0] - 1)
                    pixel_instance = int(mask[y, x])
                    candidates = [item for item in candidates if item["semantic_id"] == pixel_instance]
                else:
                    unknown_pixel = True
        evidence = entity.get("evidence", [])
        supported = [item for item in candidates if any(
            frame in masks and np.any(masks[frame] == item["semantic_id"]) for frame in evidence)]
        # Height-separated objects can share a footprint. Preserve a verified
        # initial binding only while its location/category and old evidence survive.
        initial = baseline.get(identifier)
        bound = label.get("baseline_bindings", {}).get(identifier)
        inherited = False
        if (len(supported) > 1 and initial and not unknown_pixel
                and point == initial.get("point")
                and category == _category(initial.get("category") or initial.get("name"))):
            retained = set(evidence) & set(initial.get("evidence", []))
            matched = [item for item in supported if item["id"] == bound and any(
                frame in masks and np.any(masks[frame] == item["semantic_id"]) for frame in retained)]
            if matched:
                supported, inherited = matched, True
        # An unannotated new image is not evidence of a wrong instance.
        if pixel_instance is not None and not candidates:
            failures.append(identifier)
        elif not supported and any(frame not in masks for frame in evidence):
            review.append(identifier)
        elif len(supported) == 1:
            bindings[identifier] = supported[0]["id"]
            if unknown_pixel:
                review.append(identifier)
        elif len(supported) > 1:
            review.append(identifier)
        else:
            failures.append(identifier)
        rows.append({"id": identifier, "candidates": [item["id"] for item in supported],
                     "selected_semantic_id": pixel_instance,
                     "preserved_initial_binding": inherited,
                     "description_present": bool(entity.get("description", "").strip())})
    target_ids = [identifier for identifier, instance in bindings.items() if instance == target]
    lost = [identifier for identifier, instance in label.get("baseline_bindings", {}).items()
            if bindings.get(identifier) != instance and identifier not in review]
    duplicates = {instance: [identifier for identifier, match in bindings.items() if match == instance]
                  for instance in set(bindings.values())
                  if sum(match == instance for match in bindings.values()) > 1}
    expected = label.get("expected_existing_id")
    wrong_existing = bool(expected and expected not in target_ids and expected not in review)
    hard_failure = bool(failures or lost or duplicates or wrong_existing)
    if not target_ids and not review:
        hard_failure = True
    status = "incorrect" if hard_failure else "needs_review" if review else "correct"
    return {"success": False if hard_failure else None if review else len(target_ids) == 1,
            "status": status, "target_ids": target_ids, "bindings": bindings,
            "duplicates": duplicates, "unmatched_records": failures,
            "missing_or_rebound_baseline": lost, "review_records": review, "records": rows,
            "description_check": "Presence reported; free-form semantic accuracy requires visual review."}


def score_motion(environment, label, *, submitted, start_steps, start_length, limits):
    distance = environment.distance(label["positions"])
    travelled = environment.path_length - start_length
    actions = environment.steps - start_steps
    within_budget = (actions <= limits.max_actions and
                     (limits.max_path_length is None or travelled <= limits.max_path_length + 1e-6))
    success = (submitted and math.isfinite(distance) and distance < label["success_distance"]
               and travelled >= label.get("minimum_motion_m", .1) and within_budget)
    return {"success": bool(success), "status": "correct" if success else "incorrect",
            "distance": distance, "success_distance": label["success_distance"],
            "submitted": bool(submitted), "actions": actions, "path_length": travelled,
            "within_budget": within_budget}


def run_task(repo, task_path, model_config, output, gpu=0):
    selection = Toolsets.resolve(model_config["toolsets"])
    task_path, output = Path(task_path).resolve(), Path(output).resolve()
    if task_path.is_dir():
        task_path /= "task.json"
    case, task = task_path.parent, read_json(task_path)
    if task["kind"] not in {"ground", "recognize", "revisit", "recover"}:
        raise ValueError("Unknown frozen short-task kind")
    if type(gpu) is not int or gpu not in (0, 1):
        raise ValueError("Short tasks require GPU 0 or 1")
    output.mkdir(parents=True, exist_ok=False)
    label = read_json(case / task["label"])
    if task["kind"] == "recover" and label.get("initial_blocked") is not True:
        raise ValueError("Recovery case lacks a verified blocked initial state")
    episode = EpisodeSpec(**read_json(case / task["episode"]))
    state = read_snapshot(case / task["snapshot"])
    limits = PromptLimits(**task["limits"])
    result = {"episode_id": task["id"], "benchmark": "short_tasks", "kind": task["kind"],
              "model": model_config["model"], "reasoning_effort": model_config["reasoning_effort"],
              "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "task": str(task_path), "gpu": gpu}
    write_json(output / "result.json", result)
    write_json(output / "private/task.json", task)
    environment = session = None
    start_steps, start_length = 0, 0
    try:
        environment = Environment(episode, {"gpu": gpu, "scenes": str(Path(repo) / "data/nas/scene_datasets"),
                                            "depth": selection.geometry},
                                  output / "private/environment")
        session = open_session(repo, model_config, output, environment,
                               Path(state["workspace"]) / "scene/map.json", initial_state=state)
        start_steps, start_length = environment.steps, environment.path_length
        images = [image if image in session.navigation.observations.items else str(case / image)
                  for image in task.get("images", [])]
        prompt = session.prompt(task["instruction"], images=images, limits=limits)
        result.update(prompt=prompt.to_dict(), finish_reason=prompt.reason)
        if prompt.reason in INFRASTRUCTURE_REASONS:
            result["status"] = "infrastructure_error"
        else:
            result["status"] = "completed"
            if task["kind"] in {"ground", "recognize"}:
                score = score_memory(session, label, case)
                within_budget = (environment.steps - start_steps <= limits.max_actions and
                                 (limits.max_path_length is None or
                                  environment.path_length - start_length <= limits.max_path_length + 1e-6))
                score.update(submitted=prompt.submitted, within_budget=within_budget)
                if not prompt.submitted or not within_budget:
                    score.update(success=False, status="incorrect")
            else:
                score = score_motion(environment, label, submitted=prompt.submitted,
                                     start_steps=start_steps, start_length=start_length, limits=limits)
            result["score"] = score
    except Exception as error:
        result.update(status="infrastructure_error", finish_reason="worker_error",
                      error=redact(str(error)), error_type=type(error).__name__)
        (output / "private/error.txt").write_text(traceback.format_exc())
    finally:
        if environment is not None:
            result["environment"] = {"steps": environment.steps - start_steps,
                                     "path_length": environment.path_length - start_length,
                                     "trajectory": environment.motion_trajectory}
        for resource in (session, environment):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    result.update(status="infrastructure_error")
                    result.setdefault("cleanup_errors", []).append(redact(str(error)))
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, action="append", type=Path)
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=0)
    args = parser.parse_args(argv)
    result = run_task(REPO, args.task, read_config(args.config), args.output, args.gpu)
    print(json.dumps({"episode_id": result["episode_id"], "status": result["status"],
                      "score": result.get("score")}, ensure_ascii=False))
    return 1 if result["status"] == "infrastructure_error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
