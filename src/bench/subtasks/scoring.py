"""Private scores for frozen tasks; no labels or scores are tool feedback."""

from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image


DIRECTIONS = ("front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left")
ALIASES = {"couch": "sofa", "settee": "sofa", "armchair": "chair", "seat": "chair",
           "television": "tv", "tv monitor": "tv", "potted plant": "plant",
           "houseplant": "plant", "chest of drawers": "dresser"}


def direction_label(degrees):
    """Convert a right-positive relative bearing to the nearest of eight sectors."""
    if not math.isfinite(degrees):
        raise ValueError("Direction must be finite")
    return DIRECTIONS[int(math.floor((degrees % 360 + 22.5) / 45)) % 8]


def _category(value):
    word = re.sub(r"[\s_-]+", " ", str(value or "").strip().lower())
    return ALIASES.get(word, word)


def _answer(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Missing answer")
    text = text.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, flags=re.DOTALL)
        if not match:
            raise ValueError("Invalid answer fence")
        text = match[1].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _scalar(value, key):
    if isinstance(value, dict):
        if set(value) != {key}:
            raise ValueError(f"Expected only {key}")
        return value[key]
    return value


def _pixel_answer(value, frame):
    actual_frame = frame
    if isinstance(value, dict):
        if not {"pixel"} <= set(value) <= {"frame", "pixel"}:
            raise ValueError("Expected pixel and optional frame")
        actual_frame = value.get("frame", frame)
        point = value["pixel"]
    elif isinstance(value, str):
        match = re.fullmatch(r"(f_\d{4,})\s*[:：]?\s*(\[[^\]]+\])", value)
        if not match:
            raise ValueError("Expected [x,y] or frame plus [x,y]")
        actual_frame, point = match[1], json.loads(match[2])
    else:
        point = value
    if (not isinstance(point, list) or len(point) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in point)):
        raise ValueError("Pixel must contain two finite numbers")
    return actual_frame, point


def match_entities(entities, instances, transform, frame_visibility=None, *, tolerance=.3,
                   location_sources=None, baseline_bindings=None):
    """Bind object records only when geometry, category and evidence agree uniquely.

    Multiple records can match one instance: that is a duplicate, not a matching
    algorithm's reason to assign one of them to a different physical object.
    Unmatched records remain unverified; incomplete semantic labels do not prove
    that an unmatched record is a hallucination.
    """
    origin, scale = transform["origin"], transform["size_m"]
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid map scale")
    visibility = {key: {str(i) for i in value} for key, value in (frame_visibility or {}).items()}
    baseline = {key: str(value) for key, value in (baseline_bindings or {}).items()}
    bindings, ambiguous, unmatched = {}, [], []
    for entity in entities:
        if entity.get("type") != "object":
            continue
        identifier, point = entity["id"], entity.get("point")
        if (not isinstance(point, list) or len(point) != 2 or any(
                type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in point)):
            unmatched.append(identifier)
            continue
        position = [origin[i] + point[i] * scale for i in (0, 1)]
        evidence = entity.get("evidence", [])
        visible = set().union(*(visibility.get(frame, set()) for frame in evidence)) if evidence else set()
        source = (location_sources or {}).get(identifier, {})
        source_uv, source_height = source.get("uv"), source.get("height")
        height = None
        if (isinstance(source_uv, list) and len(source_uv) == 2
                and all(type(v) in (int, float) and math.isfinite(v) for v in source_uv)
                and math.dist(source_uv, point) <= 1e-5
                and source.get("frame") in evidence
                and type(source_height) in (int, float) and math.isfinite(source_height)):
            height = transform.get("floor_height", 0.) + source_height
        category = _category(entity.get("category") or entity.get("name"))
        candidates = []
        for instance in instances:
            instance_id = str(instance["id"])
            if category != _category(instance["category"]):
                continue
            if frame_visibility is not None and instance_id not in visible:
                continue
            center, size = instance["center"], instance["size"]
            if height is not None and not center[1] - size[1] / 2 - .15 <= height <= center[1] + size[1] / 2 + .15:
                continue
            dx = max(0., abs(position[0] - center[0]) - size[0] / 2)
            dz = max(0., abs(position[1] - center[2]) - size[2] / 2)
            if math.hypot(dx, dz) <= tolerance:
                candidates.append(instance_id)
        if len(candidates) == 1:
            bindings[identifier] = candidates[0]
        elif baseline.get(identifier) in candidates:
            # Identity was known before the task. Reuse it only while current
            # geometry, category, evidence and any valid surface height agree.
            bindings[identifier] = baseline[identifier]
        elif candidates:
            ambiguous.append({"id": identifier, "candidates": sorted(candidates)})
        else:
            unmatched.append(identifier)
    by_instance = {}
    for identifier, instance in bindings.items():
        by_instance.setdefault(instance, []).append(identifier)
    return {"bindings": bindings, "ambiguous": ambiguous, "unmatched": unmatched,
            "duplicates": {key: value for key, value in by_instance.items() if len(value) > 1}}


def _result(success, reason=None, **metrics):
    return {"success": bool(success), "reason": reason or ("correct" if success else "incorrect"),
            "metrics": metrics}


def score_sample(sample, label, outcome):
    """Score one frozen sample from its answer or actual final environment state.

    Mask paths must already be resolved by the trusted loader. For 14b the
    environment supplies geodesic distance to the frozen verified approach set;
    this function never substitutes an endpoint's planar distance to an object.
    """
    sample = asdict(sample) if is_dataclass(sample) else sample
    task = sample["task_id"]
    if outcome.get("reason") in {"runtime_error", "execution_uncertain", "model_error", "transport_error"}:
        return _result(False, "execution_error", execution_reason=outcome["reason"])
    if task in {"1b", "1n"}:
        return _score_memory(task, label, outcome)
    if task == "14b":
        return _score_approach(sample, label, outcome)
    if task not in {"5a", "5c", "6a", "6c", "13d"}:
        raise ValueError(f"Unsupported task: {task}")
    if not outcome.get("submitted"):
        return _result(False, "not_submitted")
    if outcome.get("actions", 0):
        return _result(False, "qa_state_changed")
    try:
        answer = _answer(outcome.get("answer"))
        if task in {"5a", "5c"}:
            if label.get("answer_kind", "pixel") == "none":
                return _result(answer is None or isinstance(answer, str) and answer.lower().rstrip(".") == "none")
            frame, point = _pixel_answer(answer, label["frame"])
            if "mask" in label:
                mask = np.asarray(label["mask"])
            else:
                path = Path(label["mask_file"])
                mask = np.load(path, allow_pickle=False) if path.suffix == ".npy" else np.asarray(Image.open(path))
            if mask.ndim != 2:
                raise ValueError("Instance mask must have two dimensions")
            x, y = point
            if not 0 <= x < mask.shape[1] or not 0 <= y < mask.shape[0]:
                return _result(False, "pixel_out_of_bounds")
            ix, iy = int(x), int(y)
            pixel_matches = bool(mask[iy, ix] == label["instance_id"]) if "instance_id" in label else bool(mask[iy, ix])
            return _result(frame == label["frame"] and pixel_matches, frame=frame, pixel=point)
        if task == "6a":
            answer = _scalar(answer, "id")
            if not isinstance(answer, str) or not re.fullmatch(r"[RDO]\d{2,4}|new", answer):
                raise ValueError("Expected an entity ID or new")
            return _result(answer in label["accepted_ids"])
        if task == "6c":
            answer = _scalar(answer, "same_instance")
            if isinstance(answer, str) and answer.lower().rstrip(".") in {"yes", "no"}:
                answer = answer.lower().rstrip(".") == "yes"
            if not isinstance(answer, bool):
                raise ValueError("Expected yes/no or a boolean")
            return _result(answer == label["same_instance"])
        answer = _scalar(answer, "direction")
        if not isinstance(answer, str):
            raise ValueError("Expected an eight-way direction")
        answer = answer.lower().rstrip(".").replace("_", "-")
        if answer not in DIRECTIONS:
            raise ValueError("Unknown eight-way direction")
        return _result(answer == label["direction"])
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        return _result(False, "format_error", detail=str(error))


def _score_memory(task, label, outcome):
    entities = outcome.get("entities", [])
    baseline = label.get("baseline_bindings", {})
    matched = match_entities(entities, label["instances"], label["transform"],
                             label.get("frame_visibility"),
                             location_sources=outcome.get("location_sources"),
                             baseline_bindings=baseline)
    ids = {entity["id"] for entity in entities}
    # Restored baseline bindings are known, but an edited baseline object must
    # still pass the current geometry/evidence check above.
    target = str(label["target_instance"])
    target_ids = [identifier for identifier, instance in matched["bindings"].items() if instance == target]
    new_ids = ids - set(label.get("baseline_ids", baseline))
    unverified = sorted(set(matched["unmatched"]) & new_ids)
    unverified.extend(entity["id"] for entity in entities
                      if entity["id"] in new_ids and entity.get("type") != "object")
    ambiguous_new = [row for row in matched["ambiguous"] if row["id"] in new_ids]
    final_instances = set(matched["bindings"].values())
    missing_instances = sorted({str(instance) for instance in baseline.values()
                                if str(instance) != target and str(instance) not in final_instances})
    missing_baseline = sorted(identifier for identifier, instance in baseline.items()
                              if str(instance) in missing_instances)
    metrics = {"target_records": len(target_ids), "target_ids": target_ids,
               "duplicates": matched["duplicates"], "unverified_new": unverified,
               "ambiguous_new": ambiguous_new, "missing_baseline": missing_baseline,
               "missing_baseline_instances": missing_instances,
               "bindings": matched["bindings"]}
    if len(target_ids) != 1:
        return _result(False, "missing_target_memory" if not target_ids else "duplicate_target_memory", **metrics)
    if task == "1b" and not any(identifier in new_ids for identifier in target_ids):
        return _result(False, "target_not_created", **metrics)
    if missing_baseline:
        return _result(False, "unrelated_memory_lost", **metrics)
    if matched["duplicates"]:
        return _result(False, "duplicate_memory", **metrics)
    if unverified or ambiguous_new:
        return _result(False, "memory_requires_review", **metrics)
    return _result(True, **metrics)


def _score_approach(sample, label, outcome):
    limits = sample.get("limits", {})
    actions, length = outcome.get("actions", 0), outcome.get("path_length", 0.)
    if actions > limits.get("max_actions", math.inf) or length > limits.get("max_path_length", math.inf) + 1e-6:
        return _result(False, "movement_budget", actions=actions, path_length=length)
    if not outcome.get("submitted"):
        return _result(False, "not_submitted")
    distance = outcome.get("distance_to_approach")
    if type(distance) not in (int, float) or not math.isfinite(distance):
        return _result(False, "no_reachable_approach")
    threshold = label.get("approach_tolerance", .25)
    return _result(distance <= threshold, distance_to_approach=distance,
                   approach_tolerance=threshold, actions=actions, path_length=length)
