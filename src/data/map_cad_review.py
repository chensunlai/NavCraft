"""Batch small visual point tasks; proposals never directly edit wall geometry.

Each task asks for the two ground-level jambs of one visible opening. Normalized
image coordinates avoid ambiguous resized-image pixel conventions. Independent
camera/depth and target checks reject points on another doorway or object.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .map_model_review import MODEL_PROFILES, ReviewClient, backproject_pixel


POINT_TASK_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "visibility": {"type": "string", "enum": ["complete", "partial", "not_visible"]},
        "points": {"type": "array", "maxItems": 2, "items": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "label": {"type": "string", "enum": ["left", "right"]},
                "xy_normalized": {"type": "array", "minItems": 2, "maxItems": 2,
                                  "items": {"type": "number", "minimum": 0, "maximum": 1}},
            }, "required": ["label", "xy_normalized"],
        }},
        "reason": {"type": "string"},
    }, "required": ["visibility", "points", "reason"],
}

WALL_PAIR_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "relation": {"type": "string", "enum": ["same_wall_faces", "separate_walls", "uncertain"]},
        "visible_wall_evidence": {"type": "boolean"},
        "reason": {"type": "string"},
    }, "required": ["relation", "visible_wall_evidence", "reason"],
}


def wall_pair_reviewer(records, primitives):
    """Bind saved visual decisions to exactly the geometry that was reviewed."""
    lookup = {item["id"]: item for item in primitives}
    records = list(records)
    indexed = {tuple(sorted(item["source_primitive_ids"])): item for item in records}
    if len(indexed) != len(records):
        raise ValueError("Resolve duplicate wall-pair reviews before applying geometry decisions")

    def review(pair):
        record = indexed.get(tuple(sorted(pair["source_primitive_ids"])))
        if record is None:
            return {"pass": False, "evidence": {"reason": "no_review_for_this_wall_pair"}}
        if not all(key in lookup for key in record["source_primitive_ids"]):
            return {"pass": False, "evidence": {"reason": "reviewed_geometry_has_changed"}}
        source = [{"id": item["id"], "type": item["type"], "start": item["start"], "end": item["end"]}
                  for item in [lookup[key] for key in record["source_primitive_ids"]]]
        digest = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
        if digest != record["geometry_sha256"]:
            return {"pass": False, "evidence": {"reason": "reviewed_geometry_has_changed"}}
        response = record["response"]
        parsed = response.get("parsed") or {}
        approved = response.get("response_model") in {"Qwen3.8-27B", "gpt-6.1-sol"}
        accepted = (response.get("ok") is True and approved
                    and parsed.get("relation") == "same_wall_faces"
                    and parsed.get("visible_wall_evidence") is True)
        return {"pass": accepted, "evidence": {"task_id": record["task_id"],
                "geometry_sha256": digest, "model": response.get("response_model"),
                "record_path": response.get("record_path"), "visual_decision": parsed,
                "input_images": response.get("images", []),
                "scope": "same_physical_wall_confirmation_only_not_a_geometric_measurement"}}
    return review


def review_wall_pairs(tasks, output, *, model="Qwen3.8-27B", client=None):
    """Ask one relation per local crop; retain the exact input geometry hash."""
    if model not in {"Qwen3.8-27B", "gpt-6.1-sol"}:
        raise ValueError("Use a verified model backend for CAD pair tasks")
    tasks = list(tasks)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    client = client or ReviewClient(env_file=".env.harness-test")
    (output / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n")

    def run(task):
        prompt = ("只判断一件事：标为A、B的两条近似平行线，是同一堵建筑墙的两个扫描表面，"
                  "还是两面独立的墙？第一图是未经标线的真实俯视RGB，第二图只叠加候选线。"
                  "只根据实际建筑布局和可见墙体判断，不因两条线很近就说同墙；柜体、门板也不当永久墙。"
                  "若原RGB不能支持判断，回答uncertain且visible_wall_evidence=false。"
                  "不画图、不移动点、不规划路线，一句话说明可见依据。")
        response = client.request(prompt, task["images"], model=model, schema=WALL_PAIR_SCHEMA,
                    metadata={"mode": "cad_atomic_wall_pair", "task_id": task["id"]})
        return {"task_id": task["id"], "scene": task["scene"], "source_primitive_ids": task["source_primitive_ids"],
                "geometry_sha256": task["geometry_sha256"], "response": response, "maps_modified": False}

    workers = min(len(tasks), MODEL_PROFILES[model]["concurrency"])
    if workers:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run, tasks))
    else:
        results = []
    (output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    return results


def validate_jamb_points(answer, view, target_segment, floor_height):
    """Check original-image calibration and independently known local target.

    Passing means a usable local geometric proposal, not visual truth. A point
    on a curtain can still need human/second-view review even when nearby.
    """
    points = answer["points"]
    if answer["visibility"] != "complete" or len(points) != 2:
        return {"pass": False, "reason": "both_ground_jambs_not_visible", "points": []}
    if {point["label"] for point in points} != {"left", "right"}:
        return {"pass": False, "reason": "duplicate_or_missing_jamb_label", "points": []}
    with Image.open(view["rgb"]) as image:
        size = np.array(image.size)
    loaded = np.load(view["depth"])
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            depth = loaded["depth"].copy()
        finally:
            loaded.close()
    else:
        depth = loaded
    if depth.shape != tuple(size[::-1]):
        raise ValueError("RGB and depth must have the same original resolution")
    proposals = []
    for point in points:
        normalized = np.asarray(point["xy_normalized"], dtype=float)
        if normalized.shape != (2,) or not np.isfinite(normalized).all() or not ((normalized >= 0) & (normalized <= 1)).all():
            raise ValueError("Normalized coordinates must be finite and within [0, 1]")
        pixel = normalized * (size - 1)
        try:
            world = np.asarray(backproject_pixel(pixel, depth, view["intrinsics"], view["world_from_optical"]))
        except ValueError as error:
            return {"pass": False, "reason": "invalid_depth_or_calibration", "error": str(error),
                    "points": proposals}
        proposals.append({**point, "pixel_xy": pixel.tolist(), "world_xyz": world.tolist(),
                          "height_above_floor_m": float(world[1] - floor_height)})
    world = np.asarray([point["world_xyz"] for point in proposals])
    segment = np.asarray(target_segment, dtype=float)
    if segment.shape != (2, 2) or not np.isfinite(segment).all():
        raise ValueError("Target must be a finite local world-XZ segment")
    errors = np.linalg.norm(world[:, None, [0, 2]] - segment[None, :, :], axis=2)
    order = [0, 1] if errors[0, 0] + errors[1, 1] <= errors[0, 1] + errors[1, 0] else [1, 0]
    distances = [float(errors[index, endpoint]) for index, endpoint in enumerate(order)]
    grounded = bool(np.max(np.abs(world[:, 1] - floor_height)) <= .15)
    near_target = max(distances) <= .30
    distinct = float(np.linalg.norm(world[0, [0, 2]] - world[1, [0, 2]])) >= .25
    return {"pass": bool(grounded and near_target and distinct),
            "reason": "local_geometry_consistent_pending_visual_review" if grounded and near_target and distinct
                      else "wrong_height_target_or_degenerate_span",
            "points": proposals, "endpoint_distances_m": distances,
            "grounded": grounded, "near_target": near_target, "distinct": distinct,
            "maps_modified": False}


def review_point_tasks(tasks, output, *, model="Qwen3.8-27B", client=None):
    if model not in {"Qwen3.8-27B", "gpt-6.1-sol"}:
        raise ValueError("Use a verified model backend for CAD point tasks")
    tasks = list(tasks)
    if len({task["id"] for task in tasks}) != len(tasks):
        raise ValueError("Point-task IDs must be unique")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    client = client or ReviewClient(env_file=".env.harness-test")
    (output / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n")

    def run(task):
        prompt = ("只做一个小任务：在这张原始RGB中，标出指定门洞左右门框与地面的交点。"
                  "左右按图片左右。点必须是实际可见的固定门框落地位置，不是门板、家具、窗帘或推测的遮挡点。"
                  "若一侧看不到，用partial且仅标可见点；都看不到用not_visible和空points。"
                  "坐标xy_normalized=[x/(图宽-1), y/(图高-1)]，范围[0,1]。"
                  "不要画整张地图，不给路线，不补猜测点。具体目标：" + task["hint"])
        response = client.request(prompt, [task["view"]["rgb"]], model=model,
                    schema=POINT_TASK_SCHEMA, metadata={"mode": "cad_atomic_jamb_points", "task_id": task["id"]})
        result = {"task_id": task["id"], "response": response, "maps_modified": False}
        if response["ok"]:
            try:
                result["validation"] = validate_jamb_points(response["parsed"], task["view"],
                                                           task["target_segment"], task["floor_height"])
            except (ValueError, OSError, KeyError) as error:
                result["validation"] = {"pass": False, "reason": "evidence_error", "error": str(error)}
            image = Image.open(task["view"]["rgb"]).convert("RGB")
            draw = ImageDraw.Draw(image)
            for point in response["parsed"]["points"]:
                x, y = np.asarray(point["xy_normalized"]) * (np.array(image.size) - 1)
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline="#ff3355", width=2)
                draw.text((x + 8, y), point["label"], fill="#ff3355", stroke_width=1, stroke_fill="white")
            image.save(output / (task["id"] + ".jpg"))
        return result

    workers = min(len(tasks), MODEL_PROFILES[model]["concurrency"])
    if not workers:
        results = []
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run, deepcopy(tasks)))
    (output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    summary = {"tasks": len(tasks), "model": model, "concurrency": workers,
               "valid_structured_answers": sum(result["response"]["ok"] for result in results),
               "locally_consistent_proposals": sum(result.get("validation", {}).get("pass", False) for result in results),
               "maps_modified": False, "results": "results.json"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--env-file", default=".env.harness-test")
    parser.add_argument("--model", choices=["Qwen3.8-27B", "gpt-6.1-sol"], default="Qwen3.8-27B")
    args = parser.parse_args()
    print(json.dumps(review_point_tasks(json.loads(Path(args.tasks).read_text()), args.output,
                                       model=args.model, client=ReviewClient(env_file=args.env_file))), flush=True)


if __name__ == "__main__":
    main()
