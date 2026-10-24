"""Actual model-assisted wall-map experiments with conservative geometry gates."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from src.data.map_exploration import (write_json, rasterize, raster_pixels, overlay_image,
                                      PARAMETERS)
from src.data.map_geometry import load_scene_geometry, multiheight_slices
from src.data.map_model_review import ReviewClient, run_suite, project_point_votes
from src.data.map_quality import erode_one, validate_map
from src.data.map_scene import MapScene


def _function(name, description, properties, required):
    return {"type": "function", "name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": required, "additionalProperties": False}}


TOOLS = [
    _function("render_panorama", "Render four real horizontal RGB views at a requested world X/Z position; must be within 0.35m of this floor's NavMesh.",
              {"x": {"type": "number"}, "z": {"type": "number"}}, ["x", "z"]),
    _function("render_topdown", "Inspect a crop of the real orthographic RGB rendering. Coordinates are original map pixel columns/rows, not world metres.",
              {key: {"type": "integer", "minimum": 0} for key in ("left", "top", "right", "bottom")},
              ["left", "top", "right", "bottom"]),
    _function("mark", "Place a proposed issue marker on the private overlay for further visual inspection; does not edit the wall map.",
              {"x": {"type": "number"}, "z": {"type": "number"}, "label": {"type": "string"}},
              ["x", "z", "label"]),
]


def _issues(method):
    if method.get("parsed"):
        return method["parsed"].get("issues", [])
    if "vote" in method:
        return [issue for group in method["vote"]["groups"] if group["consensus"]
                for issue in group["evidence"][:1]]
    return []


def geometry_gate(walls, support, navmesh, issues, *, origin, side, allowed_images):
    """Only restore already measured raw surfaces or clear actual walkable cells.

    A point label cannot define a new wall segment. Furniture labels cannot
    erase measured geometry. These rejected proposals remain in the report.
    """
    updated = walls.copy()
    decisions = []
    allowed = {Path(name).name for name in allowed_images}
    for issue in issues:
        record = {"issue": issue, "accepted": False, "changed_pixels": 0}
        kind = issue["type"]
        if Path(issue["image"]).name not in allowed:
            record["reason"] = "perspective_point_needs_3d_correspondence_not_map_coordinates"
        elif kind not in ("missing_wall", "overdrawn_wall"):
            record["reason"] = "label_alone_cannot_change_geometry"
        else:
            x, y = issue["pixel_hint"]
            if not (0 <= x < walls.shape[1] and 0 <= y < walls.shape[0]):
                record["reason"] = "outside_map"
            else:
                yy, xx = np.ogrid[:walls.shape[0], :walls.shape[1]]
                local = (xx - x) ** 2 + (yy - y) ** 2 <= 8 ** 2
                if kind == "missing_wall":
                    change = local & support & ~updated & ~erode_one(navmesh)
                    updated |= change
                    record["reason"] = "measured_raw_surface_support_only"
                else:
                    change = local & updated & erode_one(navmesh)
                    updated &= ~change
                    record["reason"] = "only_demonstrably_walkable_wall_cells_can_be_erased"
                record["changed_pixels"] = int(change.sum())
                record["accepted"] = bool(change.any())
                if not change.any():
                    record["reason"] += ":no_supported_change"
        decisions.append(record)
    return updated, decisions


def experiment(folder, output, model, *, gpu=1, env_file=".env.harness", modes=None):
    folder, output = Path(folder).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    info = json.loads((folder / "result.json").read_text())
    candidate = folder / "raw_vote"
    original_meta = json.loads((candidate / "meta.json").read_text())
    origin, side, floor = info["origin"], info["side_m"], info["floor_height"]
    views = [{**v, "image": v["rgb"]} for v in info["views"][::4]]
    summary = {"origin_xz": origin, "side_m": side, "metres_per_pixel": .05,
               "axes": "image right +X; down +Z; world floor_y=" + str(floor),
               "safe_initial_world_xz": original_meta["snapped_start"][::2],
               "quality": {name: {"pass": value["quality"]["pass"],
                     "flagged_pairs": value["quality"]["leaks"]["flagged_count"],
                     "blocked_components": value["quality"]["traversability"]["reasons"]}
                     for name, value in info["methods"].items()},
               "flagged_pairs": [p for p in original_meta["quality"]["leaks"]["pairs"] if p["flagged"]][:4]}
    call_index = 0
    tool_log = []
    with MapScene(info["scene"], origin=origin, side=side, gpu=gpu) as simulator:
        def panorama(args):
            nonlocal call_index
            requested = [args["x"], floor, args["z"]]
            snapped = np.asarray(simulator.sim.pathfinder.snap_point(np.asarray(requested, dtype=np.float32)))
            if (not np.isfinite(snapped).all() or abs(snapped[1] - floor) > .25
                    or np.linalg.norm(snapped[[0, 2]] - requested[::2]) > .35):
                return {"text": "Requested location is not on the selected floor's walkable surface.", "images": []}
            call_index += 1
            observations = [simulator.view(snapped, yaw, output / "active_views" / f"tool{call_index}-yaw{yaw}")
                            for yaw in (0, 90, 180, 270)]
            result = {"data": {"snapped_position": snapped.tolist(), "views": observations},
                      "images": [v["rgb"] for v in observations]}
            tool_log.append({"name": "render_panorama", "arguments": args, "result": result})
            write_json(output / "active_tool_log.json", tool_log)
            return result

        def topdown(args):
            nonlocal call_index
            call_index += 1
            image = Image.open(folder / "topdown.png")
            box = [args[key] for key in ("left", "top", "right", "bottom")]
            if not (0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height):
                return {"text": "Crop must be a nonempty region inside the original map pixels."}
            path = output / f"topdown-crop-{call_index}.png"
            image.crop(box).save(path)
            result = {"data": {"original_image": str(folder / "topdown.png"), "crop": box}, "images": [str(path)]}
            tool_log.append({"name": "render_topdown", "arguments": args, "result": result})
            write_json(output / "active_tool_log.json", tool_log)
            return result

        def mark(args):
            nonlocal call_index
            call_index += 1
            image = Image.open(candidate / "overlay.png").convert("RGB")
            x, y = raster_pixels([args["x"], args["z"]], origin, side, image.width)
            draw = ImageDraw.Draw(image)
            draw.ellipse([x - 6, y - 6, x + 6, y + 6], outline="yellow", width=2)
            draw.text((x + 8, y), args["label"], fill="yellow")
            path = output / f"marker-{call_index}.png"
            image.save(path)
            result = {"text": "Proposal marked only; wall geometry unchanged.", "images": [str(path)]}
            tool_log.append({"name": "mark", "arguments": args, "result": result})
            write_json(output / "active_tool_log.json", tool_log)
            return result

        started = time.monotonic()
        suite = run_suite(candidate, views, summary, client=ReviewClient(env_file), model=model,
                          tools=TOOLS, tool_callbacks={"render_panorama": panorama,
                                                     "render_topdown": topdown, "mark": mark}, modes=modes)
    write_json(output / "model_results.json", suite)
    return score_experiment(folder, output, suite, time.monotonic() - started)


def score_experiment(folder, output, suite, elapsed=0):
    """Rescore actual stored model outputs without issuing a new request."""
    folder, output = Path(folder), Path(output)
    info = json.loads((folder / "result.json").read_text())
    candidate = folder / "raw_vote"
    original_meta = json.loads((candidate / "meta.json").read_text())
    origin, side, floor = info["origin"], info["side_m"], info["floor_height"]
    walls = np.asarray(Image.open(candidate / "walls.png")) < 128
    geometry = load_scene_geometry(info["scene"], include_semantics=False)
    slices = multiheight_slices(geometry.triangles, floor)
    support = np.any([rasterize(layer, origin, side) for layer in slices.values()], axis=0)
    points = np.concatenate(list(slices.values())).reshape(-1, 2)
    navmesh = np.asarray(Image.open(folder / "navmesh.png")) > 0
    pairs = [p for p in json.loads((folder / "geodesic_pairs.json").read_text())
             if not p.get("out_of_floor_band", False)]
    images = [str(candidate / "walls.png"), str(candidate / "overlay.png")]
    tool_log = json.loads((output / "active_tool_log.json").read_text()) if (output / "active_tool_log.json").exists() else []
    outcomes = {}
    rgb = np.asarray(Image.open(folder / "topdown.png").convert("RGB"))
    for name, method in suite["methods"].items():
        updated, decisions = geometry_gate(walls, support, navmesh, _issues(method), origin=origin,
                                          side=side, allowed_images=images)
        quality = validate_map(updated, navmesh, origin=origin, resolution=.05, pairs=pairs,
                               construction_points=points,
                               raster_pixels=raster_pixels(points, origin, side, len(walls)))
        method_output = output / name
        method_output.mkdir(exist_ok=True)
        Image.fromarray(np.where(updated, 0, 255).astype(np.uint8)).save(method_output / "walls.png")
        overlay_image(rgb, navmesh, updated).save(method_output / "overlay.png")
        outcome = {"model_output_valid": method.get("ok", False), "geometry_decisions": decisions,
                   "changed_pixels": int((updated != walls).sum()), "before": original_meta["quality"],
                   "after": quality, "note": "No prose rule or unsupported point automatically changes a map."}
        if name == "point_depth" and method.get("ok"):
            projected = method.get("projected_points", [])
            outcome["projected_points"] = projected
            outcome["point_count"] = len(projected)
            outcome["valid_depth_points"] = sum(p.get("ok", False) for p in projected)
            # Votes are in world space; different perspective pixels must not
            # be treated as votes for one map cell without reprojection.
            world_points = [p for p in projected if p.get("ok")]
            outcome["multiview_world_agreements"] = int(sum(
                a["image"] != b["image"] and a["type"] == b["type"]
                and np.linalg.norm(np.asarray(a["world_xyz"])[[0, 2]] - np.asarray(b["world_xyz"])[[0, 2]]) < .2
                for i, a in enumerate(world_points) for b in world_points[i + 1:]))
        write_json(method_output / "outcome.json", outcome)
        outcomes[name] = outcome
    result = {"scene": info["scene"], "candidate": str(candidate), "model": suite["model"],
              "seconds": elapsed, "tool_calls": len(tool_log), "methods": outcomes}
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-folder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--env", default=".env.harness")
    parser.add_argument("--score-existing", action="store_true")
    parser.add_argument("--modes", nargs="+", choices=("baseline", "point_depth", "active", "repeatvote", "analyst", "hybrid"))
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    if args.score_existing:
        suite = json.loads((Path(args.output) / "model_results.json").read_text())
        score_experiment(args.scene_folder, args.output, suite)
    else:
        experiment(args.scene_folder, args.output, args.model, gpu=args.gpu, env_file=args.env, modes=args.modes)


if __name__ == "__main__":
    main()
