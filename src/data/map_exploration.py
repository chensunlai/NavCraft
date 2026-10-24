"""Geometry-first wall-map experiments; private inspection products stay in runs.

Example: python -m src.data.map_exploration development --manifest scenes.json
    --output runs/map-exploration-geometric --gpu 1
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from src.data.map_geometry import load_scene_geometry, multiheight_slices, slice_triangles
from src.data.map_quality import erode_one, validate_map, traversability_report

PIXEL_SIZE_M = .05
HEIGHTS = (.9, 1.3, 1.7, 2.)
PARAMETERS = {"slice_heights_m": HEIGHTS, "minimum_votes": 2,
              "height_vote_tolerance_px": 1, "minimum_component_pixels": 4,
              "pixel_size_m": .05, "canvas_padding_m": 2,
              "minimum_region_area_m2": .1,
              "navmesh": {"agent_radius": .18, "agent_height": .88,
                          "agent_max_climb": .2, "cell_height": .05}}
METHODS = ("raw_vote", "navmesh_boundary", "depth_height", "semantic_verified")
BUILD_ID = "geometric-20261024"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def canvas(segments, padding_m=2.):
    points = np.asarray(segments, dtype=float).reshape(-1, 2)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("Cannot choose a canvas without finite geometry")
    low, high = points.min(0) - padding_m, points.max(0) + padding_m
    side = max(10., math.ceil(float(max(high - low)) / 10) * 10.)
    return ((low + high - side) / 2).tolist(), side


def raster_pixels(points, origin, side, size):
    """Actual rasterizer coordinates, independently checked by map_quality."""
    return np.rint((np.asarray(points) - np.asarray(origin)) / (side / size) - .5).astype(int)


def rasterize(segments, origin, side, pixel_size=PIXEL_SIZE_M):
    if pixel_size <= 0 or not math.isfinite(pixel_size):
        raise ValueError("Pixel size must be positive")
    size = int(round(side / pixel_size))
    if size < 2 or not math.isclose(size * pixel_size, side):
        raise ValueError("Canvas side must be a multiple of the pixel size")
    image = Image.new("1", (size, size))
    draw = ImageDraw.Draw(image)
    for segment in raster_pixels(segments, origin, side, size).reshape(-1, 2, 2):
        draw.line([tuple(segment[0]), tuple(segment[1])], fill=1, width=1)
    return np.asarray(image, dtype=bool)


def flood_regions(free, *, minimum_pixels=4):
    from scipy import ndimage
    labels, count = ndimage.label(free)
    result, regions = np.full(labels.shape, -1, dtype=np.int32), []
    for label in range(1, count + 1):
        rows, cols = np.nonzero(labels == label)
        if len(rows) < minimum_pixels:
            continue
        identifier = len(regions)
        result[rows, cols] = identifier
        regions.append({"id": f"R{identifier:03d}", "pixels": len(rows),
                        "area_m2": len(rows) * .05 ** 2,
                        "center_uv": [(float(cols.mean()) + .5) / labels.shape[1],
                                      (float(rows.mean()) + .5) / labels.shape[0]]})
    return result, regions


def _small_components(mask, minimum=4):
    from scipy import ndimage
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.reshape(-1))
    keep = sizes >= minimum
    keep[0] = False
    return keep[labels]


def _mask_image(mask, path):
    Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).save(path)


def region_image(labels, walls):
    palette = np.random.default_rng(19).integers(80, 235, (int(labels.max()) + 1, 3), dtype=np.uint8)
    palette[0] = 245
    colors = palette[labels]
    colors[walls] = 25
    return Image.fromarray(colors)


def floor_domain(triangles, floor, origin, side, floor_range=None):
    """Rasterize measured near-horizontal ground; do not fill unscanned holes."""
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    horizontal = np.abs(normals[:, 1]) > .9 * lengths
    low, high = triangles[:, :, 1].min(1), triangles[:, :, 1].max(1)
    height_low, height_high = floor_range if floor_range is not None else (floor - .35, floor + .2)
    selected = triangles[horizontal & (lengths > 1e-8) & (low >= height_low) & (high <= height_high)]
    size = int(round(side / .05))
    image = Image.new("1", (size, size))
    draw = ImageDraw.Draw(image)
    for polygon in raster_pixels(selected[:, :, [0, 2]], origin, side, size):
        draw.polygon([tuple(p) for p in polygon], fill=1)
    return np.asarray(image, dtype=bool), len(selected)


def overlay_image(rgb, navmesh, walls):
    values = rgb.astype(float).copy()
    values[navmesh] = .65 * values[navmesh] + .35 * np.array([70, 170, 245])
    values[walls] = [255, 70, 65]
    return Image.fromarray(np.clip(values, 0, 255).astype(np.uint8))


def _semantic_raster(geometry, floor, origin, side):
    pieces = []
    for triangles in geometry.categories.values():
        pieces.extend(slice_triangles(triangles, floor + h) for h in HEIGHTS)
    pieces = [piece for piece in pieces if len(piece)]
    return rasterize(np.concatenate(pieces) if pieces else np.empty((0, 2, 2)), origin, side)


def episode_floor_range(seed, start_y):
    """Group measured training-start heights, retaining genuine split levels."""
    heights = sorted(row["rounded_height_m"] for row in (seed or {}).get("training_start_height_summary", []))
    if not heights:
        return [start_y - .35, start_y + .35]
    groups = [[heights[0]]]
    for height in heights[1:]:
        if height - groups[-1][-1] > .65:
            groups.append([])
        groups[-1].append(height)
    group = min(groups, key=lambda values: min(abs(value - start_y) for value in values))
    return [min(group) - .2, max(group) + .2]


def build_map(scene, episode_start, output, *, gpu=1, seed=None):
    """Build baselines in one real scene, retaining independent evidence."""
    from scipy import ndimage
    from src.data.map_scene import MapScene
    from src.data.map_openings import opening_candidates, verify_openings, region_partition

    started = time.monotonic()
    output, scene = Path(output).resolve(), Path(scene).resolve(strict=True)
    output.mkdir(parents=True, exist_ok=True)
    geometry = load_scene_geometry(scene)
    initial_slices = multiheight_slices(geometry.triangles, float(episode_start[1]))
    segments = np.concatenate([s for s in initial_slices.values() if len(s)])
    origin, side = canvas(segments)
    floor_range = episode_floor_range(seed, float(episode_start[1]))
    with MapScene(scene, origin=origin, side=side, gpu=gpu, floor_range=floor_range) as simulator:
        floor, snapped = simulator.floor(episode_start)
        slices = multiheight_slices(geometry.triangles, floor)
        segments = np.concatenate([s for s in slices.values() if len(s)])
        points = segments.reshape(-1, 2)
        if (points.min(0) < np.array(origin)).any() or (points.max(0) >= np.array(origin) + side).any():
            raise ValueError("Snapped floor geometry falls outside the selected canvas")
        navmesh = simulator.navmesh_mask(floor)
        Image.fromarray((navmesh * 255).astype(np.uint8)).save(output / "navmesh.png")
        rgb, height, camera = simulator.topdown(floor)
        Image.fromarray(rgb).save(output / "topdown.png")
        np.save(output / "topdown_height.npy", height)
        write_json(output / "topdown_camera.json", camera)
        layers = [rasterize(value, origin, side) for value in slices.values()]
        raw_union = np.any(layers, axis=0)
        votes = np.sum([ndimage.binary_dilation(layer, iterations=1) for layer in layers], axis=0)
        raw = _small_components(raw_union & (votes >= 2))
        semantic = _semantic_raster(geometry, floor, origin, side)
        _mask_image(semantic, output / "semantic_draft.png")
        nav_interior = erode_one(navmesh)
        candidates = {
            "raw_vote": raw,
            "navmesh_boundary": navmesh & ~erode_one(navmesh),
            "depth_height": np.isfinite(height) & (height >= 1.3) & (height <= 2.1),
            "semantic_verified": raw | (semantic & ndimage.binary_dilation(raw_union, iterations=1)),
        }
        sampled_points = simulator.points(floor)
        pairs = simulator.pairs(sampled_points)
        write_json(output / "geodesic_pairs.json", pairs)
        np.save(output / "navmesh_samples.npy", sampled_points)
        view_points = [np.asarray(snapped)]
        for point in sampled_points:
            if min(np.linalg.norm(point[[0, 2]] - other[[0, 2]]) for other in view_points) > 2:
                view_points.append(point)
            if len(view_points) >= 4:
                break
        views = [simulator.view(point, yaw, output / "views" / f"p{index}-yaw{yaw}")
                 for index, point in enumerate(view_points) for yaw in (0, 90, 180, 270)]
        write_json(output / "views.json", views)
        opening_drafts = opening_candidates(geometry.categories, slices, floor)

        def passable(segment):
            evidence = simulator.passage(segment, floor)
            return {**evidence, "passable": evidence["pass"]}

        openings = verify_openings(opening_drafts, slices, passable)
        write_json(output / "opening_candidates.json", openings)
        domain, ground_triangles = floor_domain(geometry.triangles, floor, origin, side, floor_range)
        Image.fromarray((domain * 255).astype(np.uint8)).save(output / "floor_domain.png")
        valid_pairs = [pair for pair in pairs if not pair.get("out_of_floor_band", False)]
        write_json(output / "sampling_audit.json", {
            "navmesh": getattr(simulator, "navmesh_audit", {}),
            "samples": getattr(simulator, "sample_audit", {}),
            "pairs": getattr(simulator, "pair_audit", {}),
            "excluded_cross_floor_pairs": len(pairs) - len(valid_pairs)})
        methods = {}
        for name, candidate in candidates.items():
            method_output = output / name
            method_output.mkdir(exist_ok=True)
            walls = candidate & ~nav_interior
            quality = validate_map(walls, navmesh, origin=origin, resolution=.05, pairs=valid_pairs,
                                   construction_points=points,
                                   raster_pixels=raster_pixels(points, origin, side, len(walls)))
            quality["before_navmesh_cleanup"] = traversability_report(candidate, navmesh)
            quality["rgb_alignment"] = camera.get("alignment", {"pass": False, "reason": "actual_camera_projection_not_checked"})
            scope = dict(getattr(simulator, "navmesh_audit", {}))
            scope["pass"] = bool(scope.get("selected_triangle_count", 0) > 0
                                 and not scope.get("outside_canvas", True)
                                 and not scope.get("requires_floor_review", True))
            quality["navmesh_scope"] = scope
            quality["pass"] = (quality["pass"] and quality["rgb_alignment"]["pass"]
                               and quality["navmesh_scope"].get("pass", False))
            accepted = [opening for opening in openings if opening["status"] == "accepted"]
            labels, regions, assigned_openings = region_partition(domain, walls, accepted,
                                                                  origin=origin, resolution=.05)
            _mask_image(walls, method_output / "walls.png")
            overlay_image(rgb, navmesh, walls).save(method_output / "overlay.png")
            region_image(labels, walls).save(method_output / "regions.png")
            write_json(method_output / "regions.json", regions)
            write_json(method_output / "openings.json", assigned_openings)
            write_json(method_output / "quality.json", quality)
            meta = {"scene": str(scene), "dataset": "hm3d" if scene.name.endswith(".basis.glb") else "mp3d",
                    "floor_height": floor, "floor_source": "real_training_episode_start_snapped_to_lite_navmesh",
                    "episode_start": episode_start, "snapped_start": snapped,
                    "floor_height_range": floor_range,
                    "origin": origin, "side_m": side, "pixel_size_m": .05,
                    "axes": "top:-Z, right:+X; origin is top-left outer corner [xmin,zmin]",
                    "method": name, "version": BUILD_ID, "parameters": PARAMETERS,
                    "wall_pixels_before_navmesh": int(candidate.sum()),
                    "removed_navmesh_pixels": int((candidate & nav_interior).sum()),
                    "region_domain": {"source": "raw_horizontal_ground_triangles", "triangles": ground_triangles,
                        "pixels": int(domain.sum()), "navmesh_covered_fraction": float((navmesh & domain).sum() / navmesh.sum()),
                        "status": "geometry_partition_candidates_pending_visual_review"},
                    "semantic_comparison": {"draft_pixels": int(semantic.sum()),
                        "removed_pixels": int((semantic & ~walls).sum()),
                        "added_pixels": int((walls & ~semantic).sum())},
                    "quality": quality, "source_geometry": geometry.metadata,
                    "wall_sha256": hashlib.sha256((method_output / "walls.png").read_bytes()).hexdigest(),
                    "status": "automatic_checks_passed_pending_visual_review" if quality["pass"] else "needs_review"}
            write_json(method_output / "meta.json", meta)
            methods[name] = {"output": str(method_output), "wall_pixels": int(walls.sum()),
                             "quality": quality, "semantic_comparison": meta["semantic_comparison"]}
            render_flagged_pairs(method_output, valid_pairs, quality, rgb, navmesh, walls, origin, side)
        result = {"scene": str(scene), "floor_height": floor, "origin": origin, "side_m": side,
                  "gpu_physical": gpu, "gpu_logical": 0, "methods": methods,
                  "seed": seed, "views": views, "seconds": time.monotonic() - started}
        write_json(output / "result.json", result)
        return result


def render_flagged_pairs(output, pairs, quality, rgb, navmesh, walls, origin, side):
    entries = []
    for index, (pair, checked) in enumerate(zip(pairs, quality["leaks"]["pairs"])):
        if not checked["flagged"]:
            continue
        image = overlay_image(rgb, navmesh, walls)
        draw = ImageDraw.Draw(image)
        endpoints = np.array(pair["a"])[[0, 2]], np.array(pair["b"])[[0, 2]]
        positions = raster_pixels(endpoints, origin, side, len(walls))
        draw.line([tuple(p) for p in positions], fill=(255, 225, 0), width=3)
        route = raster_pixels(np.asarray(pair["reference_path"])[:, [0, 2]], origin, side, len(walls))
        draw.line([tuple(p) for p in route], fill=(235, 50, 230), width=2)
        for x, y in positions:
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 225, 0))
        name = f"leak-{index:03d}.png"
        folder = output / "leak_review"
        folder.mkdir(exist_ok=True)
        image.save(folder / name)
        entries.append({"pair_index": index, "image": "leak_review/" + name,
                        "review": "unresolved", "geodesic_m": pair["geodesic_m"]})
    write_json(output / "leak_reviews.json", entries)


def development(manifest, output, gpu=1):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    document = json.loads(Path(manifest).read_text())
    seeds = document if isinstance(document, list) else document["scenes"]
    write_json(output / "development_manifest.json", document)
    results = []
    for index, seed in enumerate(seeds):
        scene = seed["scene_path"]
        folder = output / f"{index:02d}-{seed['dataset']}-{Path(scene).stem.replace('.basis', '')}"
        instruction = output / f"worker-{index:02d}.json"
        write_json(instruction, {"scene": scene, "episode_start": seed["start_pose"]["position"],
                                 "output": str(folder), "gpu": gpu, "seed": seed})
        with (output / f"worker-{index:02d}.log").open("w") as log:
            completed = subprocess.run([sys.executable, "-m", "src.data.map_exploration", "worker",
                                        "--instruction", str(instruction)], stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode == 0 and (folder / "result.json").exists():
            results.append({"status": "built", "output": str(folder),
                            "result": json.loads((folder / "result.json").read_text())})
        else:
            results.append({"status": "error", "scene": scene, "returncode": completed.returncode,
                            "log": f"worker-{index:02d}.log"})
        write_json(output / "summary.json", {"build": BUILD_ID, "scenes": results})
        gallery(output, results)
        print(f"{index + 1}/{len(seeds)}: {results[-1]['status']} {scene}", flush=True)
    return results


def audit_development(output):
    """Re-evaluate saved real captures with raw geometry, without another model/GPU run."""
    from src.data.map_leaks import review_leak_pairs
    from src.data.map_openings import region_partition

    output = Path(output).resolve()
    summary = json.loads((output / "summary.json").read_text())
    for record in summary["scenes"]:
        if record["status"] != "built":
            continue
        audit_started = time.monotonic()
        folder = Path(record["output"])
        result = json.loads((folder / "result.json").read_text())
        geometry = load_scene_geometry(result["scene"], include_semantics=False)
        origin, side, floor = result["origin"], result["side_m"], result["floor_height"]
        raw_pairs = json.loads((folder / "geodesic_pairs.json").read_text())
        pairs = [p for p in raw_pairs if not p.get("out_of_floor_band", False)]
        floor_range = episode_floor_range(result["seed"], floor)
        domain, triangle_count = floor_domain(geometry.triangles, floor, origin, side, floor_range)
        Image.fromarray((domain * 255).astype(np.uint8)).save(folder / "floor_domain.png")
        openings = json.loads((folder / "opening_candidates.json").read_text())
        accepted = [opening for opening in openings if opening["status"] == "accepted"]
        for name, method in result["methods"].items():
            target = folder / name
            meta = json.loads((target / "meta.json").read_text())
            walls = np.asarray(Image.open(target / "walls.png")) < 128
            quality = meta["quality"]
            flags = [i for i, p in enumerate(quality["leaks"]["pairs"]) if p["flagged"]]
            reviews = review_leak_pairs(geometry.triangles, pairs, floor, origin, side, domain, walls,
                                        flagged_indices=flags)
            write_json(target / "leak_geometry_review.json", reviews)
            review_cards = []
            for item in reviews:
                index = item["input_pair_position"]
                review_cards.append(f'<section><h2>Pair {index}: {html.escape(item["classification"])}</h2>'
                    f'<p>{html.escape(item["review"])}; floor support={item["floor_domain_support_fraction"]:.3f}</p>'
                    f'<img src="leak_review/leak-{index:03d}.png">'
                    '<details><summary>Measured geometry evidence</summary><pre>'
                    + html.escape(json.dumps(item, ensure_ascii=False, indent=2)) + '</pre></details></section>')
            (target / "leak_review.html").write_text('<!doctype html><meta charset="utf-8"><title>Leak geometry review</title>'
                '<style>body{font:15px system-ui;max-width:1100px;margin:30px auto}img{max-width:100%}pre{white-space:pre-wrap}</style>'
                '<h1>Flag-by-flag geometry review</h1><p>Yellow: direct chord; magenta: actual NavMesh geodesic. No map pixels were changed by this review.</p>'
                + (''.join(review_cards) if review_cards else '<p>No detour-without-wall flags.</p>'))
            unresolved = [review for review in reviews if review["review"] != "reviewed_false_flag"]
            review_pass = bool(quality["leaks"]["pair_count"]) and not unresolved
            # The original flagged ratio stays visible. A reviewed low obstacle
            # is not repaired into a wall; missing scanned floor remains unresolved.
            quality["leak_review"] = {"flagged": len(flags), "reviewed_false_flags": len(reviews) - len(unresolved),
                "unresolved": len(unresolved), "pass_after_geometry_review": review_pass,
                "geometry_evidence": "leak_geometry_review.json"}
            quality["pass_after_geometry_review"] = (quality["traversability"]["pass"]
                and quality["alignment"]["pass"] and quality["rgb_alignment"]["pass"]
                and quality["navmesh_scope"]["pass"] and review_pass)
            labels, regions, assigned = region_partition(domain, walls, accepted, origin, .05)
            write_json(target / "regions.json", regions)
            write_json(target / "openings.json", assigned)
            region_image(labels, walls).save(target / "regions.png")
            meta["region_domain"].update(triangles=triangle_count, pixels=int(domain.sum()))
            meta["parameters"] = PARAMETERS
            meta["status"] = ("geometry_checks_passed_pending_visual_review" if quality["pass_after_geometry_review"]
                              else "unresolved_geometry_review")
            write_json(target / "quality.json", quality)
            write_json(target / "meta.json", meta)
            method["quality"] = quality
        result["audit_seconds"] = time.monotonic() - audit_started
        write_json(folder / "result.json", result)
        record["result"] = result
    write_json(output / "summary.json", summary)
    gallery(output, summary["scenes"])
    return summary


def freeze_development(output, destination):
    """Package qualified development candidates; no batch scenes are generated."""
    output, destination = Path(output).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    summary = json.loads((output / "summary.json").read_text())
    manifest = {"build": BUILD_ID, "method": "raw_vote", "parameters": PARAMETERS,
                "stage": "development_candidates_awaiting_user_review", "maps": [], "excluded": []}
    for record in summary["scenes"]:
        if record["status"] != "built":
            manifest["excluded"].append(record)
            continue
        result = record["result"]
        source = Path(record["output"]) / "raw_vote"
        meta = json.loads((source / "meta.json").read_text())
        if not meta["quality"].get("pass_after_geometry_review", False):
            manifest["excluded"].append({"scene": result["scene"], "quality": meta["quality"],
                                         "reason": "unresolved_geometry_evidence"})
            continue
        scene_id = result["seed"]["scene"]
        relative = Path(BUILD_ID) / meta["dataset"] / scene_id / f"floor_{meta['floor_height']:.3f}"
        target = destination / relative
        target.mkdir(parents=True)
        for filename in ("walls.png", "openings.json", "regions.png", "regions.json", "overlay.png", "leak_geometry_review.json"):
            shutil.copyfile(source / filename, target / filename)
        actual_hash = hashlib.sha256((target / "walls.png").read_bytes()).hexdigest()
        if actual_hash != meta["wall_sha256"]:
            raise ValueError("Wall image changed since geometry validation: " + str(source))
        fields = ("scene", "dataset", "floor_height", "floor_height_range", "floor_source", "origin", "side_m",
                  "pixel_size_m", "axes", "method", "version", "parameters", "quality", "status", "wall_sha256")
        frozen_meta = {key: value for key, value in meta.items() if key in fields}
        frozen_meta["parameters"] = PARAMETERS
        write_json(target / "meta.json", frozen_meta)
        manifest["maps"].append({"scene": scene_id, "dataset": meta["dataset"], "path": str(relative),
                                  "wall_sha256": meta["wall_sha256"],
                                  "regions": "geometry_candidates_not_room_ground_truth"})
    write_json(destination / "manifest.json", manifest)
    return manifest


def gallery(output, records):
    output = Path(output).resolve()
    cards = []
    for record in records:
        if record["status"] != "built":
            cards.append("<pre>" + html.escape(str(record)) + "</pre>")
            continue
        result = record["result"]
        relative = Path(record["output"]).relative_to(output)
        parts = []
        for name, data in result["methods"].items():
            q = data["quality"]
            review = q.get("leak_review", {})
            text = (f"overlap={q['traversability']['wall_overlap_fraction']}; "
                    f"raw detour flags={q['leaks']['flagged_count']}/{q['leaks']['pair_count']}; "
                    f"unresolved={review.get('unresolved', 'not reviewed')}; "
                    f"pass after geometry review={q.get('pass_after_geometry_review', q['pass'])}")
            parts.append(f'<article><h3>{html.escape(name)}</h3><p>{html.escape(text)}</p>'
                         f'<a href="{relative}/{name}/meta.json">metadata</a> · '
                         f'<a href="{relative}/{name}/leak_review.html">flag-by-flag geometry review</a>'
                         f'<img loading="lazy" src="{relative}/{name}/overlay.png"></article>')
        cards.append(f"<section><h2>{html.escape(Path(result['scene']).stem)}</h2>"
                     f"<p>Training episode floor={result['floor_height']:.3f}m; {result['seconds']:.1f}s</p>"
                     '<div class="grid">' + "".join(parts) + "</div></section>")
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Wall-map experiments</title>'
        '<style>body{font:15px system-ui;margin:24px;background:#f6f7f9;color:#15171a}.grid{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:12px}article{background:white;padding:12px;border-radius:8px}img{width:100%}p{overflow-wrap:anywhere}section{margin:28px 0}</style>'
        '<h1>Geometry-first wall-map development set</h1>'
        '<p>Red: candidate walls; blue: rebuilt robot NavMesh. A flag is a review request, not a confirmed missing wall. No model or semantic output is accepted without geometry.</p>'
        + "".join(cards))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--instruction", required=True)
    dev = sub.add_parser("development")
    dev.add_argument("--manifest", required=True)
    dev.add_argument("--output", required=True)
    dev.add_argument("--gpu", type=int, choices=(0, 1), default=1)
    audit = sub.add_parser("audit")
    audit.add_argument("--output", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--destination", required=True)
    args = parser.parse_args()
    if args.command == "worker":
        build_map(**json.loads(Path(args.instruction).read_text()))
    elif args.command == "development":
        development(args.manifest, args.output, args.gpu)
    elif args.command == "audit":
        audit_development(args.output)
    else:
        freeze_development(args.output, args.destination)


if __name__ == "__main__":
    main()
