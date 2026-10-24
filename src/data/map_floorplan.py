"""Fit architectural wall geometry and validate it against independent raw mesh.

Runs on individual development scenes only. Raster projections are evidence;
the exported wall image is rendered exclusively from fitted primitives.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from .map_exploration import (floor_domain, overlay_image, raster_pixels,
                              region_image, write_json)
from .map_geometry import load_scene_geometry, multiheight_slices
from .map_instances import load_semantic_instances
from .map_quality import erode_one, line_intersects_wall, validate_map

HEIGHTS = (.3, .6, .9, 1.2, 1.5, 1.8, 2.1, 2.4)
PARAMETERS = {"raw_heights_m": HEIGHTS, "support_distance_m": .10,
              "raw_sample_spacing_m": .025, "fit_max_error_m": .05,
              "fit_min_length_m": .15, "fit_max_gap_m": .12,
              "junction_max_shift_m": .15, "minimum_support_fraction": .65,
              "minimum_support_heights": 2, "body_support_max_height_m": 1.8,
              "pixel_size_m": .05}


def segment_points(segment, spacing=.025):
    a, b = np.asarray(segment, dtype=float)
    length = np.linalg.norm(b - a)
    return a + np.linspace(0, 1, max(2, math.ceil(length / spacing) + 1))[:, None] * (b - a)


class RawWallEvidence:
    """A semantic label cannot manufacture raw surface or passage evidence."""
    def __init__(self, raw_triangles, floor, origin, side, navmesh):
        self.origin, self.side, self.floor = np.asarray(origin), float(side), float(floor)
        self.navmesh = np.asarray(navmesh, dtype=bool)
        self.interior = erode_one(navmesh)
        self.layers = multiheight_slices(raw_triangles, floor, HEIGHTS)
        self.body_heights = np.array([height <= 1.8 for height in self.layers])
        self.trees = {}
        for height, segments in self.layers.items():
            points = [segment_points(segment) for segment in segments]
            self.trees[height] = cKDTree(np.concatenate(points)) if points else None

    def profile(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        return np.stack([tree.query(points)[0] <= .10 if tree is not None
                         else np.zeros(len(points), dtype=bool) for tree in self.trees.values()], axis=1)

    def support(self, segment):
        points = segment_points(segment)
        hits = self.profile(points)
        fractions = hits.mean(axis=0)
        supported = hits[:, self.body_heights].sum(axis=1) >= 2
        return {"pass": bool(supported.mean() >= .65),
                "supported_fraction": float(supported.mean()),
                "height_support_fractions": dict(zip(map(str, self.trees), fractions.tolist())),
                "raw_source": "independent_render_mesh_height_cuts",
                "support_scope": "wall_body_heights_at_or_below_1.8m",
                "sample_count": len(points), "distance_tolerance_m": .10}

    def not_blocks_navmesh(self, segment):
        intersects = line_intersects_wall(self.interior, *np.asarray(segment),
                                         origin=self.origin, resolution=.05)
        return {"pass": not intersects, "intersects_walkable_interior": intersects,
                "source": "saved_rebuilt_lite_navmesh"}

    def cell_values(self, mask, points):
        cells = np.floor((np.asarray(points) - self.origin) / .05).astype(int)
        valid = (cells >= 0).all(1) & (cells[:, 0] < mask.shape[1]) & (cells[:, 1] < mask.shape[0])
        result = np.zeros(len(cells), dtype=bool)
        result[valid] = mask[cells[valid, 1], cells[valid, 0]]
        return result


def primitive_points(primitive, spacing=.025):
    if primitive["type"] == "line":
        return segment_points([primitive["start"], primitive["end"]], spacing)
    radius = primitive["radius_m"]
    start = math.radians(primitive["start_angle_degrees"])
    sweep = math.radians(primitive.get("sweep_degrees", primitive["end_angle_degrees"] - primitive["start_angle_degrees"]))
    angles = np.linspace(start, start + sweep, max(3, math.ceil(abs(sweep) * radius / spacing) + 1))
    return np.asarray(primitive["center"]) + radius * np.column_stack([np.cos(angles), np.sin(angles)])


def split_primitive(primitive, points, selected):
    """Retain fitted analytic geometry while carving real passable intervals."""
    starts = np.flatnonzero(selected & np.r_[True, ~selected[:-1]])
    ends = np.flatnonzero(selected & np.r_[~selected[1:], True])
    result = []
    for start, end in zip(starts, ends):
        length = float(np.linalg.norm(np.diff(points[start:end + 1], axis=0), axis=1).sum())
        if length < .12:
            continue
        value = copy.deepcopy(primitive)
        value["id"] = primitive["id"] + f"-part{len(result)}"
        value["start"], value["end"] = points[start].tolist(), points[end].tolist()
        value["endpoints"] = [value["start"], value["end"]]
        value["length_m"] = length
        if primitive["type"] == "arc":
            angle = primitive["start_angle_degrees"]
            sweep = primitive.get("sweep_degrees", primitive["end_angle_degrees"] - angle)
            value["start_angle_degrees"] = angle + sweep * start / (len(points) - 1)
            value["end_angle_degrees"] = angle + sweep * end / (len(points) - 1)
            value["sweep_degrees"] = value["end_angle_degrees"] - value["start_angle_degrees"]
            value["start_angle_radians"] = math.radians(value["start_angle_degrees"])
            value["end_angle_radians"] = math.radians(value["end_angle_degrees"])
            value["sweep_radians"] = math.radians(value["sweep_degrees"])
            value["length_m"] = abs(value["sweep_radians"]) * value["radius_m"]
        result.append(value)
    return result


def verify_primitive(primitive, evidence):
    points = primitive_points(primitive)
    profile = evidence.profile(points)
    supported = profile[:, evidence.body_heights].sum(1) >= 2
    # Only clear *measured walkable* intersections. Missing scan support alone
    # never proves a door; a semantically suggested window is not auto-opened.
    passable = evidence.cell_values(evidence.navmesh, points) & ~supported
    intervals = []
    for start, end in zip(np.flatnonzero(passable & np.r_[True, ~passable[:-1]]),
                          np.flatnonzero(passable & np.r_[~passable[1:], True])):
        low, high = start, end
        while low > 0 and not supported[low - 1]:
            low -= 1
        while high + 1 < len(points) and not supported[high + 1]:
            high += 1
        passable[low:high + 1] = True
        intervals.append({"start": points[low].tolist(), "end": points[high].tolist(),
                          "length_m": float(np.linalg.norm(points[high] - points[low])),
                          "source": "raw_clear_interval_connected_to_walkable_crossing"})
    retained = ~passable
    nonpassable_support = supported[retained].mean() if retained.any() else 0.
    # A proposed long line must have enough independent evidence as a whole.
    # Unsupported portions remain explicitly reported rather than being called GT.
    accepted = bool(retained.any() and nonpassable_support >= .65)
    decision = {"primitive_id": primitive["id"], "accepted": accepted,
                "raw_supported_fraction": float(supported.mean()),
                "retained_supported_fraction": float(nonpassable_support),
                "carved_passages": intervals, "unsupported_sample_count": int((~supported & retained).sum()),
                "semantic_alone_accepted": False}
    values = split_primitive(primitive, points, retained) if accepted else []
    for value in values:
        value["evidence"] = decision
    return values, decision


def fit_scene(source, output):
    """Fit and inspect one saved development scene, without bulk generation."""
    from .map_wall_fit import fit_wall_primitives
    from .map_wall_structure import reconnect_primitives, rasterize_primitives, structure_svg
    from .map_wall_recovery import recover_collinear_gaps, raw_gap_support
    from .map_window_boundaries import build_window_boundaries
    from .map_openings import region_partition
    from .map_leaks import review_leak_pairs

    started = time.monotonic()
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline = json.loads((source / "raw_vote/meta.json").read_text())
    scene = baseline["scene"]
    floor, origin, side = baseline["floor_height"], baseline["origin"], baseline["side_m"]
    geometry = load_scene_geometry(scene, include_semantics=False)
    instances, prior_meta = load_semantic_instances(scene, wanted_categories={"wall", "door", "window", "furniture"})
    # Window/door labels remain separate priors: curtain meshes or open leaves
    # must not become permanent walls. The render mesh independently checks
    # every proposed architectural wall.
    wall_parts, provenance = [], []
    for instance in instances:
        if instance.category != "wall":
            continue
        for height, segments in multiheight_slices(instance.triangles, floor,
                 relative_heights=(.5, .9, 1.3, 1.7, 2., 2.3)).items():
            wall_parts.extend(segments)
            provenance.extend({"instance_id": instance.id, "region_id": instance.region_id,
                               "category": instance.category,
                               "relative_height_m": height} for _ in segments)
    if not wall_parts:
        raise ValueError("No architectural wall candidates on the selected floor")
    segments = np.asarray(wall_parts)
    np.savez_compressed(output / "candidate_segments.npz", segments=segments)
    fit = fit_wall_primitives(segments, max_error=.05, min_length=.15, max_gap=.12)
    write_json(output / "candidate_fit.json", fit)
    write_json(output / "candidate_provenance.json", provenance)
    Image.fromarray(np.where(rasterize_primitives(fit["primitives"], origin, side), 0, 255)
                    .astype(np.uint8)).save(output / "candidate_walls.png")
    navmesh = np.asarray(Image.open(source / "navmesh.png")) > 0
    evidence = RawWallEvidence(geometry.triangles, floor, origin, side, navmesh)
    accepted, decisions = [], []
    for index, primitive in enumerate(fit["primitives"]):
        primitive.setdefault("id", f"W{index:04d}")
        if "start" not in primitive:
            primitive["start"], primitive["end"] = primitive["endpoints"]
        values, checked = verify_primitive(primitive, evidence)
        accepted.extend(values)
        decisions.append(checked)
    write_json(output / "geometry_decisions.json", decisions)
    portals = json.loads((source / "opening_candidates.json").read_text())
    portals = [p for p in portals if p["status"] == "accepted"]
    priors = []
    for instance in instances:
        if instance.category not in {"door", "window"}:
            continue
        heights = instance.triangles[:, :, 1]
        selected = instance.triangles[(heights.max(1) >= floor + .2) &
                                      (heights.min(1) <= floor + 2.4)]
        if not len(selected):
            continue
        points = selected[:, :, [0, 2]].reshape(-1, 2)
        centre = points.mean(0)
        _, _, axes = np.linalg.svd(points - centre, full_matrices=False)
        tangent = axes[0]
        along = (points - centre) @ tangent
        if np.ptp(along) >= .1:
            priors.append({"id": f"{instance.category}:{instance.id}",
                           "category": instance.category,
                           "segment": (centre + np.array([along.min(), along.max()])[:, None] * tangent).tolist(),
                           "source": "semantic_instance_projection_prior_not_verified_geometry"})
    recovery = recover_collinear_gaps(accepted,
                    raw_support=lambda segment: raw_gap_support(segment, evidence.layers),
                    not_blocks_navmesh=evidence.not_blocks_navmesh,
                    semantic_priors=priors, portals=portals)
    write_json(output / "gap_recovery.json", recovery)
    accepted = recovery["primitives"]
    windows = build_window_boundaries(geometry.triangles, accepted,
                    [instance for instance in instances if instance.category == "window"],
                    floor_height=floor,
                    passage=lambda segment: {"pass": False, "probes": [],
                        "reason": "robot_crossing_not_probed_in_offline_frame_reconstruction"},
                    not_blocks_navmesh=evidence.not_blocks_navmesh)
    write_json(output / "window_reconstruction.json", windows)
    accepted = windows["primitives"]
    structure = reconnect_primitives(accepted, raw_support=evidence.support,
                    not_blocks_navmesh=evidence.not_blocks_navmesh, portals=portals,
                    max_shift_m=.15, collinear_gap_m=.4)
    primitives = structure["primitives"]
    for primitive in primitives:
        primitive.setdefault("thickness_m", None)
        primitive.setdefault("evidence_kind", "raw_surface_verified")
        primitive.setdefault("representation", "reconstructed_wall_alignment"
                             if primitive["evidence_kind"] == "structural_inference"
                             else "fitted_wall_surface_line")
    walls = rasterize_primitives(primitives, origin, side, resolution=.05)
    points = np.concatenate([primitive_points(primitive) for primitive in primitives])
    pairs = [pair for pair in json.loads((source / "geodesic_pairs.json").read_text())
             if not pair.get("out_of_floor_band", False)]
    quality = validate_map(walls, navmesh, origin=origin, resolution=.05, pairs=pairs,
                          construction_points=points,
                          raster_pixels=raster_pixels(points, origin, side, len(walls)))
    quality["rgb_alignment"] = baseline["quality"]["rgb_alignment"]
    flags = [i for i, pair in enumerate(quality["leaks"]["pairs"]) if pair["flagged"]]
    domain, _ = floor_domain(geometry.triangles, floor, origin, side, baseline.get("floor_height_range"))
    reviews = review_leak_pairs(geometry.triangles, pairs, floor, origin, side, domain, walls, flagged_indices=flags)
    labels, regions, openings = region_partition(domain, walls, portals, origin, .05)
    rgb = np.asarray(Image.open(source / "topdown.png").convert("RGB"))
    Image.fromarray(np.where(walls, 0, 255).astype(np.uint8)).save(output / "walls.png")
    Image.fromarray(rgb).save(output / "topdown.png")
    overlay_image(rgb, navmesh, walls).save(output / "overlay.png")
    region_image(labels, walls).save(output / "regions.png")
    (output / "walls.svg").write_text(structure_svg(primitives, origin, side, resolution=.05))
    write_json(output / "wall_geometry.json", {"parameters": PARAMETERS, "primitives": primitives,
                    "junctions": structure["nodes"], "connections": structure["connections"],
                    "unexplained_endpoints": structure["unexplained_endpoints"],
                    "coordinate_frame": "HabitatXZ: right+X, down+Z", "status": "development_not_qualified"})
    write_json(output / "connection_rejections.json", structure["rejected"])
    write_json(output / "quality.json", quality)
    write_json(output / "leak_geometry_review.json", reviews)
    write_json(output / "regions.json", regions)
    write_json(output / "openings.json", openings)
    write_json(output / "semantic_sources.json", prior_meta)
    write_json(output / "raw_source.json", geometry.metadata)
    meta = {"scene": scene, "floor_height": floor, "origin": origin, "side_m": side,
            "pixel_size_m": .05, "method": "fitted_architectural_walls", "parameters": PARAMETERS,
            "wall_sha256": hashlib.sha256((output / "walls.png").read_bytes()).hexdigest(),
            "implementation_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ("map_floorplan.py", "map_wall_fit.py", "map_wall_structure.py",
                                     "map_wall_recovery.py", "map_window_boundaries.py",
                                     "map_instances.py", "map_geometry.py", "map_openings.py")},
            "status": "development_pending_structural_review", "quality": quality,
            "counts": {"input_segments": len(segments), "fitted_primitives": len(fit["primitives"]),
                "verified_primitives": len(accepted), "final_primitives": len(primitives),
                "junctions": len(structure["nodes"]), "connections": len(structure["connections"]),
                "unexplained_endpoints": len(structure["unexplained_endpoints"]), "region_candidates": len(regions),
                "reconstructed_window_boundaries": len(windows["accepted"]),
                "window_candidates_pending": sum(item["status"] == "needs_review" for item in windows["candidates"])},
            "seconds": time.monotonic() - started}
    write_json(output / "meta.json", meta)
    before = source / "raw_vote/walls.png"
    page = ('<!doctype html><meta charset="utf-8"><title>Architectural wall fitting</title>'
        '<style>body{font:16px system-ui;margin:28px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}img{width:100%;background:#fff;border:1px solid #ddd}pre{white-space:pre-wrap}</style>'
        '<h1>Fitted architectural wall plan — development candidate</h1><p>Geometry-verified candidates; structural correctness still needs inspection. Raw projections are not final wall geometry.</p>'
        '<div class="grid"><section><h2>Previous projection</h2><img src="baseline.png"></section>'
        '<section><h2>Fitted straight / curved walls</h2><img src="walls.svg"></section>'
        '<section><h2>Actual RGB + NavMesh + fitted walls</h2><img src="overlay.png"></section>'
        '<section><h2>Candidate room regions</h2><img src="regions.png"></section></div>'
        '<pre>' + html.escape(json.dumps(meta, ensure_ascii=False, indent=2)) + '</pre>')
    Image.open(before).save(output / "baseline.png")
    (output / "index.html").write_text(page)
    print(json.dumps({"output": str(output), **meta["counts"], "seconds": meta["seconds"]}), flush=True)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    fit_scene(args.source, args.output)


if __name__ == "__main__":
    main()
