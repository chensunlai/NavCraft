"""Audit gaps between fitted wall lines with raw geometry and real passage probes.

This is an offline builder step.  A gap is recorded as an opening only when its
interior has raw-mesh clearance and a real robot-width probe crosses it.  The
tool never fills an uncertain gap and never edits the source scene.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .map_exploration import floor_domain, region_image, write_json
from .map_floorplan import RawWallEvidence, segment_points
from .map_geometry import load_scene_geometry
from .map_scene import MapScene
from .map_openings import region_partition


def _compact_view(view):
    """Keep RGB plus a compressed depth evidence array; discard bulky .npy."""
    depth = Path(view["depth"])
    values = np.load(depth)
    compact = depth.with_suffix(".npz")
    np.savez_compressed(compact, depth=values)
    depth.unlink()
    view = dict(view)
    view["depth"] = str(compact)
    view["depth_dtype"] = str(values.dtype)
    return view


def line_gap_candidates(primitives, *, min_width=.45, max_width=1.6, max_offset=.08):
    """Return non-overlapping, nearly collinear gaps between fitted lines."""
    lines = [item for item in primitives if item.get("type") == "line"]
    candidates = []
    for first, left in enumerate(lines):
        a = np.asarray([left["start"], left["end"]], dtype=float)
        tangent = a[1] - a[0]
        length = np.linalg.norm(tangent)
        if length <= 1e-8:
            continue
        tangent /= length
        normal = np.array([-tangent[1], tangent[0]])
        for right in lines[first + 1:]:
            b = np.asarray([right["start"], right["end"]], dtype=float)
            other = b[1] - b[0]
            other_length = np.linalg.norm(other)
            if other_length <= 1e-8:
                continue
            other /= other_length
            if abs(float(tangent @ other)) < math.cos(math.radians(5)):
                continue
            if float(tangent @ other) < 0:
                b = b[::-1]
                other = -other
            if max(abs((b - a.mean(0)) @ normal)) > max_offset:
                continue
            first_interval = np.sort(a @ tangent)
            second_interval = np.sort(b @ tangent)
            if first_interval[1] < second_interval[0]:
                gap = second_interval[0] - first_interval[1]
                segment = np.stack((a[np.argmax(a @ tangent)], b[np.argmin(b @ tangent)]))
            elif second_interval[1] < first_interval[0]:
                gap = first_interval[0] - second_interval[1]
                segment = np.stack((b[np.argmax(b @ tangent)], a[np.argmin(a @ tangent)]))
            else:
                continue
            if min_width <= gap <= max_width:
                candidates.append({
                    "segment": segment.tolist(),
                    "width_m": float(gap),
                    "primitives": [left["id"], right["id"]],
                    "source": "fitted_collinear_wall_endpoint_gap",
                })

    # Multiple fragments can describe one opening. Keep the shortest local
    # candidate at each centre; do not duplicate openings in the room graph.
    result = []
    for item in sorted(candidates, key=lambda value: value["width_m"]):
        centre = np.mean(item["segment"], axis=0)
        if any(np.linalg.norm(centre - np.mean(old["segment"], axis=0)) < .15 for old in result):
            continue
        item["id"] = f"G{len(result) + 1:03d}"
        result.append(item)
    return result


def clear_gap_interval(segment, evidence):
    """Locate a contiguous measured clear interval, not an average over a gap."""
    points = segment_points(segment)
    profile = evidence.profile(points)
    heights = [height for height in evidence.trees if height <= 1.7 + 1e-9]
    body = profile[:, [height <= 1.7 + 1e-9 for height in evidence.trees]]
    clear = body.sum(axis=1) <= 1
    starts = np.flatnonzero(clear & np.r_[True, ~clear[:-1]])
    ends = np.flatnonzero(clear & np.r_[~clear[1:], True])
    intervals = [(float(np.linalg.norm(points[end] - points[start])), start, end)
                 for start, end in zip(starts, ends)]
    width, start, end = max(intervals, default=(0., 0, 0))
    return {"raw_clear_body_fraction": float(clear.mean()),
            "raw_height_support": {str(height): float(values.mean())
                                   for height, values in zip(heights, body.T)},
            "clear_segment": [points[start].tolist(), points[end].tolist()] if width > 0 else None,
            "clear_width_m": width}


def same_opening(first, second):
    """Overlapping evidence spans on the same wall describe one opening."""
    a, b = np.asarray(first["segment"], dtype=float), np.asarray(second["segment"], dtype=float)
    tangent = a[1] - a[0]
    length = np.linalg.norm(tangent)
    other = b[1] - b[0]
    if length < 1e-8 or np.linalg.norm(other) < 1e-8:
        return False
    tangent /= length
    normal = np.array([-tangent[1], tangent[0]])
    if (abs(float(tangent @ (other / np.linalg.norm(other)))) < math.cos(math.radians(8))
            or np.max(abs((b - a.mean(0)) @ normal)) > .2):
        return False
    lo, hi = np.sort((b - a[0]) @ tangent)
    overlap = min(length, hi) - max(0., lo)
    return overlap >= .6 * min(length, hi - lo)


def passed_probe_count(passage, segment):
    """Count distinct, complete successful crossings of this measured span."""
    if passage.get("pass") is not True:
        return 0
    a, b = np.asarray(segment, dtype=float)
    tangent = b - a
    width = np.linalg.norm(tangent)
    if width <= 1e-8:
        return 0
    tangent /= width
    normal = np.array([-tangent[1], tangent[0]])
    crossings = set()
    for probe in passage.get("probes", []):
        if probe.get("pass") is not True:
            continue
        points = [np.asarray(probe.get(key, []), dtype=float)
                  for key in ("start", "requested_end", "actual_end")]
        if any(point.shape != (3,) or not np.isfinite(point).all() for point in points):
            continue
        start, end, actual = [point[[0, 2]] for point in points]
        if np.linalg.norm(actual - end) >= .06:
            continue
        sides = np.array([(start - a) @ normal, (end - a) @ normal])
        if sides.prod() >= 0:
            continue
        crossing = start + sides[0] / (sides[0] - sides[1]) * (end - start)
        if not -1e-8 <= (crossing - a) @ tangent <= width + 1e-8:
            continue
        crossings.add(tuple(sorted((tuple(np.round(points[0], 6)), tuple(np.round(points[1], 6))))))
    return len(crossings)


def merge_audited_openings(existing, audited, audit_path):
    """Retain prior records and attach repeated checks to one opening ID."""
    result = deepcopy(existing)
    identifiers = {item["id"] for item in result}
    added = 0
    for item in audited:
        if item["status"] != "accepted":
            continue
        opening = {**item, "candidate_segment": item["segment"],
                   "segment": item["evidence"]["clear_segment"],
                   "width_m": item["evidence"]["clear_width_m"]}
        duplicate = next((old for old in result if old.get("status") == "accepted"
                          and same_opening(old, opening)), None)
        if duplicate is not None:
            item["duplicate_of"] = duplicate["id"]
            target = duplicate
        else:
            index = 1
            while f"A{index:03d}" in identifiers:
                index += 1
            opening["id"] = f"A{index:03d}"
            opening["sources"] = [{"kind": "fitted_gap"}, {"kind": "raw_mesh"},
                                  {"kind": "actual_robot_width_probe"}]
            identifiers.add(opening["id"])
            result.append(opening)
            target = opening
            added += 1
        item["opening_id"] = target["id"]
        target.setdefault("gap_audit_evidence", []).append({
            "path": str(audit_path), "candidate_id": item["id"],
            "segment": opening["segment"], "evidence": deepcopy(item["evidence"])})
    return result, added


def audit_scene(source, fitted, output, *, gpu=0):
    source, fitted, output = Path(source).resolve(), Path(fitted).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline = json.loads((source / "raw_vote/meta.json").read_text())
    geometry_data = json.loads((fitted / "wall_geometry.json").read_text())
    primitives = geometry_data["primitives"]
    candidates = line_gap_candidates(primitives)
    raw = load_scene_geometry(baseline["scene"], include_semantics=False)
    navmesh = np.asarray(Image.open(source / "navmesh.png")) > 0
    evidence = RawWallEvidence(raw.triangles, baseline["floor_height"], baseline["origin"],
                               baseline["side_m"], navmesh)
    audited = []
    with MapScene(baseline["scene"], origin=baseline["origin"], side=baseline["side_m"], gpu=gpu,
                  floor_range=baseline.get("floor_height_range")) as scene:
        for candidate in candidates:
            clearance = clear_gap_interval(candidate["segment"], evidence)
            probe_segment = clearance["clear_segment"] or candidate["segment"]
            passage = scene.passage(probe_segment, baseline["floor_height"])
            passed_probes = passed_probe_count(passage, probe_segment)
            accepted = clearance["clear_width_m"] >= .4 and passed_probes >= 2
            record = {**candidate, "status": "accepted" if accepted else "needs_review",
                      "evidence": {**clearance,
                                   "passage": passage,
                                   "passed_probe_count": passed_probes,
                                   "acceptance": "contiguous_raw_clear_interval_ge_0.4m_and_two_real_probes",
                                   "width_is_candidate_span_not_measured_door_width": True}}
            if accepted:
                centre = np.mean(probe_segment, axis=0)
                direction = np.diff(np.asarray(probe_segment), axis=0)[0]
                direction /= np.linalg.norm(direction)
                normal = np.array([-direction[1], direction[0]])
                for side in (-1, 1):
                    requested = np.array([centre[0] + side * .7 * normal[0],
                                          baseline["floor_height"],
                                          centre[1] + side * .7 * normal[1]])
                    point = np.asarray(scene.sim.pathfinder.snap_point(requested), dtype=float)
                    if (np.isfinite(point).all()
                            and np.linalg.norm(point[[0, 2]] - requested[[0, 2]]) < .5):
                        views = []
                        for yaw in (0, 90, 180, 270):
                            view = _compact_view(scene.view(
                                point, yaw, output / f"{candidate['id']}-side{side:+d}-yaw{yaw}"))
                            views.append(view)
                        record.setdefault("evidence", {})[f"views_side_{side:+d}"] = views
            audited.append(record)

    existing = json.loads((fitted / "openings.json").read_text())
    combined, added = merge_audited_openings(existing, audited, output / "gap_audit.json")
    accepted = [item for item in combined if item.get("status") == "accepted"]
    write_json(output / "gap_audit.json", {"source": str(source), "fitted": str(fitted),
                                            "candidates": audited,
                                            "accepted_count": added,
                                            "accepted_total_count": len(accepted)})
    write_json(fitted / "openings.before_gap_audit.json", existing)

    domain, _ = floor_domain(raw.triangles, baseline["floor_height"], baseline["origin"],
                              baseline["side_m"], baseline.get("floor_height_range"))
    walls = np.asarray(Image.open(fitted / "walls.png")) < 128
    labels, regions, openings = region_partition(domain, walls, accepted,
                                                  baseline["origin"], .05)
    partitioned = {item["id"]: item for item in openings}
    write_json(fitted / "openings.json", [partitioned.get(item["id"], item) for item in combined])
    for item in regions:
        item["source"] = "floor_domain_partitioned_by_fitted_walls_and_audited_openings"
    write_json(fitted / "regions.before_gap_audit.json", json.loads((fitted / "regions.json").read_text()))
    write_json(fitted / "regions.json", regions)
    Image.fromarray(labels.astype(np.uint16)).save(fitted / "regions_audited_labels.png")
    region_image(labels, walls).save(fitted / "regions.png")
    meta_path = fitted / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta.setdefault("counts", {})["audited_openings"] = len(accepted)
        meta["counts"]["region_candidates"] = len(regions)
        meta["gap_audit"] = {"path": str(output / "gap_audit.json"),
                             "accepted_count": added,
                             "accepted_total_count": len(accepted),
                             "status": "development_pending_structural_review"}
        write_json(meta_path, meta)
    return audited


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--fitted", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0, choices=(0, 1))
    args = parser.parse_args()
    result = audit_scene(args.source, args.fitted, args.output, gpu=args.gpu)
    print(json.dumps({"output": args.output,
                      "candidates": len(result),
                      "accepted": sum(item["status"] == "accepted" for item in result)},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
