"""Infer one CAD wall axis from independently supported, outward-facing sides.

Only verified architectural line primitives belong here. Close parallel lines
are candidates, not proof: corridor faces point into their intervening strip.
Circles/arcs pass through; coincident scan-line deduplication is a separate step.
"""
from copy import deepcopy
import math

import numpy as np


PARAMETERS = {"angle_degrees": 3., "thickness_range_m": [.04, .40],
              "minimum_overlap_m": .30, "plane_tolerance_m": .025,
              "minimum_common_height_m": .80, "minimum_area_fraction": .55,
              "minimum_along_coverage": .90, "minimum_outward_fraction": .80,
              "minimum_review_face_area_m2": .20}


def _intervals(values, tolerance=1e-8):
    result = []
    for low, high in sorted(values):
        if high <= low:
            continue
        if result and low <= result[-1][1] + tolerance:
            result[-1][1] = max(high, result[-1][1])
        else:
            result.append([float(low), float(high)])
    return result


def _intersections(first, second):
    return _intervals((max(a, c), min(b, d)) for a, b in first for c, d in second)


def _subtract(interval, covered):
    low, high = interval
    result = []
    for start, end in _intervals(covered):
        if start > low + 1e-8:
            result.append([low, min(start, high)])
        low = max(low, end)
    if low < high - 1e-8:
        result.append([low, high])
    return result


def _line(item):
    points = np.asarray([item["start"], item["end"]], dtype=float)
    if points.shape != (2, 2) or not np.isfinite(points).all():
        raise ValueError("Wall lines require finite world XZ endpoints")
    direction = points[1] - points[0]
    length = float(np.linalg.norm(direction))
    if length <= 1e-8:
        raise ValueError("Wall lines must have positive length")
    tangent = direction / length
    if tangent[0] < -1e-8 or (abs(tangent[0]) <= 1e-8 and tangent[1] < 0):
        tangent = -tangent
    return points, tangent, length


def _clip_area(triangle, low, high, bottom, top):
    polygon = list(triangle)
    for axis, boundary, sign in [(0, low, 1), (0, high, -1), (1, bottom, 1), (1, top, -1)]:
        clipped = []
        for first, second in zip(polygon, polygon[1:] + polygon[:1]):
            inside = sign * (first[axis] - boundary) >= 0
            other_inside = sign * (second[axis] - boundary) >= 0
            if inside:
                clipped.append(first)
            if inside != other_inside:
                t = (boundary - first[axis]) / (second[axis] - first[axis])
                clipped.append(first + t * (second - first))
        polygon = clipped
        if len(polygon) < 3:
            return 0.
    points = np.asarray(polygon)
    return float(abs(np.sum(points[:, 0] * np.roll(points[:, 1], -1)
                            - points[:, 1] * np.roll(points[:, 0], -1))) / 2)


def _face_evidence(ids, along, heights, normal_dot, expected_sign, interval, band):
    low, high = interval
    bottom, top = band
    areas, kept = [], []
    for index in ids:
        area = _clip_area(np.column_stack([along[index], heights[index]]), low, high, bottom, top)
        if area > 1e-8:
            areas.append(area)
            kept.append(int(index))
    areas = np.asarray(areas)
    kept = np.asarray(kept, dtype=int)
    total = float(areas.sum())
    selected = normal_dot[kept] * expected_sign >= math.cos(math.radians(15))
    outward = float(areas[selected].sum() / total) if total else 0.
    coverage = _intervals((max(low, float(along[i].min())), min(high, float(along[i].max()))) for i in kept)
    along_fraction = sum(b - a for a, b in coverage) / (high - low)
    area_fraction = total / ((high - low) * (top - bottom))
    return {"pass": bool(outward >= PARAMETERS["minimum_outward_fraction"]
                         and area_fraction >= PARAMETERS["minimum_area_fraction"]
                         and along_fraction >= PARAMETERS["minimum_along_coverage"]),
            "raw_face_ids": kept.tolist(), "projected_area_m2": total,
            "area_fraction": area_fraction, "along_coverage": along_fraction,
            "outward_area_fraction": outward, "expected_normal_sign": expected_sign,
            "area_weighted_normal_dot": float(areas @ normal_dot[kept] / total) if total else None}


def _raw_pair(triangles, normals, points, tangent, interval, height_range):
    normal = np.array([-tangent[1], tangent[0]])
    offsets = [float((p @ normal).mean()) for p in points]
    thickness = offsets[1] - offsets[0]
    tolerance = min(PARAMETERS["plane_tolerance_m"], thickness * .20)
    along = triangles[:, :, [0, 2]] @ tangent
    across = triangles[:, :, [0, 2]] @ normal
    heights = triangles[:, :, 1]
    dots = normals[:, [0, 2]] @ normal
    eligible = ((abs(normals[:, 1]) <= .25) & (abs(dots) >= math.cos(math.radians(15)))
                & (along.max(1) >= interval[0]) & (along.min(1) <= interval[1]))
    if height_range is not None:
        eligible &= (heights.max(1) >= height_range[0]) & (heights.min(1) <= height_range[1])
    ids = [np.flatnonzero(eligible & (abs(across - offset).max(1) <= tolerance)) for offset in offsets]
    bands = []
    for indices in ids:
        values = [(float(heights[i].min()), float(heights[i].max())) for i in indices]
        if height_range is not None:
            values = [(max(a, height_range[0]), min(b, height_range[1])) for a, b in values]
        bands.append(_intervals(values, tolerance=.025))
    common = _intersections(*bands)
    band = max(common, key=lambda pair: pair[1] - pair[0], default=None)
    record = {"raw_source": "original_render_mesh_triangle_winding", "normal_xz": normal.tolist(),
              "plane_distance_tolerance_m": tolerance, "face_offsets_m": offsets,
              "wall_thickness_m": thickness, "common_vertical_intervals_m": common,
              "height_range_filter_m": list(height_range) if height_range is not None else None}
    if band is None or band[1] - band[0] < PARAMETERS["minimum_common_height_m"]:
        return {**record, "pass": False, "reason": "insufficient_common_raw_vertical_extent"}
    low = _face_evidence(ids[0], along, heights, dots, -1, interval, band)
    high = _face_evidence(ids[1], along, heights, dots, 1, interval, band)
    return {**record, "pass": bool(low["pass"] and high["pass"]), "support_height_band_m": band,
            "low_face": low, "high_face": high,
            "reason": None if low["pass"] and high["pass"] else "raw_area_or_outward_normals_not_confirmed"}


def _reviewable_area_failure(proof):
    """A review may resolve sparse surface coverage, never absent geometry."""
    if proof["pass"] or "support_height_band_m" not in proof:
        return False
    band = proof["support_height_band_m"]
    faces = [proof["low_face"], proof["high_face"]]
    return bool(band[1] - band[0] >= PARAMETERS["minimum_common_height_m"]
                and any(face["area_fraction"] < PARAMETERS["minimum_area_fraction"] for face in faces)
                and all(face["outward_area_fraction"] >= PARAMETERS["minimum_outward_fraction"]
                        and face["along_coverage"] >= PARAMETERS["minimum_along_coverage"]
                        and face["projected_area_m2"] >= PARAMETERS["minimum_review_face_area_m2"]
                        for face in faces))


def _portal_crossing(segment, portal):
    a, b = np.asarray(segment), np.asarray(portal, dtype=float)
    d, e = a[1] - a[0], b[1] - b[0]
    cross = lambda p, q: float(p[0] * q[1] - p[1] * q[0])
    determinant = cross(d, e)
    if abs(determinant) < 1e-10:
        if abs(cross(b[0] - a[0], d)) > 1e-8:
            return False
        values = sorted(float((point - a[0]) @ d / (d @ d)) for point in b)
        return min(1., values[1]) - max(0., values[0]) > 1e-8
    t, u = cross(b[0] - a[0], e) / determinant, cross(b[0] - a[0], d) / determinant
    return -1e-8 <= t <= 1 + 1e-8 and 1e-8 < u < 1 - 1e-8


def _endcaps(lines, excluded, triangles, normals, tangent, normal, low, high, intervals, band):
    """Remove only measured outward caps across the two confirmed wall faces."""
    caps = []
    thickness = high - low
    tolerance = min(.025, thickness * .20)
    along = triangles[:, :, [0, 2]] @ normal
    across = triangles[:, :, [0, 2]] @ tangent
    heights = triangles[:, :, 1]
    dots = normals[:, [0, 2]] @ tangent
    for index, (points, direction, length) in lines.items():
        if index in excluded or abs(direction @ tangent) > math.sin(math.radians(5)):
            continue
        side_positions = sorted(points @ normal)
        if max(abs(side_positions[0] - low), abs(side_positions[1] - high)) > tolerance:
            continue
        if abs(length - thickness) > tolerance:
            continue
        for start, end in intervals:
            for position, sign in [(start, -1), (end, 1)]:
                if abs(points @ tangent - position).max() > tolerance:
                    continue
                ids = np.flatnonzero((abs(normals[:, 1]) <= .25)
                        & (abs(dots) >= math.cos(math.radians(15)))
                        & (abs(across - position).max(1) <= tolerance)
                        & (along.max(1) >= low) & (along.min(1) <= high))
                proof = _face_evidence(ids, along, heights, dots, sign, [low, high], band)
                if proof["pass"]:
                    caps.append({"index": index, "axis_position_m": position,
                                 "raw_evidence": proof, "source_role": "measured_wall_endcap"})
    return caps


def _occupies_portal(segment, portal, tangent, normal, low, high):
    if _portal_crossing(segment, portal):
        return True
    # A portal may be recorded on either face. Moving the display line to the
    # wall centre must not bypass that opening just because its offset changed.
    points = np.asarray(portal, dtype=float)
    positions, offsets = points @ tangent, points @ normal
    axis_positions = segment @ tangent
    overlap = min(positions.max(), axis_positions.max()) - max(positions.min(), axis_positions.min())
    return bool(overlap > 1e-8 and offsets.min() <= high + .025 and offsets.max() >= low - .025)


def _covered_fragments(lines, excluded, tangent, normal, low, high, intervals, portals):
    """Find redundant verified scan edges inside this finite physical wall.

    No axis extension is allowed, even by the normal-direction fit tolerance.
    Sources already belonging to another supported wall pair are excluded.
    """
    tolerance = min(PARAMETERS["plane_tolerance_m"], (high - low) * .20)
    covered = []
    for index, (points, _, _) in lines.items():
        if index in excluded:
            continue
        offsets, positions = points @ normal, points @ tangent
        if offsets.min() < low - tolerance or offsets.max() > high + tolerance:
            continue
        interval = next(([start, end] for start, end in intervals
                         if positions.min() >= start - 1e-8 and positions.max() <= end + 1e-8), None)
        if interval is None or any(_occupies_portal(points, p["segment"], tangent, normal, low, high)
                                   for p in portals if p.get("status", "accepted") != "rejected"):
            continue
        covered.append({"index": index, "source_role": "covered_wall_volume_scan_edge",
                        "along_interval_m": [float(positions.min()), float(positions.max())],
                        "covered_by_axis_interval_m": interval,
                        "normal_extent_m": [float(offsets.min()), float(offsets.max())],
                        "physical_wall_face_offsets_m": [float(low), float(high)],
                        "normal_fit_tolerance_m": tolerance,
                        "axis_extended": False})
    return covered


def merge_wall_faces(primitives, raw_triangles, *, not_blocks_navmesh=None, portals=(), height_range=None,
                     review_pair=None):
    """Return axes for outward-supported wall sides and known NavMesh.

    Normals must point away from the strip between the two sides. The occupied
    union of connected source intervals defines the finite axis: supported
    single-face tails use a stated constant-axis inference, never bridge a gap.
    ``height_range`` is optional absolute Y, not one hard-coded global floor.
    Missing NavMesh context leaves candidates pending. Rendering thickness stays
    unset; physical thickness is evidence, not the CAD line's display width.
    ``review_pair(record)`` may confirm same-wall identity only when insufficient
    raw area fraction is the sole geometric failure. It must return an explicit
    boolean ``pass`` and auditable ``evidence`` including its review source.
    Such acceptance is structural inference, never a pure geometry pass; raw
    normals, common height, surface area, portals and NavMesh remain mandatory.
    Fully covered unpaired scan edges are absorbed into that finite wall slab
    with separate provenance; they cannot extend its axis or cross a gap.
    """
    original = deepcopy(list(primitives))
    if len({p["id"] for p in original}) != len(original):
        raise ValueError("Wall primitive IDs must be unique")
    triangles = np.asarray(raw_triangles, dtype=float).reshape(-1, 3, 3)
    if not np.isfinite(triangles).all():
        raise ValueError("Raw triangles must be finite")
    if height_range is not None:
        if len(height_range) != 2 or not all(math.isfinite(v) for v in height_range) or height_range[0] >= height_range[1]:
            raise ValueError("height_range requires increasing finite absolute Y bounds")
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    normals = np.divide(cross, lengths[:, None], out=np.zeros_like(cross), where=lengths[:, None] > 1e-10)
    lines = {i: _line(p) for i, p in enumerate(original) if p["type"] == "line"}
    pending, pairs = [], []
    parent = list(range(len(original)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i in lines:
        for j in lines:
            if j <= i:
                continue
            first, t1, _ = lines[i]
            second, t2, _ = lines[j]
            if abs(t1 @ t2) < math.cos(math.radians(PARAMETERS["angle_degrees"])):
                continue
            if t1 @ t2 < 0:
                t2 = -t2
            tangent = (t1 + t2) / np.linalg.norm(t1 + t2)
            normal = np.array([-tangent[1], tangent[0]])
            offsets = [float((p @ normal).mean()) for p in (first, second)]
            low, high = (i, j) if offsets[0] < offsets[1] else (j, i)
            thickness = abs(offsets[1] - offsets[0])
            if not PARAMETERS["thickness_range_m"][0] <= thickness <= PARAMETERS["thickness_range_m"][1]:
                continue
            interval = [max(float((p @ tangent).min()) for p in (first, second)),
                        min(float((p @ tangent).max()) for p in (first, second))]
            if interval[1] - interval[0] < PARAMETERS["minimum_overlap_m"]:
                continue
            record = {"source_primitive_ids": [original[low]["id"], original[high]["id"]],
                      "overlap_interval_m": interval, "wall_thickness_m": thickness}
            if any(np.ptp(p @ normal) > .03 for p in (first, second)):
                pending.append({**record, "reason": "faces_drift_from_a_common_parallel_wall"})
                continue
            proof = _raw_pair(triangles, normals, [lines[low][0], lines[high][0]], tangent, interval, height_range)
            record.update(strict_geometry_pass=proof["pass"], strict_failure=proof["reason"],
                          acceptance="raw_geometry" if proof["pass"] else None)
            if not proof["pass"]:
                if review_pair is not None and _reviewable_area_failure(proof):
                    review = review_pair(deepcopy({**record, "raw_evidence": proof}))
                    if (not isinstance(review, dict) or type(review.get("pass")) is not bool
                            or review.get("evidence") is None):
                        raise ValueError("review_pair must return explicit boolean pass and auditable evidence")
                    record["same_wall_review"] = deepcopy(review)
                    if review["pass"]:
                        record["acceptance"] = "structural_inference"
                if record["acceptance"] is None:
                    pending.append({**record, "reason": proof["reason"], "raw_evidence": proof})
                    continue
            pairs.append({**record, "indices": [low, high], "raw_evidence": proof})
            parent[find(high)] = find(low)

    groups = {}
    for pair in pairs:
        groups.setdefault(find(pair["indices"][0]), []).append(pair)
    merged, replacements = [], {}
    consumed = set()
    paired_indices = {index for pair in pairs for index in pair["indices"]}
    for group in groups.values():
        indices = sorted({index for pair in group for index in pair["indices"]})
        record = {"source_primitive_ids": [original[i]["id"] for i in indices],
                  "raw_pairs": [{k: v for k, v in pair.items() if k != "indices"} for pair in group]}
        record["strict_geometry_pass"] = all(pair["raw_evidence"]["pass"] for pair in group)
        record["acceptance"] = "raw_geometry" if record["strict_geometry_pass"] else "structural_inference"
        reference = lines[indices[0]][1]
        tangent = sum(lines[i][1] * (1 if lines[i][1] @ reference >= 0 else -1) * lines[i][2]
                      for i in indices)
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        offsets = {i: float((lines[i][0] @ normal).mean()) for i in indices}
        low_indices = {min(pair["indices"], key=lambda i: offsets[i]) for pair in group}
        high_indices = {max(pair["indices"], key=lambda i: offsets[i]) for pair in group}
        if low_indices & high_indices or any(np.ptp([offsets[i] for i in side]) > .035 for side in (low_indices, high_indices)):
            pending.append({**record, "reason": "inconsistent_wall_side_or_thickness_group"})
            continue
        low = np.average([offsets[i] for i in low_indices], weights=[lines[i][2] for i in low_indices])
        high = np.average([offsets[i] for i in high_indices], weights=[lines[i][2] for i in high_indices])
        intervals = _intervals((float((lines[i][0] @ tangent).min()), float((lines[i][0] @ tangent).max())) for i in indices)
        paired = _intervals((max(float((lines[i][0] @ tangent).min()) for i in pair["indices"]),
                             min(float((lines[i][0] @ tangent).max()) for i in pair["indices"])) for pair in group)
        band = max((pair["raw_evidence"]["support_height_band_m"] for pair in group), key=lambda p: p[1] - p[0])
        caps = _endcaps(lines, paired_indices | consumed, triangles, normals, tangent, normal,
                        low, high, intervals, band)
        cap_indices = {cap["index"] for cap in caps}
        record["paired_face_primitive_ids"] = record["source_primitive_ids"]
        record["cap_primitive_ids"] = [original[i]["id"] for i in sorted(cap_indices)]
        record["endcap_evidence"] = [{**{k: v for k, v in cap.items() if k != "index"},
                                       "primitive_id": original[cap["index"]]["id"]} for cap in caps]
        covered = _covered_fragments(lines, paired_indices | consumed | cap_indices, tangent, normal,
                                     low, high, intervals, portals)
        covered_indices = {fragment["index"] for fragment in covered}
        record["covered_primitive_ids"] = [original[i]["id"] for i in sorted(covered_indices)]
        record["covered_fragment_evidence"] = [{**{k: v for k, v in fragment.items() if k != "index"},
                                                "primitive_id": original[fragment["index"]]["id"]}
                                               for fragment in covered]
        source_indices = set(indices) | cap_indices | covered_indices
        record["source_primitive_ids"] = [original[i]["id"] for i in sorted(source_indices)]
        proposed, checks, reason = [], [], None
        for start, end in intervals:
            segment = np.array([start, end])[:, None] * tangent + (low + high) / 2 * normal
            if any(_occupies_portal(segment, p["segment"], tangent, normal, low, high)
                   for p in portals if p.get("status", "accepted") != "rejected"):
                reason = "known_portal_would_be_occupied"
                break
            if not_blocks_navmesh is None:
                reason = "navmesh_context_required"
                break
            check = not_blocks_navmesh(segment.tolist())
            if not isinstance(check, dict) or type(check.get("pass")) is not bool:
                raise ValueError("not_blocks_navmesh must return a dict containing explicit boolean pass")
            checks.append(check)
            if not check["pass"]:
                reason = "axis_would_occupy_measured_navmesh"
                break
            identifier = f"AXIS{len(merged) + 1:04d}-{len(proposed)}"
            while any(p["id"] == identifier for p in original):
                identifier += "-axis"
            single = _subtract([start, end], _intersections([[start, end]], paired))
            proposed.append({"id": identifier, "type": "line", "start": segment[0].tolist(), "end": segment[1].tolist(),
                             "endpoints": segment.tolist(), "length_m": end - start,
                             "source_primitive_ids": record["source_primitive_ids"],
                             "source_segment_indices": sorted({s for i in source_indices for s in original[i].get("source_segment_indices", [])}),
                             "representation": "wall_axis",
                             "evidence_kind": ("outward_raw_face_pair_and_constant_axis_inference"
                                               if record["strict_geometry_pass"]
                                               else "reviewed_same_wall_structural_inference"),
                             "thickness_m": None, "wall_thickness_m": float(high - low),
                             "axis_evidence": {"acceptance": record["acceptance"],
                                               "strict_geometry_pass": record["strict_geometry_pass"],
                                               "paired_intervals_m": _intersections([[start, end]], paired),
                                               "single_face_intervals_m": single,
                                               "single_face_tail_assumption": "constant axis offset along existing continuous architectural line; local tail thickness unmeasured",
                                               "normal_xz": normal.tolist(), "raw_pairs": record["raw_pairs"],
                                               "endcap_evidence": record["endcap_evidence"],
                                               "covered_fragment_evidence": record["covered_fragment_evidence"]}})
        if reason is not None:
            pending.append({**record, "reason": reason, "navmesh_checks": checks})
            continue
        consumed.update(source_indices)
        replacements[min(source_indices)] = proposed
        merged.append({**record, "axis_ids": [p["id"] for p in proposed],
                       "wall_thickness_m": float(high - low), "occupied_intervals_m": intervals,
                       "paired_intervals_m": paired, "navmesh_checks": checks})
    output = []
    for index, primitive in enumerate(original):
        if index in replacements:
            output.extend(replacements[index])
        if index not in consumed:
            if primitive["type"] == "line":
                primitive["thickness_m"] = None
                primitive["axis_status"] = "unpaired_face_thickness_unmeasured"
            output.append(primitive)
    return {"primitives": output, "merged": merged, "pending": pending,
            "parameters": deepcopy(PARAMETERS), "height_range_m": list(height_range) if height_range is not None else None,
            "limitations": ["Only verified architectural primitives are valid inputs.",
                            "Raw winding must be reliable; ambiguous/double-sided orientation stays pending.",
                            "Review-assisted same-wall acceptance is structural inference, not a raw geometry pass.",
                            "Unfiltered raw height is shared face support, not an automatic floor assignment.",
                            "Physical thickness is evidence; final CAD rendering remains a single line."]}
