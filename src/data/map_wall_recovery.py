"""Recover missing straight architectural spans only from aligned raw surfaces.

Input primitives must already be accepted architectural walls. Semantic windows
are traceable hints, never geometric proof. Doors and verified portals protect
their openings. Existing walls, including arcs and short jambs, stay unchanged.
"""

from copy import deepcopy
import math

import numpy as np


def _segment(value):
    points = np.asarray(value, dtype=float)
    if points.shape != (2, 2) or not np.isfinite(points).all():
        raise ValueError("Expected two finite world [X, Z] points")
    length = float(np.linalg.norm(points[1] - points[0]))
    if length <= 1e-8:
        raise ValueError("Wall segments must have positive length")
    return points, length


def _union(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1] + 1e-10:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([float(start), float(end)])
    return result


def raw_gap_support(segment, raw_slices, *, distance_m=.025, angle_degrees=5.):
    """Measure exact interval coverage by raw cuts on this wall plane.

    ``raw_slices`` maps relative heights in metres to raw world-XZ segments,
    e.g. ``RawWallEvidence.layers``. Nearby transverse surfaces do not count.
    At least 90% of the gap must be supported at two heights simultaneously;
    strong cuts must include a low (<= .9 m) and high (>= 1.5 m) surface, with
    >= .8 m vertical separation. This is surface evidence, not movement proof.
    """
    points, length = _segment(segment)
    if not 0 < distance_m <= .025 or not 0 < angle_degrees <= 5:
        raise ValueError("Raw alignment tolerances must be in (0, .025 m] and (0, 5 degrees]")
    tangent = (points[1] - points[0]) / length
    normal = np.array([-tangent[1], tangent[0]])
    fractions, intervals_by_height, events = {}, {}, []
    for height, values in sorted(raw_slices.items(), key=lambda item: float(item[0])):
        height = float(height)
        lines = np.asarray(values, dtype=float).reshape(-1, 2, 2)
        if not math.isfinite(height) or height <= 0 or not np.isfinite(lines).all():
            raise ValueError("Raw relative heights must be positive and all cuts finite")
        if str(height) in fractions:
            raise ValueError("Raw slices must have unique actual heights")
        delta = lines[:, 1] - lines[:, 0]
        lengths = np.linalg.norm(delta, axis=1)
        directions = np.divide(delta, lengths[:, None], out=np.zeros_like(delta), where=lengths[:, None] > 1e-10)
        aligned = ((lengths > 1e-10)
                   & (np.abs(directions @ tangent) >= math.cos(math.radians(angle_degrees)))
                   & (np.abs((lines - points[0]) @ normal).max(1) <= distance_m))
        projected = np.sort((lines[aligned] - points[0]) @ tangent, axis=1)
        projected = np.clip(projected, 0, length)
        intervals = _union(projected.tolist())
        intervals_by_height[str(height)] = intervals
        fractions[str(height)] = sum(end - start for start, end in intervals) / length
        events.extend((value, change) for start, end in intervals for value, change in ((start, 1), (end, -1)))
    covered_twice, count, previous = 0., 0, 0.
    for position, change in sorted(events):
        if count >= 2:
            covered_twice += position - previous
        count += change
        previous = position
    strong = [float(height) for height, value in fractions.items() if value >= .9 - 1e-9]
    supported = covered_twice / length
    span = max(strong) - min(strong) if strong else 0.
    passed = (supported >= .9 - 1e-9 and len(strong) >= 2 and min(strong) <= .9
              and max(strong) >= 1.5 and span >= .8 - 1e-9)
    return {"pass": bool(passed), "source": "independent_raw_render_mesh_height_cuts",
            "aligned_height_support_fractions": fractions, "supported_fraction": supported,
            "aligned_intervals_m": intervals_by_height, "strong_heights_m": strong,
            "strong_height_span_m": span, "same_plane_and_direction": True,
            "distance_tolerance_m": distance_m, "angle_tolerance_degrees": angle_degrees,
            "coverage_method": "exact_union_intervals_and_at_least_two_height_overlap"}


def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _intersects_interior(segment, protected):
    """Collinear overlap or crossing the protected interior; jamb touch is OK."""
    a, _ = _segment(segment)
    b, _ = _segment(protected)
    direction, other = a[1] - a[0], b[1] - b[0]
    determinant = _cross(direction, other)
    if abs(determinant) < 1e-10:
        if abs(_cross(b[0] - a[0], direction)) > 1e-8:
            return False
        positions = sorted(float((point - a[0]) @ direction / (direction @ direction)) for point in b)
        return min(1., positions[1]) - max(0., positions[0]) > 1e-8
    delta = b[0] - a[0]
    t, u = _cross(delta, other) / determinant, _cross(delta, direction) / determinant
    return 1e-8 < t < 1 - 1e-8 and 1e-8 < u < 1 - 1e-8


def _prior_near_gap(segment, prior, tolerance):
    a, length = _segment(segment)
    b, _ = _segment(prior)
    tangent = (a[1] - a[0]) / length
    normal = np.array([-tangent[1], tangent[0]])
    along = (b - a[0]) @ tangent
    return (np.abs((b - a[0]) @ normal).max() <= tolerance
            and min(length, float(along.max())) - max(0., float(along.min())) > 1e-8)


def _check(callback, segment, name):
    result = callback(segment.tolist())
    if not isinstance(result, dict) or type(result.get("pass")) is not bool:
        raise ValueError(name + " must return a dict containing an explicit boolean pass")
    return result


def _strict_support(evidence):
    fractions = evidence.get("aligned_height_support_fractions", {})
    strong = [float(height) for height, value in fractions.items() if float(value) >= .9 - 1e-9]
    return (evidence["pass"] and evidence.get("same_plane_and_direction") is True
            and evidence.get("supported_fraction", 0.) >= .9 - 1e-9
            and len(strong) >= 2 and min(strong) <= .9 and max(strong) >= 1.5
            and max(strong) - min(strong) >= .8 - 1e-9
            and 0 < evidence.get("distance_tolerance_m", math.inf) <= .025
            and 0 < evidence.get("angle_tolerance_degrees", math.inf) <= 5.)


def recover_collinear_gaps(primitives, *, raw_support, not_blocks_navmesh,
                           semantic_priors=(), portals=(), min_gap_m=.12, max_gap_m=2.,
                           plane_tolerance_m=.015, angle_tolerance_degrees=3.):
    """Add analytic straight spans with strict raw proof; preserve all originals.

    Both callbacks receive a world-XZ segment and return explicit boolean
    ``pass`` plus evidence. ``raw_support`` must provide the strict profile
    returned by :func:`raw_gap_support`, not the looser nearest-surface score.
    Semantic priors are ``{id, category: door|window, segment}`` records. Doors
    are conservatively protected, including closed-door geometry. Accepted or
    undecided portals are protected; a rejected portal is not passage proof.
    """
    if (not all(math.isfinite(v) for v in (min_gap_m, max_gap_m, plane_tolerance_m, angle_tolerance_degrees))
            or not 0 < min_gap_m <= max_gap_m <= 2.
            or not 0 < plane_tolerance_m <= .015 or not 0 < angle_tolerance_degrees <= 3.):
        raise ValueError("Require 0 < min_gap <= max_gap <= 2 m and conservative plane/angle limits")
    result = deepcopy(list(primitives))
    if len({wall["id"] for wall in result}) != len(result):
        raise ValueError("Wall primitive IDs must be unique")
    lines = [(wall, *_segment([wall["start"], wall["end"]])) for wall in result if wall["type"] == "line"]
    priors = deepcopy(list(semantic_priors))
    for prior in priors:
        if prior["category"] not in {"door", "window"}:
            raise ValueError("Semantic recovery priors must be doors or windows")
        _segment(prior["segment"])
    protected = [value for value in portals if value.get("status", "accepted") != "rejected"]
    for value in protected:
        _segment(value["segment"])
    candidates, recovered, rejected = [], [], []
    for index, (first, a, a_length) in enumerate(lines):
        tangent = (a[1] - a[0]) / a_length
        normal = np.array([-tangent[1], tangent[0]])
        for second, b, b_length in lines[index + 1:]:
            if abs(tangent @ ((b[1] - b[0]) / b_length)) < math.cos(math.radians(angle_tolerance_degrees)):
                continue
            positions = (b - a[0]) @ tangent
            if positions.min() > a_length:
                proposed = np.stack((a[1], b[positions.argmin()]))
            elif positions.max() < 0:
                proposed = np.stack((b[positions.argmax()], a[0]))
            else:
                continue
            _, gap = _segment(proposed)
            if not min_gap_m - 1e-9 <= gap <= max_gap_m + 1e-9:
                continue
            other_tangent = (b[1] - b[0]) / b_length
            other_normal = np.array([-other_tangent[1], other_tangent[0]])
            offset = max(float(np.abs((b - a[0]) @ normal).max()),
                         float(np.abs((a - b[0]) @ other_normal).max()))
            candidates.append({"segment": proposed.tolist(), "length_m": gap,
                               "adjacent_primitive_ids": [first["id"], second["id"]],
                               "plane_offset_m": offset})
    for candidate in sorted(candidates, key=lambda item: item["length_m"]):
        segment = np.asarray(candidate["segment"])
        record = {**candidate, "status": "rejected", "evidence": {}}
        reason = None
        if candidate["plane_offset_m"] > plane_tolerance_m:
            reason = "adjacent_walls_not_on_same_plane"
        matching = [prior for prior in priors if _prior_near_gap(segment, prior["segment"], .05)]
        record["semantic_prior_ids"] = [prior["id"] for prior in matching]
        record["evidence"]["semantic_priors_are_geometric_proof"] = False
        if reason is None and (any(prior["category"] == "door" for prior in matching)
                               or any(_intersects_interior(segment, value["segment"]) for value in protected)):
            reason = "protected_door_or_portal"
        if reason is None and any(wall["id"] not in candidate["adjacent_primitive_ids"]
                                  and _prior_near_gap(segment, [wall["start"], wall["end"]], plane_tolerance_m)
                                  for wall in result if wall["type"] == "line"):
            reason = "existing_collinear_wall_inside_gap"
        if reason is None:
            evidence = _check(raw_support, segment, "raw_support")
            record["evidence"]["raw_support"] = evidence
            if not _strict_support(evidence):
                reason = "insufficient_aligned_multilevel_raw_support"
        if reason is None:
            navigation = _check(not_blocks_navmesh, segment, "not_blocks_navmesh")
            record["evidence"]["navmesh"] = navigation
            if not navigation["pass"]:
                reason = "would_block_navmesh"
        if reason is not None:
            record["reason"] = reason
            rejected.append(record)
            continue
        identifier = f"REC{len(recovered) + 1:04d}"
        existing_ids = {wall["id"] for wall in result}
        while identifier in existing_ids:
            identifier = "R" + identifier
        record.update(status="accepted", reason="aligned_raw_wall_surface_and_no_walkable_crossing",
                      primitive_id=identifier)
        result.append({"id": identifier, "type": "line", "start": segment[0].tolist(),
                       "end": segment[1].tolist(), "endpoints": segment.tolist(),
                       "length_m": candidate["length_m"], "source_primitive_ids": candidate["adjacent_primitive_ids"],
                       "recovery": deepcopy(record)})
        recovered.append(record)
    return {"primitives": result, "recovered": recovered, "rejected": rejected,
            "parameters": {"min_gap_m": min_gap_m, "max_gap_m": max_gap_m,
                           "plane_tolerance_m": plane_tolerance_m, "angle_tolerance_degrees": angle_tolerance_degrees},
            "original_primitives_modified": False}
