"""Conservative raw-geometry review of detour-without-wall flags.

This is a local heuristic about the exact sampled chord, not proof that an
entire scene has no missing walls.  It never edits walls or changes raw scores.
"""

from __future__ import annotations

import math

import numpy as np

from .map_geometry import slice_triangles
from .map_quality import line_intersects_wall


HEIGHTS = (.2, .5, .9, 1.3, 1.7, 2., 2.3)
FOOTPRINT_RADIUS_M = .18  # Same fixed radius as the Lite robot's rebuilt NavMesh.
FOOTPRINT_OFFSETS_M = (-.18, -.15, -.10, -.05, -.02, 0., .02, .05, .10, .15, .18)


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _intersections(a, b, segments):
    """Finite line intersections, including endpoint touches and collinear edges."""
    segments = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
    delta, length = b - a, float(np.linalg.norm(b - a))
    if not len(segments):
        return []
    direction, offsets = segments[:, 1] - segments[:, 0], segments[:, 0] - a
    denominator = _cross(delta, direction)
    ordinary = np.abs(denominator) > 1e-10
    t = np.divide(_cross(offsets, direction), denominator, out=np.zeros(len(segments)), where=ordinary)
    u = np.divide(_cross(offsets, delta), denominator, out=np.zeros(len(segments)), where=ordinary)
    valid = ordinary & (t >= -1e-8) & (t <= 1 + 1e-8) & (u >= -1e-8) & (u <= 1 + 1e-8)
    hits = [(float(np.clip(value, 0, 1)), "cross_or_touch") for value in t[valid]]
    collinear = ~ordinary & (np.abs(_cross(offsets, delta)) <= 1e-7 * length)
    for segment in segments[collinear]:
        positions = (segment - a) @ delta / length ** 2
        low, high = max(0., float(positions.min())), min(1., float(positions.max()))
        if low <= high + 1e-8:
            hits.extend([(low, "collinear_overlap"), (high, "collinear_overlap")])
    result = []
    for value, kind in sorted(hits):
        if result and (value - result[-1]["fraction"]) * length <= .005:
            continue
        result.append({"fraction": value, "distance_from_a_m": value * length,
                       "xz": (a + value * delta).tolist(), "kind": kind})
    return result


def _section(triangles, a, b, floor):
    """Exact vertical chord-plane sections in [distance from a, relative height]."""
    delta = b - a
    length = np.linalg.norm(delta)
    tangent = delta / length
    normal = np.array([-tangent[1], tangent[0]])
    horizontal = triangles[:, :, [0, 2]] - a
    coordinates = np.stack([horizontal @ tangent, horizontal @ normal, triangles[:, :, 1] - floor], axis=-1)
    segments = slice_triangles(coordinates, 0.)
    # Entire triangles in the query plane do not form ordinary plane cuts. Keep
    # their edges as conservative evidence rather than silently dropping them.
    coplanar = coordinates[np.abs(coordinates[:, :, 1]).max(1) <= 1e-7]
    if len(coplanar):
        edges = np.concatenate([coplanar[:, [first, second]][:, :, [0, 2]]
                                for first, second in ((0, 1), (1, 2), (2, 0))])
        segments = np.concatenate([segments, edges])
    keep = (segments[:, :, 0].max(1) > .1) & (segments[:, :, 0].min(1) < length - .1)
    return segments[keep]


def _floor_support(domain, a, b, origin, resolution):
    length = np.linalg.norm(b - a)
    points = a + np.linspace(0, 1, max(3, math.ceil(length / (resolution / 2)) + 1))[:, None] * (b - a)
    cells = np.floor((points - origin) / resolution).astype(int)
    valid = (cells >= 0).all(1) & (cells[:, 0] < domain.shape[1]) & (cells[:, 1] < domain.shape[0])
    supported = np.zeros(len(points), dtype=bool)
    supported[valid] = domain[cells[valid, 1], cells[valid, 0]]
    return float(supported.mean()), len(points)


def _band_high_surfaces(triangles, a, b, floor):
    """Exact clipping prevents a thin high surface hiding between offset rays."""
    tangent = (b - a) / np.linalg.norm(b - a)
    normal = np.array([-tangent[1], tangent[0]])
    horizontal = triangles[:, :, [0, 2]] - a
    local = np.stack([horizontal @ tangent, horizontal @ normal, triangles[:, :, 1] - floor], axis=-1)
    low, high = np.array([.1, -FOOTPRINT_RADIUS_M, 1.1]), np.array([np.linalg.norm(b - a) - .1, FOOTPRINT_RADIUS_M, 2.3])
    selected = local[(local.max(1) >= low).all(1) & (local.min(1) <= high).all(1)]
    polygons = []
    for triangle in selected:
        polygon = list(triangle)
        for axis in range(3):
            for boundary, sign in ((low[axis], 1), (high[axis], -1)):
                clipped = []
                for first, second in zip(polygon, polygon[1:] + polygon[:1]):
                    inside_a = sign * (first[axis] - boundary) >= -1e-9
                    inside_b = sign * (second[axis] - boundary) >= -1e-9
                    if inside_a:
                        clipped.append(first)
                    if inside_a != inside_b:
                        fraction = (boundary - first[axis]) / (second[axis] - first[axis])
                        clipped.append(first + fraction * (second - first))
                polygon = clipped
                if not polygon:
                    break
            if not polygon:
                break
        if polygon:
            polygons.append(np.asarray(polygon).tolist())
    return polygons


def _footprint_review(triangles, cuts, a, b, floor, domain, walls, origin, resolution):
    delta = b - a
    normal = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
    contexts = []
    for offset in FOOTPRINT_OFFSETS_M:
        start, end = a + offset * normal, b + offset * normal
        support, samples = _floor_support(domain, start, end, origin, resolution)
        section = _section(triangles, start, end, floor)
        low = section[(section[:, :, 1].max(1) > .15) & (section[:, :, 1].min(1) < 1.1)]
        high = section[(section[:, :, 1].max(1) >= 1.1) & (section[:, :, 1].min(1) <= 2.3)]
        inside_canvas = (((np.array([start, end]) - origin) / resolution >= 0).all()
                         and ((np.array([start, end]) - origin) / resolution <= domain.shape[0]).all())
        wall_hit = (line_intersects_wall(walls, start, end, origin=origin, resolution=resolution)
                    if inside_canvas else None)
        intersections = {str(height): _intersections(start, end, values) for height, values in cuts.items()}
        contexts.append({"offset_m": offset, "a_xz": start.tolist(), "b_xz": end.tolist(),
                         "floor_domain_support_fraction": support, "floor_domain_samples": samples,
                         "line_intersects_existing_wall": wall_hit,
                         "raw_intersections": intersections,
                         "raw_intersection_counts": {height: len(value) for height, value in intersections.items()},
                         "low_section_distance_height_segments": low.tolist(),
                         "high_guard_distance_height_segments": high.tolist()})
    high_polygons = _band_high_surfaces(triangles, a, b, floor)
    low_offsets = [row["offset_m"] for row in contexts if row["low_section_distance_height_segments"]]
    eligible = (bool(low_offsets) and not high_polygons
                and all(row["floor_domain_support_fraction"] >= .9
                        and not row["high_guard_distance_height_segments"]
                        and row["line_intersects_existing_wall"] is False for row in contexts))
    return {"radius_m": FOOTPRINT_RADIUS_M, "normal_xz": normal.tolist(),
            "offsets_m": list(FOOTPRINT_OFFSETS_M), "parallel_chords": contexts,
            "continuous_band_high_surface_polygons_distance_offset_height": high_polygons,
            "low_obstacle_offsets_m": low_offsets,
            "boundary_touch_only": bool(low_offsets) and all(math.isclose(abs(value), FOOTPRINT_RADIUS_M) for value in low_offsets),
            "nonarchitectural_clearance_evidence": eligible,
            "action_replayed": False, "movement_feasibility_claimed": False}


def review_leak_pairs(raw_triangles, pairs, floor, origin, side, floor_domain, walls, *, flagged_indices=None):
    """Review flags with raw surfaces and an independently constructed floor mask.

    Pass quality records with ``flagged=True`` or provide explicit positional
    ``flagged_indices`` for the original geodesic pairs. Semantic labels are not
    consumed. Pixel rows increase Z, columns X; origin is the outer top-left XZ.
    ``reviewed_false_flag`` means this heuristic found a low obstruction on this
    chord or in the fixed robot-radius strip, with no measured high surface.
    It does not mean the obstacle is a table, movement is feasible, an action was
    replayed, or that map quality passed. Original flags/scores stay unchanged.
    """
    triangles = np.asarray(raw_triangles, dtype=float).reshape(-1, 3, 3)
    domain, walls = np.asarray(floor_domain, dtype=bool), np.asarray(walls, dtype=bool)
    origin = np.asarray(origin, dtype=float)
    if (domain.ndim != 2 or domain.shape != walls.shape or not domain.size
            or domain.shape[0] != domain.shape[1] or origin.shape != (2,)):
        raise ValueError("Need matching nonempty square rasters and an XZ origin")
    if not np.isfinite(triangles).all() or not np.isfinite(origin).all() or not math.isfinite(floor) or not math.isfinite(side) or side <= 0:
        raise ValueError("Geometry, origin, floor and positive side must be finite")
    selected = ([index for index, pair in enumerate(pairs) if pair.get("flagged") is True]
                if flagged_indices is None else list(flagged_indices))
    if any(type(index) is not int or not 0 <= index < len(pairs) for index in selected) or len(set(selected)) != len(selected):
        raise ValueError("flagged_indices must contain unique valid pair positions")
    cuts = {height: slice_triangles(triangles, floor + height) for height in HEIGHTS} if selected else {}
    resolution = side / domain.shape[0]
    result = []
    for index in selected:
        pair = pairs[index]
        endpoints = np.asarray([pair["a"], pair["b"]], dtype=float)
        if endpoints.shape != (2, 3) or not np.isfinite(endpoints).all():
            raise ValueError("Pairs need two finite 3D endpoints")
        a, b = endpoints[:, [0, 2]]
        length = float(np.linalg.norm(b - a))
        if length <= .2:
            raise ValueError("Leak review needs a chord longer than 0.2m")
        intersects_wall = line_intersects_wall(walls, a, b, origin=origin, resolution=resolution)
        intersections = {str(height): _intersections(a, b, segments) for height, segments in cuts.items()}
        # Exclude endpoint touches from obstruction evidence, while keeping them
        # in the reported intersections for transparent boundary diagnostics.
        interior = {height: [hit for hit in hits if .1 < hit["distance_from_a_m"] < length - .1]
                    for height, hits in intersections.items()}
        support, floor_samples = _floor_support(domain, a, b, origin, resolution)
        section = _section(triangles, a, b, floor)
        # A surface missed by discrete cuts (for example a sloped panel) still
        # blocks classification when its exact chord-plane section spans this
        # body-height range. Horizontal ceilings above 2.3m are outside this gate.
        high_guard = section[(section[:, :, 1].max(1) >= 1.1) & (section[:, :, 1].min(1) <= 2.3)]
        low_sections = section[(section[:, :, 1].max(1) > .15) & (section[:, :, 1].min(1) < 1.1)]
        low_count = sum(len(interior[str(height)]) for height in (.2, .5, .9))
        high_count = sum(len(interior[str(height)]) for height in (1.3, 1.7, 2., 2.3))
        footprint = None
        if support < .7:
            classification, review = "unobserved_floor_gap", "unresolved"
            reason = "less_than_70_percent_floor_domain_support; exterior_or_scan_void_not_disambiguated"
        elif support >= .9 and low_count and not high_count and not len(high_guard) and not intersects_wall:
            classification, review = "low_obstacle_detour", "reviewed_false_flag"
            reason = "low_raw_surface_intersection_with_clear_upper_cuts_and_exact_high_surface_guard"
        else:
            classification, review, reason = "unresolved", "unresolved", "insufficient_evidence_to_exclude_missing_architectural_wall"
            if support >= .9 and not low_count and not high_count and not len(high_guard) and not intersects_wall:
                footprint = _footprint_review(triangles, cuts, a, b, floor, domain, walls, origin, resolution)
                if footprint["nonarchitectural_clearance_evidence"]:
                    classification, review = "low_obstacle_clearance_detour", "reviewed_false_flag"
                    reason = "low_raw_surface_in_robot_radius_strip_without_measured_high_surface; no_action_replay_or_feasibility_claim"
        result.append({
            "pair_index": pair.get("index", index), "input_pair_position": index,
            "classification": classification, "review": review, "reason": reason,
            "a": endpoints[0].tolist(), "b": endpoints[1].tolist(), "chord_length_m": length,
            "geodesic_m": pair.get("geodesic_m"), "line_intersects_existing_wall": intersects_wall,
            "floor_domain_support_fraction": support, "floor_domain_samples": floor_samples,
            "relative_slice_heights_m": list(HEIGHTS), "raw_intersections": intersections,
            "raw_intersection_counts": {height: len(hits) for height, hits in intersections.items()},
            "low_obstacle_proof": {"interior_low_cut_hits": low_count, "interior_upper_cut_hits": high_count,
                                   "exact_vertical_section_segments": len(section),
                                   "low_section_distance_height_segments": low_sections.tolist(),
                                   "high_guard_distance_height_segments": high_guard.tolist()},
            "footprint_review": footprint,
            "rule": {"kind": "local_conservative_geometry_heuristic", "semantic_labels_used": False,
                     "low_obstacle_max_height_m": 1.1, "high_surface_guard_m": [1.1, 2.3],
                     "minimum_low_obstacle_floor_support": .9, "floor_gap_support_threshold": .7,
                     "robot_radius_m": FOOTPRINT_RADIUS_M,
                     "context": "local chord/robot-radius geometry evidence only; no movement-feasibility or replay claim; no automatic wall patch or original score change"},
        })
    return result
