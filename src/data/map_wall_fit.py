"""Fit supported straight walls or circular arcs without Manhattan snapping.

Input segments remain evidence: every primitive identifies its original segment
indices. Matching only facing, mutually nearest endpoints preserves door gaps,
parallel walls and sharp corners. Fitting does not invent junctions or fill gaps.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree


def _coincident_runs(source, source_ids, max_error):
    """Collapse overlapping co-planar cuts, keeping every original support ID."""
    tolerance = min(.01, max_error / 5)
    delta = source[:, 1] - source[:, 0]
    angles = np.arctan2(delta[:, 1], delta[:, 0]) % math.pi
    angle_ids = np.rint(angles / math.radians(5)).astype(int) % 36
    result, groups = [], []
    for angle_id in np.unique(angle_ids):
        tangent = np.array([math.cos(angle_id * math.pi / 36), math.sin(angle_id * math.pi / 36)])
        normal = np.array([-tangent[1], tangent[0]])
        selected = np.flatnonzero(angle_ids == angle_id)
        offsets = source[selected].mean(1) @ normal
        offset_ids = np.rint(offsets / tolerance).astype(int)
        for offset_id in np.unique(offset_ids):
            members = selected[offset_ids == offset_id]
            intervals = np.sort(source[members] @ tangent, axis=1)
            sequence = np.argsort(intervals[:, 0])
            runs = []
            for position in sequence:
                identifier = members[position]
                low, high = intervals[position]
                if runs and low <= runs[-1][1] + 1e-7:
                    runs[-1][1] = max(runs[-1][1], float(high))
                    runs[-1][2].append(identifier)
                else:
                    runs.append([float(low), float(high), [identifier]])
            for _, _, run in runs:
                points = source[run].reshape(-1, 2)
                lengths = np.linalg.norm(source[run, 1] - source[run, 0], axis=1)
                center, direction, residuals, along = _line(points, np.repeat(lengths, 2) / 2)
                if len(run) > 1 and residuals.max() <= tolerance:
                    result.append(center + np.array([along.min(), along.max()])[:, None] * direction)
                    groups.append([int(source_ids[index]) for index in run])
                else:
                    result.extend(source[index] for index in run)
                    groups.extend([[int(source_ids[index])] for index in run])
    return np.asarray(result).reshape(-1, 2, 2), groups


def _line(points, weights):
    center = np.average(points, axis=0, weights=weights)
    centered = points - center
    covariance = (centered * weights[:, None]).T @ centered / weights.sum()
    _, axes = np.linalg.eigh(covariance)
    tangent = axes[:, -1]
    if tangent[0] < -1e-10 or abs(tangent[0]) <= 1e-10 and tangent[1] < 0:
        tangent *= -1
    normal = np.array([-tangent[1], tangent[0]])
    distances = np.abs(centered @ normal)
    along = centered @ tangent
    return center, tangent, distances, along


def _connections(segments, lengths, max_error, max_gap):
    points = segments.reshape(-1, 2)
    inward = np.empty_like(points)
    inward[::2] = (segments[:, 1] - segments[:, 0]) / lengths[:, None]
    inward[1::2] = -inward[::2]
    tree = cKDTree(points)
    chosen = {}
    cosine = math.cos(math.radians(30))
    for endpoint, neighbours in enumerate(tree.query_ball_point(points, max_gap + 1e-9)):
        candidates = []
        for other in neighbours:
            if other // 2 == endpoint // 2:
                continue
            agreement = float(inward[endpoint] @ inward[other])
            if agreement > -cosine:
                continue
            delta = points[other] - points[endpoint]
            distance = float(np.linalg.norm(delta))
            # Longer connectors must follow the wall; short endpoint noise can
            # be lateral. Mutual nearest matching avoids hopping to a parallel
            # wall when a continuation on the same wall exists.
            if distance > max_error / 2 and (inward[endpoint] @ delta / distance > -cosine
                                            or inward[other] @ delta / distance < cosine):
                continue
            candidates.append((distance + .05 * (1 + agreement), other))
        if candidates:
            chosen[endpoint] = min(candidates)[1]
    return {endpoint: other for endpoint, other in chosen.items() if chosen.get(other) == endpoint}


def _chains(segments, connections):
    remaining = set(range(len(segments)))
    result = []
    while remaining:
        seed = min(remaining)
        # Walk to an unmatched end first; a complete circular chain has none.
        endpoint, seen = 2 * seed, set()
        while endpoint in connections and endpoint // 2 not in seen:
            seen.add(endpoint // 2)
            endpoint = connections[endpoint] ^ 1
        if endpoint // 2 in seen:
            endpoint = 2 * seed
        order, points, visited = [], [], set()
        while endpoint // 2 not in visited:
            identifier = endpoint // 2
            visited.add(identifier)
            remaining.discard(identifier)
            order.append(identifier)
            points.extend([segments[identifier, endpoint % 2], segments[identifier, 1 - endpoint % 2]])
            exit_endpoint = endpoint ^ 1
            if exit_endpoint not in connections:
                break
            endpoint = connections[exit_endpoint]
        result.append((order, np.asarray(points)))
    return result


def _arc(points, weights, line_rms, max_error, ordered_points):
    center_hint = np.average(points, axis=0, weights=weights)
    local = points - center_hint
    design = np.column_stack([2 * local, np.ones(len(points))])
    coefficients, _, rank, _ = np.linalg.lstsq(design * np.sqrt(weights[:, None]),
                                              np.sum(local ** 2, axis=1) * np.sqrt(weights), rcond=None)
    if rank < 3:
        return None
    center = coefficients[:2] + center_hint
    radius_squared = float(coefficients[2] + coefficients[:2] @ coefficients[:2])
    if radius_squared <= 0:
        return None
    radius = math.sqrt(radius_squared)
    residuals = np.abs(np.linalg.norm(points - center, axis=1) - radius)
    rms = math.sqrt(float(np.average(residuals ** 2, weights=weights)))
    angles = np.unwrap(np.arctan2(ordered_points[:, 1] - center[1], ordered_points[:, 0] - center[0]))
    changes = np.diff(angles)
    sweep = float(angles[-1] - angles[0])
    travelled = float(np.abs(changes).sum())
    backward = (travelled - abs(sweep)) / max(travelled, 1e-12)
    if (residuals.max() > max_error or abs(sweep) < math.radians(25) or radius < .3
            or backward > .1 or rms >= .35 * line_rms):
        return None
    start, end = float(angles[0]), float(angles[-1])
    endpoints = center + radius * np.column_stack([np.cos([start, end]), np.sin([start, end])])
    extrema = [start, end]
    low, high = sorted((start, end))
    for index in range(math.ceil(low / (math.pi / 2)), math.floor(high / (math.pi / 2)) + 1):
        extrema.append(index * math.pi / 2)
    extent = center + radius * np.column_stack([np.cos(extrema), np.sin(extrema)])
    return {"type": "arc", "endpoints": endpoints.tolist(), "start": endpoints[0].tolist(), "end": endpoints[1].tolist(),
            "center": center.tolist(), "radius_m": radius,
            "start_angle_radians": start, "end_angle_radians": end, "sweep_radians": sweep,
            "start_angle_degrees": math.degrees(start), "end_angle_degrees": math.degrees(end),
            "sweep_degrees": math.degrees(sweep), "direction": "counterclockwise" if sweep > 0 else "clockwise",
            "length_m": abs(sweep) * radius, "extent_xz": [*extent.min(0).tolist(), *extent.max(0).tolist()],
            "fitting": {"max_residual_m": float(residuals.max()), "rms_m": rms,
                        "line_rms_m": line_rms, "circle_to_line_rms_ratio": rms / max(line_rms, 1e-12),
                        "angular_backtracking_fraction": backward, "method": "length_weighted_circle_least_squares"}}


def _consolidate_lines(primitives, source, lengths, max_error, max_gap):
    """Merge co-planar runs, vectorizing cheap tests before source refitting."""
    result = list(primitives)
    tolerance = min(.01, max_error / 5)
    changed = True
    while changed:
        changed = False
        lines = [index for index, wall in enumerate(result) if wall["type"] == "line"]
        if len(lines) < 2:
            break
        ends = np.asarray([result[index]["endpoints"] for index in lines])
        centers = ends.mean(1)
        directions = ends[:, 1] - ends[:, 0]
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        for position, first in enumerate(lines[:-1]):
            a, tangent = ends[position], directions[position]
            normal = np.array([-tangent[1], tangent[0]])
            following = np.arange(position + 1, len(lines))
            intervals = np.sort(ends[following] @ tangent, axis=1)
            own = np.sort(a @ tangent)
            gaps = np.maximum(np.maximum(own[0] - intervals[:, 1], intervals[:, 0] - own[1]), 0.)
            candidates = following[(np.abs(directions[following] @ tangent) >= math.cos(math.radians(5)))
                                   & (np.abs((centers[following] - centers[position]) @ normal) <= tolerance)
                                   & (gaps <= max_gap + 1e-9)]
            for other_position in candidates:
                second = lines[other_position]
                ids = sorted(set(result[first]["source_segment_indices"] + result[second]["source_segment_indices"]))
                points, weights = source[ids].reshape(-1, 2), np.repeat(lengths[ids], 2) / 2
                center, direction, residuals, along = _line(points, weights)
                if residuals.max() > max_error:
                    continue
                endpoints = center + np.array([along.min(), along.max()])[:, None] * direction
                gap = gaps[other_position - position - 1]
                result[first] = {"type": "line", "start": endpoints[0].tolist(), "end": endpoints[1].tolist(),
                                 "endpoints": endpoints.tolist(), "length_m": float(np.ptp(along)),
                                 "source_segment_indices": ids, "source_segment_count": len(ids),
                                 "max_source_endpoint_gap_m": max(float(gap), result[first]["max_source_endpoint_gap_m"], result[second]["max_source_endpoint_gap_m"]),
                                 "fitting": {"max_residual_m": float(residuals.max()),
                                             "rms_m": math.sqrt(float(np.average(residuals ** 2, weights=weights))),
                                             "method": "coincident_run_length_weighted_total_least_squares"}}
                result.pop(second)
                changed = True
                break
            if changed:
                break
    return result


def _retain_coincident_support(primitives, source, lengths, valid_ids, max_error):
    """Retain short cuts explained by unchanged long walls, in bounded batches."""
    covered = {index for primitive in primitives for index in primitive["source_segment_indices"]}
    remaining = np.asarray([index for index in valid_ids if index not in covered], dtype=int)
    lines = [wall for wall in primitives if wall["type"] == "line"]
    if not lines or not len(remaining):
        return
    tolerance = min(.01, max_error / 5)
    endpoints = np.asarray([wall["endpoints"] for wall in lines])
    wall_lengths = np.asarray([wall["length_m"] for wall in lines])
    tangents = (endpoints[:, 1] - endpoints[:, 0]) / wall_lengths[:, None]
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    additions = [[] for _ in lines]
    for batch in np.array_split(remaining, max(1, math.ceil(len(remaining) / 512))):
        delta = source[batch, None] - endpoints[None, :, None, 0]
        along = np.einsum("blpc,lc->blp", delta, tangents)
        errors = np.abs(np.einsum("blpc,lc->blp", delta, normals)).max(2)
        directions = (source[batch, 1] - source[batch, 0]) / lengths[batch, None]
        valid = ((errors <= tolerance) & (along.min(2) >= -tolerance)
                 & (along.max(2) <= wall_lengths[None] + tolerance)
                 & (np.abs(directions @ tangents.T) >= math.cos(math.radians(15))))
        errors[~valid] = np.inf
        chosen = errors.argmin(1)
        for identifier, position, error in zip(batch, chosen, errors[np.arange(len(batch)), chosen]):
            if math.isfinite(error):
                additions[position].append(int(identifier))
    for wall, extra in zip(lines, additions):
        if not extra:
            continue
        ids = sorted(wall["source_segment_indices"] + extra)
        wall["source_segment_indices"] = ids
        wall["source_segment_count"] = len(ids)
        wall["fitting"]["additional_coincident_support_not_refitted_count"] = len(extra)
        a, b = np.asarray(wall["endpoints"])
        tangent = (b - a) / wall["length_m"]
        residuals = (source[ids].reshape(-1, 2) - a) @ np.array([-tangent[1], tangent[0]])
        wall["fitting"]["max_residual_m"] = float(np.abs(residuals).max())
        wall["fitting"]["rms_m"] = math.sqrt(float(np.average(residuals ** 2, weights=np.repeat(lengths[ids], 2))))


def fit_wall_primitives(segments, *, max_error=.05, min_length=.25, max_gap=.15):
    """Return fitted ``primitives`` plus coverage/error ``metrics`` in metres.

    Curves need at least six facing-connected source segments and strong circle
    evidence. Other bends are split into supported lines; no right-angle grid is
    assumed. A gap larger than max_gap never receives a connecting primitive.
    """
    source = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
    if not np.isfinite(source).all():
        raise ValueError("Wall segments must be finite")
    if not all(math.isfinite(value) for value in (max_error, min_length, max_gap)) or min(max_error, min_length, max_gap) <= 0:
        raise ValueError("Fitting limits must be positive and finite")
    lengths = np.linalg.norm(source[:, 1] - source[:, 0], axis=1)
    valid_ids = np.flatnonzero(lengths > 1e-8)
    segments, support_groups = _coincident_runs(source[valid_ids], valid_ids, max_error) if len(valid_ids) else (np.empty((0, 2, 2)), [])
    weights = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    connections = _connections(segments, weights, max_error, max_gap) if len(segments) else {}
    chains = _chains(segments, connections)
    primitives, discarded = [], []

    def fit(order, ordered_points):
        source_ids = sorted({index for identifier in order for index in support_groups[identifier]})
        points = source[source_ids].reshape(-1, 2)
        point_weights = np.repeat(lengths[source_ids], 2) / 2
        center, tangent, residuals, along = _line(points, point_weights)
        rms = math.sqrt(float(np.average(residuals ** 2, weights=point_weights)))
        if residuals.max() <= max_error:
            endpoints = center + np.array([along.min(), along.max()])[:, None] * tangent
            primitive = {"type": "line", "endpoints": endpoints.tolist(),
                         "start": endpoints[0].tolist(), "end": endpoints[1].tolist(),
                         "length_m": float(np.ptp(along)),
                         "fitting": {"max_residual_m": float(residuals.max()), "rms_m": rms,
                                     "method": "length_weighted_total_least_squares"}}
        elif len(order) >= 6:
            # Point order is essential: a U-turn or several unrelated sections
            # on a common circle are not automatically one supported arc.
            primitive = _arc(points, point_weights, rms, max_error, ordered_points)
        else:
            primitive = None
        if primitive is not None:
            if primitive["length_m"] >= min_length:
                gaps = np.linalg.norm(ordered_points[2::2] - ordered_points[1:-1:2], axis=1)
                primitive.update(source_segment_indices=source_ids, source_segment_count=len(source_ids),
                                 max_source_endpoint_gap_m=float(gaps.max()) if len(gaps) else 0.)
                primitives.append(primitive)
            else:
                discarded.extend(source_ids)
            return
        if len(order) == 1:
            discarded.extend(source_ids)
            return
        middle = len(order) // 2
        fit(order[:middle], ordered_points[:2 * middle])
        fit(order[middle:], ordered_points[2 * middle:])

    for order, points in chains:
        fit(order, points)
    primitives = _consolidate_lines(primitives, source, lengths, max_error, max_gap)
    _retain_coincident_support(primitives, source, lengths, valid_ids, max_error)
    for index, primitive in enumerate(primitives, start=1):
        primitive["id"] = f"W{index:04d}"
    covered = sorted({index for primitive in primitives for index in primitive["source_segment_indices"]})
    total_length = float(lengths.sum())
    return {"primitives": primitives, "metrics": {
        "input_segment_count": len(source), "nondegenerate_segment_count": len(valid_ids),
        "prepared_segment_count": len(segments), "overlap_merged_segment_count": len(valid_ids) - len(segments),
        "continuous_chain_count": len(chains), "primitive_count": len(primitives),
        "line_count": sum(item["type"] == "line" for item in primitives),
        "arc_count": sum(item["type"] == "arc" for item in primitives),
        "covered_segment_count": len(covered), "discarded_segment_indices": sorted(set(discarded) - set(covered)),
        "degenerate_segment_indices": np.flatnonzero(lengths <= 1e-8).tolist(),
        "source_length_m": total_length, "covered_source_length_m": float(lengths[covered].sum()),
        "source_length_coverage": float(lengths[covered].sum()) / total_length if total_length else None,
        "max_residual_m": max((item["fitting"]["max_residual_m"] for item in primitives), default=None),
        "max_error_m": max_error, "min_length_m": min_length, "max_gap_m": max_gap,
        "manhattan_snapping": False, "raster_gap_filling_performed": False,
        "primitive_support_may_bridge_gaps_up_to_m": max_gap,
    }}
