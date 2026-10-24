"""Complete short wall breaks only where a connected raw 3D wall proves them.

This is an offline geometry operation. It neither closes a gap because it is
small nor treats a lintel, sill, semantic label or NavMesh boundary as a wall.
Coordinates are Habitat XYZ; exported wall coordinates are world [X, Z].
"""

from copy import deepcopy
import math

import numpy as np

from .map_wall_structure import junction_graph


def _segment(value):
    points = np.asarray(value, dtype=float)
    if points.shape != (2, 2) or not np.isfinite(points).all():
        raise ValueError("Expected two finite world [X, Z] endpoints")
    length = float(np.linalg.norm(points[1] - points[0]))
    if length <= 1e-8:
        raise ValueError("Segments need positive length")
    return points, length


def _clip(polygon, axis, boundary, greater):
    result = []
    for first, second in zip(polygon, polygon[1:] + polygon[:1]):
        inside = first[axis] >= boundary if greater else first[axis] <= boundary
        other = second[axis] >= boundary if greater else second[axis] <= boundary
        if inside:
            result.append(first)
        if inside != other:
            result.append(first + (second - first) * (
                (boundary - first[axis]) / (second[axis] - first[axis])))
    return result


def _area(polygons, left, right, bottom, top):
    """Exact polygon-union area by a sweep through all edge-order changes.

    Projected clipped triangles are convex. Between consecutive vertex heights
    and edge intersections their horizontal union length is linear, so its
    trapezoid integral is exact. Overlapping triangles are counted once.
    """
    pieces, edges = [], []
    for polygon in polygons:
        value = list(polygon)
        for axis, boundary, greater in ((0, left, True), (0, right, False),
                                         (1, bottom, True), (1, top, False)):
            if len(value) < 3:
                break
            value = _clip(value, axis, boundary, greater)
        if len(value) < 3:
            continue
        current = []
        for first, second in zip(value, value[1:] + value[:1]):
            delta = second - first
            if abs(delta[1]) <= 1e-12:
                continue
            slope = delta[0] / delta[1]
            edge = (min(first[1], second[1]), max(first[1], second[1]),
                    slope, first[0] - slope * first[1])
            current.append(edge)
            edges.append(edge)
        if len(current) >= 2:
            pieces.append(current)
    if not pieces:
        return 0.
    edges = np.asarray(edges)
    events = [bottom, top, *edges[:, 0], *edges[:, 1]]
    for index, edge in enumerate(edges[:-1]):
        others = edges[index + 1:]
        denominator = edge[2] - others[:, 2]
        crossing = np.divide(others[:, 3] - edge[3], denominator,
                             out=np.full(len(others), np.inf), where=abs(denominator) > 1e-12)
        valid = ((crossing > np.maximum(edge[0], others[:, 0]) + 1e-12)
                 & (crossing < np.minimum(edge[1], others[:, 1]) - 1e-12))
        events.extend(crossing[valid])
    events = np.unique(np.clip(events, bottom, top))
    area = 0.
    for low, high in zip(events[:-1], events[1:]):
        if high - low <= 1e-12:
            continue
        middle = (low + high) / 2
        active = [[edge for edge in piece if edge[0] < middle < edge[1]] for piece in pieces]

        def union_length(height):
            intervals = sorted((min(edge[2] * height + edge[3] for edge in piece),
                                max(edge[2] * height + edge[3] for edge in piece))
                               for piece in active if len(piece) >= 2)
            length, end = 0., -math.inf
            for begin, finish in intervals:
                length += max(0., finish - max(begin, end))
                end = max(end, finish)
            return length

        area += (high - low) * (union_length(low) + union_length(high)) / 2
    return float(max(0., area))


def wall_surface_evidence(segment, raw_triangles, *, floor_height, anchor_before_m=.1,
                          anchor_after_m=.1):
    """Measure one connected wall face over its full continuous body rectangle.

    A 5 cm plane band accommodates the existing <=5 cm source fit residual;
    it never permits geometry to move by 5 cm. Original mesh topology and
    winding separate neighbouring sheets before their projected areas are
    combined. At least 95% total body area and 90% of every 40 cm height band
    must exist in one connected component, including both requested anchors.
    """
    points, length = _segment(segment)
    raw = np.asarray(raw_triangles, dtype=float).reshape(-1, 3, 3)
    if (not np.isfinite(raw).all() or not math.isfinite(floor_height)
            or not 0 <= anchor_before_m <= .2 or not 0 <= anchor_after_m <= .2):
        raise ValueError("Require finite raw triangles/floor and anchors within 0..20 cm")
    tangent = (points[1] - points[0]) / length
    normal = np.array([-tangent[1], tangent[0]])
    local = np.stack(((raw[:, :, [0, 2]] - points[0]) @ tangent,
                      raw[:, :, 1] - floor_height,
                      (raw[:, :, [0, 2]] - points[0]) @ normal), axis=2)
    cross = np.cross(raw[:, 1] - raw[:, 0], raw[:, 2] - raw[:, 0])
    magnitudes = np.linalg.norm(cross, axis=1)
    normals = np.divide(cross, magnitudes[:, None], out=np.zeros_like(cross),
                        where=magnitudes[:, None] > 1e-12)
    facing = normals[:, [0, 2]] @ normal
    selected = ((local[:, :, 0].max(1) >= -anchor_before_m)
                & (local[:, :, 0].min(1) <= length + anchor_after_m)
                & (local[:, :, 1].max(1) >= .2) & (local[:, :, 1].min(1) <= 1.8)
                & (local[:, :, 2].max(1) >= -.05) & (local[:, :, 2].min(1) <= .05)
                & (abs(facing) >= math.cos(math.radians(25))))
    indices = np.flatnonzero(selected)
    parents, seen = list(range(len(indices))), {}

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for index, identifier in enumerate(indices):
        for vertex in raw[identifier]:
            key = (bool(facing[identifier] > 0), *np.rint(vertex / 1e-6).astype(np.int64))
            if key in seen:
                parents[root(index)] = root(seen[key])
            else:
                seen[key] = index
    groups = {}
    for index, identifier in enumerate(indices):
        groups.setdefault(root(index), []).append(int(identifier))
    components, polygons = [], []
    for identifiers in groups.values():
        projected, used = [], []
        for identifier in identifiers:
            polygon = list(local[identifier])
            for axis, value, greater in ((0, -anchor_before_m, True),
                                         (0, length + anchor_after_m, False),
                                         (1, .2, True), (1, 1.8, False),
                                         (2, -.05, True), (2, .05, False)):
                if len(polygon) < 3:
                    break
                polygon = _clip(polygon, axis, value, greater)
            if len(polygon) >= 3:
                projected.append(np.asarray(polygon)[:, :2])
                used.append(identifier)
        if not projected:
            continue
        bands = [min(1., _area(projected, 0, length, low, high) / (length * (high - low)))
                 for low, high in ((.2, .6), (.6, 1.), (1., 1.4), (1.4, 1.8))]
        coverage = min(1., _area(projected, 0, length, .2, 1.8) / (length * 1.6))
        anchors = []
        for low, high in ((-anchor_before_m, 0), (length, length + anchor_after_m)):
            if high > low:
                anchors.append(min(1., _area(projected, low, high, .2, 1.8) / ((high - low) * 1.6)))
        passed = coverage >= .95 - 1e-9 and min(bands) >= .9 - 1e-9 and min(anchors, default=1.) >= .9 - 1e-9
        components.append({"pass": bool(passed), "body_area_coverage": coverage,
                           "height_band_coverage": bands, "anchor_area_coverage": anchors,
                           "raw_triangle_indices": used,
                           "median_normal_offset_m": float(np.median(local[used, :, 2]))})
        polygons.append(projected)
    best = max(range(len(components)),
               key=lambda i: (components[i]["pass"], components[i]["body_area_coverage"]), default=None)
    passed = best is not None and components[best]["pass"]
    reason = "connected_raw_wall_body_and_continuation" if passed else "insufficient_connected_raw_wall_body"
    conflicts = []
    if best is not None:
        a = _area(polygons[best], 0, length, .2, 1.8)
        for index in range(len(components)):
            if index == best:
                continue
            b = _area(polygons[index], 0, length, .2, 1.8)
            overlap = a + b - _area(polygons[best] + polygons[index], 0, length, .2, 1.8)
            separation = abs(components[best]["median_normal_offset_m"] - components[index]["median_normal_offset_m"])
            if overlap > 1e-6 and separation > .001:
                conflicts.append({"component_index": index, "projected_overlap_m2": overlap,
                                  "median_sheet_separation_m": separation})
        if conflicts:
            passed, reason = False, "ambiguous_multiple_raw_surface_sheets"
    return {"pass": bool(passed), "reason": reason, "source": "original_render_mesh_connected_3d_triangles",
            "raw_triangle_id_space": "row_in_input_raw_triangles", "segment": points.tolist(),
            "body_height_range_m": [.2, 1.8], "plane_band_m": .05,
            "surface_angle_limit_degrees": 25., "anchor_extents_m": [anchor_before_m, anchor_after_m],
            "area_method": "exact_union_of_clipped_triangle_projections_continuous_vertical_sweep",
            "selected_component_index": best, "components": components, "sheet_conflicts": conflicts}


def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _protected(segment, openings):
    points, length = _segment(segment)
    tangent = (points[1] - points[0]) / length
    normal = np.array([-tangent[1], tangent[0]])
    for opening in openings:
        if opening.get("status") == "rejected" or opening.get("category") == "window":
            continue
        other, width = _segment(opening["segment"])
        direction = other[1] - other[0]
        along = (other - points[0]) @ tangent
        overlap = min(length, along.max()) - max(0., along.min())
        if (abs(tangent @ (direction / width)) >= math.cos(math.radians(25))
                and abs((other - points[0]) @ normal).max() <= .05 and overlap > 1e-8):
            return opening.get("id", "unnamed_opening")
        determinant = _cross(points[1] - points[0], direction)
        if abs(determinant) > 1e-12:
            delta = other[0] - points[0]
            t, u = _cross(delta, direction) / determinant, _cross(delta, points[1] - points[0]) / determinant
            if 1e-8 < t < 1 - 1e-8 and 1e-8 < u < 1 - 1e-8:
                return opening.get("id", "unnamed_opening")
    return None


def _navigation(callback, segment):
    value = callback(np.asarray(segment).tolist())
    if not isinstance(value, dict) or type(value.get("pass")) is not bool:
        raise ValueError("not_blocks_navmesh must return an explicit boolean pass")
    return value


def _joint_line(items, source_segments, fixed_points):
    ids = sorted({i for item in items for i in item.get("source_segment_indices", [])})
    if source_segments is not None and all(item.get("source_segment_indices") for item in items):
        if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= i < len(source_segments) for i in ids):
            raise ValueError("Invalid original source segment indices")
        observed = source_segments[ids]
        method = "joint_original_segment_tls"
    else:
        ids = []
        observed = np.asarray([[item["start"], item["end"]] for item in items])
        method = "joint_accepted_line_endpoint_tls"
    lengths = np.linalg.norm(observed[:, 1] - observed[:, 0], axis=1)
    if (lengths <= 1e-10).any():
        raise ValueError("Original source segments must have positive length")
    points, weights = observed.reshape(-1, 2), np.repeat(lengths / 2, 2)
    fixed = np.unique(np.asarray(fixed_points).reshape(-1, 2), axis=0)
    center = fixed[0] if len(fixed) else np.average(points, axis=0, weights=weights)
    local = points - center
    if len(fixed) >= 2:
        tangent = fixed[np.argmax(np.linalg.norm(fixed - center, axis=1))] - center
        tangent /= np.linalg.norm(tangent)
    else:
        _, vectors = np.linalg.eigh((local * weights[:, None]).T @ local)
        tangent = vectors[:, -1]
    normal = np.array([-tangent[1], tangent[0]])
    old = np.asarray([[item["start"], item["end"]] for item in items])
    displacements = abs((old - center) @ normal).max(1)
    residual = abs(local @ normal)
    position = (old - center) @ tangent
    segment = center + np.asarray([position.min(), position.max()])[:, None] * tangent
    fit = {"method": method, "max_residual_m": float(residual.max()),
           "rms_m": float(np.sqrt(np.average(residual ** 2, weights=weights))),
           "max_original_line_displacement_m": float(displacements.max()),
           "original_line_displacements_m": dict(zip((item["id"] for item in items), displacements.tolist())),
           "fixed_existing_junctions": fixed.tolist(), "source_segment_indices": ids}
    passed = residual.max() <= .05 + 1e-10 and displacements.max() <= .03 + 1e-10
    if len(fixed) and abs((fixed - center) @ normal).max() > 1e-6:
        passed = False
    return segment, center, tangent, fit, bool(passed)


def complete_wall_gaps(primitives, raw_triangles, *, floor_height, not_blocks_navmesh,
                       source_segments=None, portals=(), semantic_priors=(), max_gap_m=.45):
    """Recover short breaks and missing corners, returning auditable geometry.

    Straight-break candidates need two facing free endpoints and a connected
    full-height raw face with continuation anchors on both sides. They replace
    the two fragments with one joint line (<=3 cm old-line movement, <=5 cm
    source residual); existing junction coordinates are fixed. A corner extends
    a free endpoint on its existing line to another existing finite wall, with
    the same raw-body proof and an anchor behind it. Original input is immutable.

    Accepted/undecided portals and semantic door priors protect real openings,
    including closed leaves. The mandatory NavMesh callback checks every output
    line; it is never used as evidence that a wall exists.
    """
    if not math.isfinite(max_gap_m) or not 0 < max_gap_m <= .45:
        raise ValueError("max_gap_m must be within (0, .45]")
    raw = np.asarray(raw_triangles, dtype=float).reshape(-1, 3, 3)
    if not np.isfinite(raw).all() or not math.isfinite(floor_height):
        raise ValueError("Raw triangles and floor_height must be finite")
    result = deepcopy(list(primitives))
    if len({item["id"] for item in result}) != len(result):
        raise ValueError("Primitive IDs must be unique")
    source = None if source_segments is None else np.asarray(source_segments, dtype=float).reshape(-1, 2, 2)
    if source is not None and not np.isfinite(source).all():
        raise ValueError("Source segments must be finite")
    portal_values = deepcopy(list(portals))
    protected = portal_values + [deepcopy(p) for p in semantic_priors if p["category"] == "door"]
    # Door labels and undecided portals veto wall additions, but only measured
    # accepted portals may label graph endpoints as physical opening jambs.
    graph_portals = [deepcopy(p) for p in portal_values if p.get("status") == "accepted"]
    nodes, _, unexplained_before = junction_graph(result, graph_portals)
    by_id = {item["id"]: item for item in result}
    ends, fixed = [], {}
    for node in nodes:
        for incidence in node["incidences"]:
            item = by_id[incidence["primitive_id"]]
            if item["type"] != "line" or item.get("representation") == "wall_axis":
                continue
            if incidence["at"] == "interior":
                fixed.setdefault(item["id"], []).append(node["point"])
                continue
            if node["kind"] not in {"endpoint", "portal_jamb"}:
                fixed.setdefault(item["id"], []).append(node["point"])
                continue
            if node["kind"] == "portal_jamb":
                fixed.setdefault(item["id"], []).append(node["point"])
            at = incidence["at"]
            point = np.asarray(item[at])
            direction = point - np.asarray(item["end" if at == "start" else "start"])
            ends.append((item["id"], at, point, direction / np.linalg.norm(direction)))
    candidates = []
    for index, (identifier, at, point, direction) in enumerate(ends):
        for other_id, other_at, other_point, other_direction in ends[index + 1:]:
            delta = other_point - point
            length = float(np.linalg.norm(delta))
            if identifier == other_id or not 1e-5 < length <= max_gap_m:
                continue
            if min(direction @ (delta / length), other_direction @ (-delta / length)) >= math.cos(math.radians(25)):
                candidates.append({"kind": "short_break", "input_ids": [identifier, other_id],
                                   "segment": [point.tolist(), other_point.tolist()], "length_m": length})
        for other_id, other in by_id.items():
            if (other_id == identifier or other["type"] != "line"
                    or other.get("representation") == "wall_axis"):
                continue
            carrier, _ = _segment([other["start"], other["end"]])
            other_direction = carrier[1] - carrier[0]
            determinant = _cross(direction, other_direction)
            if abs(determinant) < .5 * np.linalg.norm(other_direction):
                continue
            shift = _cross(carrier[0] - point, other_direction) / determinant
            fraction = _cross(carrier[0] - point, direction) / determinant
            if 1e-5 < shift <= max_gap_m and -1e-8 <= fraction <= 1 + 1e-8:
                candidates.append({"kind": "missing_corner", "input_ids": [identifier, other_id],
                                   "at": at, "segment": [point.tolist(), (point + shift * direction).tolist()],
                                   "length_m": shift})
    accepted, rejected, consumed = [], [], set()
    for candidate in sorted(candidates, key=lambda c: (c["length_m"], c["input_ids"])):
        record = deepcopy(candidate)
        items = [by_id[i] for i in candidate["input_ids"]]
        reason = None
        if any(identifier in consumed for identifier in candidate["input_ids"]):
            reason = "source_already_changed_by_an_earlier_completion"
        missing = np.asarray(candidate["segment"])
        if reason is None and candidate["kind"] == "short_break":
            segment, center, tangent, fit, passed = _joint_line(
                items, source, [point for item in items for point in fixed.get(item["id"], [])])
            record["joint_fit"] = fit
            if not passed:
                reason = "joint_fit_residual_or_displacement_exceeded"
            missing = center + ((missing - center) @ tangent)[:, None] * tangent
        elif reason is None:
            segment = np.asarray([items[0]["start"], items[0]["end"]], dtype=float)
            segment[("start", "end").index(candidate["at"])] = missing[1]
        if reason is None:
            protected_id = _protected(segment, protected)
            if protected_id is not None:
                reason = "protected_door_or_portal"
                record["protected_opening_id"] = protected_id
        if reason is None:
            tangent = (missing[1] - missing[0]) / np.linalg.norm(missing[1] - missing[0])
            normal = np.array([-tangent[1], tangent[0]])
            for other in result:
                if other["id"] in candidate["input_ids"] or other["type"] != "line":
                    continue
                line, size = _segment([other["start"], other["end"]])
                along = (line - missing[0]) @ tangent
                overlap = min(np.linalg.norm(missing[1] - missing[0]), along.max()) - max(0., along.min())
                if (abs((line[1] - line[0]) @ tangent) / size >= math.cos(math.radians(25))
                        and abs((line - missing[0]) @ normal).max() <= .03 and overlap > .001):
                    reason = "missing_span_already_covered_by_an_existing_wall"
                    record["overlapping_primitive_id"] = other["id"]
                    break
        if reason is None:
            proof = wall_surface_evidence(missing, raw, floor_height=floor_height,
                anchor_after_m=.1 if candidate["kind"] == "short_break" else 0.)
            record["raw_surface"] = proof
            record["completed_missing_segment"] = missing.tolist()
            if not proof["pass"]:
                reason = proof["reason"]
        if reason is None:
            navigation = _navigation(not_blocks_navmesh, segment)
            record["navmesh"] = navigation
            if not navigation["pass"]:
                reason = "would_block_navmesh"
        if reason is not None:
            record.update(status="rejected", reason=reason)
            rejected.append(record)
            continue
        carrier = max(items, key=lambda item: np.linalg.norm(np.subtract(item["end"], item["start"]))) if candidate["kind"] == "short_break" else items[0]
        value = deepcopy(carrier)
        for key in ("start_node", "end_node", "interior_nodes", "cad", "evidence"):
            value.pop(key, None)
        record.update(status="accepted", reason="connected_full_3d_wall_body_and_protected_passages",
                      output_primitive_id=value["id"], original_primitives=deepcopy(items))
        value.update(start=segment[0].tolist(), end=segment[1].tolist(), endpoints=segment.tolist(),
                     length_m=float(np.linalg.norm(segment[1] - segment[0])), completion=deepcopy(record),
                     graph_requires_rebuild=True, evidence_kind="connected_raw_3d_wall_completion")
        if candidate["kind"] == "short_break":
            value.update(source_ids=sorted({s for item in items for s in item.get("source_ids", [item["id"]])}),
                         source_primitive_ids=candidate["input_ids"], fitting=deepcopy(fit),
                         source_segment_indices=fit["source_segment_indices"],
                         source_segment_count=len(fit["source_segment_indices"]))
            result = [item for item in result if item["id"] not in candidate["input_ids"]] + [value]
            consumed.update(candidate["input_ids"])
        else:
            result = [value if item["id"] == value["id"] else item for item in result]
            consumed.add(value["id"])
        accepted.append(record)
    nodes, edges, unexplained = junction_graph(result, graph_portals)
    return {"primitives": result, "accepted": accepted, "rejected": rejected,
            "nodes": nodes, "edges": edges, "unexplained_endpoints": unexplained,
            "unexplained_endpoints_before": unexplained_before,
            "parameters": {"max_gap_m": max_gap_m, "max_joint_fit_movement_m": .03,
                           "max_source_residual_m": .05},
            "input_primitives_modified": False, "all_openings_closed": False}
