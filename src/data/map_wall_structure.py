"""Evidence-gated wall junctions and CAD-style line/arc/circle rendering.

Coordinates are world [X, Z] metres. Arc parameters are immutable: this module
does not approximate curves by straight walls or infer unsupported boundaries.
"""

from copy import deepcopy
import html
import math

import numpy as np
from PIL import Image, ImageDraw


COLLINEAR_OFFSET_M = .015
NODE_TOLERANCE_M = .000001


def _point(value):
    point = np.asarray(value, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("Wall points must contain finite world [X, Z] coordinates")
    return point


def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _intersection(first, second):
    direction, other = first[1] - first[0], second[1] - second[0]
    determinant = _cross(direction, other)
    if abs(determinant) < 1e-12:
        return None
    return first[0] + _cross(second[0] - first[0], other) / determinant * direction


def _on_segment(point, segment):
    direction = segment[1] - segment[0]
    length2 = float(direction @ direction)
    if length2 < NODE_TOLERANCE_M ** 2:
        return False
    t = float((point - segment[0]) @ direction / length2)
    return (-1e-8 <= t <= 1 + 1e-8
            and np.linalg.norm(point - segment[0] - t * direction) <= NODE_TOLERANCE_M)


def _curve_angles(primitive):
    if primitive["type"] == "circle":
        return 0., 360.
    start = float(primitive["start_angle_degrees"])
    return start, float(primitive.get("sweep_degrees", primitive["end_angle_degrees"] - start))


def _arc_points(primitive, step_m=.025):
    center, radius = _point(primitive["center"]), float(primitive["radius_m"])
    start, sweep = _curve_angles(primitive)
    if not all(math.isfinite(value) for value in (radius, start, sweep)) or radius <= 0:
        raise ValueError("Arcs require finite angles and a positive radius_m")
    if math.radians(abs(sweep)) * radius < NODE_TOLERANCE_M:
        raise ValueError("Wall arcs must have nonzero length")
    count = max(2, math.ceil(math.radians(abs(sweep)) * radius / step_m) + 1)
    angles = np.radians(np.linspace(start, start + sweep, count))
    return center + radius * np.column_stack((np.cos(angles), np.sin(angles)))


def _endpoints(primitive):
    if primitive["type"] in {"arc", "circle"}:
        return _arc_points(primitive)[[0, -1]]
    if primitive["type"] != "line":
        raise ValueError("Wall primitives must have type line, arc or circle")
    points = np.stack((_point(primitive["start"]), _point(primitive["end"])))
    if np.linalg.norm(points[1] - points[0]) < NODE_TOLERANCE_M:
        raise ValueError("Wall lines must have nonzero length")
    return points


def _arc_parameter(primitive, point):
    """Position on a directed circular arc, or None outside its actual sweep."""
    relative = np.asarray(point) - _point(primitive["center"])
    radius = float(primitive["radius_m"])
    if abs(np.linalg.norm(relative) - radius) > NODE_TOLERANCE_M:
        return None
    start, sweep = _curve_angles(primitive)
    angle = math.degrees(math.atan2(relative[1], relative[0]))
    delta = ((angle - start) if sweep > 0 else (start - angle)) % 360
    tolerance = math.degrees(NODE_TOLERANCE_M / radius)
    if 360 - delta <= tolerance:
        delta = 0.
    if delta > abs(sweep) + tolerance:
        return None
    return min(1., delta / abs(sweep))


def _line_arc_junctions(segment, primitive):
    """Analytic intersections; neither the circle nor its angular extent moves."""
    a, b = segment
    direction = b - a
    relative = a - _point(primitive["center"])
    aa, bb = float(direction @ direction), 2 * float(relative @ direction)
    cc = float(relative @ relative) - float(primitive["radius_m"]) ** 2
    discriminant = bb * bb - 4 * aa * cc
    if discriminant < -1e-10:
        return []
    values = []
    for t in sorted({(-bb - math.sqrt(max(0., discriminant))) / (2 * aa),
                     (-bb + math.sqrt(max(0., discriminant))) / (2 * aa)}):
        point = a + t * direction
        if _arc_parameter(primitive, point) is not None:
            values.append(point)
    return values


def _portal_intersection(segment, portal):
    """Protect the portal interior; touching its jamb endpoint is allowed."""
    a, b = np.asarray(segment), np.asarray(portal)
    direction, other = a[1] - a[0], b[1] - b[0]
    determinant = _cross(direction, other)
    if abs(determinant) < 1e-10:
        if abs(_cross(b[0] - a[0], direction)) > 1e-8:
            return False
        length2 = float(direction @ direction)
        if length2 < 1e-16:
            return False
        positions = sorted(float((point - a[0]) @ direction / length2) for point in b)
        return min(1., positions[1]) - max(0., positions[0]) > 1e-8
    offset = b[0] - a[0]
    t, u = _cross(offset, other) / determinant, _cross(offset, direction) / determinant
    return -1e-8 <= t <= 1 + 1e-8 and 1e-8 < u < 1 - 1e-8


def _candidates(primitives, max_shift, collinear_gap):
    lines = [(index, _endpoints(item)) for index, item in enumerate(primitives) if item["type"] == "line"]
    ends = [(index, endpoint, points[endpoint]) for index, points in lines for endpoint in (0, 1)]
    positions = np.array([point for _, _, point in ends])
    candidates = []
    for i, (index, endpoint, point) in enumerate(ends):
        prior = _endpoints(primitives[index])
        direction = prior[1] - prior[0]
        close = np.linalg.norm(positions - point, axis=1) <= max(2 * max_shift, collinear_gap)
        for j in np.flatnonzero(close):
            other_index, other_endpoint, other_point = ends[j]
            if j <= i or other_index == index or np.linalg.norm(point - other_point) < NODE_TOLERANCE_M:
                continue
            other = _endpoints(primitives[other_index])
            other_direction = other[1] - other[0]
            scale = np.linalg.norm(direction) * np.linalg.norm(other_direction)
            sine = abs(_cross(direction, other_direction)) / scale
            kind = "endpoint_corner"
            if sine < math.sin(math.radians(15)):
                offset = abs(_cross(other_point - point, direction)) / np.linalg.norm(direction)
                outward = point - prior[1 - endpoint]
                other_outward = other_point - other[1 - other_endpoint]
                if (sine > math.sin(math.radians(5)) or offset > COLLINEAR_OFFSET_M
                        or float(outward @ (other_point - point)) <= 0
                        or float(other_outward @ (point - other_point)) <= 0):
                    continue
                junction, kind = (point + other_point) / 2, "collinear_gap"
            else:
                t = _cross(other[0] - prior[0], other_direction) / _cross(direction, other_direction)
                junction = prior[0] + t * direction
            shifts = [float(np.linalg.norm(junction - point)), float(np.linalg.norm(junction - other_point))]
            if kind != "collinear_gap" and max(shifts) > max_shift + 1e-9:
                continue
            candidates.append({"kind": kind, "ends": [(index, endpoint), (other_index, other_endpoint)],
                               "junction": junction, "maximum_shift_m": max(shifts)})

        for carrier_index, carrier in lines:
            if carrier_index == index:
                continue
            carrier_direction = carrier[1] - carrier[0]
            determinant = _cross(direction, carrier_direction)
            if abs(determinant) < math.sin(math.radians(15)) * np.linalg.norm(direction) * np.linalg.norm(carrier_direction):
                continue
            t = _cross(carrier[0] - prior[0], carrier_direction) / determinant
            junction = prior[0] + t * direction
            u = float((junction - carrier[0]) @ carrier_direction / (carrier_direction @ carrier_direction))
            shift = float(np.linalg.norm(junction - point))
            if 1e-6 < u < 1 - 1e-6 and NODE_TOLERANCE_M < shift <= max_shift + 1e-9:
                candidates.append({"kind": "endpoint_to_interior", "ends": [(index, endpoint)],
                                   "carrier": carrier_index, "junction": junction, "maximum_shift_m": shift})
        for carrier_index, carrier in enumerate(primitives):
            if carrier["type"] not in {"arc", "circle"}:
                continue
            for junction in _line_arc_junctions(prior, carrier):
                shift = float(np.linalg.norm(junction - point))
                if NODE_TOLERANCE_M < shift <= max_shift + 1e-9:
                    candidates.append({"kind": "endpoint_to_arc", "ends": [(index, endpoint)],
                                       "carrier": carrier_index, "junction": junction,
                                       "maximum_shift_m": shift})
    return sorted(candidates, key=lambda item: (item["maximum_shift_m"], item["kind"], item["ends"]))


def _evidence(callback, segment, name):
    result = callback(np.asarray(segment).tolist())
    if not isinstance(result, dict) or type(result.get("pass")) is not bool:
        raise ValueError(name + " must return a dict with an explicit boolean pass")
    return result


def _compatible_near_endpoints(first, second, primitives):
    """Close parallel faces are not evidence of one architectural junction."""
    index, endpoint, point = first
    other_index, other_endpoint, other_point = second
    if np.linalg.norm(point - other_point) <= NODE_TOLERANCE_M:
        return True
    segment, other = _endpoints(primitives[index]), _endpoints(primitives[other_index])
    direction, other_direction = segment[1] - segment[0], other[1] - other[0]
    sine = abs(_cross(direction, other_direction)) / (np.linalg.norm(direction) * np.linalg.norm(other_direction))
    if sine >= math.sin(math.radians(15)):
        return True
    offset = abs(_cross(other_point - point, direction)) / np.linalg.norm(direction)
    return (offset <= NODE_TOLERANCE_M
            and float((point - segment[1 - endpoint]) @ (other_point - other[1 - other_endpoint])) < 0)


def junction_graph(primitives, portals):
    """Attach node IDs to the supplied primitives without changing geometry."""
    nodes, indexed = [], {}
    for primitive in primitives:
        ids = []
        for endpoint, point in zip(("start", "end"), _endpoints(primitive)):
            key = tuple(np.floor(point / NODE_TOLERANCE_M).astype(np.int64))
            nearby = [index for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                      for index in indexed.get((key[0] + dx, key[1] + dy), [])
                      if np.linalg.norm(np.asarray(nodes[index]["point"]) - point) <= NODE_TOLERANCE_M]
            if nearby:
                index = min(nearby, key=lambda index: np.linalg.norm(np.asarray(nodes[index]["point"]) - point))
            else:
                index = len(nodes)
                indexed.setdefault(key, []).append(index)
                nodes.append({"id": f"J{len(nodes) + 1:04d}", "point": point.tolist(), "incidences": []})
            node = nodes[index]
            node["incidences"].append({"primitive_id": primitive["id"], "at": endpoint})
            ids.append(node["id"])
        primitive["start_node"], primitive["end_node"] = ids
        primitive["interior_nodes"] = []

    edges = []
    for primitive in primitives:
        ordered = [(0., primitive["start_node"]), (1., primitive["end_node"])]
        if primitive["type"] == "line":
            a, b = _endpoints(primitive)
            direction = b - a
        for node in nodes:
            if primitive["type"] == "line":
                point = np.asarray(node["point"])
                t = float((point - a) @ direction / (direction @ direction))
                on_curve = np.linalg.norm(point - a - t * direction) <= NODE_TOLERANCE_M
            else:
                t = _arc_parameter(primitive, node["point"])
                on_curve = t is not None
            if on_curve and 1e-6 < t < 1 - 1e-6:
                ordered.append((t, node["id"]))
                primitive["interior_nodes"].append(node["id"])
                node["incidences"].append({"primitive_id": primitive["id"], "at": "interior", "parameter": t})
        ordered.sort()
        edges.extend({"primitive_id": primitive["id"], "start_node": a[1], "end_node": b[1]}
                     for a, b in zip(ordered, ordered[1:]) if a[1] != b[1])
    unexplained = []
    for node in nodes:
        incident = {item["primitive_id"] for item in node["incidences"]}
        closed = len(node["incidences"]) == 2 and len(incident) == 1
        jamb = any(np.linalg.norm(np.asarray(node["point"]) - point) <= .15
                   for portal in portals for point in np.asarray(portal["segment"]))
        node["kind"] = "junction" if len(incident) > 1 else "closed_primitive" if closed else "portal_jamb" if jamb else "endpoint"
        if node["kind"] == "endpoint":
            unexplained.append(deepcopy(node))
    return nodes, edges, unexplained


def reconnect_primitives(primitives, *, raw_support, not_blocks_navmesh,
                         portals=(), max_shift_m=.15, collinear_gap_m=.4,
                         min_line_length_m=.05, endpoint_merge_tolerance_m=.05):
    """Connect straight walls to lines or fixed arcs after independent checks.

    Callbacks receive a proposed world-XZ connector and must return
    ``{"pass": bool, ...evidence}``. Larger collinear gaps are explicit rejected
    candidates, never a reason to exceed the endpoint displacement limit.
    """
    if not 0 < max_shift_m <= .15 or collinear_gap_m < max_shift_m:
        raise ValueError("Require 0 < max_shift_m <= .15 and collinear_gap_m >= max_shift_m")
    if not math.isfinite(min_line_length_m) or min_line_length_m <= NODE_TOLERANCE_M:
        raise ValueError("min_line_length_m must be positive and exceed node tolerance")
    if not math.isfinite(endpoint_merge_tolerance_m) or not 0 < endpoint_merge_tolerance_m <= .05:
        raise ValueError("endpoint_merge_tolerance_m must be in (0, .05]")
    result = deepcopy(list(primitives))
    if len({item["id"] for item in result}) != len(result):
        raise ValueError("Wall primitive IDs must be unique")
    for item in result:
        _endpoints(item)
    protected = [portal for portal in portals if portal.get("status", "accepted") != "rejected"]
    for portal in protected:
        np.stack([_point(point) for point in portal["segment"]]).reshape(2, 2)
    original = [_endpoints(item) for item in result]
    connections, rejected, fixed, interiors = [], [], {}, {}
    initial_nodes, _, _ = junction_graph(result, protected)
    indices = {item["id"]: index for index, item in enumerate(result)}
    for node in initial_nodes:
        if node["kind"] != "junction":
            continue
        for incidence in node["incidences"]:
            index = indices[incidence["primitive_id"]]
            if incidence["at"] == "interior":
                interiors.setdefault(index, []).append(np.asarray(node["point"]))
            else:
                fixed[(index, ("start", "end").index(incidence["at"]))] = np.asarray(node["point"])

    # Multi-height cuts often end a few millimetres apart at one architectural
    # corner. Merge a compact endpoint cluster only after the same raw-surface
    # and walkability checks used for longer connectors.
    endpoint_items = [(index, endpoint, _endpoints(item)[endpoint])
                      for index, item in enumerate(result) if item["type"] == "line"
                      for endpoint in (0, 1)]
    used = set()
    for seed, (index, endpoint, point) in enumerate(endpoint_items):
        if seed in used:
            continue
        member_numbers, members = [seed], [(index, endpoint, point)]
        for item_no, item in enumerate(endpoint_items):
            if item_no in used or any(item[0] == member[0] for member in members):
                continue
            if (all(np.linalg.norm(item[2] - member[2]) <= endpoint_merge_tolerance_m for member in members)
                    and all(_compatible_near_endpoints(item, member, result) for member in members)):
                member_numbers.append(item_no)
                members.append(item)
        if len(members) < 2:
            continue
        anchors = [fixed[(i, e)] for i, e, _ in members if (i, e) in fixed]
        centre = anchors[0].copy() if anchors else np.mean([item[2] for item in members], axis=0)
        records = [{"primitive_id": result[i]["id"], "at": ("start", "end")[e]} for i, e, _ in members]
        maximum_shift = max(float(np.linalg.norm(original[i][e] - centre)) for i, e, _ in members)
        segments, reason = [], None
        if any(np.linalg.norm(anchor - centre) > NODE_TOLERANCE_M for anchor in anchors):
            reason = "endpoint_already_assigned_to_another_junction"
        elif maximum_shift > max_shift_m + 1e-9:
            reason = "endpoint_displacement_limit"
        for member_index, member_endpoint, member_point in members:
            if reason is not None:
                break
            proposed = _endpoints(result[member_index])
            proposed[member_endpoint] = centre
            minimum = min(min_line_length_m,
                          float(np.linalg.norm(original[member_index][1] - original[member_index][0])))
            segment = np.stack((member_point, centre))
            if np.linalg.norm(proposed[1] - proposed[0]) < minimum - NODE_TOLERANCE_M:
                reason = "primitive_below_minimum_length"
                break
            if not all(_on_segment(point, proposed) for point in interiors.get(member_index, [])):
                reason = "would_break_existing_interior_junction"
                break
            if any(_portal_intersection(segment, portal["segment"]) for portal in protected):
                reason = "protected_portal_interior"
                break
            if np.linalg.norm(segment[1] - segment[0]) > NODE_TOLERANCE_M:
                segments.append(segment)
        checks = [{"segment": segment.tolist(),
                   "raw_support": _evidence(raw_support, segment, "raw_support"),
                   "not_blocks_navmesh": _evidence(not_blocks_navmesh, segment, "not_blocks_navmesh")}
                  for segment in segments] if reason is None else []
        if reason is None and not all(check["raw_support"]["pass"] and
                                      check["not_blocks_navmesh"]["pass"] for check in checks):
            reason = "independent_evidence_failed"
        record = {"kind": "near_endpoint_cluster", "junction": centre.tolist(),
                  "endpoints": records,
                  "maximum_shift_m": maximum_shift}
        if reason is not None:
            rejected.append({**record, "reason": reason, "checks": checks})
            continue
        if not segments:
            used.update(member_numbers)
            continue
        for member_index, member_endpoint, _ in members:
            item = result[member_index]
            item[("start", "end")[member_endpoint]] = centre.tolist()
            item["endpoints"] = [item["start"], item["end"]]
            item["length_m"] = float(np.linalg.norm(np.asarray(item["end"]) - item["start"]))
            fixed[(member_index, member_endpoint)] = centre.copy()
        used.update(member_numbers)
        connections.append({**record, "checks": checks})
    for candidate in _candidates(result, max_shift_m, collinear_gap_m):
        lines = [_endpoints(result[index]) for index, _ in candidate["ends"]]
        if candidate["kind"] == "collinear_gap":
            junction = np.mean([line[endpoint] for line, (_, endpoint) in zip(lines, candidate["ends"])], axis=0)
        elif candidate["kind"] == "endpoint_to_arc":
            carrier_primitive = result[candidate["carrier"]]
            intersections = _line_arc_junctions(lines[0], carrier_primitive)
            junction = min(intersections, key=lambda point: np.linalg.norm(point - candidate["junction"])) if intersections else None
        else:
            carrier = _endpoints(result[candidate["carrier"]]) if "carrier" in candidate else lines[1]
            junction = _intersection(lines[0], carrier)
        if junction is None:
            continue
        maximum_shift = max(float(np.linalg.norm(original[index][endpoint] - junction))
                            for index, endpoint in candidate["ends"])
        record = {"kind": candidate["kind"], "junction": junction.tolist(),
                  "endpoints": [{"primitive_id": result[index]["id"], "at": ("start", "end")[endpoint]}
                                for index, endpoint in candidate["ends"]],
                  "maximum_shift_m": maximum_shift}
        if "carrier" in candidate:
            record["carrier_primitive_id"] = result[candidate["carrier"]]["id"]
        segments = [np.stack((_endpoints(result[index])[endpoint], junction))
                    for index, endpoint in candidate["ends"]]
        segments = [segment for segment in segments if np.linalg.norm(segment[1] - segment[0]) > NODE_TOLERANCE_M]
        reason = None
        if any(end in fixed and np.linalg.norm(fixed[end] - junction) > NODE_TOLERANCE_M
               for end in candidate["ends"]):
            reason = "endpoint_already_assigned_to_another_junction"
        elif any(_portal_intersection(segment, portal["segment"]) for segment in segments for portal in protected):
            reason = "protected_portal_interior"
        elif maximum_shift > max_shift_m + 1e-9:
            reason = "endpoint_displacement_limit"
        elif ("carrier" in candidate and candidate["kind"] != "endpoint_to_arc"
              and not _on_segment(junction, carrier)):
            reason = "carrier_no_longer_contains_junction"
        for index, endpoint in candidate["ends"]:
            proposed = _endpoints(result[index])
            proposed[endpoint] = junction
            minimum = min(min_line_length_m, float(np.linalg.norm(original[index][1] - original[index][0])))
            if np.linalg.norm(proposed[1] - proposed[0]) < minimum - NODE_TOLERANCE_M:
                reason = reason or "primitive_below_minimum_length"
            elif not all(_on_segment(point, proposed) for point in interiors.get(index, [])):
                reason = reason or "would_break_existing_interior_junction"
        if reason:
            rejected.append({**record, "reason": reason})
            continue
        if not segments:
            continue
        checks = [{"segment": segment.tolist(), "raw_support": _evidence(raw_support, segment, "raw_support"),
                   "not_blocks_navmesh": _evidence(not_blocks_navmesh, segment, "not_blocks_navmesh")}
                  for segment in segments]
        if not all(check["raw_support"]["pass"] and check["not_blocks_navmesh"]["pass"] for check in checks):
            rejected.append({**record, "reason": "independent_evidence_failed", "checks": checks})
            continue
        for index, endpoint in candidate["ends"]:
            item = result[index]
            item[("start", "end")[endpoint]] = junction.tolist()
            item["endpoints"] = [item["start"], item["end"]]
            item["length_m"] = float(np.linalg.norm(np.asarray(item["end"]) - item["start"]))
            fixed[(index, endpoint)] = junction.copy()
        if "carrier" in candidate:
            interiors.setdefault(candidate["carrier"], []).append(junction.copy())
        connections.append({**record, "checks": checks})
    nodes, edges, unexplained = junction_graph(result, protected)
    return {"primitives": result, "nodes": nodes, "edges": edges,
            "shared_junctions": [node for node in nodes if node["kind"] == "junction"],
            "connections": connections, "rejected": rejected,
            "unexplained_endpoints": unexplained,
            "parameters": {"max_shift_m": max_shift_m, "collinear_gap_m": collinear_gap_m,
                           "collinear_offset_m": COLLINEAR_OFFSET_M, "min_line_length_m": min_line_length_m,
                           "endpoint_merge_tolerance_m": endpoint_merge_tolerance_m}}


def rasterize_primitives(primitives, origin, side_m, resolution=.05):
    """Render one CAD stroke per wall; physical thickness stays in metadata."""
    origin = _point(origin)
    size = round(side_m / resolution)
    if resolution <= 0 or size < 1 or not math.isclose(size * resolution, side_m):
        raise ValueError("Canvas side must be a positive exact multiple of resolution")
    image = Image.new("1", (size, size))
    draw = ImageDraw.Draw(image)
    for item in primitives:
        points = _arc_points(item, resolution / 2) if item["type"] in {"arc", "circle"} else _endpoints(item)
        measured = item.get("thickness_m")
        if measured is not None and (not math.isfinite(float(measured)) or float(measured) <= 0):
            raise ValueError("Measured thickness_m must be positive and finite")
        pixels = np.rint((points - origin) / resolution - .5).astype(int)
        draw.line([tuple(point) for point in pixels], fill=1, width=1)
    return np.asarray(image, dtype=bool)


def structure_svg(primitives, origin, side_m, resolution=.05):
    """Preserve true circular arcs as SVG arcs, with physical world-unit width."""
    origin = _point(origin)
    elements = []
    for item in primitives:
        start, end = _endpoints(item) - origin
        width = resolution
        identifier = html.escape(str(item["id"]), quote=True)
        style = f'id="{identifier}" fill="none" stroke="black" stroke-width="{width:g}"'
        if item["type"] == "line":
            elements.append(f'<path {style} d="M {start[0]:g} {start[1]:g} L {end[0]:g} {end[1]:g}"/>')
        else:
            radius = float(item["radius_m"])
            _, sweep = _curve_angles(item)
            if abs(sweep) >= 359.999999:
                center = np.asarray(item["center"]) - origin
                elements.append(f'<circle {style} cx="{center[0]:g}" cy="{center[1]:g}" r="{radius:g}"/>')
            else:
                elements.append(f'<path {style} d="M {start[0]:g} {start[1]:g} A {radius:g} {radius:g} 0 {int(abs(sweep)>180)} {int(sweep>0)} {end[0]:g} {end[1]:g}"/>')
    size = round(side_m / resolution)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {side_m:g} {side_m:g}">'
            '<rect width="100%" height="100%" fill="white"/>' + "".join(elements) + '</svg>')
