"""Canonical CAD lines and exact circular coverage, without inventing walls.

``canonicalize_walls`` unions overlap on an existing fitted carrier. Numerical
seams up to 1 micrometre may join; larger straight gaps need ``allow_gap_join``
evidence. Distinct parallel faces stay separate. Overlapping circular fits may
share a minimax circle only within a strict measured displacement budget.
The optional callbacks receive a world-XZ line segment and return {pass: bool}.
No raster, NavMesh, model, or raw scene is loaded by this module.
"""

from copy import deepcopy
import math

import numpy as np


NUMERICAL_M = 1e-6
TAU = 2 * math.pi


def _point(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError("CAD points must be finite world-XZ pairs")
    return result


def _line(item):
    points = np.stack((_point(item["start"]), _point(item["end"])))
    length = float(np.linalg.norm(points[1] - points[0]))
    if length <= 1e-10:
        raise ValueError("CAD lines must have nonzero length")
    tangent = (points[1] - points[0]) / length
    if tangent[0] < -1e-12 or abs(tangent[0]) <= 1e-12 and tangent[1] < 0:
        tangent *= -1
    return points, length, tangent


def _circle(item):
    center, radius = _point(item["center"]), float(item["radius_m"])
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("CAD circles require a finite positive radius_m")
    if item["type"] == "circle":
        return center, radius, 0., TAU
    start = math.radians(float(item["start_angle_degrees"]))
    sweep = math.radians(float(item["sweep_degrees"] if "sweep_degrees" in item else
                               float(item["end_angle_degrees"]) - math.degrees(start)))
    if not math.isfinite(start + sweep) or not 1e-12 < abs(sweep) <= TAU + 1e-12:
        raise ValueError("CAD arcs require a nonzero sweep no greater than one circle")
    if sweep < 0:
        start += sweep
    return center, radius, start % TAU, min(abs(sweep), TAU)


def _length(item):
    return _line(item)[1] if item["type"] == "line" else _circle(item)[1] * _circle(item)[3]


def _check(callback, segment):
    try:
        result = callback(np.asarray(segment).tolist())
        if not isinstance(result, dict) or type(result.get("pass")) is not bool:
            raise ValueError("Geometry callback must return a dict containing boolean pass")
        return deepcopy(result)
    except Exception as error:
        return {"pass": False, "error": f"{type(error).__name__}: {error}"}


def _residual_budget(first, second, tolerance):
    residuals = [item.get("fitting", {}).get("max_residual_m") for item in (first, second)]
    known = [float(value) for value in residuals if value is not None]
    if any(not math.isfinite(value) or value < 0 for value in known):
        raise ValueError("Fitting residuals must be finite and nonnegative")
    return min(tolerance, max(NUMERICAL_M, sum(known))) if known else tolerance


def _same_line(carrier, item, tolerance, angle, *, allow_parallel_offset=False):
    a, _, tangent = _line(carrier)
    b, _, other = _line(item)
    cosine = abs(float(tangent @ other))
    if cosine < math.cos(math.radians(angle)) - 1e-12:
        return False
    normal = np.array([-tangent[1], tangent[0]])
    distances = np.abs((b - a[0]) @ normal)
    # Close, exactly parallel lines can be opposite physical wall faces. That
    # interpretation belongs to the wall-axis stage, not duplicate removal.
    parallel = abs(float(tangent[0] * other[1] - tangent[1] * other[0])) <= 1e-10
    if parallel and distances.max() > NUMERICAL_M and not allow_parallel_offset:
        return False
    return bool(distances.max() <= _residual_budget(carrier, item, tolerance) + 1e-12)


def _same_circle(first, second):
    center, radius, _, _ = _circle(first)
    other, other_radius, _, _ = _circle(second)
    # Equality needs no refit. Nearby but different parameters require the
    # separately verified joint fit below.
    separation = np.linalg.norm(center - other)
    return separation + abs(radius - other_radius) <= NUMERICAL_M and separation < min(radius, other_radius)


def _projected_arc(item, center):
    original, radius, start, sweep = _circle(item)
    if np.linalg.norm(original - center) >= radius:
        return None  # Projection could fold or reverse the angular coverage.
    if sweep >= TAU:
        return 0., TAU
    angles = np.array([start, start + sweep])
    points = original + radius * np.column_stack((np.cos(angles), np.sin(angles))) - center
    projected = np.arctan2(points[:, 1], points[:, 0])
    return float(projected[0] % TAU), float((projected[1] - projected[0]) % TAU)


def _near_circle(first, second):
    center, radius, start, sweep = _circle(first)
    other, other_radius, _, _ = _circle(second)
    if np.linalg.norm(center - other) > .05 or abs(radius - other_radius) > .05:
        return False
    projected = _projected_arc(second, center)
    if projected is None:
        return False
    begin, width = projected
    overlap = sum(max(0., min(start + sweep, begin + shift + width) - max(start, begin + shift))
                  for shift in (-TAU, 0., TAU))
    return overlap * min(radius, other_radius) > NUMERICAL_M


def _arc_displacement(item, center, radius):
    """Exact maximum radial projection distance, including interior extrema."""
    original, old_radius, start, sweep = _circle(item)
    direction = math.atan2((original - center)[1], (original - center)[0])
    angles = [start, start + sweep]
    for extremum in (direction, direction + math.pi):
        position = (extremum - start) % TAU
        if position <= sweep + 1e-12:
            angles.append(start + position)
    points = original + old_radius * np.column_stack((np.cos(angles), np.sin(angles)))
    distances = np.abs(np.linalg.norm(points - center, axis=1) - radius)
    return float(distances.max()), points


def _joint_circle(items, maximum):
    """Fit close overlapping arcs and analytically verify the complete curves."""
    from scipy.optimize import minimize

    definitions = [_circle(item) for item in items]
    if any(np.linalg.norm(first[0] - second[0]) > .05 + 1e-10
           or abs(first[1] - second[1]) > .05 + 1e-10
           for index, first in enumerate(definitions) for second in definitions[index + 1:]):
        return {"pass": False, "reason": "source_circle_parameter_spread_over_5cm"}
    hint = np.mean([value[0] for value in definitions], axis=0)
    samples = []
    for center, radius, start, sweep in definitions:
        angles = np.linspace(start, start + sweep, min(256, max(12, math.ceil(radius * sweep / .025) + 1)))
        samples.append(center + radius * np.column_stack((np.cos(angles), np.sin(angles))))
    points = np.concatenate(samples)
    initial_radius = float(np.mean([value[1] for value in definitions]))
    initial_error = float(np.abs(np.linalg.norm(points - hint, axis=1) - initial_radius).max())
    initial = np.array([0., 0., initial_radius, initial_error])
    bounds = [(-.05, .05), (-.05, .05),
              (max(1e-8, min(value[1] for value in definitions) - .05),
               max(value[1] for value in definitions) + .05), (0., None)]
    solution = None
    for _ in range(3):
        local = points - hint
        def constraints(value):
            residual = np.linalg.norm(local - value[:2], axis=1) - value[2]
            return np.r_[value[3] - residual, value[3] + residual]
        solution = minimize(lambda value: value[3], initial, method="SLSQP", bounds=bounds,
                            constraints=[{"type": "ineq", "fun": constraints}],
                            options={"ftol": 1e-12, "maxiter": 200})
        if not solution.success or not np.isfinite(solution.x).all():
            return {"pass": False, "reason": "joint_circle_optimization_failed",
                    "message": str(solution.message)}
        center, radius = hint + solution.x[:2], float(solution.x[2])
        checked = [_arc_displacement(item, center, radius) for item in items]
        error = max(value[0] for value in checked)
        if error <= solution.x[3] + 1e-8:
            break
        # Add exact interior extrema missed by the sample grid, then refit.
        points = np.concatenate([points] + [value[1] for value in checked])
        initial = solution.x.copy()
        initial[3] = error
    shifts = {item["id"]: value[0] for item, value in zip(items, checked)}
    monotone = all(_projected_arc(item, center) is not None for item in items)
    parameters_bounded = all(np.linalg.norm(center - value[0]) <= .05 + 1e-10
                             and abs(radius - value[1]) <= .05 + 1e-10 for value in definitions)
    passed = error <= maximum + 1e-10 and monotone and parameters_bounded
    return {"pass": bool(passed),
            "reason": "bounded_joint_circle" if passed
                      else "joint_circle_displacement_exceeded_or_projection_folds",
            "center": center.tolist(), "radius_m": radius,
            "max_displacement_m": error, "source_displacements_m": shifts,
            "maximum_allowed_displacement_m": maximum, "sample_count": len(points),
            "circle_parameter_shifts_within_5cm": parameters_bounded,
            "verification": "analytic_extrema_over_each_entire_source_arc",
            "angular_coverage": "union_of_radially_projected_source_intervals_without_gap_fill"}


def _sources(items):
    return sorted({source for item in items for source in item.get("source_ids", [item["id"]])})


def _provenance(value, items, carrier_id):
    value = deepcopy(value)
    value["source_ids"] = _sources(items)
    indices = sorted({index for item in items for index in item.get("source_segment_indices", [])})
    if indices:
        value.update(source_segment_indices=indices, source_segment_count=len(indices))
    else:
        value.pop("source_segment_indices", None)
        value.pop("source_segment_count", None)
    value["cad"] = {"carrier_source_id": carrier_id, "input_ids": [item["id"] for item in items],
                    "input_count": len(items)}
    return value


def _arc_value(carrier, start, sweep):
    value = deepcopy(carrier)
    center, radius, _, _ = _circle(carrier)
    for key in ("start", "end", "endpoints", "start_angle_degrees", "end_angle_degrees",
                "sweep_degrees", "start_angle_radians", "end_angle_radians", "sweep_radians",
                "direction", "extent_xz"):
        value.pop(key, None)
    if sweep >= TAU:
        value.update(type="circle", center=center.tolist(), radius_m=radius, length_m=TAU * radius)
    else:
        angles = np.array([start, start + sweep])
        ends = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
        value.update(type="arc", start=ends[0].tolist(), end=ends[1].tolist(), endpoints=ends.tolist(),
                     start_angle_degrees=math.degrees(start), end_angle_degrees=math.degrees(start + sweep),
                     sweep_degrees=math.degrees(sweep), start_angle_radians=start,
                     end_angle_radians=start + sweep, sweep_radians=sweep,
                     direction="counterclockwise", length_m=radius * sweep)
    return value


def _endpoints(item):
    if item["type"] == "line":
        return _line(item)[0]
    center, radius, start, sweep = _circle(item)
    if item["type"] == "circle":
        return None
    # Source start/end node names follow the original arc direction.
    source_sweep = (item["sweep_degrees"] if "sweep_degrees" in item else
                    item["end_angle_degrees"] - item["start_angle_degrees"])
    if float(source_sweep) < 0:
        start, sweep = start + sweep, -sweep
    angles = np.array([start, start + sweep])
    return center + radius * np.column_stack((np.cos(angles), np.sin(angles)))


def _node_parameter(value, point, tolerance):
    if value["type"] == "line":
        ends, length, _ = _line(value)
        tangent = (ends[1] - ends[0]) / length
        position = float((point - ends[0]) @ tangent)
        offset = np.linalg.norm(point - ends[0] - position * tangent)
        return position if offset <= tolerance and -NUMERICAL_M <= position <= length + NUMERICAL_M else None
    center, radius, start, sweep = _circle(value)
    if abs(np.linalg.norm(point - center) - radius) > tolerance:
        return None
    angle = (math.atan2(*(point - center)[::-1]) - start) % TAU
    if TAU - angle <= NUMERICAL_M / radius:
        angle = 0.
    return angle * radius if angle <= sweep + NUMERICAL_M / radius else None


def _nodes(value, items, junctions, tolerance):
    entries, interior = [], set()
    for item in items:
        ends = _endpoints(item)
        if ends is not None:
            entries.extend((item[key], point) for key, point in zip(("start_node", "end_node"), ends)
                           if item.get(key) is not None)
        interior.update(item.get("interior_nodes", []))
    entries.extend((node["id"], _point(node["point"])) for node in junctions)
    for key in ("start_node", "end_node", "interior_nodes"):
        value.pop(key, None)
    projections = []
    if value["type"] != "line":
        tolerance = max(NUMERICAL_M, value["cad"].get("max_projection_distance_m", 0.) + NUMERICAL_M)
    for identifier, point in entries:
        position = _node_parameter(value, point, tolerance)
        if position is None:
            continue
        if value["type"] == "line":
            ends, length, _ = _line(value)
            projected = ends[0] + position * (ends[1] - ends[0]) / length
        else:
            center, radius, _, _ = _circle(value)
            projected = center + radius * (point - center) / np.linalg.norm(point - center)
        projections.append({"id": identifier, "original_point": point.tolist(),
                            "projected_point": projected.tolist(),
                            "displacement_m": float(np.linalg.norm(projected - point))})
        if value["type"] == "circle":
            interior.add(identifier)
        elif position <= NUMERICAL_M:
            value.setdefault("start_node", identifier)
        elif _length(value) - position <= NUMERICAL_M:
            value.setdefault("end_node", identifier)
        else:
            interior.add(identifier)
    interior.difference_update([value.get("start_node"), value.get("end_node")])
    value["interior_nodes"] = sorted(interior)
    value["cad"]["node_projections"] = projections


def _line_results(group, allow_join, navigation, conflicts, decisions, surface_checks):
    carrier = group[0]
    ends, _, tangent = _line(carrier)
    normal = np.array([-tangent[1], tangent[0]])
    intervals = []
    for item in group:
        positions = np.sort((_line(item)[0] - ends[0]) @ tangent)
        intervals.append((float(positions[0]), float(positions[1]), item))
    runs = []
    for low, high, item in sorted(intervals, key=lambda entry: (entry[0], entry[1], entry[2]["id"])):
        gap = low - runs[-1][1] if runs else math.inf
        join = None
        numerical_join = gap <= NUMERICAL_M
        if runs and gap >= -NUMERICAL_M:
            previous_points = np.concatenate([_line(old)[0] for old in runs[-1][2]])
            previous = previous_points[np.argmin(abs((previous_points - ends[0]) @ tangent - runs[-1][1]))]
            incoming = _line(item)[0]
            following = incoming[np.argmin(abs((incoming - ends[0]) @ tangent - low))]
            segment = np.stack((previous, following))
            actual_gap = float(np.linalg.norm(following - previous))
            numerical_join = actual_gap <= NUMERICAL_M
            if not numerical_join and allow_join is not None:
                join = {"segment": segment.tolist(), "length_m": actual_gap,
                        "added_projected_gap_m": max(0., gap), "evidence": _check(allow_join, segment)}
        if runs and (numerical_join or join is not None and join["evidence"]["pass"]):
            runs[-1][1] = max(runs[-1][1], high)
            runs[-1][2].append(item)
            if gap > 0:
                runs[-1][3].append(join or {"length_m": gap, "added_projected_gap_m": gap,
                                           "source": "numerical_seam"})
            elif join is not None:
                runs[-1][3].append(join)
        else:
            if join is not None:
                decisions.append({"action": "gap_preserved", "source_ids": _sources([runs[-1][2][-1], item]),
                                  "join": join})
            runs.append([low, high, [item], []])
    result = []
    for low, high, items, joins in runs:
        if len(items) == 1:
            value = _provenance(items[0], items, items[0]["id"])
            value["cad"].update(max_projection_distance_m=0., added_gap_length_m=0.)
            result.append((value, items))
            continue
        segment = ends[0] + np.array([low, high])[:, None] * tangent
        check = _check(navigation, segment) if navigation is not None else None
        if check is not None and not check["pass"]:
            conflicts.append({"severity": "hard", "reason": "not_blocks_navmesh_failed",
                              "source_ids": _sources(items), "proposed_segment": segment.tolist(), "evidence": check})
            result.extend((_provenance(item, [item], item["id"]), [item]) for item in items)
            continue
        value = _provenance(carrier, items, carrier["id"])
        value.update(type="line", start=segment[0].tolist(), end=segment[1].tolist(),
                     endpoints=segment.tolist(), length_m=high - low)
        projection = max(float(np.abs((_line(item)[0] - ends[0]) @ normal).max()) for item in items)
        value["cad"].update(max_projection_distance_m=projection, joins=joins,
                            added_gap_length_m=sum(join["added_projected_gap_m"] for join in joins), navigation_check=check)
        input_ids = {item["id"] for item in items}
        checks = [check for pair, check in surface_checks.items() if set(pair) <= input_ids]
        value["cad"]["same_surface_checks"] = checks
        decisions.append({"action": "line_union", "source_ids": _sources(items),
                          "max_projection_distance_m": projection, "joins": joins,
                          "same_surface_checks": checks})
        result.append((value, items))
    return result


def _arc_results(group, decisions, conflicts, navigation, maximum):
    carrier = group[0]
    exact = all(_same_circle(carrier, item) for item in group)
    fit = None if exact else _joint_circle(group, maximum)
    if fit is not None and not fit["pass"]:
        conflicts.append({"severity": "hard", "reason": "near_circular_overlap_unresolved",
                          "source_ids": _sources(group), "evidence": fit})
        return [(_provenance(item, [item], item["id"]), [item]) for item in group]
    fitted = deepcopy(carrier)
    if fit is not None:
        fitted.update(center=fit["center"], radius_m=fit["radius_m"])
    center, radius, _, _ = _circle(fitted)
    intervals = []
    for index, item in enumerate(group):
        start, sweep = _projected_arc(item, center)
        if sweep >= TAU:
            intervals.append([0., TAU, {index}])
        elif start + sweep <= TAU:
            intervals.append([start, start + sweep, {index}])
        else:
            intervals.extend([[start, TAU, {index}], [0., start + sweep - TAU, {index}]])
    runs = []
    for low, high, members in sorted(intervals, key=lambda entry: entry[0]):
        # A missing angular interval is never completed into a circle. Only
        # floating-point angle roundoff (far below the line seam allowance)
        # may connect adjacent circular coverage.
        if runs and low - runs[-1][1] <= 1e-14:
            runs[-1][1] = max(runs[-1][1], high)
            runs[-1][2].update(members)
        else:
            runs.append([low, high, set(members)])
    if len(runs) > 1 and runs[0][0] + TAU - runs[-1][1] <= 1e-14:
        last, first = runs.pop(), runs.pop(0)
        runs.append([last[0], first[1] + TAU, last[2] | first[2]])
    if fit is not None and any(high - low >= TAU for low, high, _ in runs):
        original_intervals = []
        for item in group:
            _, _, start, sweep = _circle(item)
            original_intervals.append([start, min(TAU, start + sweep)])
            if start + sweep > TAU:
                original_intervals.append([0., start + sweep - TAU])
        covered, end = 0., 0.
        for low, high in sorted(original_intervals):
            covered += max(0., high - max(end, low))
            end = max(end, high)
        if covered < TAU - 1e-14:
            conflicts.append({"severity": "hard", "reason": "circular_projection_would_close_original_gap",
                              "source_ids": _sources(group), "original_uncovered_angle_degrees": math.degrees(TAU - covered),
                              "evidence": fit})
            return [(_provenance(item, [item], item["id"]), [item]) for item in group]
    result = []
    for low, high, members in runs:
        items = [group[index] for index in sorted(members)]
        value = _provenance(_arc_value(fitted, low, high - low), items, carrier["id"])
        displacement = (fit["max_displacement_m"] if fit is not None else
                        max(_arc_displacement(item, center, radius)[0] for item in items))
        value["cad"].update(circle_parameters_refitted=not exact, added_gap_length_m=0., joint_fit=fit,
                            max_projection_distance_m=displacement)
        if fit is not None and navigation is not None:
            begin, sweep = _circle(value)[2:]
            angles = np.linspace(begin, begin + sweep, max(3, math.ceil(radius * sweep / .025) + 1))
            points = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
            checks = [_check(navigation, [a, b]) for a, b in zip(points, points[1:])]
            value["cad"]["navigation_checks"] = checks
            if not all(check["pass"] for check in checks):
                conflicts.append({"severity": "hard", "reason": "not_blocks_navmesh_failed",
                                  "source_ids": _sources(items), "evidence": checks})
                result.extend((_provenance(item, [item], item["id"]), [item]) for item in items)
                continue
        decisions.append({"action": "circular_coverage_union", "source_ids": _sources(items),
                          "closed": value["type"] == "circle", "circle_parameters_refitted": not exact,
                          "joint_fit": fit})
        result.append((value, items))
    return result


def _overlap(first, second):
    if first["type"] == second["type"] == "line":
        if not _same_line(first, second, NUMERICAL_M, .000001):
            return 0.
        ends, length, tangent = _line(first)
        positions = np.sort((_line(second)[0] - ends[0]) @ tangent)
        own = np.sort((ends - ends[0]) @ tangent)
        return max(0., min(own[1], positions[1]) - max(own[0], positions[0]))
    if first["type"] == "line" or second["type"] == "line" or not _same_circle(first, second):
        return 0.
    _, radius, a, length = _circle(first)
    _, _, b, other_length = _circle(second)
    return sum(max(0., min(a + length, b + shift + other_length) - max(a, b + shift))
               for shift in (-TAU, 0., TAU)) * radius


def _near_line_overlap(first, second, distance_budget):
    a, length, tangent = _line(first)
    b, other_length, direction = _line(second)
    cosine = min(1., abs(float(tangent @ direction)))
    if cosine < math.cos(math.radians(2)):
        return None
    own = np.sort((a - a[0]) @ tangent)
    projected = (b - a[0]) @ tangent
    low, high = max(own[0], min(projected)), min(own[1], max(projected))
    overlap = float(high - low)
    fraction = overlap / min(length, abs(float(projected[1] - projected[0])))
    if overlap <= NUMERICAL_M or fraction < .5:
        return None
    along = np.array([low, high])
    on_first = a[0] + along[:, None] * tangent
    on_second = b[0] + ((along - projected[0]) / (projected[1] - projected[0]))[:, None] * (b[1] - b[0])
    normal = np.array([-tangent[1], tangent[0]])
    signed = (on_second - on_first) @ normal
    maximum = float(np.abs(signed).max())
    if maximum > .03 + 1e-12:
        return None
    # An automatically created crossing node is not independent evidence
    # about whether a long acute overlap is a real branch or duplicate scans.
    budget = _residual_budget(first, second, distance_budget)
    return {"severity": "hard", "reason": "unresolved_near_line_overlap",
            "canonical_ids": [first["id"], second["id"]], "source_ids": _sources([first, second]),
            "overlap_m": overlap, "shorter_span_coverage_fraction": float(fraction),
            "angle_degrees": math.degrees(math.acos(cosine)),
            "normal_distances_at_overlap_ends_m": np.abs(signed).tolist(),
            "max_normal_distance_m": maximum,
            "overlap_segment": on_first.tolist(), "other_overlap_segment": on_second.tolist(),
            "diagnostic_distance_limit_m": .03, "minimum_coverage_fraction": .5,
            "fit_constrained_projection_budget_m": budget,
            "within_projection_budget": maximum <= budget + 1e-12,
            "same_surface_verified": False,
            "required_evidence": "independent_same_facing_same_surface_evidence_or_distinct_wall_confirmation"}


def overlap_report(primitives, *, distance_budget_m=.02):
    """Diagnose exact and subpixel near coverage without changing IDs or geometry.

    Near-parallel lines with at least half-shorter-span overlap and at most 3 cm
    separation remain pending until their surface relationship is established.
    This diagnostic limit is deliberately distinct from the tighter residual
    budget needed to move geometry. No geometric merge is authorized here.
    """
    if not math.isfinite(distance_budget_m) or not 0 <= distance_budget_m <= .03:
        raise ValueError("distance_budget_m must be between 0 and .03 metres")
    items, exact, near = list(primitives), [], []
    for index, first in enumerate(items):
        for second in items[index + 1:]:
            overlap = _overlap(first, second)
            if overlap > NUMERICAL_M:
                exact.append({"severity": "hard", "reason": "unresolved_overlap",
                              "canonical_ids": [first["id"], second["id"]], "overlap_m": overlap,
                              "source_ids": _sources([first, second])})
            elif first["type"] == second["type"] == "line":
                conflict = _near_line_overlap(first, second, distance_budget_m)
                if conflict is not None:
                    near.append(conflict)
    return {"exact": exact, "near_lines": near, "exact_overlap_count": len(exact),
            "near_line_overlap_count": len(near), "visible_overlap_count": len(exact) + len(near),
            "qualified": not exact and not near}


def _surface_permission(first, second, callback, cache, tolerance, angle):
    """Only independent evidence can reclassify overlapping parallel offsets."""
    if callback is None or not _same_line(first, second, tolerance, angle, allow_parallel_offset=True):
        return False
    a, _, tangent = _line(first)
    b, _, direction = _line(second)
    if abs(float(tangent[0] * direction[1] - tangent[1] * direction[0])) > 1e-10:
        return False
    normal = np.array([-tangent[1], tangent[0]])
    if np.abs((b - a[0]) @ normal).max() <= NUMERICAL_M:
        return False
    own, other = np.sort(a @ tangent), np.sort(b @ tangent)
    if min(own[1], other[1]) - max(own[0], other[0]) <= NUMERICAL_M:
        return False
    pair = tuple(sorted([first["id"], second["id"]]))
    if pair not in cache:
        try:
            evidence = callback(deepcopy(first), deepcopy(second))
            if not isinstance(evidence, dict) or type(evidence.get("pass")) is not bool:
                raise ValueError("same_surface must return a dict containing boolean pass")
            evidence = deepcopy(evidence)
        except Exception as error:
            evidence = {"pass": False, "error": f"{type(error).__name__}: {error}"}
        cache[pair] = {"input_ids": list(pair), "source_ids": _sources([first, second]),
                       "evidence": evidence}
    return cache[pair]["evidence"]["pass"]


def canonicalize_walls(primitives, *, distance_tolerance_m=.02, angle_tolerance_degrees=2.,
                       arc_max_displacement_m=.03, allow_gap_join=None,
                       not_blocks_navmesh=None, same_surface=None, junctions=()):
    """Return canonical primitives, source mapping, decisions and hard conflicts.

    The longest existing line is the carrier; projected endpoints never extend
    beyond the union of source intervals except explicit or numerical joins.
    Known fitting residuals further bound the projection tolerance. Nonparallel
    directions must differ by at most 2 degrees; parallel offset faces remain
    separate even within the distance budget unless ``same_surface(first,
    second)`` independently verifies that overlapping pair. This callback never
    authorizes a gap or relaxes the residual budget. Close overlapping arcs may use a
    jointly fitted circle, verified over their full curves within at most 3 cm.
    Circular coverage is projected then unioned; missing angles are not filled.
    Existing endpoint/T-junction IDs become interior nodes of a single long
    primitive. ``metrics.qualified`` is false for unresolved exact/near overlap
    or rejected navigation checks; it is a CAD consistency result, not map quality.
    """
    if not math.isfinite(distance_tolerance_m) or not 0 <= distance_tolerance_m <= .03:
        raise ValueError("distance_tolerance_m must be between 0 and .03 metres")
    if not math.isfinite(angle_tolerance_degrees) or not 0 <= angle_tolerance_degrees <= 2:
        raise ValueError("angle_tolerance_degrees must be between 0 and 2 degrees")
    if not math.isfinite(arc_max_displacement_m) or not 0 <= arc_max_displacement_m <= .03:
        raise ValueError("arc_max_displacement_m must be between 0 and .03 metres")
    originals = deepcopy(list(primitives))
    if len({item["id"] for item in originals}) != len(originals):
        raise ValueError("Source primitive IDs must be unique")
    for item in originals:
        if item.get("type") not in {"line", "arc", "circle"}:
            raise ValueError("CAD primitives must be line, arc or circle")
        _length(item)
    groups, decisions, conflicts, surface_checks = [], [], [], {}
    for item in sorted(originals, key=lambda value: (-_length(value), value["id"])):
        for group in groups:
            carrier = group[0]
            match = (item["type"] == carrier["type"] == "line"
                     and _same_line(carrier, item, distance_tolerance_m, angle_tolerance_degrees))
            if not match and item["type"] == carrier["type"] == "line":
                match = _surface_permission(carrier, item, same_surface, surface_checks,
                                            distance_tolerance_m, angle_tolerance_degrees)
            if match:
                # Avoid transitive merging of separate parallel faces or a
                # chain whose total direction spread exceeds the angle cap.
                for old in group:
                    old_points, _, old_direction = _line(old)
                    points, _, direction = _line(item)
                    cosine = abs(float(old_direction @ direction))
                    parallel = abs(float(old_direction[0] * direction[1] - old_direction[1] * direction[0])) <= 1e-10
                    offset = np.abs((points - old_points[0]) @ np.array([-old_direction[1], old_direction[0]])).max()
                    if cosine < math.cos(math.radians(angle_tolerance_degrees)) - 1e-12:
                        match = False
                        break
                    if parallel and offset > NUMERICAL_M and not _surface_permission(
                            old, item, same_surface, surface_checks, distance_tolerance_m, angle_tolerance_degrees):
                        match = False
                        break
            elif item["type"] != "line" and carrier["type"] != "line":
                if all(_same_circle(old, item) for old in group):
                    match = True
                elif any(_near_circle(old, item) for old in group):
                    fit = _joint_circle(group + [item], arc_max_displacement_m)
                    match = fit["pass"]
                    if not match:
                        conflicts.append({"severity": "hard", "reason": "near_circular_overlap_unresolved",
                                          "source_ids": _sources(group + [item]), "evidence": fit})
            if match:
                group.append(item)
                break
        else:
            groups.append([item])
    generated = []
    decisions.extend({"action": "same_surface_check", **check} for check in surface_checks.values())
    for group in groups:
        generated.extend(_line_results(group, allow_gap_join, not_blocks_navmesh,
                                       conflicts, decisions, surface_checks) if group[0]["type"] == "line"
                         else _arc_results(group, decisions, conflicts, not_blocks_navmesh, arc_max_displacement_m))
    mapping, result = {}, []
    for index, (value, items) in enumerate(generated, start=1):
        value["id"] = f"CAD{index:04d}"
        _nodes(value, items, junctions, distance_tolerance_m)
        for item in items:
            mapping.setdefault(item["id"], []).append(value["id"])
        result.append(value)
    coverage = overlap_report(result, distance_budget_m=distance_tolerance_m)
    conflicts.extend(coverage["exact"] + coverage["near_lines"])
    def counts(items):
        return {"primitive_count": len(items), "line_count": sum(item["type"] == "line" for item in items),
                "arc_count": sum(item["type"] == "arc" for item in items),
                "circle_count": sum(item["type"] == "circle" for item in items),
                "sum_length_m": sum(_length(item) for item in items)}
    return {"primitives": result, "source_to_canonical": mapping, "decisions": decisions,
            "conflicts": conflicts, "metrics": {"before": counts(originals), "after": counts(result),
            "max_projection_distance_m": max((item["cad"].get("max_projection_distance_m", 0.) for item in result), default=0.),
            "added_gap_length_m": sum(item["cad"].get("added_gap_length_m", 0.) for item in result),
            "unresolved_overlap_count": coverage["exact_overlap_count"],
            "unresolved_near_line_overlap_count": coverage["near_line_overlap_count"],
            "visible_overlap_count": coverage["visible_overlap_count"], "hard_conflict_count": len(conflicts),
            "unresolved_near_circular_overlap_count": sum(item["reason"] == "near_circular_overlap_unresolved" for item in conflicts),
            "qualified": not conflicts,
            "scope": "CAD coverage consistency only; not scene or navigation acceptance"}}
