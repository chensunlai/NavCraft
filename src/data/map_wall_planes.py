"""Simplify one connected measured wall surface at a common reference height.

Different horizontal cuts of a slightly leaning/rough wall are not independent
2D walls. This operation verifies original source cuts against render geometry,
connectivity and directed normals, then fits their actual 3D points. It never
relaxes the separate single-sheet check or fills an uncovered opening.
"""
from copy import deepcopy
import math

import numpy as np

from .map_wall_cad import overlap_report


def _cuts(triangles, height):
    intersections = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        first, second = triangles[:, a], triangles[:, b]
        dy = second[:, 1] - first[:, 1]
        valid = (abs(dy) > 1e-10) & ((first[:, 1] - height) * (second[:, 1] - height) <= 0)
        ratio = np.divide(height - first[:, 1], dy, out=np.zeros_like(dy), where=valid)
        point = first[:, [0, 2]] + ratio[:, None] * (second[:, [0, 2]] - first[:, [0, 2]])
        point[~valid] = np.nan
        intersections.append(point)
    points = np.stack(intersections, axis=1)
    pairs = np.stack([points[:, [a, b]] for a, b in ((0, 1), (1, 2), (2, 0))], axis=1)
    length2 = ((pairs[:, :, 1] - pairs[:, :, 0]) ** 2).sum(2)
    length2[~np.isfinite(pairs).all(axis=(2, 3))] = -1
    valid = length2.max(1) > 1e-16
    return pairs[np.arange(len(triangles)), length2.argmax(1)][valid], np.flatnonzero(valid)


def _components(triangles):
    parent = list(range(len(triangles)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    edges = {}
    # Weld scan/UV seams only to 0.1mm, far below separate wall-sheet scales.
    for i, triangle in enumerate(np.rint(triangles / .0001).astype(np.int64)):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((tuple(triangle[a]), tuple(triangle[b]))))
            if edge in edges:
                parent[find(i)] = find(edges[edge])
            else:
                edges[edge] = i
    return np.array([find(i) for i in range(len(triangles))])


def _crosses_portal(segment, portal):
    a, b = np.asarray(segment), np.asarray(portal, dtype=float)
    d, e = a[1] - a[0], b[1] - b[0]
    cross = lambda p, q: float(p[0] * q[1] - p[1] * q[0])
    determinant = cross(d, e)
    if abs(determinant) < 1e-10:
        if abs(cross(b[0] - a[0], d)) > 1e-8:
            return False
        positions = sorted(float((point - a[0]) @ d / (d @ d)) for point in b)
        return min(1., positions[1]) - max(0., positions[0]) > 1e-8
    along, opening = cross(b[0] - a[0], e) / determinant, cross(b[0] - a[0], d) / determinant
    return -1e-8 <= along <= 1 + 1e-8 and 1e-8 < opening < 1 - 1e-8


def refit_wall_planes(primitives, raw_triangles, source_segments, source_heights, *, floor_height,
                     reference_height=None, not_blocks_navmesh=None, portals=()):
    """Return primitives and accepted/rejected 3D surface refit records.

    ``source_heights`` contains absolute world Y for every candidate segment.
    Only verified architectural surface lines with original source indices are
    valid inputs. The default reference height is floor_height + 1.2m.
    Acceptance is explicitly a bounded connected-surface simplification, not a
    claim that all scan triangles are one exact mathematical plane.
    """
    original = deepcopy(list(primitives))
    source = np.asarray(source_segments, dtype=float)
    heights = np.asarray(source_heights, dtype=float)
    triangles = np.asarray(raw_triangles, dtype=float).reshape(-1, 3, 3)
    reference = float(floor_height + 1.2 if reference_height is None else reference_height)
    if (source.ndim != 3 or source.shape[1:] != (2, 2) or heights.shape != (len(source),)
            or not all(np.isfinite(value).all() for value in (source, heights, triangles))
            or not math.isfinite(reference)):
        raise ValueError("Finite original XZ segments, matching absolute heights and XYZ triangles are required")
    by_id = {item["id"]: item for item in original}
    if len(by_id) != len(original):
        raise ValueError("Wall primitive IDs must be unique")
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = np.divide(cross, np.linalg.norm(cross, axis=1)[:, None], out=np.zeros_like(cross),
                        where=np.linalg.norm(cross, axis=1)[:, None] > 1e-12)
    candidates = overlap_report([p for p in original if p["type"] == "line"])["near_lines"]
    consumed, replacements, accepted, rejected = set(), {}, [], []
    for candidate in candidates:
        ids = candidate["canonical_ids"]
        if consumed.intersection(ids):
            continue  # Keep shared candidates for the caller's next geometry pass.
        items = [by_id[key] for key in ids]
        record = {"input_ids": ids, "candidate": candidate, "reference_height_m": reference}
        if any(p.get("representation") == "wall_axis" for p in items):
            rejected.append({**record, "reason": "wall_axes_are_not_surface_sources"})
            continue
        points, weights, owners, used_sources = [], [], [], set()
        valid = True
        for owner, item in enumerate(items):
            indices = item.get("source_segment_indices", [])
            if not indices or any(isinstance(i, bool) or not isinstance(i, (int, np.integer))
                                  or not 0 <= i < len(source) for i in indices):
                valid = False
                break
            ends = np.array([item["start"], item["end"]])
            tangent = ends[1] - ends[0]
            length = np.linalg.norm(tangent)
            tangent /= length
            for i in set(indices):
                segment = source[i]
                along = (segment - ends[0]) @ tangent
                low, high = max(0., along.min()), min(length, along.max())
                if high <= low or abs(along[1] - along[0]) < 1e-10:
                    continue
                # Source IDs can retain geometry removed by door carving.
                # Clip to the accepted old line, never import that removed part.
                segment = segment[0] + ((np.array([low, high]) - along[0]) /
                           (along[1] - along[0]))[:, None] * (segment[1] - segment[0])
                points.extend(np.column_stack([segment[:, 0], np.full(2, heights[i]), segment[:, 1]]))
                weights.extend([float(np.linalg.norm(segment[1] - segment[0])) / 2] * 2)
                owners.extend([owner] * 2)
                used_sources.add(int(i))
        if not valid or not points:
            rejected.append({**record, "reason": "missing_original_source_geometry_or_heights"})
            continue
        points, weights, owners = np.array(points), np.array(weights), np.array(owners)
        if len(np.unique(points[:, 1])) < 3 or np.ptp(points[:, 1]) < .8:
            rejected.append({**record, "reason": "insufficient_measured_height_extent"})
            continue
        tangent = np.array(items[0]["end"]) - items[0]["start"]
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        along, across = triangles[:, :, [0, 2]] @ tangent, triangles[:, :, [0, 2]] @ normal
        own_along, own_across = points[:, [0, 2]] @ tangent, points[:, [0, 2]] @ normal
        selected = np.flatnonzero((along.max(1) >= own_along.min() - .03)
                    & (along.min(1) <= own_along.max() + .03)
                    & (across.max(1) >= own_across.min() - .06)
                    & (across.min(1) <= own_across.max() + .06)
                    & (triangles[:, :, 1].max(1) >= points[:, 1].min() - .05)
                    & (triangles[:, :, 1].min(1) <= points[:, 1].max() + .05))
        if not len(selected):
            rejected.append({**record, "reason": "no_independent_raw_surface"})
            continue
        local, local_normals = triangles[selected], normals[selected]
        snapped, matched, distances = points.copy(), np.full(len(points), -1), np.full(len(points), np.inf)
        usable = (abs(local_normals[:, 1]) <= .65) & (abs(local_normals[:, [0, 2]] @ normal) >= .5)
        for height in np.unique(points[:, 1]):
            cuts, face_ids = _cuts(local[usable], height)
            if not len(cuts):
                continue
            face_ids = np.flatnonzero(usable)[face_ids]
            rows = np.flatnonzero(points[:, 1] == height)
            delta = cuts[:, 1] - cuts[:, 0]
            ratios = np.clip(((points[rows, None][:, :, [0, 2]] - cuts[None, :, 0]) * delta).sum(2)
                             / (delta * delta).sum(1), 0, 1)
            nearest = cuts[None, :, 0] + ratios[:, :, None] * delta
            errors = np.linalg.norm(nearest - points[rows, None][:, :, [0, 2]], axis=2)
            # A point on a shared edge has several exact raw owners. Prefer
            # the wall-facing owner over its bevel, independent of file order.
            alignment = abs(local_normals[face_ids][:, [0, 2]] @ normal)
            choices = np.where(errors <= errors.min(1)[:, None] + 1e-8,
                               alignment[None, :], -1.).argmax(1)
            snapped[rows[:, None], [0, 2]] = nearest[np.arange(len(rows)), choices]
            matched[rows] = face_ids[choices]
            distances[rows] = errors[np.arange(len(rows)), choices]
        record["max_raw_registration_distance_m"] = float(distances.max())
        if distances.max() > .005:
            rejected.append({**record, "reason": "source_cuts_do_not_register_to_raw_geometry"})
            continue
        center = np.average(snapped, axis=0, weights=weights)
        local_points = snapped - center
        plane = np.linalg.eigh((local_points * weights[:, None]).T @ local_points / weights.sum())[1][:, 0]
        dots = local_normals[matched] @ plane
        if np.average(dots, weights=weights) < 0:
            plane, dots = -plane, -dots
        orientation = [float(np.average(dots[owners == side] >= math.cos(math.radians(20)),
                                        weights=weights[owners == side])) for side in (0, 1)]
        residual = abs(local_points @ plane)
        record.update(plane_normal_xyz=plane.tolist(), plane_center_xyz=center.tolist(),
                      max_raw_plane_residual_m=float(residual.max()),
                      same_directed_normal_fractions=orientation,
                      height_extent_m=[float(snapped[:, 1].min()), float(snapped[:, 1].max())],
                      raw_triangle_ids=sorted(set(map(int, selected[matched]))))
        if min(orientation) < .8 or abs(plane[1]) > math.sin(math.radians(10)) or residual.max() > .05:
            rejected.append({**record, "reason": "raw_plane_or_directed_normal_not_bounded"})
            continue
        # A scene-wide mesh component is not a wall identity. Endcaps, ceiling
        # bridges and corners cannot connect two otherwise separate wall sheets.
        consistent = ((local_normals @ plane >= math.cos(math.radians(45)))
                      & (abs((local - center) @ plane).max(1) <= .05))
        components = np.full(len(local), -1)
        components[consistent] = _components(local[consistent])
        component = components[matched]
        common = (set(component[owners == 0]) & set(component[owners == 1])) - {-1}
        best = max(common, key=lambda key: weights[component == key].sum(), default=None)
        fractions = [float(weights[(owners == side) & (component == best)].sum() /
                           weights[owners == side].sum()) for side in (0, 1)]
        record.update(shared_raw_component_fractions=fractions,
                      connectivity_scope="local same-directed-normal faces within 45deg and all vertices within 5cm of plane; no perpendicular endcap/ceiling bridges",
                      primary_source_normal_degrees=20., minimum_primary_normal_fraction=.8)
        if min(fractions) < .95:
            rejected.append({**record, "reason": "independent_or_disconnected_raw_surfaces"})
            continue
        xz_normal = plane[[0, 2]]
        scale = np.linalg.norm(xz_normal)
        constant = (plane @ center - plane[1] * reference) / scale
        xz_normal /= scale
        direction = np.array([-xz_normal[1], xz_normal[0]])
        ends = [np.array([p["start"], p["end"]]) for p in items]
        shifts = {p["id"]: float(abs(e @ xz_normal - constant).max()) for p, e in zip(items, ends)}
        record["old_line_displacements_m"] = shifts
        if max(shifts.values()) > .03:
            rejected.append({**record, "reason": "reference_height_displacement_exceeded"})
            continue
        intervals = []
        for low, high in sorted(sorted(e @ direction) for e in ends):
            if intervals and low <= intervals[-1][1] + 1e-8:
                intervals[-1][1] = max(intervals[-1][1], float(high))
            else:
                intervals.append([float(low), float(high)])
        proposed = [np.array(interval)[:, None] * direction + constant * xz_normal for interval in intervals]
        if any(_crosses_portal(segment, portal["segment"]) for segment in proposed
               for portal in portals if portal.get("status", "accepted") != "rejected"):
            rejected.append({**record, "reason": "projected_wall_crosses_protected_opening"})
            continue
        if not_blocks_navmesh is None:
            rejected.append({**record, "reason": "navmesh_evidence_required"})
            continue
        checks = [not_blocks_navmesh(segment.tolist()) for segment in proposed]
        if any(not isinstance(check, dict) or type(check.get("pass")) is not bool for check in checks):
            raise ValueError("not_blocks_navmesh must return explicit boolean pass")
        record["navmesh_checks"] = deepcopy(checks)
        if not all(check["pass"] for check in checks):
            rejected.append({**record, "reason": "projected_wall_blocks_navmesh"})
            continue
        record.update(original_primitives=deepcopy(items), source_segment_indices=sorted(used_sources),
                      acceptance="bounded_connected_raw_3d_surface_simplification",
                      coverage_intervals_m=intervals, coverage_scope="projected old coverage union; no gap filling")
        values = []
        for run, segment in enumerate(proposed):
            identifier = ids[0] if len(proposed) == 1 else f"{ids[0]}-plane-{run}"
            values.append({"id": identifier, "type": "line", "start": segment[0].tolist(),
                           "end": segment[1].tolist(), "endpoints": segment.tolist(),
                           "length_m": float(np.linalg.norm(segment[1] - segment[0])),
                           "source_primitive_ids": ids, "source_segment_indices": sorted(used_sources),
                           "source_ids": sorted({s for item in items for s in item.get("source_ids", [item["id"]])}),
                           "thickness_m": None, "representation": "reference_height_wall_surface_line",
                           "evidence_kind": record["acceptance"], "plane_refit": deepcopy(record),
                           "graph_requires_rebuild": True})
        consumed.update(ids)
        replacements[ids[0]] = values
        accepted.append(record)
    output = []
    for item in original:
        output.extend(replacements.get(item["id"], [] if item["id"] in consumed else [item]))
    return {"primitives": output, "accepted": accepted, "rejected": rejected,
            "reference_height_m": reference,
            "limitations": ["Only original verified architectural surface lines are valid inputs.",
                            "Shared mesh connectivity and bounded plane fitting are not room ground truth.",
                            "Reference-height simplification does not assert an exactly planar scan."]}
