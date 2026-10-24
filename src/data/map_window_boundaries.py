"""Reconstruct architectural window-wall lines from an observed raised frame.

Raw upper/lower spans and both vertical sides constrain a rectangular opening.
The inferred wall line follows its adjacent architectural walls, not curtains or
different sill/header recess depths. This is explicitly structural inference;
it does not claim a measured solid surface or a physically closed glass pane.
"""

from copy import deepcopy
import math

import numpy as np

from .map_geometry import slice_triangles


def _intervals(values, low, high, tolerance=1e-9):
    result = []
    for start, end in sorted(np.asarray(values, dtype=float).reshape(-1, 2).tolist()):
        start, end = max(low, start), min(high, end)
        if end <= start:
            continue
        if result and start <= result[-1][1] + tolerance:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def _coverage(intervals, length):
    return sum(end - start for start, end in intervals) / length


def _gaps(primitives):
    lines = [wall for wall in primitives if wall["type"] == "line"]
    for index, first in enumerate(lines):
        a = np.asarray([first["start"], first["end"]], dtype=float)
        tangent = a[1] - a[0]
        length = np.linalg.norm(tangent)
        if length <= 1e-8:
            continue
        tangent /= length
        normal = np.array([-tangent[1], tangent[0]])
        for second in lines[index + 1:]:
            b = np.asarray([second["start"], second["end"]], dtype=float)
            other = b[1] - b[0]
            if np.linalg.norm(other) <= 1e-8 or abs(tangent @ (other / np.linalg.norm(other))) < math.cos(math.radians(3)):
                continue
            if np.abs((b - a[0]) @ normal).max() > .015:
                continue
            positions = (b - a[0]) @ tangent
            if positions.min() > length:
                span = np.stack((a[1], b[positions.argmin()]))
            elif positions.max() < 0:
                span = np.stack((b[positions.argmax()], a[0]))
            else:
                continue
            gap = np.linalg.norm(span[1] - span[0])
            if .12 <= gap <= 2.:
                covered = False
                for carrier in lines:
                    if carrier["id"] in (first["id"], second["id"]):
                        continue
                    points = np.asarray([carrier["start"], carrier["end"]], dtype=float)
                    positions = (points - span[0]) @ tangent
                    if (np.abs((points - span[0]) @ normal).max() <= .015
                            and min(gap, float(positions.max())) - max(0., float(positions.min())) > .025):
                        covered = True
                        break
                if covered:
                    continue
                yield span, [first["id"], second["id"]]


def _horizontal_groups(ids, u, v, y, normals, length, window_y):
    centers = y.mean(1)
    eligible = ids[(np.abs(normals[ids, 1]) >= .9) & (np.ptp(y[ids], axis=1) <= .025)]
    groups = []
    for height_id in np.unique(np.rint(centers[eligible] / .025).astype(int)):
        selected = eligible[np.rint(centers[eligible] / .025).astype(int) == height_id]
        intervals = _intervals(np.stack((u[selected].min(1), u[selected].max(1)), axis=1), 0, length)
        if _coverage(intervals, length) < .9 - 1e-9:
            continue
        height = float(np.median(y[selected]))
        kind = ("sill" if abs(height - window_y[0]) <= .25 else
                "header" if abs(height - window_y[1]) <= .4 else None)
        if kind is not None:
            corner_depths = {}
            for side, point in (("left", intervals[0][0]), ("right", intervals[-1][1])):
                close = selected[(u[selected].min(1) <= point + .05) & (u[selected].max(1) >= point - .05)]
                corner_depths[side] = [float(v[close].min()), float(v[close].max())]
            groups.append({"kind": kind, "height_m": height, "projected_coverage": _coverage(intervals, length),
                           "along_intervals_m": intervals,
                           "corner_normal_ranges_m": corner_depths,
                           "normal_range_m": [float(v[selected].min()), float(v[selected].max())],
                           "source_raw_face_ids": selected.tolist()})
    return groups


def _plane_profile(triangles, groups, a, tangent, normal, length):
    best = None
    bottom, top = groups
    # This bounded 5 mm search measures the actual frame depth; it never moves
    # the original wall plane to obtain a favourable nearest-surface score.
    low = max(-.35, bottom["normal_range_m"][0], top["normal_range_m"][0])
    high = min(.35, bottom["normal_range_m"][1], top["normal_range_m"][1])
    if high <= low:
        return None
    transformed = []
    for group in groups:
        faces = triangles[group["source_raw_face_ids"]]
        transformed.append(np.stack(((faces[:, :, [0, 2]] - a) @ tangent,
                                     (faces[:, :, [0, 2]] - a) @ normal, faces[:, :, 1]), axis=-1))
    for offset in np.linspace(low + 1e-7, high - 1e-7, max(3, math.ceil((high - low) / .005) + 1)):
        intervals = []
        for faces in transformed:
            cuts = slice_triangles(faces, float(offset))
            intervals.append(_intervals(np.sort(cuts[:, :, 0], axis=1), 0, length))
        fractions = [_coverage(values, length) for values in intervals]
        score = (min(fractions), sum(fractions), -abs(offset))
        if best is None or score > best[0]:
            both = _intervals([[max(a0, b0), min(a1, b1)] for a0, a1 in intervals[0]
                               for b0, b1 in intervals[1] if min(a1, b1) > max(a0, b0)], 0, length)
            best = (score, {"normal_offset_m": float(offset), "sill_coverage": fractions[0],
                            "header_coverage": fractions[1], "sill_intervals_m": intervals[0],
                            "header_intervals_m": intervals[1], "joint_intervals_m": both,
                            "joint_coverage": _coverage(both, length)})
    return best[1]


def _jambs(ids, u, v, y, normals, tangent, length, bottom, top):
    result = []
    side_normals = normals[:, [0, 2]] @ tangent
    for name, position in (("left", 0.), ("right", length)):
        selected = ids[(np.abs(side_normals[ids]) >= .85)
                       & (np.abs(u[ids].mean(1) - position) <= .15)]
        intervals = _intervals(np.stack((y[selected].min(1), y[selected].max(1)), axis=1),
                               bottom, top, tolerance=.025)
        longest = max((end - start for start, end in intervals), default=0.)
        corners, depths = {}, {}
        for corner, height in (("lower", bottom), ("upper", top)):
            close = selected[(y[selected].min(1) <= height + .05) & (y[selected].max(1) >= height - .05)]
            corners[corner] = [float(u[close].min()), float(u[close].max())] if len(close) else None
            depths[corner] = [float(v[close].min()), float(v[close].max())] if len(close) else None
        result.append({"side": name, "continuous_height_span_m": longest, "height_intervals_m": intervals,
                       "height_coverage": _coverage(intervals, top - bottom), "corner_along_ranges_m": corners,
                       "corner_normal_ranges_m": depths,
                       "normal_range_m": [float(v[selected].min()), float(v[selected].max())] if len(selected) else None,
                       "source_raw_face_ids": selected.tolist()})
    return result


def _rectangle(bottom, top, jambs):
    """Check that measured edge fragments close the same raised rectangle."""
    checks = []
    for side, jamb in enumerate(jambs):
        for name, edge in (("lower", bottom), ("upper", top)):
            point = edge["along_intervals_m"][0][0] if side == 0 else edge["along_intervals_m"][-1][1]
            bounds = jamb["corner_along_ranges_m"][name]
            distance = max(bounds[0] - point, point - bounds[1], 0.) if bounds else None
            horizontal_depth = edge["corner_normal_ranges_m"][jamb["side"]]
            vertical_depth = jamb["corner_normal_ranges_m"][name]
            depth_gap = (max(horizontal_depth[0] - vertical_depth[1], vertical_depth[0] - horizontal_depth[1], 0.)
                         if vertical_depth else None)
            checks.append({"side": jamb["side"], "edge": name, "horizontal_endpoint_m": point,
                           "raw_jamb_range_m": bounds, "separation_m": distance,
                           "normal_separation_m": depth_gap,
                           "pass": distance is not None and distance <= .05 + 1e-9
                           and depth_gap is not None and depth_gap <= .05 + 1e-9})
    vertical = []
    for jamb in jambs:
        intervals = jamb["height_intervals_m"]
        lower_distance = intervals[0][0] - bottom["height_m"] if intervals else math.inf
        upper_distance = top["height_m"] - intervals[-1][1] if intervals else math.inf
        vertical.append({"side": jamb["side"], "height_coverage": jamb["height_coverage"],
                         "continuous_height_span_m": jamb["continuous_height_span_m"],
                         "reaches_sill_and_header": max(lower_distance, upper_distance) <= .05,
                         "pass": jamb["height_coverage"] >= .9 - 1e-9
                         and jamb["continuous_height_span_m"] >= 1.4 - 1e-9
                         and max(lower_distance, upper_distance) <= .05})
    geometry_closes = bool(all(check["pass"] for check in checks + vertical))
    floor_associated = .25 <= bottom["height_m"] <= 1.8 and top["height_m"] <= 3.2
    return {"pass": bool(geometry_closes and floor_associated), "geometry_closes": geometry_closes,
            "current_floor_window_association": floor_associated,
            "current_floor_auto_inference_range_m": {"sill": [.25, 1.8], "header_max": 3.2},
            "outside_range_is_unknown_not_absent_window": True,
            "height_m": top["height_m"] - bottom["height_m"],
            "raised_sill_height_m": bottom["height_m"], "corner_tolerance_m": .05,
            "corner_checks": checks, "vertical_checks": vertical,
            "rule": "two_90_percent_horizontal_edges_and_two_90_percent_vertical_edges_with_four_closing_corners",
            "recess_depth_equality_required": False}


def build_window_boundaries(raw_triangles, primitives, window_instances, *, floor_height, passage,
                            not_blocks_navmesh=None):
    """Return unchanged walls plus evidence-supported window-wall reconstructions.

    ``window_instances`` are SemanticInstance objects. ``passage(segment)`` uses
    MapScene's ``{pass, probes}`` format. Empty probes mean unknown, not a blocked
    window. Missing probes do not veto architectural inference. Any recorded
    successful crossing vetoes it. Every accepted addition is explicitly an
    inferred wall line spanning a measured frame, not a measured solid surface.
    ``not_blocks_navmesh`` optionally checks the reconstructed line against
    measured walkable space. A known overlap vetoes it independently of probes.
    """
    triangles = np.asarray(raw_triangles, dtype=float).reshape(-1, 3, 3)
    if not np.isfinite(triangles).all() or not math.isfinite(floor_height):
        raise ValueError("Raw triangles and floor height must be finite")
    original = deepcopy(list(primitives))
    window_instances = list(window_instances)
    if len({item["id"] for item in original}) != len(original):
        raise ValueError("Wall primitive IDs must be unique")
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    sizes = np.linalg.norm(cross, axis=1)
    normals = np.divide(cross, sizes[:, None], out=np.zeros_like(cross), where=sizes[:, None] > 1e-10)
    y = triangles[:, :, 1] - floor_height
    candidates, accepted, rejected, additions = [], [], [], []
    seen = set()
    for span, adjacent in _gaps(original):
        a, b = span
        length = float(np.linalg.norm(b - a))
        tangent = (b - a) / length
        normal = np.array([-tangent[1], tangent[0]])
        u = (triangles[:, :, [0, 2]] - a) @ tangent
        v = (triangles[:, :, [0, 2]] - a) @ normal
        for instance in window_instances:
            if instance.category != "window":
                continue
            semantic = np.asarray(instance.triangles)
            su = (semantic[:, :, [0, 2]] - a) @ tangent
            sv = (semantic[:, :, [0, 2]] - a) @ normal
            sy = semantic[:, :, 1] - floor_height
            selected = (sv.min(1) >= -.35) & (sv.max(1) <= .35) & (sy.max(1) > .2)
            intervals = _intervals(np.stack((su[selected].min(1), su[selected].max(1)), axis=1), 0, length)
            prior_coverage = _coverage(intervals, length)
            if prior_coverage < .8 - 1e-9:
                continue
            key = (instance.id, tuple(sorted(adjacent)))
            if key in seen:
                continue
            seen.add(key)
            window_y = [float(sy[selected].min()), float(sy[selected].max())]
            local = ((u.max(1) >= -.15) & (u.min(1) <= length + .15)
                     & (v.min(1) >= -.35) & (v.max(1) <= .35)
                     & (y.min(1) >= max(.2, window_y[0] - .35)) & (y.max(1) <= window_y[1] + .45))
            ids = np.flatnonzero(local)
            groups = _horizontal_groups(ids, u, v, y, normals, length, window_y)
            pairs = [(bottom, top) for bottom in groups for top in groups
                     if bottom["kind"] == "sill" and top["kind"] == "header"
                     and top["height_m"] - bottom["height_m"] >= 1.4]
            number = len(candidates) + 1
            identifier = f"WINDOW{number:04d}"
            while any(item["id"] == identifier + "-0" for item in original):
                number += 1
                identifier = f"WINDOW{number:04d}"
            record = {"id": identifier, "segment": span.tolist(), "adjacent_primitive_ids": adjacent,
                      "semantic_instance_id": instance.id, "source_kind": "reconstructed_window_wall",
                      "status": "needs_review", "solid_wall_measured": False, "proposed_primitives": [],
                      "evidence_kind": "structural_inference", "glass_open_or_closed": "not_determined",
                      "evidence": {"semantic_prior_coverage": prior_coverage,
                                   "semantic_source": deepcopy(instance.source),
                                   "semantic_prior_is_raw_proof": False, "horizontal_groups": groups,
                                   "raw_geometry_source_kind": "original_render_mesh",
                                   "raw_face_id_space": "row_in_input_raw_triangles",
                                   "raw_triangle_count": len(triangles)}}
            candidates.append(record)
            if not pairs:
                record["reason"] = "missing_full_width_raw_sill_or_header"
                rejected.append(record)
                continue
            choices = []
            for bottom, top in pairs:
                jambs = _jambs(ids, u, v, y, normals, tangent, length, bottom["height_m"], top["height_m"])
                closure = _rectangle(bottom, top, jambs)
                choices.append((bottom, top, jambs, closure))
            bottom, top, jambs, closure = max(choices, key=lambda item: (
                item[3]["pass"], sum(check["pass"] for check in item[3]["corner_checks"]),
                min(item[0]["projected_coverage"], item[1]["projected_coverage"])))
            # A shared recess plane is diagnostic only: a real sill and header
            # can have different depths while delimiting the same wall opening.
            profile = _plane_profile(triangles, (bottom, top), a, tangent, normal, length)
            record["evidence"].update(sill=bottom, header=top, jambs=jambs, frame_rectangle=closure,
                best_frame_plane=profile, reconstruction_plane="continuation_of_adjacent_architectural_wall_lines",
                reconstructed_span_is_inferred_not_raw_surface=True)
            source_ids = sorted(set(bottom["source_raw_face_ids"] + top["source_raw_face_ids"]
                                    + [index for jamb in jambs for index in jamb["source_raw_face_ids"]]))
            record["source_raw_face_ids"] = source_ids
            record["proposed_primitives"].append({"id": identifier + "-0", "type": "line",
                "start": a.tolist(), "end": b.tolist(), "endpoints": span.tolist(), "length_m": length,
                "role": "architectural_window_wall", "source_kind": "reconstructed_window_wall",
                "evidence_kind": "structural_inference", "solid_wall_measured": False,
                "source_raw_face_ids": source_ids, "window_candidate_id": identifier})
            record["inferred_boundary_spans_m"] = [[0., length]]
            navigation = passage(span.tolist())
            if not isinstance(navigation, dict) or type(navigation.get("pass")) is not bool:
                raise ValueError("passage must return a dict containing explicit boolean pass")
            record["evidence"]["navmesh"] = navigation
            record["evidence"]["robot_crossing_status"] = (
                "successful_crossing_recorded" if navigation["pass"] else
                "blocked_probes_recorded" if navigation.get("probes") else "not_established")
            walkability = ({"pass": None, "source": "not_provided"} if not_blocks_navmesh is None
                           else not_blocks_navmesh(span.tolist()))
            if not isinstance(walkability, dict) or (not_blocks_navmesh is not None and type(walkability.get("pass")) is not bool):
                raise ValueError("not_blocks_navmesh must return a dict containing explicit boolean pass")
            record["evidence"]["walkable_space_check"] = walkability
            if navigation["pass"] or any(probe.get("pass") is True for probe in navigation.get("probes", [])):
                record.update(status="rejected", reason="measured_direct_passage_exists")
                rejected.append(record)
            elif walkability["pass"] is False:
                record.update(status="rejected", reason="would_block_measured_walkable_space")
                rejected.append(record)
            elif closure["pass"]:
                record.update(status="accepted", reason="raised_rectangular_frame_and_adjacent_wall_continuity")
                accepted.append(record)
                additions.extend(record["proposed_primitives"])
            else:
                record["reason"] = ("incomplete_or_nonclosing_raised_rectangular_frame"
                                    if closure["current_floor_window_association"] else
                                    "requires_floor_or_window_type_confirmation")
                record["interpretation"] = "window_frame_needs_additional_geometry_or_visual_evidence"
            for primitive in record["proposed_primitives"]:
                # Keep per-segment provenance usable without an external index.
                primitive["evidence"] = deepcopy(record["evidence"])
    return {"primitives": original + deepcopy(additions), "candidates": candidates,
            "accepted": accepted, "rejected": rejected, "original_primitives_modified": False}
