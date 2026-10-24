"""Raw-frame-supported window boundary candidates, separate from solid walls.

Semantic instances select a possible window; raw sill/header/jamb geometry must
independently support its frame. A proposed inset boundary remains reviewable
when one plane is incomplete or NavMesh probes do not establish obstruction.
"""

from copy import deepcopy
import math

import numpy as np

from .map_geometry import multiheight_slices, slice_triangles
from .map_wall_recovery import raw_gap_support


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
            groups.append({"kind": kind, "height_m": height, "projected_coverage": _coverage(intervals, length),
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
        result.append({"side": name, "continuous_height_span_m": longest, "height_intervals_m": intervals,
                       "normal_range_m": [float(v[selected].min()), float(v[selected].max())] if len(selected) else None,
                       "source_raw_face_ids": selected.tolist()})
    return result


def build_window_boundaries(raw_triangles, primitives, window_instances, *, floor_height, passage):
    """Return unchanged walls plus accepted inset boundaries and all candidates.

    ``window_instances`` are SemanticInstance objects. ``passage(segment)`` uses
    MapScene's ``{pass, probes}`` format. Empty probes mean unknown, not a blocked
    window. Review candidates keep ``proposed_primitives`` for independent RGB
    confirmation; they are not added to the accepted ``primitives`` list.
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
            identifier = f"WINDOW{len(candidates) + 1:04d}"
            record = {"id": identifier, "segment": span.tolist(), "adjacent_primitive_ids": adjacent,
                      "semantic_instance_id": instance.id, "source_kind": "frame_supported_window_boundary",
                      "status": "needs_review", "solid_wall_measured": False, "proposed_primitives": [],
                      "evidence": {"semantic_prior_coverage": prior_coverage,
                                   "semantic_source": deepcopy(instance.source),
                                   "semantic_prior_is_raw_proof": False, "horizontal_groups": groups}}
            candidates.append(record)
            if not pairs:
                record["reason"] = "missing_full_width_raw_sill_or_header"
                rejected.append(record)
                continue
            choices = [(bottom, top, _plane_profile(triangles, (bottom, top), a, tangent, normal, length))
                       for bottom, top in pairs]
            choices = [item for item in choices if item[2] is not None]
            if not choices:
                record["reason"] = "no_common_recess_frame_depth"
                rejected.append(record)
                continue
            bottom, top, profile = max(choices, key=lambda item: min(item[2]["sill_coverage"], item[2]["header_coverage"]))
            jambs = _jambs(ids, u, v, y, normals, tangent, length, bottom["height_m"], top["height_m"])
            record["evidence"].update(sill=bottom, header=top, jambs=jambs, best_frame_plane=profile)
            offset = profile["normal_offset_m"]
            inset = span + offset * normal
            source_ids = sorted(set(bottom["source_raw_face_ids"] + top["source_raw_face_ids"]
                                    + [index for jamb in jambs for index in jamb["source_raw_face_ids"]]))
            record["source_raw_face_ids"] = source_ids
            proposed = [("window_frame_boundary", inset)]
            connector_checks = []
            if abs(offset) > 1e-6:
                raw_layers = multiheight_slices(triangles[ids], floor_height, (.3, .6, .9, 1.2, 1.5, 1.8, 2.1, 2.4))
                for side in range(2):
                    connector = np.stack((span[side], inset[side]))
                    connector_checks.append(raw_gap_support(connector, raw_layers))
                    proposed.append(("recess_jamb_connection", connector))
            record["evidence"]["recess_connection_support"] = connector_checks
            for number, (role, points) in enumerate(proposed):
                record["proposed_primitives"].append({"id": f"{identifier}-{number}", "type": "line",
                    "start": points[0].tolist(), "end": points[1].tolist(), "endpoints": points.tolist(),
                    "length_m": float(np.linalg.norm(points[1] - points[0])), "role": role,
                    "source_kind": "frame_supported_window_boundary", "solid_wall_measured": False,
                    "source_raw_face_ids": source_ids, "window_candidate_id": identifier})
            missing = []
            previous = 0.
            for start, end in profile["joint_intervals_m"]:
                if start > previous + 1e-8:
                    missing.append([previous, start])
                previous = end
            if previous < length - 1e-8:
                missing.append([previous, length])
            record["uncertain_boundary_spans_m"] = missing
            navigation = passage(span.tolist())
            if not isinstance(navigation, dict) or type(navigation.get("pass")) is not bool:
                raise ValueError("passage must return a dict containing explicit boolean pass")
            record["evidence"]["navmesh"] = navigation
            blocked = [probe for probe in navigation.get("probes", []) if probe.get("pass") is False
                       and all(np.asarray(probe.get(key, [])).shape == (3,)
                               and np.isfinite(probe[key]).all() for key in ("start", "requested_end", "actual_end"))]
            geometry_complete = (min(profile["sill_coverage"], profile["header_coverage"]) >= .9 - 1e-9
                                 and profile["joint_coverage"] >= .9 - 1e-9
                                 and all(jamb["continuous_height_span_m"] >= 1.4 - 1e-9 for jamb in jambs)
                                 and all(check["pass"] for check in connector_checks))
            if navigation["pass"] or any(probe.get("pass") is True for probe in navigation.get("probes", [])):
                record.update(status="rejected", reason="measured_direct_passage_exists")
                rejected.append(record)
            elif geometry_complete and blocked:
                record.update(status="accepted", reason="raw_frame_and_actual_blocked_probe")
                accepted.append(record)
                additions.extend(record["proposed_primitives"])
            else:
                record["reason"] = ("incomplete_single_plane_or_jamb_geometry" if not geometry_complete
                                    else "no_walkable_crossing_found_without_actual_blocked_probe")
                record["interpretation"] = "inferred_window_boundary_pending_visual_confirmation"
            for number, primitive in enumerate(record["proposed_primitives"]):
                # Keep per-segment provenance usable without an external index.
                primitive["evidence"] = deepcopy(record["evidence"])
                if number:
                    primitive["source_raw_face_ids"] = jambs[number - 1]["source_raw_face_ids"]
    return {"primitives": original + deepcopy(additions), "candidates": candidates,
            "accepted": accepted, "rejected": rejected, "original_primitives_modified": False}
