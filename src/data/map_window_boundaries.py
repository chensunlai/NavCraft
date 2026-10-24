"""Reconstruct only wall gaps supported by complete raised window frames.

The raw frame is fitted over the complete current-floor instance, independently
of the width of a scanner-induced wall fragment gap. Four raw edge bands must
close a 3D frame and each support at least 90% of its fitted edge. Glass/frame
instances are grouped; adjacent-floor and floor-level glazing stay explicitly
outside automatic raised-window inference. Only missing wall spans are added.
"""

from copy import deepcopy
import math

import numpy as np


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
        for second in lines[index + 1 :]:
            b = np.asarray([second["start"], second["end"]], dtype=float)
            other = b[1] - b[0]
            if np.linalg.norm(other) <= 1e-8 or abs(
                tangent @ (other / np.linalg.norm(other))
            ) < math.cos(math.radians(3)):
                continue
            if np.abs((b - a[0]) @ normal).max() > 0.015:
                continue
            positions = (b - a[0]) @ tangent
            if positions.min() > length:
                span = np.stack((a[1], b[positions.argmin()]))
            elif positions.max() < 0:
                span = np.stack((b[positions.argmax()], a[0]))
            else:
                continue
            gap = np.linalg.norm(span[1] - span[0])
            if 0.12 <= gap <= 2.0:
                covered = False
                for carrier in lines:
                    if carrier["id"] in (first["id"], second["id"]):
                        continue
                    points = np.asarray([carrier["start"], carrier["end"]], dtype=float)
                    positions = (points - span[0]) @ tangent
                    if (
                        np.abs((points - span[0]) @ normal).max() <= 0.015
                        and min(gap, float(positions.max())) - max(0.0, float(positions.min())) > 0.025
                    ):
                        covered = True
                        break
                if covered:
                    continue
                yield span, [first["id"], second["id"]]


def _raw_edges(ids, u, v, y):
    """Columns are u0/u1, y0/y1, normal0/normal1, raw face ID, local edge ID."""
    data = []
    for edge, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
        data.append(
            np.column_stack(
                [
                    u[ids, i],
                    u[ids, j],
                    y[ids, i],
                    y[ids, j],
                    v[ids, i],
                    v[ids, j],
                    ids,
                    np.full(len(ids), edge),
                ]
            )
        )
    return np.concatenate(data)


def _bands(edges, axis, low, high, side):
    """Keep separated raw edge-band peaks near a complete frame's four sides."""
    start, end = (0, 1) if axis == "horizontal" else (2, 3)
    cross_start, cross_end = (2, 3) if axis == "horizontal" else (0, 1)
    along = np.abs(edges[:, end] - edges[:, start])
    across = np.abs(edges[:, cross_end] - edges[:, cross_start])
    selected = edges[(along > 0.005) & (across <= 0.04) & (across <= along * math.tan(math.radians(12)))]
    centers = selected[:, [cross_start, cross_end]].mean(1)
    bands = []
    for center in np.unique(np.round(centers / 0.025) * 0.025):
        if not low <= center <= high:
            continue
        evidence = selected[np.abs(centers - center) <= 0.026]
        intervals = _intervals(np.sort(evidence[:, [start, end]], axis=1), -1e6, 1e6)
        total = sum(b - a for a, b in intervals)
        bands.append(
            {"coordinate": float(center), "edges": evidence, "seed_length": total, "side": side}
        )
    chosen = []
    for band in sorted(bands, key=lambda b: b["seed_length"], reverse=True):
        if all(abs(band["coordinate"] - old["coordinate"]) > 0.05 for old in chosen):
            chosen.append(band)
        if len(chosen) >= 5:
            break
    return chosen


def _edge_evidence(band, low, high, axis):
    cols = (0, 1) if axis == "horizontal" else (2, 3)
    intervals = _intervals(np.sort(band["edges"][:, cols], axis=1), low, high)
    return {
        "coordinate_m": band["coordinate"],
        "coverage": _coverage(intervals, high - low),
        "intervals_m": intervals,
        "source_raw_face_ids": np.unique(band["edges"][:, 6]).astype(int).tolist(),
        "source_raw_edges": band["edges"][:, 6:8].astype(int).tolist(),
        "normal_span_m": float(np.ptp(band["edges"][:, 4:6])),
        "normal_range_m": [float(band["edges"][:, 4:6].min()), float(band["edges"][:, 4:6].max())],
    }


def _corner(horizontal, vertical, u, y):
    h = horizontal["edges"]
    v = vertical["edges"]
    h = h[(h[:, :2].max(1) >= u - 0.06) & (h[:, :2].min(1) <= u + 0.06)]
    v = v[(v[:, 2:4].max(1) >= y - 0.06) & (v[:, 2:4].min(1) <= y + 0.06)]
    if not len(h) or not len(v):
        return {"pass": False, "normal_gap_m": None}
    gap = max(float(h[:, 4:6].min() - v[:, 4:6].max()), float(v[:, 4:6].min() - h[:, 4:6].max()), 0.0)
    return {
        "pass": gap <= 0.06,
        "normal_gap_m": gap,
        "horizontal_raw_face_ids": np.unique(h[:, 6]).astype(int).tolist(),
        "vertical_raw_face_ids": np.unique(v[:, 6]).astype(int).tolist(),
    }


def _fit_raw_frame(raw, instance, a, tangent, floor):
    """Fit an elevation rectangle; measure each complete edge's relative support."""
    normal = np.array([-tangent[1], tangent[0]])
    semantic = instance.triangles.reshape(-1, 3)
    su = (semantic[:, [0, 2]] - a) @ tangent
    sy = semantic[:, 1] - floor
    left, right = float(su.min()), float(su.max())
    bottom, top = float(sy.min()), float(sy.max())
    width = right - left
    height = top - bottom
    if width < 0.4 or height < 0.4:
        return None
    u = (raw[:, :, [0, 2]] - a) @ tangent
    v = (raw[:, :, [0, 2]] - a) @ normal
    y = raw[:, :, 1] - floor
    local = (
        (u.max(1) >= left - 0.15)
        & (u.min(1) <= right + 0.15)
        & (v.min(1) >= -0.35)
        & (v.max(1) <= 0.35)
        & (y.min(1) >= max(0.2, bottom - 0.3))
        & (y.max(1) <= top + 0.4)
    )
    ids = np.flatnonzero(local)
    edges = _raw_edges(ids, u, v, y)
    lowers = _bands(edges, "horizontal", bottom - 0.25, bottom + 0.25, "sill")
    uppers = _bands(edges, "horizontal", top - 0.4, top + 0.2, "header")
    lefts = _bands(edges, "vertical", left - 0.15, left + min(0.25, width * 0.25), "left")
    rights = _bands(edges, "vertical", right - min(0.25, width * 0.25), right + 0.15, "right")
    best = None
    for lo in lowers:
        for hi in uppers:
            H = hi["coordinate"] - lo["coordinate"]
            if H < 0.4 or H < 0.6 * height:
                continue
            for l in lefts:
                for r in rights:
                    W = r["coordinate"] - l["coordinate"]
                    if W < 0.4 or W < 0.6 * width:
                        continue
                    evidence = [
                        _edge_evidence(lo, l["coordinate"], r["coordinate"], "horizontal"),
                        _edge_evidence(hi, l["coordinate"], r["coordinate"], "horizontal"),
                        _edge_evidence(l, lo["coordinate"], hi["coordinate"], "vertical"),
                        _edge_evidence(r, lo["coordinate"], hi["coordinate"], "vertical"),
                    ]
                    coverage = [e["coverage"] for e in evidence]
                    corners = [
                        _corner(h, z, z["coordinate"], h["coordinate"]) for h in (lo, hi) for z in (l, r)
                    ]
                    depth_supported = all(e["normal_span_m"] >= 0.025 for e in evidence)
                    passed = (
                        min(coverage) >= 0.9 - 1e-9
                        and all(c["pass"] for c in corners)
                        and depth_supported
                    )
                    score = (passed, min(coverage), sum(coverage), -abs(W - width) - abs(H - height))
                    if best is None or score > best[0]:
                        best = (
                            score,
                            {
                                "pass": bool(passed),
                                "left_m": l["coordinate"],
                                "right_m": r["coordinate"],
                                "sill_height_m": lo["coordinate"],
                                "header_height_m": hi["coordinate"],
                                "width_m": W,
                                "height_m": H,
                                "edges": dict(zip(["sill", "header", "left", "right"], evidence)),
                                "corners": corners,
                                "source_raw_face_ids": sorted(
                                    {i for e in evidence for i in e["source_raw_face_ids"]}
                                ),
                                "minimum_edge_coverage": min(coverage),
                                "three_dimensional_frame_depth_support": bool(depth_supported),
                                "semantic_bbox": {
                                    "left": left,
                                    "right": right,
                                    "bottom": bottom,
                                    "top": top,
                                },
                            },
                        )
    return best[1] if best else None


def _window_groups(instances):
    values = []
    for instance in instances:
        if instance.category != "window":
            continue
        points = np.asarray(instance.triangles, dtype=float).reshape(-1, 3)
        if not len(points) or not np.isfinite(points).all():
            continue
        center = points[:, [0, 2]].mean(0)
        _, _, axes = np.linalg.svd(points[:, [0, 2]] - center, full_matrices=False)
        axis = axes[0]
        along = (points[:, [0, 2]] - center) @ axis
        values.append(
            (
                instance,
                points,
                center,
                axis,
                float(np.ptp(along) * (points[:, 1].max() - points[:, 1].min())),
            )
        )
    groups = []
    for item in sorted(values, key=lambda v: ("frame" not in v[0].label.casefold(), -v[4])):
        instance, points, center, axis, _ = item
        assigned = False
        for group in groups:
            original = group[0]
            other = original[1]
            base = original[2]
            t = original[3]
            n = np.array([-t[1], t[0]])
            a = (other[:, [0, 2]] - base) @ t
            b = (points[:, [0, 2]] - base) @ t
            W = float(np.ptp(b))
            H = float(np.ptp(points[:, 1]))
            overlap = min(a.max(), b.max()) - max(a.min(), b.min())
            vertical = min(other[:, 1].max(), points[:, 1].max()) - max(
                other[:, 1].min(), points[:, 1].min()
            )
            if (
                W > 0
                and H > 0
                and overlap >= 0.8 * W
                and vertical >= 0.8 * H
                and abs((center - base) @ n) <= 0.35
                and abs(axis @ t) >= math.cos(math.radians(20))
            ):
                group.append(item)
                assigned = True
                break
        if not assigned:
            groups.append([item])
    return groups


def _body_support(primitives, a, tangent, left, right):
    normal = np.array([-tangent[1], tangent[0]])
    sides = {"left": [], "right": []}
    for wall in primitives:
        if wall["type"] != "line":
            continue
        points = np.asarray([wall["start"], wall["end"]])
        delta = points[1] - points[0]
        length = np.linalg.norm(delta)
        if length <= 1e-8 or abs(delta @ tangent) / length < math.cos(math.radians(5)):
            continue
        positions = (points - a) @ tangent
        lo, hi = np.sort(positions)
        clipped = np.array([max(lo, left - 0.15), min(hi, right + 0.15)])
        if clipped[1] < clipped[0]:
            continue
        offsets = (points - a) @ normal
        local_offsets = offsets[0] + (clipped - positions[0]) * (offsets[1] - offsets[0]) / (
            positions[1] - positions[0]
        )
        if np.abs(local_offsets).max() > 0.05:
            continue
        if lo < left - 0.05 and hi >= left - 0.15:
            sides["left"].append(wall["id"])
        if hi > right + 0.05 and lo <= right + 0.15:
            sides["right"].append(wall["id"])
    return {
        "pass": bool(sides["left"] and sides["right"]),
        "architectural_wall_ids": sides,
        "plane_tolerance_m": 0.05,
        "frame_outer_span_m": [left, right],
    }


def build_window_boundaries(
    raw_triangles, primitives, window_instances, *, floor_height, passage, not_blocks_navmesh=None
):
    """Infer only existing wall gaps covered by an independently fitted raw frame.

    ``passage`` returns MapScene's ``{pass, probes}``; any successful crossing is
    a veto. ``not_blocks_navmesh`` optionally checks existing walkable space.
    Missing probes stay unknown. Returned ``not_applicable`` groups include
    neighbouring-floor windows and floor-level glazing, separately from local
    pending candidates. Complete-frame support does not measure a solid wall.
    """
    raw = np.asarray(raw_triangles, dtype=float).reshape(-1, 3, 3)
    if not np.isfinite(raw).all() or not math.isfinite(floor_height):
        raise ValueError("Raw triangles and floor height must be finite")
    original = deepcopy(list(primitives))
    gaps = list(_gaps(original))
    candidates = []
    accepted = []
    rejected = []
    not_applicable = []
    additions = []
    built = []
    for group in _window_groups(window_instances):
        instance, points, _, _, _ = group[0]
        ymin = float(points[:, 1].min() - floor_height)
        ymax = float(points[:, 1].max() - floor_height)
        aliases = [v[0].id for v in group]
        if not 0.25 <= ymin <= 1.8 or ymax > 3.2:
            not_applicable.append(
                {
                    "semantic_instance_ids": aliases,
                    "height_above_floor_m": [ymin, ymax],
                    "reason": "outside_supported_current_floor_raised_window_range",
                    "is_absent_window": False,
                }
            )
            continue
        for span, adjacent in gaps:
            a, b = span
            L = float(np.linalg.norm(b - a))
            t = (b - a) / L
            n = np.array([-t[1], t[0]])
            u = (points[:, [0, 2]] - a) @ t
            v = (points[:, [0, 2]] - a) @ n
            projected = (instance.triangles[:, :, [0, 2]] - a) @ t
            semantic_intervals = _intervals(np.stack([projected.min(1), projected.max(1)], axis=1), 0, L)
            coverage = _coverage(semantic_intervals, L)
            if coverage < 0.8 - 1e-9 or max(abs(np.quantile(v, [0.05, 0.95]))) > 0.35:
                continue
            if any(
                np.linalg.norm(span.mean(0) - old.mean(0)) < 0.05
                and np.linalg.norm(span[1] - span[0]) <= np.linalg.norm(old[1] - old[0]) + 0.05
                for old in built
            ):
                continue
            serial = len(candidates) + 1
            identifier = f"WINDOW{serial:04d}"
            while any(wall["id"] == identifier + "-0" for wall in original):
                serial += 1
                identifier = f"WINDOW{serial:04d}"
            record = {
                "id": identifier,
                "segment": span.tolist(),
                "adjacent_primitive_ids": adjacent,
                "semantic_instance_id": instance.id,
                "semantic_instance_ids": aliases,
                "source_kind": "reconstructed_window_wall",
                "evidence_kind": "structural_inference",
                "solid_wall_measured": False,
                "glass_open_or_closed": "not_determined",
                "status": "needs_review",
                "proposed_primitives": [],
                "evidence": {
                    "semantic_prior_coverage": coverage,
                    "semantic_prior_is_raw_proof": False,
                    "semantic_source": deepcopy(instance.source),
                    "raw_geometry_source_kind": "original_render_mesh",
                    "raw_face_id_space": "row_in_input_raw_triangles",
                    "frame_coordinate_origin_world_xz": a.tolist(),
                    "frame_tangent_world_xz": t.tolist(),
                    "floor_height_world_y": floor_height,
                },
            }
            candidates.append(record)
            frame = _fit_raw_frame(raw, instance, a, t, floor_height)
            record["evidence"]["full_raw_frame"] = frame
            if frame is None:
                record["reason"] = "no_raw_rectangular_frame_fit"
                continue
            rectangle_ok = (
                frame["pass"]
                and 0.25 <= frame["sill_height_m"] <= 1.8
                and frame["header_height_m"] <= 3.2
                and frame["left_m"] - 0.15 <= 1e-9
                and frame["right_m"] + 0.15 >= L - 1e-9
            )
            body = _body_support(original, a, t, float(u.min()), float(u.max()))
            record["evidence"]["wall_body_continuity"] = body
            record["evidence"]["frame_rectangle"] = {
                "pass": bool(rectangle_ok),
                "width_m": frame["width_m"],
                "height_m": frame["height_m"],
                "edge_coverage": {k: x["coverage"] for k, x in frame["edges"].items()},
                "corners": frame["corners"],
                "absolute_minimum_jamb_height_required": False,
            }
            record["source_raw_face_ids"] = frame["source_raw_face_ids"]
            record["proposed_primitives"] = [
                {
                    "id": identifier + "-0",
                    "type": "line",
                    "start": a.tolist(),
                    "end": b.tolist(),
                    "endpoints": span.tolist(),
                    "length_m": L,
                    "role": "architectural_window_wall",
                    "source_kind": "reconstructed_window_wall",
                    "evidence_kind": "structural_inference",
                    "solid_wall_measured": False,
                    "source_raw_face_ids": frame["source_raw_face_ids"],
                    "window_candidate_id": identifier,
                }
            ]
            navigation = passage(span.tolist())
            if not isinstance(navigation, dict) or type(navigation.get("pass")) is not bool:
                raise ValueError("passage must return explicit boolean pass")
            walkability = (
                {"pass": None, "source": "not_provided"}
                if not_blocks_navmesh is None
                else not_blocks_navmesh(span.tolist())
            )
            if (
                not isinstance(walkability, dict)
                or not_blocks_navmesh is not None
                and type(walkability.get("pass")) is not bool
            ):
                raise ValueError("not_blocks_navmesh must return explicit boolean pass")
            record["evidence"].update(
                navmesh=navigation,
                walkable_space_check=walkability,
                robot_crossing_status=(
                    "not_established" if not navigation.get("probes") else "recorded_probes"
                ),
                reconstruction_plane="adjacent_architectural_walls_only_missing_gap_is_added",
            )
            if navigation["pass"] or any(
                probe.get("pass") is True for probe in navigation.get("probes", [])
            ):
                record.update(status="rejected", reason="measured_direct_passage_exists")
                rejected.append(record)
            elif walkability["pass"] is False:
                record.update(status="rejected", reason="would_block_measured_walkable_space")
                rejected.append(record)
            elif rectangle_ok and body["pass"]:
                record.update(
                    status="accepted", reason="full_raw_frame_and_architectural_wall_body_continuity"
                )
                accepted.append(record)
                additions.extend(record["proposed_primitives"])
                built.append(span)
            else:
                record["reason"] = "incomplete_full_frame_or_wall_body_connection"
            for line in record["proposed_primitives"]:
                line["evidence"] = deepcopy(record["evidence"])
    return {
        "primitives": original + deepcopy(additions),
        "candidates": candidates,
        "accepted": accepted,
        "rejected": rejected,
        "not_applicable": not_applicable,
        "original_primitives_modified": False,
    }
