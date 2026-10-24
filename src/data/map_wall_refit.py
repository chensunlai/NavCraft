"""Jointly refit verified near-coincident wall surfaces from original cuts.

This is an upstream geometry operation, not a wall detector or gap repair.
Independent same-surface evidence and NavMesh checks remain caller-owned.
"""

from copy import deepcopy

import numpy as np

from .map_wall_cad import overlap_report


def refit_overlapping_lines(primitives, source_segments, *, same_surface=None, not_blocks_navmesh=None):
    """Return ``primitives``, ``accepted`` and ``rejected`` refit records.

    Candidates come from the CAD near-overlap diagnostic. Each accepted join
    needs an explicit ``same_surface(first, second)`` pass on the original fits,
    valid original segment indices, maximum raw residual <=5cm and movement of
    every old line <=3cm. ``not_blocks_navmesh(segment)`` must pass for each
    finite projected coverage interval. Original gaps are never filled.

    Curves, wall axes and rejected/unverifiable fits retain their geometry.
    Graph metadata on refitted lines is removed for the caller to rebuild.
    """
    original = deepcopy(list(primitives))
    if len({item["id"] for item in original}) != len(original):
        raise ValueError("Wall primitive IDs must be unique")
    source = np.asarray(source_segments, dtype=float)
    if source.size == 0:
        source = source.reshape(0, 2, 2)
    if source.ndim != 3 or source.shape[1:] != (2, 2) or not np.isfinite(source).all():
        raise ValueError("source_segments requires finite original world-XZ segments")
    index = {item["id"]: i for i, item in enumerate(original)}
    lines = [item for item in original if item["type"] == "line"]
    candidates = overlap_report(lines, distance_budget_m=.03)["near_lines"]
    groups = {i: [i] for i in range(len(original))}
    owner = list(range(len(original)))
    fitted, surface_checks, accepted, rejected = {}, {}, [], []

    def checked(callback, *args):
        value = callback(*deepcopy(args))
        if not isinstance(value, dict) or type(value.get("pass")) is not bool:
            raise ValueError("Geometry callbacks must return an explicit boolean pass")
        return deepcopy(value)

    for candidate in candidates:
        first, second = (index[key] for key in candidate["canonical_ids"])
        left, right = owner[first], owner[second]
        if left == right:
            continue
        members = sorted(groups[left] + groups[right])
        items = [original[i] for i in members]
        record = {"input_ids": [item["id"] for item in items], "candidate": candidate}
        if any(item.get("representation") == "wall_axis" for item in items):
            rejected.append({**record, "reason": "wall_axes_are_not_raw_surface_fits"})
            continue
        ids, valid = set(), True
        for item in items:
            values = item.get("source_segment_indices")
            if not isinstance(values, (list, tuple)) or not values:
                valid = False
                break
            if any(isinstance(v, bool) or not isinstance(v, (int, np.integer))
                   or v < 0 or v >= len(source) for v in values):
                valid = False
                break
            ids.update(map(int, values))
        if not valid:
            rejected.append({**record, "reason": "missing_or_invalid_original_source_indices"})
            continue
        if same_surface is None:
            rejected.append({**record, "reason": "independent_same_surface_evidence_required"})
            continue
        proof = checked(same_surface, original[first], original[second])
        record["same_surface_check"] = proof
        if not proof["pass"]:
            rejected.append({**record, "reason": "independent_same_surface_check_failed"})
            continue
        record["same_surface_checks"] = (surface_checks.get(left, []) + surface_checks.get(right, [])
                                          + [{"input_ids": [original[first]["id"], original[second]["id"]],
                                              "evidence": proof}])
        ids = sorted(ids)
        raw = source[ids]
        lengths = np.linalg.norm(raw[:, 1] - raw[:, 0], axis=1)
        if np.any(lengths <= 1e-10):
            rejected.append({**record, "reason": "degenerate_original_source_segment"})
            continue
        points, weights = raw.reshape(-1, 2), np.repeat(lengths / 2, 2)
        center = np.average(points, axis=0, weights=weights)
        local = points - center
        _, vectors = np.linalg.eigh((local * weights[:, None]).T @ local / weights.sum())
        tangent = vectors[:, -1]
        if tangent[0] < -1e-10 or abs(tangent[0]) <= 1e-10 and tangent[1] < 0:
            tangent *= -1
        normal = np.array([-tangent[1], tangent[0]])
        residual = abs(local @ normal)
        old = [np.asarray([item["start"], item["end"]], dtype=float) for item in items]
        shifts = {item["id"]: float(abs((ends - center) @ normal).max())
                  for item, ends in zip(items, old)}
        fit = {"method": "length_weighted_joint_original_segment_tls",
               "max_residual_m": float(residual.max()),
               "rms_m": float(np.sqrt(np.average(residual ** 2, weights=weights))),
               "original_line_displacements_m": shifts,
               "max_original_line_displacement_m": max(shifts.values()),
               "source_segment_indices": ids,
               "center_xz": center.tolist(), "tangent_xz": tangent.tolist()}
        record["joint_fit"] = fit
        if residual.max() > .05 + 1e-10 or max(shifts.values()) > .03 + 1e-10:
            rejected.append({**record, "reason": "raw_residual_or_old_geometry_displacement_exceeded"})
            continue
        intervals = []
        for low, high in sorted(sorted((ends - center) @ tangent) for ends in old):
            if intervals and low <= intervals[-1][1] + 1e-8:
                intervals[-1][1] = max(intervals[-1][1], float(high))
            else:
                intervals.append([float(low), float(high)])
        segments = [center + np.asarray(interval)[:, None] * tangent for interval in intervals]
        if not_blocks_navmesh is None:
            rejected.append({**record, "reason": "navmesh_evidence_required"})
            continue
        checks = [checked(not_blocks_navmesh, segment.tolist()) for segment in segments]
        record["navmesh_checks"] = checks
        if not all(check["pass"] for check in checks):
            rejected.append({**record, "reason": "refitted_coverage_blocks_navmesh"})
            continue
        record.update(original_primitives=deepcopy(items), coverage_intervals_m=intervals,
                      coverage_scope="union_of_projected_old_segments; no gap fill or raw-span extension")
        carrier = max(items, key=lambda item: np.linalg.norm(np.subtract(item["end"], item["start"])))
        replacements = []
        for run, segment in enumerate(segments):
            value = deepcopy(carrier)
            for key in ("start_node", "end_node", "interior_nodes", "cad"):
                value.pop(key, None)
            identifier = carrier["id"] if len(segments) == 1 else f"{carrier['id']}-refit-{run}"
            while identifier in index and identifier not in record["input_ids"]:
                identifier += "-refit"
            value.update(id=identifier, start=segment[0].tolist(), end=segment[1].tolist(),
                         endpoints=segment.tolist(), length_m=float(np.linalg.norm(segment[1] - segment[0])),
                         source_ids=sorted({s for item in items for s in item.get("source_ids", [item["id"]])}),
                         source_primitive_ids=record["input_ids"], source_segment_indices=ids,
                         source_segment_count=len(ids), fitting=deepcopy(fit),
                         joint_refit=deepcopy(record), graph_requires_rebuild=True)
            replacements.append(value)
        destination = min(members)
        for member in members:
            owner[member] = destination
        del groups[left]
        del groups[right]
        groups[destination] = members
        fitted.pop(left, None)
        fitted.pop(right, None)
        fitted[destination] = replacements
        surface_checks.pop(left, None)
        surface_checks.pop(right, None)
        surface_checks[destination] = record["same_surface_checks"]
        accepted.append(record)
    output = []
    for i, item in enumerate(original):
        if owner[i] != i:
            continue
        output.extend(fitted.get(i, [item]))
    return {"primitives": output, "accepted": accepted, "rejected": rejected,
            "parameters": {"max_raw_residual_m": .05, "max_old_line_displacement_m": .03,
                           "source_weighting": "original segment length split equally between its endpoints"}}
