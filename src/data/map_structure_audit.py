"""Machine-readable structural audit for a fitted floor-plan candidate.

The audit does not repair geometry.  It makes unresolved joins, gaps and room
claims explicit so a NavMesh pass cannot be mistaken for an architectural pass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .map_gap_audit import line_gap_candidates


def near_endpoint_clusters(primitives, *, tolerance=.05):
    endpoints = []
    for primitive in primitives:
        if primitive.get("type") != "line":
            continue
        for at in ("start", "end"):
            endpoints.append((primitive["id"], at, np.asarray(primitive[at], dtype=float)))
    clusters, used = [], set()
    for index, item in enumerate(endpoints):
        if index in used:
            continue
        members = [j for j, other in enumerate(endpoints)
                   if j not in used and np.linalg.norm(item[2] - other[2]) <= tolerance]
        spread = max((np.linalg.norm(endpoints[a][2] - endpoints[b][2])
                      for a in members for b in members), default=0.0)
        if len(members) > 1 and spread > 1e-5:
            used.update(members)
            clusters.append({"endpoints": [{"primitive_id": endpoints[j][0], "at": endpoints[j][1],
                                             "point": endpoints[j][2].tolist()} for j in members],
                             "center": np.mean([endpoints[j][2] for j in members], axis=0).tolist(),
                             "tolerance_m": tolerance})
    return clusters


def audit_fitted(fitted):
    fitted = Path(fitted).resolve()
    geometry = json.loads((fitted / "wall_geometry.json").read_text())
    meta = json.loads((fitted / "meta.json").read_text())
    openings = json.loads((fitted / "openings.json").read_text())
    gaps = line_gap_candidates(geometry["primitives"])
    accepted = [item for item in openings if item.get("status") == "accepted"]
    for gap in gaps:
        centre = np.mean(gap["segment"], axis=0)
        gap["opening_record"] = next((item.get("id") for item in accepted
                                       if np.linalg.norm(centre - np.mean(item["segment"], axis=0)) < .2), None)
        gap["status"] = "explained_opening" if gap["opening_record"] else "unresolved_gap"
    unresolved = [item for item in gaps if item["status"] == "unresolved_gap"]
    clusters = near_endpoint_clusters(geometry["primitives"])
    result = {
        "fitted": str(fitted),
        "scene": meta.get("scene"),
        "status": "pending_structural_review" if unresolved or clusters else "needs_visual_review",
        "counts": {"primitives": len(geometry["primitives"]),
                   "connections": len(geometry.get("connections", [])),
                   "unexplained_endpoints": len(geometry.get("unexplained_endpoints", [])),
                   "near_endpoint_clusters": len(clusters),
                   "candidate_gaps": len(gaps),
                   "unresolved_gaps": len(unresolved),
                   "accepted_openings": len(accepted),
                   "regions": len(json.loads((fitted / "regions.json").read_text()))},
        "near_endpoint_clusters": clusters,
        "gaps": gaps,
        "limitations": [
            "Endpoint proximity is a review cue, not proof of a common wall junction.",
            "A room candidate has no identity or adjacency claim until visual and entry evidence is checked.",
            "A one-pixel centerline does not measure wall thickness.",
        ],
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fitted", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit_fitted(args.fitted)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["counts"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
