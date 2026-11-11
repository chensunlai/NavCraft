"""Derive instance reference images and node visibility review proposals."""
from collections import defaultdict
from math import dist


def different_view(a, b):
    return dist(a["pose"]["position"], b["pose"]["position"]) >= .2


def instance_targets(bundle):
    by_object = defaultdict(list)
    for s in bundle["sightings"]:
        f = bundle["frames"][s["frame"]]
        if (s["kind"] == "object" and s.get("identity_verified") is True
                and s.get("semantic_verified") is True
                and not f["stream"].startswith(("branch-", "replay-"))):
            by_object[s["id"]].append(s)
    for obj, views in sorted(by_object.items()):
        for ref in views:
            if any(different_view(bundle["frames"][ref["frame"]], bundle["frames"][x["frame"]])
                   for x in views):
                yield {"id": f"judgment-instance-{obj}", "object_id": obj,
                       "reference_frames": [ref["frame"]], "reference_pixel": ref["pixel"],
                       "identity_verified": True, "source_benchmark": None}
                break


def register_instance_targets(bundle):
    targets = bundle.setdefault("targets", [])
    known = {t["id"] for t in targets}
    targets.extend(t for t in instance_targets(bundle) if t["id"] not in known)


def target_reviews(bundle):
    for target, state, sight, threshold, _ in arrival_candidates(bundle):
        yield {"kind": "judgment_target_arrival", "name": "Arrival region",
               "frames": [target["reference_frames"][0], sight["frame"]],
               "markers": [target["reference_pixel"], sight["pixel"]],
               "prompt": f"Is the starting camera visibly within or outside {threshold} m horizontally "
                         "from the footprint of the marked object, with clear scale cues? Reject if uncertain.",
               "check_keys": ["answerable"],
               "fact": {"section": "judgment_evidence", "key": "arrivals", "value": {
                   "target": target["id"], "state": state["id"]}}}


def arrival_candidates(bundle):
    """Use a verified region and stay away from its boundary."""
    from ..navigation.scoring import footprint_distance, in_approach_region

    legal = {t["object_id"]: t for t in bundle.get("targets", [])
             if t.get("object_id") and t.get("approach_region") and t.get("positions")}
    sightings = {(s["frame"], s["id"]): s for s in bundle["sightings"]
                 if s.get("semantic_verified") and s.get("identity_verified")}
    for target in instance_targets(bundle):
        goal = legal.get(target["object_id"])
        if not goal:
            continue
        region = goal["approach_region"]
        ref = target["reference_frames"][0]
        threshold = region["max_footprint_distance"]
        for state in bundle["states"]:
            if ref not in state["frame_keys"]:
                continue
            key = state["current_frame"]
            frame = bundle["frames"][key]
            s = sightings.get((key, target["object_id"]))
            if not s or not different_view(frame, bundle["frames"][ref]) or state.get("room") != region["room"]:
                continue
            position = frame["pose"]["position"]
            distance = footprint_distance(position, region["rectangles"])
            if abs(distance - threshold) <= .25:
                continue
            arrived = in_approach_region(bundle["scene"], position, region)
            if arrived != (distance < threshold):
                continue
            yield target, state, s, threshold, arrived
