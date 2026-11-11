"""Derive instance reference images and node visibility review proposals."""
from collections import defaultdict
from math import dist


def different_view(a, b):
    return dist(a["pose"]["position"], b["pose"]["position"]) >= .2


def instance_targets(bundle,root=None):
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
                from ..review.instances import frame_visibility
                scopes={record["completeness_binding"]["scope"]
                        for key,row in bundle.get("instance_evidence",{}).items()
                        for identifier,record in row.get("objects",{}).items()
                        if identifier==obj and record.get("completeness_binding") and
                        frame_visibility(bundle,key,obj,root=root)!="unknown"}
                extra={"instance_scope":next(iter(scopes))} if len(scopes)==1 else {}
                yield {"id": f"judgment-instance-{obj}", "object_id": obj,
                       "reference_frames": [ref["frame"]], "reference_pixel": ref["pixel"],
                       "identity_verified": True, "source_benchmark": None, **extra}
                break


def register_instance_targets(bundle,root=None):
    targets = bundle.setdefault("targets", [])
    known = {t["id"] for t in targets}
    targets.extend(t for t in instance_targets(bundle,root) if t["id"] not in known)


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


def build_instance_correspondence(c):
    """Instance identity uses reviewed native/body association, not identical surface points."""
    from itertools import combinations
    from .construction import BOOL,mark,row
    groups=defaultdict(list)
    for key in c.state["frame_keys"]:
        for sight in c.sights[key]:
            if sight["kind"] in {"object","opening"} and sight.get("identity_verified"):
                groups[(sight["kind"],sight["id"])].append(sight)
    for group in groups.values():
        for first,second in combinations(group,2):
            if not different_view(c.frames[first["frame"]],c.frames[second["frame"]]):
                continue
            c.put("cross_view_identity","Do the marked instances A and B denote the same physical entity?",
                  BOOL,[True],[row(first["frame"],"First view",[mark(first,"A")]),
                               row(second["frame"],"Second view",[mark(second,"B")])],
                  private={"variant":"reviewed_instance_identity","measurements":[first,second],
                           "rule":"same independently verified physical instance; markings need not be same surface point"})
    for fact in c.bundle.get("navigation_evidence",{}).get("distinct_instances",[]):
        keys=fact.get("reference_frames",[])
        if not fact.get("review_candidate") or fact.get("same_category") is not True or len(keys)!=1:
            continue
        a,b=keys[0],fact["current_frame"]
        if {a,b}-c.keys:
            continue
        first=next((s for s in c.sights[a] if s["id"]==fact["requested_object"] and s.get("identity_verified")),None)
        second=next((s for s in c.sights[b] if s["id"]==fact["observed_object"] and s.get("identity_verified")),None)
        if first and second:
            c.put("cross_view_identity","Do the marked instances A and B denote the same physical entity?",
                  BOOL,[False],[row(a,"First view",[mark(first,"A")]),row(b,"Second view",[mark(second,"B")])],
                  private={"variant":"reviewed_distinct_instances","measurements":[first,second],"pair_fact":fact})
