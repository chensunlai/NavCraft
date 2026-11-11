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


def arrival_pictures(c, target, sight):
    """Give current surroundings and a public baseline for the actual target surface."""
    from .construction import row,mark
    from .calibration import shared_pair,public_camera
    other=shared_pair(c,sight)
    if other is None:
        return None
    ref=target["reference_frames"][0]
    pictures=[row(ref,"Target reference",[("T",target["reference_pixel"])])]
    for point,title in ((sight,"Current target surface A"),(other,"Same target surface A from a known baseline")):
        pictures.append(row(point["frame"],title+"; "+public_camera(c,point["frame"]),[mark(point,"A")]))
    for key in c.current_panorama:
        pictures.append(row(key,"Current surroundings; "+public_camera(c,key)))
    return pictures


def target_reviews(bundle,root):
    from .construction import Context
    for target, state, sight, threshold, _ in arrival_candidates(bundle):
        c=Context(bundle,state,root,{})
        pictures=arrival_pictures(c,target,sight)
        if pictures is None:
            continue
        yield {"kind": "judgment_target_arrival", "name": "Arrival region",
               "frames": [p[0] for p in pictures],
               "markers": [p[2][0][1] if p[2] else None for p in pictures],
               "prompt": f"Can these public images, intrinsics and camera poses establish whether the current camera is within or outside {threshold} m horizontally "
                         "from the marked target object's physical footprint (same floor/room)? A marks one actual common surface, not the whole footprint. "
                         "The answer need not be yes; accept only if the extent and metric separation are actually inferable. Reject hidden/unclear footprint or uncertain scale. "
                         + " ".join(f"Image {i+1}: {p[1]}" for i,p in enumerate(pictures)),
               "check_keys": ["answerable"],
               "fact": {"section": "judgment_evidence", "key": "arrivals", "value": {
                   "target": target["id"], "state": state["id"],"frames":[p[0] for p in pictures]}}}


def arrival_candidates(bundle):
    """Physical task region truth, with current-panorama reviewed or bound-native identity."""
    from ..navigation.scoring import footprint_distance, in_approach_region
    legal = {t["object_id"]: t for t in bundle.get("targets", [])
             if t.get("object_id") and t.get("approach_region") and t.get("positions")}
    sightings = [s for s in bundle["sightings"]+bundle.get("derived_sightings",[])
                 if s.get("identity_verified") and (s.get("semantic_verified") or
                    s.get("method")=="native_instance_reprojection" and s.get("evidence_binding"))]
    for target in instance_targets(bundle):
        goal = legal.get(target["object_id"])
        if not goal:
            continue
        region = goal["approach_region"]
        ref = target["reference_frames"][0]
        threshold = region["max_footprint_distance"]
        for state in bundle["states"]:
            if ref not in state["frame_keys"] or state.get("room") != region["room"]:
                continue
            current=bundle["frames"][state["current_frame"]]
            keys=[k for k in state["frame_keys"] if k==state["current_frame"] or current.get("group") and
                  bundle["frames"][k].get("group")==current["group"]]
            candidates=[s for s in sightings if s["frame"] in keys and s["id"]==target["object_id"]
                        and different_view(bundle["frames"][s["frame"]],bundle["frames"][ref])]
            if not candidates:
                continue
            position = current["pose"]["position"]
            distance = footprint_distance(position, region["rectangles"])
            if abs(distance - threshold) <= .25:
                continue
            arrived = in_approach_region(bundle["scene"], position, region)
            if arrived != (distance < threshold):
                continue
            for sight in candidates:
                yield target, state, sight, threshold, arrived


def pair_context(c, fact):
    """Keep exactly the reviewed public surroundings, or reject an illegal prefix."""
    from .construction import row
    context = fact.get("public_context", [])
    if any(item["frame"] not in c.keys for item in context):
        return None
    return [row(item["frame"], item["caption"]) for item in context]

PAIR_CONTEXT_CONDITION = " These extra panoramas depict the same static scene; objects have not moved. Use their public capture poses and surroundings."


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
        context = pair_context(c, fact)
        if first and second and context is not None:
            c.put("cross_view_identity","Do the marked instances A and B denote the same physical entity?" + (PAIR_CONTEXT_CONDITION if context else ""),
                  BOOL,[False],[row(a,"First view",[mark(first,"A")]),row(b,"Second view",[mark(second,"B")])] + context,
                  private={"variant":"reviewed_distinct_instances","measurements":[first,second],"pair_fact":fact})
