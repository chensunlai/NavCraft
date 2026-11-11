"""Current-view correction evidence, separated from the prior room-assignment review."""
import json
from ..bundle import snapshot_digest


def current_correction_reviews(bundle,root):
    from ..navigation.construction import correction_applicable
    from .construction import Context,_node_snapshot
    for state in bundle["states"]:
        provenance=state.get("correction_provenance",{})
        proposed=provenance.get("fact",{})
        canonical=next((r for r in bundle.get("navigation_evidence",{}).get("corrections",[])
                        if r.get("review_candidate")==proposed.get("review_candidate")
                        and r.get("review_source")==proposed.get("review_source")),None)
        original=({key:canonical[key] for key in ("state","target","node","field","wrong","correct",
                   "evidence_frame","comparison_frames","review_candidate","review_source")}
                  if canonical else None)
        if not original or not correction_applicable(bundle,state,original,root):
            continue
        c=Context(bundle,state,root,{})
        current=c.current_panorama
        if len(current)!=6 or set(current)&set(original["comparison_frames"]):
            continue
        memory={n["id"]:n for n in _node_snapshot(c)["nodes"]}
        comparison=original["comparison_frames"]
        groups={}
        for node,keys in state["node_frames"].items():
            if set(keys)<=set(comparison):
                groups[node]={"room":memory[node]["room"],"pose":c.frames[keys[0]]["public_pose"],
                              "images":[6+comparison.index(k)+1 for k in keys]}
        rooms=sorted({group["room"] for group in groups.values()})
        if len(groups)<3 or len(rooms)<2:
            continue
        frames=[*current,*comparison]
        yield {"kind":"judgment_current_correction","name":"Current room association",
               "frames":frames,"markers":None,
               "captions":[f"Current panorama view {c.frames[k].get('yaw')}°" for k in current]+[
                   "Previously recorded room comparison" for _ in comparison],
               "prompt":"The first six images show the actual current panorama. The following independently "
                        "reviewed node panoramas are grouped into public room labels: "+json.dumps(groups)+
                        ". Determine which labelled physical room matches the current observation. "
                        "Set association_supported only when the actual current images and comparisons "
                        "distinguish the room; do not infer from the existence of a proposed correction.",
               "check_keys":["association_supported"],
               "value_keys":{"current_room":rooms},
               "annotations":[original,{"state":state["id"],"snapshot_sha256":snapshot_digest(c.root/state["snapshot"])}],
               "fact":{"section":"judgment_evidence","key":"current_corrections","value":{
                   "state":state["id"],"original":original,"current_frames":current}}}


def build_current_contradiction(c):
    from ..navigation.construction import correction_applicable,_altered_cut
    from .construction import _node_snapshot,enum,row
    memory={n["id"]:n for n in _node_snapshot(c)["nodes"]}
    for review in c.bundle.get("judgment_evidence",{}).get("current_corrections",[]):
        fact=review["original"]
        if (not review.get("review_candidate") or review["state"]!=c.state["id"]
                or review.get("current_room")!=fact["correct"]
                or set(review["current_frames"])!=set(c.current_panorama)
                or not correction_applicable(c.bundle,c.state,fact,c.root)
                or memory.get(fact["node"],{}).get(fact["field"])!=fact["correct"]):
            continue
        altered=_altered_cut(c.bundle,c.root,c.state,fact)
        c.put("memory_contradiction","Which node's recorded room contradicts the current panorama "
              "and the independent room comparisons? Return its node ID.",
              enum(*memory),[fact["node"]],[row(k,"Actual current panorama") for k in review["current_frames"]],
              memory=True,private={"altered_state":altered,"correction":fact,
                                   "current_association_review":review,"variant":"current_room_record_contradiction"})
