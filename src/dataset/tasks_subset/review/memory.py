"""Derive public node descriptions from reviewed RGB in each saved prefix."""
from copy import deepcopy
from pathlib import Path
import shutil

from src.common.io import read_json

from ..bundle import snapshot_digest, write_json
from ..sampling.scene import room_at


def public_room(state, room_id):
    return state.get("room_aliases", {}).get(room_id)


def enrich_node_memory(bundle, root, output_bundle=None):
    root = Path(root).resolve()
    target = Path(output_bundle) if output_bundle else root / "private/enriched_bundle.json"
    if target.exists():
        raise FileExistsError(target)
    result = deepcopy(bundle)
    frames = bundle["frames"]
    for state in result["states"]:
        raw = state["snapshot"]
        source = root / raw
        directory = root / "states_enriched" / state["id"] / "snapshot"
        memory_file = Path("workspace/memory/nodes.json")
        memory = read_json(source / memory_file)
        state["room_aliases"] = {room: f"Room {i}" for i, room in
                                 enumerate(dict.fromkeys(state["visited_rooms"]), 1)}
        prefix = {k for k in state["frame_keys"] if frames[k]["step"] <= state["step"]}
        provenance = {}
        for node in memory["nodes"]:
            groups = {node["anchor"], node["panorama"], *node["history"]}
            local = {k for k in prefix if frames[k]["group"] in groups}
            anchor = next((k for k in sorted(local) if frames[k]["group"] == node["anchor"]
                           and frames[k]["yaw"] == 0), None)
            room = room_at(bundle["scene"], frames[anchor]["pose"]["position"]) if anchor else None
            node["room"] = public_room(state, room)
            facts = {}
            for sight in bundle["sightings"]:
                source_binding = sight.get("review_source", {})
                if (sight["frame"] not in local or sight.get("semantic_verified") is not True
                        or not sight.get("review_candidate") or source_binding.get("frame") != sight["frame"]
                        or source_binding.get("pixel") != sight.get("pixel")):
                    continue
                name = sight.get("review_name") if sight["kind"] == "object" else "open passage"
                if name:
                    facts.setdefault(name, {"frame": sight["frame"], "review_candidate": sight["review_candidate"],
                                            "review_source": source_binding})
            facts = dict(list(facts.items())[:4])
            if facts:
                node["description"] = "Observed here: " + ", ".join(facts) + "."
            provenance[node["id"]] = {"room": {"frame": anchor, "map_region": room},
                                      "description": facts}
        # Existing room-level summaries are not introduced by this stage.
        shutil.copytree(source, directory)
        write_json(directory / memory_file, memory)
        state["snapshot"] = str(directory.relative_to(root))
        state["memory_provenance"] = {"raw_snapshot": raw, "nodes": provenance,
                                       "map_hashes": bundle["source"]["sha256"]}
    states = {state["id"]: state for state in result["states"]}
    for trial in result["trials"]:
        state = states[trial["start_state"]]
        trial["historical_start_snapshot"] = state["memory_provenance"]["raw_snapshot"]
        if not trial.get("start_snapshot_sha256"):
            trial["invalidated"] = "missing_start_snapshot_binding"
        elif trial["start_snapshot_sha256"] != snapshot_digest(root / state["snapshot"]):
            trial["invalidated"] = "start_snapshot_changed_by_enrichment"
    write_json(target, result)
    return result
