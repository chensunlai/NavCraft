from copy import deepcopy
import json

from PIL import Image
import pytest

from src.harness.trace import TraceWriter
from src.viewer.export import _memory_records, export_sequence


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def node_state(description="Sofa faces the television"):
    return {"nodes": [{"id": "N001", "anchor": "p_first", "panorama": "p_latest",
                       "room": "living_room", "description": "Beside the sofa",
                       "history": ["p_first", "p_latest"]}],
            "rooms": {"living_room": description}}


def observations():
    return {f"f_{index:04d}": {"frame": f"f_{index:04d}", "kind": "observation",
            "group": group, "yaw": yaw, "step": step, "pose": {"pos": pos, "heading": yaw},
            "path": f"observations/f_{index:04d}.jpg"}
            for index, (group, yaw, step, pos) in enumerate(
                [(group, yaw, step, pos) for group, step, pos in
                 (("p_first", 0, [.2, .3]), ("p_latest", 4, [.24, .3]))
                 for yaw in range(0, 360, 60)], 1)}


def test_node_display_uses_anchor_and_keeps_room_description_and_evidence():
    memory, frames = node_state(), observations()
    before = deepcopy(memory)
    node, room = _memory_records(memory, frames)
    assert node["type"] == "node" and node["point"] == [.2, .3]
    assert node["point"] != frames["f_0007"]["pose"]["pos"]
    assert node["evidence"] == list(frames)
    assert room == {"id": "room:living_room", "type": "room", "name": "living_room",
                    "description": "Sofa faces the television", "anchor": None,
                    "nodes": ["N001"], "evidence": ["f_0007"]}
    assert memory == before
    memory["nodes"].clear()
    assert _memory_records(memory, frames)[0]["nodes"] == []


def test_entity_display_records_are_unchanged():
    entities = [{"id": "O01", "type": "object", "name": "sofa", "point": [.1, .2]}]
    displayed = _memory_records(entities, {})
    assert displayed == entities
    displayed[0]["name"] = "Changed only in the viewer"
    assert entities[0]["name"] == "sofa"


@pytest.mark.parametrize("memory_events", [True, False])
def test_single_episode_exports_node_and_room_snapshots(tmp_path, memory_events):
    run, output = tmp_path / "node-run", tmp_path / "site"
    workspace = run / "workspace"
    frames = observations()
    for frame in frames.values():
        path = workspace / frame["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24), "gray").save(path)
    (workspace / "observations/index.jsonl").write_text(
        "".join(json.dumps(frame) + "\n" for frame in frames.values()))
    write_json(workspace / "scene/map.json", {"origin": [0, 0], "size_m": 20, "walls": []})
    before, after = node_state("Earlier room description"), node_state()
    write_json(workspace / "memory/nodes.json", after)
    write_json(run / "private/episode.json", {
        "id": "objnav:fixture:chair:1", "scene_id": "fixture", "benchmark": "objnav", "goals": [{}]})
    write_json(run / "result.json", {
        "model": "fixture", "reasoning_effort": "high", "memory_type": "node", "status": "completed",
        "environment": {"metrics": {"success": True, "steps": 4, "requests": 1,
                                     "navigation_error": .2, "success_distance": 1},
                        "motion_trajectory": [[4, 0, 6], [4.8, 0, 6]]}})
    trace = TraceWriter(run / "private")
    trace.emit("prompt_start", {"text": "Go next to the chair", "images": [frames["f_0001"]["path"]]})
    if memory_events:
        trace.emit("memory_changed", {"before": trace.store(before), "after": trace.store(after)})
    trace.emit("prompt_end", {"submitted": True})
    trace.close()
    original_trace = (run / "private/trace.jsonl").read_bytes()
    export_sequence(run, output)
    index = json.loads((output / "data/index.json").read_text())
    model = index["models"][0]
    detail = json.loads((output / model["rounds"][0]["detail"]).read_text())
    assert model["final_entities"] == 2
    node, room = detail["memory_after"]
    assert node["point"] == [.2, .3]
    assert room["description"] == after["rooms"]["living_room"]
    assert set(detail["evidence"]) == set(frames)
    if memory_events:
        assert detail["memory_before"][1]["description"] == "Earlier room description"
        assert detail["memory_changes"] == {"created": [], "deleted": [], "revised": ["room:living_room"]}
    else:
        assert detail["memory_before"] == [] and "Only the final" in detail["memory_status"]
    assert (run / "private/trace.jsonl").read_bytes() == original_trace
    assert json.loads((workspace / "memory/nodes.json").read_text()) == after
