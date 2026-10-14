"""Private verification rejects post-hoc, ungrounded, or unevidenced mapping claims."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pi_verify", ROOT / "src/harness-pi/python/verify.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_answer_parser_requires_anchor_and_unambiguous_final_answer():
    assert MODULE.parse_answer('Explanation.\n```json\n{"choice":"D03","anchor":"initial_observation"}\n```')["choice"] == "D03"
    with pytest.raises(ValueError):
        MODULE.parse_answer('{"choice":"D03"}')
    with pytest.raises(ValueError):
        MODULE.parse_answer('{"choice":"D03","anchor":"initial"} {"choice":"D02","anchor":"initial"}')


def test_qa_uses_actual_answer_and_original_time_anchor(tmp_path):
    (tmp_path / "private").mkdir()
    (tmp_path / "task.json").write_text(json.dumps({"id": "qa", "form": "retrieval_qa"}))
    (tmp_path / "private/labels.json").write_text(json.dumps({"choice": "D03", "anchor": "initial_observation", "label_method": "private geometry"}))
    run = lambda answer: MODULE.qa(workspace=tmp_path, sample_dir=tmp_path, answer=answer)
    assert run('{"choice":"D03","anchor":"initial_observation"}')["passed"]
    assert not run('{"choice":"D03","anchor":"after_motion"}')["passed"]
    assert not run('I succeeded')["passed"]


def mapping_fixture(tmp_path, monkeypatch):
    workspace, scene, trace, env = [tmp_path / name for name in ("workspace", "scene", "trace", "environment")]
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    write(scene / "raw/doors.json", {"doors": [
        {"id": "D03", "level": "L1", "segment": [[4, 5], [6, 5]], "connects": ["R01", "R03"]},
        {"id": "D02", "level": "L1", "segment": [[7, 8], [9, 8]], "connects": ["R01", "R02"]}]})
    panel = SimpleNamespace(level_id="L1", x_scale=10, y_scale=-10, x_offset=0, y_offset=100)
    import src.harness.map
    monkeypatch.setattr(src.harness.map, "MapCalibration", lambda directory: SimpleNamespace(viewbox=[0, 0, 100, 100], panels=[panel]))
    map_path = "memory/maps/L1.entities.json"
    write(workspace / "scene/maps/index.json", {"maps": [{"map_id": "map_L1", "current_entities_ref": map_path}]})
    entity = {"id": "newDoor", "map_id": "map_L1", "type": "portal", "geometry": {"segment": [[.4, .5], [.6, .5]]}}
    card_path = "memory/entities/newDoor.md"
    card1 = "# newDoor\n- 连接：R01 与 R03\n- 证据：`frame_000001`\n"
    card2 = card1.replace('`frame_000001`', '`frame_000001`, `frame_000002`') + '- 描述：新视角核对门口\n'
    write(workspace / map_path, [entity])
    (workspace / card_path).parent.mkdir(parents=True, exist_ok=True)
    (workspace / card_path).write_text(card2)
    observations, frames = [], []
    for i, x in [(1, 5), (2, 4.8)]:
        identifier = f"frame_{i:06d}"
        path = workspace / f"observations/{identifier}.png"
        path.parent.mkdir(exist_ok=True)
        Image.new("RGB", (64, 48), color=(20 * i, 40, 60)).save(path)
        private = env / f"environment/observations/rgb_{i:06d}.png"
        private.parent.mkdir(parents=True, exist_ok=True)
        private.write_bytes(path.read_bytes())
        observations.append({"id": identifier, "path": str(path.relative_to(workspace)), "source": "observation"})
        frames.append({"id": identifier, "pose": {"world_from_optical": [[1, 0, 0, x], [0, -1, 0, 1], [0, 0, -1, -3], [0, 0, 0, 1]]}, "camera": {"hfov_degrees": 90}})
    (workspace / "observations/index.jsonl").write_text(''.join(json.dumps(x) + '\n' for x in observations))
    (env / "frames.jsonl").write_text(''.join(json.dumps(x) + '\n' for x in frames))
    (trace / "blobs").mkdir(parents=True)
    def digest(value):
        text = value if isinstance(value, str) else json.dumps(value)
        name = hashlib.sha256(text.encode()).hexdigest()
        (trace / "blobs" / name).write_text(text)
        return name
    base, geom, c1, c2 = digest([]), digest([entity]), digest(card1), digest(card2)
    events = [
        {"type": "workspace_changes", "data": {"label": "initial", "changes": [{"path": map_path, "after": base}]}},
        {"type": "workspace_changes", "data": {"label": "after_tool:write", "changes": [{"path": map_path, "before": base, "after": geom}, {"path": card_path, "after": c1}]}},
        {"type": "tool_execution_end", "data": {"toolName": "nav_observe", "result": {"views": [{"id": "frame_000002"}]}, "isError": False}},
        {"type": "workspace_changes", "data": {"label": "after_tool:edit", "changes": [{"path": card_path, "before": c1, "after": c2}]}},
        {"type": "tool_execution_start", "data": {"toolCallId": "read1", "toolName": "read", "args": {"path": card_path}}},
        {"type": "tool_execution_end", "data": {"toolCallId": "read1", "toolName": "read", "result": {"content": [{"type": "text", "text": card2}]}, "isError": False}}]
    def save():
        (trace / "trace.jsonl").write_text(''.join(json.dumps(x) + '\n' for x in events))
    save()
    args = {"workspace": workspace, "scene_dir": scene, "trace_dir": trace, "environment_private_dir": env}
    return args, events, save, write


def test_mapping_requires_geometry_connections_real_evidence_and_online_use(tmp_path, monkeypatch):
    args, events, save, _ = mapping_fixture(tmp_path, monkeypatch)
    result = MODULE.mapping(**args)
    assert result["passed"] and result["entity_id"] == "newDoor"
    assert result["geometry"]["nearest_archived_portal"] == "D03"
    assert result["lifecycle"]["created_during_run"]
    assert result["lifecycle"]["revised_after_new_evidence"]
    assert result["visual_review_required"]
    events[-1]["data"]["isError"] = True
    save()
    assert not MODULE.mapping(**args)["passed"]


def test_posthoc_or_prebuilt_portal_is_not_online_construction(tmp_path, monkeypatch):
    args, events, save, _ = mapping_fixture(tmp_path, monkeypatch)
    events[0]["data"]["changes"] = deepcopy(events[1]["data"]["changes"])
    save()
    result = MODULE.mapping(**args)
    assert not result["lifecycle"]["created_during_run"]
    assert not result["passed"]


def test_revision_before_new_observation_is_not_evidence_driven(tmp_path, monkeypatch):
    args, events, save, _ = mapping_fixture(tmp_path, monkeypatch)
    events[2], events[3] = events[3], events[2]
    save()
    result = MODULE.mapping(**args)
    assert not result["lifecycle"]["revised_after_new_evidence"]
    assert not result["passed"]


def test_model_cannot_forge_evidence_image(tmp_path, monkeypatch):
    args, _, _, _ = mapping_fixture(tmp_path, monkeypatch)
    Image.new("RGB", (64, 48), "red").save(args["workspace"] / "observations/frame_000002.png")
    result = MODULE.mapping(**args)
    assert not result["passed"] and not result["evidence_ok"]
    assert not result["evidence"][1]["authentic_rgb"]


def test_geometrically_wrong_portal_and_wrong_connection_are_rejected(tmp_path, monkeypatch):
    args, _, _, write = mapping_fixture(tmp_path, monkeypatch)
    path = args["workspace"] / "memory/maps/L1.entities.json"
    entities = json.loads(path.read_text())
    entities[0]["geometry"]["segment"] = [[.7, .2], [.9, .2]]
    write(path, entities)
    result = MODULE.mapping(**args)
    assert not result["passed"] and result["geometry"]["nearest_archived_portal"] == "D02"
    card = args["workspace"] / "memory/entities/newDoor.md"
    card.write_text(card.read_text().replace("R03", "R02"))
    assert not MODULE.mapping(**args)["connections_ok"]


def test_trace_current_state_must_match_recorded_snapshots(tmp_path, monkeypatch):
    args, _, _, _ = mapping_fixture(tmp_path, monkeypatch)
    card = args["workspace"] / "memory/entities/newDoor.md"
    card.write_text(card.read_text() + "Unrecorded edit\n")
    result = MODULE.mapping(**args)
    assert not result["lifecycle"]["final_snapshot_matches"]
    assert not result["passed"]


def test_intermediate_invalid_json_does_not_erase_valid_online_history(tmp_path, monkeypatch):
    args, events, save, _ = mapping_fixture(tmp_path, monkeypatch)
    invalid = '[{"id":'
    digest = hashlib.sha256(invalid.encode()).hexdigest()
    (args["trace_dir"] / "blobs" / digest).write_text(invalid)
    restored = events[1]["data"]["changes"][0]["after"]
    path = "memory/maps/L1.entities.json"
    events[2:2] = [
        {"type": "workspace_changes", "data": {"label": "after_tool:edit", "changes": [{"path": path, "after": digest}]}},
        {"type": "workspace_changes", "data": {"label": "after_tool:edit", "changes": [{"path": path, "after": restored}]}},
    ]
    save()
    assert MODULE.mapping(**args)["passed"]


def test_image_goal_only_render_restores_camera(tmp_path):
    goal = {"modality": "image", "instruction": "Find picture", "positions": [[0, 0, 0]],
            "image_goal": {"position": [3, 4, 5], "rotation": [0, 0, 0, 1], "hfov": 90, "image_dimensions": [48, 64]}}
    episode = {"id": "goat:test", "source_id": "test", "benchmark": "goat", "scene_id": "fake.glb",
               "start_position": [0, 0, 0], "start_rotation": [0, 0, 0, 1], "source_file": "fixture", "goals": [goal]}
    (tmp_path / "manifest.json").write_text(json.dumps({"episodes": [episode]}))
    result = MODULE.image_goal(repo=tmp_path, private_dir=tmp_path / "out", env_config={"name": "fake"},
                               task_config={"episodes": {"manifest": "manifest.json"}}, episode_id="goat:test", goal_index=0)
    assert result["passed"] and result["camera_restored"]
    assert result["image_size"] == [64, 48]
