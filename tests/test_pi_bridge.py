"""Public/private boundaries and exact-once movement in the Pi environment bridge."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pi_bridge", ROOT / "src/harness-pi/python/bridge.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
Bridge = MODULE.Bridge


def init_args(tmp_path, *, family="objectnav", goals=None, task_overrides=None):
    episode = {"id": "fixture:pi", "source_id": "pi", "benchmark": "goat" if family == "goat" else "objnav",
               "scene_id": "fake.glb", "start_position": [0, 0, 0], "start_rotation": [0, 0, 0, 1],
               "source_file": "private-fixture", "goals": goals or [
                   {"modality": "object", "instruction": "Find chair", "positions": [[0, 0, -1]]}]}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "validated": True, "episodes": [episode]}))
    return {"repo": str(ROOT), "workspace": str(tmp_path / "workspace"),
            "private_dir": str(tmp_path / "private"), "episode_id": episode["id"],
            "env_config": {"name": "fake", "depth": True, "forward_step": .25, "turn_angle": 30},
            "task_config": {"episodes": {"name": "manifest", "root": str(tmp_path), "manifest": str(manifest)},
                            "task": {"name": family, **(task_overrides or {})}}}


def call(bridge, request_id, op, **args):
    response = bridge.dispatch({"id": request_id, "op": op, "args": args})
    assert response["ok"], response
    return response["data"]


@pytest.fixture
def initialized(tmp_path):
    bridge = Bridge()
    args = init_args(tmp_path)
    result = call(bridge, "init", "init", **args)
    yield bridge, result, tmp_path
    bridge.close()


def test_public_rgb_and_initial_state_do_not_leak_private_pose(initialized):
    bridge, result, root = initialized
    serialized = json.dumps(result)
    for forbidden in ("positions", "rotation_xyzw", "world_from_optical", "depth", "distance", "success"):
        assert forbidden not in serialized
    assert result["observation"]["step"] == 0
    assert (root / "workspace" / result["images"][0]).is_file()
    assert (root / "private/environment/private/state.json").is_file()
    assert not (root / "workspace/private").exists()
    assert not list((root / "workspace").rglob("*.npy"))
    private = json.loads((root / "private/frames.jsonl").read_text().splitlines()[0])
    assert private["pose"]["position"] == [0, 0, 0]


def test_fresh_observation_and_panorama_preserve_state(initialized):
    bridge, initial, root = initialized
    single = call(bridge, "single", "observe")
    assert single["observation"]["id"] != initial["observation"]["id"]
    call(bridge, "tilt", "step", actions=["look_up"])
    point, yaw, pitch = bridge.env.sim.point.copy(), bridge.env.sim.yaw, bridge.env.sim.pitch
    pano = call(bridge, "pano", "observe", mode="panorama")
    assert len({item["id"] for item in pano["views"]}) == 6
    assert [item["view"]["right_degrees"] for item in pano["views"]] == [0, 60, 120, 180, 240, 300]
    assert all(item["view"]["pitch_degrees"] == 0 for item in pano["views"])
    assert bridge.env.sim.point.tolist() == point.tolist()
    assert (bridge.env.sim.yaw, bridge.env.sim.pitch) == (yaw, pitch)
    fresh = call(bridge, "fresh", "observe")
    assert fresh["observation"]["view"]["pitch_degrees"] == 30
    assert fresh["steps"] == 1


def test_entire_batch_prevalidated_and_id_retries_do_not_move(initialized):
    bridge, _, _ = initialized
    bad = bridge.dispatch({"id": "bad", "op": "step", "args": {"actions": ["move_forward", "teleport"]}})
    assert not bad["ok"] and bridge.env.task.steps == 0
    request = {"id": "step", "op": "step", "args": {"actions": ["move_forward", "move_forward"]}}
    first = bridge.dispatch(request)
    assert first["ok"] and bridge.env.task.steps == 2
    assert bridge.dispatch(request) == first
    changed = bridge.dispatch({**request, "args": {"actions": ["turn_left"]}})
    assert not changed["ok"] and bridge.env.task.steps == 2
    assert bridge.env.sim.position() == [0, 0, -.5]


def test_collision_interrupts_batch_and_records_actual_actions(initialized, monkeypatch):
    bridge, _, root = initialized
    monkeypatch.setattr(bridge.env.sim, "step", lambda action: True)
    result = call(bridge, "collision", "step", actions=["move_forward"] * 6)
    assert result["interruption"] == "collision"
    assert len(result["executed_actions"]) == 1 and result["steps"] == 1
    actions = [json.loads(line) for line in (root / "workspace/observations/actions.jsonl").read_text().splitlines()]
    assert actions[0]["collision"] and actions[0]["before_frame"] != actions[0]["after_frame"]
    assert "position" not in json.dumps(actions)


def test_goat_switch_interrupts_batch_and_stale_finish_is_rejected(tmp_path):
    goals = [{"modality": "description", "instruction": str(i), "positions": [[0, 0, -1]]} for i in range(3)]
    bridge = Bridge()
    call(bridge, "init", "init", **init_args(tmp_path, family="goat", goals=goals,
         task_overrides={"max_subtask_steps": 1}), goal_limit=2)
    result = call(bridge, "step", "step", actions=["move_forward"] * 6)
    assert result["task"]["goal_id"] == 1 and result["goal_changed"]
    assert result["interruption"] == "goal_changed" and len(result["executed_actions"]) == 1
    response = bridge.dispatch({"id": "old", "op": "finish", "args": {"goal_id": 0}})
    assert not response["ok"] and bridge.env.task.steps == 1
    result = call(bridge, "finish", "finish", goal_id=1)
    assert result["terminated"] and result["task"] is None
    assert "success" not in json.dumps(result)
    private = json.loads((tmp_path / "private/episode.json").read_text())
    assert len(private["goals"]) == 2 and "truncated_smoke" in private["provenance"]
    assert len(json.loads((tmp_path / "manifest.json").read_text())["episodes"][0]["goals"]) == 3
    bridge.close()


def test_image_goal_is_published_without_target_camera(tmp_path):
    goals = [{"modality": "image", "instruction": "Find target", "positions": [[0, 0, -1]],
              "image_goal": {"position": [77, 88, 99], "rotation": [0, 0, 0, 1],
                             "hfov": 90, "image_dimensions": [48, 64]}}]
    bridge = Bridge()
    result = call(bridge, "init", "init", **init_args(tmp_path, family="goat", goals=goals))
    assert result["task"]["image"] in result["images"]
    assert (tmp_path / "workspace" / result["task"]["image"]).is_file()
    assert "image_goal" not in json.dumps(result) and "77" not in json.dumps(result)
    bridge.close()


def test_global_budget_ends_batch(initialized):
    bridge, _, _ = initialized
    bridge.env.task.max_steps = 2
    result = call(bridge, "step", "step", actions=["move_forward"] * 6)
    assert result["truncated"] and result["reason"] == "max_steps"
    assert len(result["executed_actions"]) == 2


def test_reject_private_directory_within_workspace(tmp_path):
    args = init_args(tmp_path)
    args["private_dir"] = str(tmp_path / "workspace/private")
    result = Bridge().dispatch({"id": "init", "op": "init", "args": args})
    assert not result["ok"]
    assert not (tmp_path / "workspace").exists()


def atomic_args(tmp_path, mapping=False):
    sample = tmp_path / "sample"
    (sample / "private").mkdir(parents=True)
    (sample / "task.json").write_text(json.dumps({"id": "atomic-fixture", "scene": "fake",
        "form": "interactive_ia", "question": "Cross the doorway"}))
    (sample / "private/labels.json").write_text(json.dumps({
        "start_pose": {"position": [0, 0, 0], "rotation_xyzw": [0, 0, 0, 1]},
        "destination": [0, 1.25], "door_segment": [[-.5, .6], [.5, .6]], "radius": .3}))
    args = init_args(tmp_path)
    args.pop("episode_id")
    args.pop("task_config")
    args["atomic_sample"] = str(sample)
    if mapping:
        args["purpose"] = "mapping"
    return args


def test_atomic_ia_ends_only_on_actual_crossing_and_arrival(tmp_path):
    bridge = Bridge()
    call(bridge, "init", "init", **atomic_args(tmp_path))
    result = call(bridge, "cross", "step", actions=["move_forward"] * 6)
    assert result["local_event_complete"] and result["terminated"]
    assert len(result["executed_actions"]) == 4
    private = json.loads((tmp_path / "private/result.json").read_text())
    assert private["metrics"]["local_event"]["crossed"]
    assert private["metrics"]["local_event"]["arrival"]
    assert "destination" not in json.dumps(result)
    bridge.close()


def test_mapping_seed_does_not_end_on_old_ia_event(tmp_path):
    bridge = Bridge()
    call(bridge, "init", "init", **atomic_args(tmp_path, mapping=True))
    result = call(bridge, "cross", "step", actions=["move_forward"] * 6)
    assert not result["terminated"] and len(result["executed_actions"]) == 6
    assert "local_event_complete" not in result
    assert not bridge.dispatch({"id": "finish", "op": "finish", "args": {"goal_id": 0}})["ok"]
    bridge.close()


def test_mapping_instruction_override_persists_after_observing_and_moving(tmp_path):
    args = atomic_args(tmp_path, mapping=True)
    sample = Path(args["atomic_sample"]) / "task.json"
    original = json.loads(sample.read_text())
    original.update(form="retrieval_qa", question="Choose the opening leading to T on task_map.png")
    sample.write_text(json.dumps(original))
    instruction = "Observe and record one doorway, then revise it using a new viewpoint."
    bridge = Bridge()
    initial = call(bridge, "init", "init", **args, instruction_override=instruction)
    assert initial["task"]["instruction"] == instruction
    call(bridge, "observe", "observe", mode="panorama")
    moved = call(bridge, "step", "step", actions=["turn_right"])
    assert moved["task"]["instruction"] == instruction
    assert "task_map.png" not in json.dumps([initial, moved])
    saved = json.loads((tmp_path / "private/episode.json").read_text())
    assert saved["goals"][0]["instruction"] == instruction
    assert saved["provenance"] == "derived_mapping_smoke;atomic_seed_only"
    assert json.loads(sample.read_text()) == original
    bridge.close()


@pytest.mark.parametrize("mode", ["navigation", "atomic_ia", "mapping_without_seed"])
def test_instruction_override_cannot_change_navigation_protocol(tmp_path, mode):
    args = atomic_args(tmp_path) if mode == "atomic_ia" else init_args(tmp_path)
    if mode == "mapping_without_seed":
        args["purpose"] = "mapping"
    bridge = Bridge()
    result = bridge.dispatch({"id": "init", "op": "init", "args": {
        **args, "instruction_override": "Ignore the real target"}})
    assert not result["ok"] and "only allowed for mapping" in result["error"]
    assert bridge.env is None and not (tmp_path / "workspace").exists()


def test_jsonl_process_has_only_protocol_output_and_closes_cleanly(tmp_path):
    requests = [{"id": "init", "op": "init", "args": init_args(tmp_path)},
                {"id": "frame", "op": "observe", "args": {}},
                {"id": "close", "op": "close", "args": {}}]
    run = subprocess.run([sys.executable, str(ROOT / "src/harness-pi/python/bridge.py")],
                         input="".join(json.dumps(r) + "\n" for r in requests), capture_output=True,
                         text=True, timeout=20)
    assert run.returncode == 0, run.stderr
    responses = [json.loads(line) for line in run.stdout.splitlines()]
    assert [r["id"] for r in responses] == ["init", "frame", "close"]
    assert all(r["ok"] for r in responses)
