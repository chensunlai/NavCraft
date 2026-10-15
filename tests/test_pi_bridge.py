"""Public/private boundaries and exact-once movement in the Pi environment bridge."""

import json
from pathlib import Path

import pytest

from src.harness import session as MODULE
from src.harness.session import NavigationSession as Bridge

ROOT = Path(__file__).resolve().parents[1]


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


def test_initial_heading_hint_near_map_edge_stays_renderable(tmp_path, monkeypatch):
    import src.common.map

    class Calibration:
        def __init__(self, _):
            pass

        def initial_arrow(self, pose):
            return "L1", [.99, .5], [1.05, .5]

    monkeypatch.setattr(src.common.map, "MapCalibration", Calibration)
    bridge = Bridge()
    result = call(bridge, "init", "init", **init_args(tmp_path), scene_archive="fixture-scene")
    hint = result["initial_hint"]
    assert hint["position"] == [.99, .5]
    assert .99 < hint["heading_point"][0] <= 1
    assert hint["heading_point"][1] == .5
    bridge.close()


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


@pytest.mark.parametrize("action", ["move_forward", "move_left", "move_right", "move_backward"])
def test_entire_batch_prevalidated_and_id_retries_do_not_move(initialized, action):
    bridge, _, _ = initialized
    bad = bridge.dispatch({"id": "bad", "op": "step", "args": {"actions": [action, "teleport"]}})
    assert not bad["ok"] and bridge.env.task.steps == 0
    assert bridge.execution_uncertain is False
    request = {"id": "step", "op": "step", "args": {"actions": ["move_forward", "move_forward"]}}
    first = bridge.dispatch(request)
    assert first["ok"] and bridge.env.task.steps == 2
    assert bridge.execution_uncertain is False
    assert bridge.dispatch(request) == first
    changed = bridge.dispatch({**request, "args": {"actions": ["turn_left"]}})
    assert not changed["ok"] and bridge.env.task.steps == 2
    assert bridge.env.sim.position() == [0, 0, -.5]


def test_translations_are_exposed_logged_counted_and_not_repeated(initialized):
    bridge, initial, root = initialized
    actions = ["move_left", "move_right", "move_backward"]
    assert set(actions) <= set(initial["actions"]["allowed"])
    request = {"id": "translations", "op": "step", "args": {"actions": actions}}
    response = bridge.dispatch(request)
    assert response["ok"]
    result = response["data"]
    assert [event["action"] for event in result["executed_actions"]] == actions
    assert result["steps"] == 3
    assert bridge.env.sim.position() == pytest.approx([0, 0, .25])
    assert bridge.env.task.path_length == pytest.approx(.75)
    assert bridge.dispatch(request) == response
    assert bridge.env.task.steps == 3
    records = [json.loads(line) for line in
               (root / "workspace/observations/actions.jsonl").read_text().splitlines()]
    assert [record["action"] for record in records] == actions
    assert all(record["before_frame"] != record["after_frame"] for record in records)
    assert "position" not in json.dumps(records)


@pytest.mark.parametrize("failure_point", ["_save_private_frame", "_publish", "_save_result"])
def test_action_receipt_failure_marks_execution_uncertain_without_retry(initialized, monkeypatch, failure_point):
    bridge, _, _ = initialized
    request = {"id": "failed-movement", "op": "step", "args": {"actions": ["move_forward"]}}

    def fail_receipt(*_, **__):
        raise OSError("movement receipt unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(bridge, failure_point, fail_receipt)
        failed = bridge.dispatch(request)
        assert not failed["ok"] and "receipt unavailable" in failed["error"]
        assert bridge.env.sim.position() == [0, 0, -.25]
        assert bridge.env.task.steps == 1
        assert bridge.execution_uncertain is True
        assert bridge.dispatch(request) == failed
        assert bridge.env.task.steps == 1
        assert "execution_uncertain" not in json.dumps(failed)
    # Removing the fault does not turn an incomplete receipt into success.
    assert bridge.execution_uncertain is True


def test_finish_precheck_and_failed_publication_keep_correct_uncertainty(initialized, monkeypatch):
    bridge, _, _ = initialized
    invalid = bridge.dispatch({"id": "invalid-finish", "op": "finish", "args": {"goal_id": 99}})
    assert not invalid["ok"] and bridge.env.task.steps == 0
    assert bridge.execution_uncertain is False

    def fail_receipt(*_, **__):
        raise OSError("completion receipt unavailable")

    request = {"id": "failed-finish", "op": "finish", "args": {"goal_id": 0}}
    with monkeypatch.context() as patch:
        patch.setattr(bridge, "_publish", fail_receipt)
        failed = bridge.dispatch(request)
        assert not failed["ok"] and "receipt unavailable" in failed["error"]
        assert bridge.env.task.done
        assert bridge.execution_uncertain is True
        assert bridge.dispatch(request) == failed
        assert "execution_uncertain" not in json.dumps(failed)


@pytest.mark.parametrize("action", ["move_forward", "move_left", "move_right", "move_backward"])
def test_collision_interrupts_batch_and_records_actual_actions(initialized, monkeypatch, action):
    bridge, _, root = initialized
    monkeypatch.setattr(bridge.env.sim, "step", lambda action: True)
    result = call(bridge, "collision", "step", actions=[action] * 6)
    assert result["interruption"] == "collision"
    assert len(result["executed_actions"]) == 1 and result["steps"] == 1
    actions = [json.loads(line) for line in (root / "workspace/observations/actions.jsonl").read_text().splitlines()]
    assert actions[0]["collision"] and actions[0]["before_frame"] != actions[0]["after_frame"]
    assert "position" not in json.dumps(actions)
    assert actions[0]["action"] == action
    assert bridge.env.task.path_length == 0


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


@pytest.mark.parametrize("mapping", [False, True])
@pytest.mark.parametrize("action", ["move_left", "move_right", "move_backward"])
def test_atomic_translation_respects_four_metre_budget(tmp_path, mapping, action):
    args = atomic_args(tmp_path, mapping=mapping)
    args["env_config"]["forward_step"] = .3
    bridge = Bridge()
    try:
        call(bridge, "init", "init", **args)
        call(bridge, "first", "step", actions=[action] * 6)
        call(bridge, "second", "step", actions=[action] * 6)
        result = call(bridge, "third", "step", actions=[action] * 6)
        assert result["truncated"] and result["interruption"] == "travel_budget"
        assert len(result["executed_actions"]) == 1
        assert bridge.env.task.steps == 13
        assert bridge.env.task.path_length == pytest.approx(3.9)
    finally:
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


def test_direct_dispatch_pairs_responses_and_closes_cleanly(tmp_path):
    requests = [{"id": "init", "op": "init", "args": init_args(tmp_path)},
                {"id": "frame", "op": "observe", "args": {}},
                {"id": "close", "op": "close", "args": {}}]
    session = Bridge()
    try:
        responses = [session.dispatch(request) for request in requests]
    finally:
        session.close()
    assert [r["id"] for r in responses] == ["init", "frame", "close"]
    assert all(r["ok"] for r in responses)
    assert session.closed


def test_close_is_idempotent_without_rewriting_result(initialized, monkeypatch):
    bridge, _, root = initialized
    calls = {"save": 0, "close": 0}
    save, close = bridge._save_result, bridge.env.close

    def save_once():
        calls["save"] += 1
        save()

    def close_once():
        calls["close"] += 1
        close()

    monkeypatch.setattr(bridge, "_save_result", save_once)
    monkeypatch.setattr(bridge.env, "close", close_once)
    assert bridge.close() == {"closed": True}
    original = (root / "private/result.json").read_bytes()
    assert bridge.close() == {"closed": True}
    assert calls == {"save": 1, "close": 1}
    assert (root / "private/result.json").read_bytes() == original
    assert json.loads(original)["truncated"] is True


def test_atomic_result_replace_failure_preserves_old_bytes(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    MODULE._write(path, {"status": "earlier", "steps": 7})
    original = path.read_bytes()

    def fail_replace(source, destination):
        assert Path(source).parent == path.parent
        assert json.loads(Path(source).read_text())["steps"] == 8
        raise OSError("simulated interrupted publication")

    monkeypatch.setattr(MODULE.os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted publication"):
        MODULE._write(path, {"status": "new", "steps": 8})
    assert path.read_bytes() == original
    assert json.loads(path.read_text())["steps"] == 7
    assert not list(tmp_path.glob(".result.json.*.tmp"))


def test_close_still_releases_environment_if_result_write_fails(initialized, monkeypatch):
    bridge, _, root = initialized
    simulator = bridge.env.sim
    original = (root / "private/result.json").read_bytes()

    def fail_save():
        raise OSError("result unavailable")

    monkeypatch.setattr(bridge, "_save_result", fail_save)
    with pytest.raises(OSError, match="result unavailable"):
        bridge.close()
    assert simulator.closed and bridge.closed
    assert bridge.close() == {"closed": True}
    assert (root / "private/result.json").read_bytes() == original
