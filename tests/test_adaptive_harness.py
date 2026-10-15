"""Adaptive movement stays one action, with private paths and public relative feedback."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.bench.harness_verify import _local_replay_event, _motion_replay
from src.harness.session import NavigationSession


ROOT = Path(__file__).resolve().parents[1]


def open_bridge(tmp_path, *, labels=None, family="objectnav", task_options=None):
    config = {"name": "fake", "depth": False, "forward_step": .25,
              "turn_angle": 30}
    args = {"repo": str(ROOT), "workspace": str(tmp_path / "workspace"),
            "private_dir": str(tmp_path / "private"), "env_config": config}
    if labels is not None:
        sample = tmp_path / "sample"
        (sample / "private").mkdir(parents=True)
        labels = {"start_pose": {"position": [0, 0, 0], "rotation_xyzw": [0, 0, 0, 1]}, **labels}
        (sample / "task.json").write_text(json.dumps({"id": "atomic-path", "scene": "fake",
            "form": "interactive_ia", "question": "Cross the designated opening"}))
        (sample / "private/labels.json").write_text(json.dumps(labels))
        args["atomic_sample"] = str(sample)
    else:
        goals = [{"modality": "object", "instruction": "Find chair", "positions": [[0, 0, -2]]}]
        if family == "goat":
            goals.append({"modality": "object", "instruction": "Find bed", "positions": [[0, 0, -3]]})
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": 1, "validated": True,
            "episodes": [{"id": "fake:adaptive", "source_id": "adaptive",
            "benchmark": "goat" if family == "goat" else "objnav", "scene_id": "fake.glb",
            "start_position": [0, 0, 0], "start_rotation": [0, 0, 0, 1],
            "source_file": "fixture", "goals": goals}]}))
        args.update(episode_id="fake:adaptive", task_config={
            "episodes": {"name": "manifest", "root": str(tmp_path), "manifest": str(manifest)},
            "task": {"name": family, **(task_options or {})}})
    bridge = NavigationSession()
    initial = bridge.init(**args)
    return bridge, initial


def scripted_motion(monkeypatch, bridge, motions):
    from src.environment.base import MotionResult

    pending = iter(motions)

    def step(action):
        motion = next(pending)
        path = motion["path"]
        assert path[0] == pytest.approx(bridge.env.sim.position())
        bridge.env.sim.point = np.asarray(path[-1], dtype=np.float64)
        yaw = motion.get("yaw", 0.0)
        bridge.env.sim.yaw += math.radians(yaw)
        rotation = [0, math.sin(bridge.env.sim.yaw / 2), 0, math.cos(bridge.env.sim.yaw / 2)]
        return MotionResult(collision=motion.get("collision", False),
            blocked=motion.get("blocked", False), auto_yaw_delta_degrees=yaw,
            samples=[{"position": point, "rotation_xyzw": rotation,
                      "collision": motion.get("collision", False),
                      "auto_yaw_delta_degrees": yaw} for point in path])

    monkeypatch.setattr(bridge.env.sim, "step", step)


def test_progressing_collision_continues_but_blocked_interrupts(tmp_path, monkeypatch):
    bridge, initial = open_bridge(tmp_path)
    try:
        assert initial["actions"]["max_auto_yaw_degrees"] == 15
        scripted_motion(monkeypatch, bridge, [
            {"path": [[0, 0, 0], [.02, 0, -.05], [.04, 0, -.1]], "collision": True, "yaw": 3},
            {"path": [[.04, 0, -.1], [.05, 0, -.2]], "collision": True, "yaw": -2},
            {"path": [[.05, 0, -.2], [.05, 0, -.2]], "collision": True, "blocked": True}])
        result = bridge.step(["move_forward"] * 6)
        assert result["interruption"] == "blocked"
        assert len(result["executed_actions"]) == 3 and result["steps"] == 3
        assert [row["auto_yaw_delta_degrees"] for row in result["executed_actions"]] == [3, -2, 0]
        assert not result["executed_actions"][0]["blocked"]
        assert result["executed_actions"][-1]["blocked"]
        text = json.dumps(result)
        assert all(private not in text for private in ("position", "rotation_xyzw", "samples", "motion_trajectory"))
        private = json.loads((tmp_path / "private/result.json").read_text())
        assert len(private["trajectory"]) == 4
        assert len(private["motion_trajectory"]) == 5
        assert len((tmp_path / "workspace/observations/index.jsonl").read_text().splitlines()) == 4
        assert _motion_replay(private)[2]
    finally:
        bridge.close()


def test_progressing_collisions_finish_batch_without_motion_switches(tmp_path, monkeypatch):
    bridge, initial = open_bridge(tmp_path)
    try:
        assert initial["actions"]["max_auto_yaw_degrees"] == 15
        assert "adaptive_motion" not in initial["actions"]
        scripted_motion(monkeypatch, bridge, [
            {"path": [[0, 0, -i / 10], [0, 0, -(i + 1) / 10]], "collision": True}
            for i in range(3)])
        result = bridge.step(["move_forward"] * 3)
        assert result["interruption"] is None and result["steps"] == 3
    finally:
        bridge.close()


@pytest.mark.parametrize("family, task_options, reason", [
    ("objectnav", {"max_steps": 1}, "max_steps"),
    ("goat", {"max_subtask_steps": 1}, "goal_changed")])
def test_task_boundaries_precede_blocked(tmp_path, monkeypatch, family, task_options, reason):
    bridge, _ = open_bridge(tmp_path, family=family, task_options=task_options)
    try:
        scripted_motion(monkeypatch, bridge, [
            {"path": [[0, 0, 0], [0, 0, 0]], "collision": True, "blocked": True}])
        result = bridge.step(["move_forward"] * 6)
        assert result["interruption"] == reason
        assert len(result["executed_actions"]) == 1
    finally:
        bridge.close()


def test_ia_checks_actual_curve_and_counts_atomic_actions(tmp_path, monkeypatch):
    labels = {"door_segment": [[-.1, .1], [.1, .1]], "destination": [0, .2], "radius": .03}
    bridge, _ = open_bridge(tmp_path, labels=labels)
    try:
        scripted_motion(monkeypatch, bridge, [
            {"path": [[0, 0, 0], [.2, 0, 0], [.2, 0, -.2], [0, 0, -.2]]},
            {"path": [[0, 0, -.2], [0, 0, 0], [0, 0, -.2]]}])
        first = bridge.step(["move_forward"])
        assert not first["local_event_complete"]
        assert not bridge.env.task.event()["crossed"]
        result = bridge.step(["move_forward"])
        assert result["local_event_complete"]
        event = bridge.env.task.event()
        assert event["actions"] == 2 and event["travel_metres"] == pytest.approx(1.0)
    finally:
        bridge.close()


def test_ia_microsteps_do_not_consume_action_budget(tmp_path, monkeypatch):
    labels = {"door_segment": [[-.1, .1], [.1, .1]], "destination": [0, .28], "radius": .01}
    bridge, _ = open_bridge(tmp_path, labels=labels)
    try:
        scripted_motion(monkeypatch, bridge, [{"path": [[0, 0, -i / 100] for i in range(29)]}])
        result = bridge.step(["move_forward"])
        assert result["local_event_complete"] and result["steps"] == 1
        assert bridge.env.task.event()["actions"] == 1
    finally:
        bridge.close()


def test_true_curve_over_four_metres_cannot_complete_ia(tmp_path, monkeypatch):
    labels = {"door_segment": [[-.1, .1], [.1, .1]], "destination": [0, .2], "radius": .03}
    bridge, _ = open_bridge(tmp_path, labels=labels)
    try:
        scripted_motion(monkeypatch, bridge, [{"path": [[0, 0, 0], [0, 0, -2.2], [0, 0, -.2]]}])
        result = bridge.step(["move_forward"])
        assert result["truncated"] and result["interruption"] == "travel_budget"
        assert not result["local_event_complete"]
        assert bridge.env.task.path_length == pytest.approx(4.2)
    finally:
        bridge.close()


def test_private_replay_rejects_disconnected_action_endpoints():
    state = {"trajectory": [[0, 0, 0], [0, 0, -.2]],
             "motion_trajectory": [[0, 0, 0], [.1, 0, -.1], [0, 0, -.2]],
             "motion_spans": [[0, 2]], "metrics": {"steps": 1}}
    assert _motion_replay(state)[2]
    state["motion_spans"] = [[0, 1]]
    assert not _motion_replay(state)[2]


def test_private_event_uses_micro_path_and_atomic_count():
    labels = {"door_segment": [[-.1, .1], [.1, .1]], "destination": [0, .28], "radius": .01}
    state = {"trajectory": [[0, 0, 0], [0, 0, -.28]],
             "motion_trajectory": [[0, 0, -i / 100] for i in range(29)],
             "metrics": {"steps": 1}}
    assert _local_replay_event(labels, state)["passed"]
    state.pop("motion_trajectory")
    assert _local_replay_event(labels, state)["passed"]


@pytest.mark.parametrize("kind", ["navigation", "explore"])
def test_sequence_scores_the_real_curve_instead_of_endpoint_chord(tmp_path, kind):
    from types import SimpleNamespace
    from src.dataset.episodes.base import EpisodeSpec
    from src.environment.tasks.base import ObjectNavTask
    from src.harness.sequence import SequenceSession

    start, end = [.5, 0, -.2], [.5, 0, .2]
    curve = [start, [1.5, 0, -.2], [1.5, 0, .2], end]
    task = ObjectNavTask({})
    episode = EpisodeSpec("curve", "curve", "campaign", "fake.glb", start, [0, 0, 0, 1],
                          [{"modality": "instruction", "instruction": "Cross door", "positions": [end]}], "fixture")
    task.reset(episode, start, .4)
    task.on_step(end, 0, motion_path=curve)
    task.stopped, task.terminated = True, True
    spec = {"kind": kind, "scope_room": "B", "route_events": [
        {"id": "door", "kind": "portal", "segment_xz": [[0, 0], [1, 0]], "to_room": "B"}]}
    polygon = ([[0, -.3], [1, -.3], [1, .3], [0, .3]] if kind == "explore"
               else [[0, 0], [1, 0], [1, -1], [0, -1]])
    scenario = {"environment": {}, "rounds": [spec], "private_rooms": [{"id": "B", "polygon": polygon}]}
    bridge = SequenceSession(ROOT, tmp_path / "workspace", tmp_path / "private", scenario)
    bridge.round_index = 0
    bridge.env = SimpleNamespace(task=task)
    result = bridge.score("stop")
    if kind == "navigation":
        assert result["independent"]["arrival"]
        assert not result["independent"]["route"]["passed"]
    else:
        assert not result["exploration"]["scope_maintained"]
