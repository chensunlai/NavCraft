"""Task progression, public/private boundaries and evaluator protocol contracts."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.dataset.episodes.base import EpisodeSpec
from src.environment.base import NavigationInputError
from src.environment.fake import FakeEnvironment, FakeSimulator
from src.environment.habitat import HabitatEnvironment
from src.environment.tasks.base import GOATTask, ObjectNavTask, VLNCETask
from src.environment.tasks.metrics import ndtw, spl


def episode(goals=None, benchmark="objnav", **kwargs):
    return EpisodeSpec(
        id="fixture:1", source_id="1", benchmark=benchmark, scene_id="fake.glb",
        start_position=[0, 0, 0], start_rotation=[0, 0, 0, 1], source_file="fixture",
        goals=goals or [{"modality": "object", "instruction": "Find chair", "positions": [[0, 0, -0.25]]}],
        **kwargs,
    )


def test_stop_is_counted_and_strict_objectnav_score(tmp_path):
    env = FakeEnvironment({}, ObjectNavTask({}), tmp_path)
    initial = env.reset(episode())
    assert initial.observation.pose["position"] == [0, 0, 0]
    env.step("move_forward")
    transition = env.finish_goal()
    assert transition.terminated and not transition.truncated
    assert transition.feedback["steps"] == 2
    assert env.metrics()["success"] == 1
    assert env.metrics()["spl"] == 1
    assert env.metrics()["success_distance"] == 0.1
    with pytest.raises(NavigationInputError):
        env.step("move_forward")
    env.close()


def test_objectnav_threshold_is_not_vlnce_threshold(tmp_path):
    task = ObjectNavTask({})
    env = FakeEnvironment({}, task, tmp_path)
    env.reset(episode(goals=[{"modality": "object", "instruction": "chair", "positions": [[0, 0, -1]]}]))
    env.finish_goal()
    assert env.metrics()["success"] == 0


def test_public_transition_never_contains_annotation_positions_or_metrics(tmp_path):
    env = FakeEnvironment({}, ObjectNavTask({}), tmp_path)
    transition = env.reset(episode(reference_path=[[99, 99, 99]], gt_path=[[98, 98, 98]]))
    value = transition.public_dict()
    assert set(value["task"]) == {"goal_id", "modality", "instruction", "image"}
    assert "positions" not in json.dumps(value)
    assert "success" not in json.dumps(value)
    assert "reference_path" not in json.dumps(value)
    assert "distance" not in json.dumps(value)
    assert Path(value["observation"]["rgb"]).exists()
    assert np.load(transition.observation.depth).dtype == np.float32


def test_goat_three_modalities_advance_at_same_pose(tmp_path):
    goals = [
        {"modality": "object", "instruction": "chair", "positions": [[0, 0, 0]]},
        {"modality": "description", "instruction": "red chair", "positions": [[0, 0, 0]]},
        {"modality": "image", "instruction": "Match target image", "positions": [[0, 0, 0]],
         "image_goal": {"position": [77, 88, 99], "rotation": [0, 0, 0, 1], "hfov": 90, "image_dimensions": [48, 64]}},
    ]
    env = FakeEnvironment({}, GOATTask({}), tmp_path)
    env.reset(episode(goals, "goat"))
    transition = env.finish_goal()
    assert transition.goal_changed and transition.task.modality == "description"
    transition = env.finish_goal()
    assert transition.task.modality == "image" and transition.task.image.exists()
    assert transition.observation.pose["position"] == [0, 0, 0]
    assert "image_goal" not in json.dumps(transition.public_dict())
    assert env.sim.position() == [0, 0, 0]
    transition = env.finish_goal()
    assert transition.terminated and transition.task is None
    scores = env.metrics()
    assert scores["partial_success"] == 1
    assert scores["composite_success"] == 1
    assert scores["steps"] == 3


def test_goat_subtask_limit_switches_without_reset(tmp_path):
    goals = [{"modality": "object", "instruction": "chair", "positions": [[0, 0, -2]]}] * 2
    env = FakeEnvironment({}, GOATTask({"max_subtask_steps": 1}), tmp_path)
    env.reset(episode(goals, "goat"))
    transition = env.step("move_forward")
    assert transition.goal_changed and transition.task.goal_id == 1
    assert transition.observation.pose["position"] == [0, 0, -0.25]
    assert env.task.goal_steps == 0
    assert env.task.shortest == 1.75
    assert env.metrics()["subtask_success"] == [0, 0]
    env.finish_goal()
    assert env.metrics()["subtask_success"] == [0, 1]


def test_global_budget_truncates_without_advancing(tmp_path):
    goals = [{"modality": "object", "instruction": "chair", "positions": [[0, 0, -2]]}] * 3
    env = FakeEnvironment({}, GOATTask({"max_steps": 1, "max_subtask_steps": 1}), tmp_path)
    env.reset(episode(goals, "goat"))
    transition = env.step("move_forward")
    assert transition.truncated and not transition.terminated
    assert transition.reason == "max_steps" and not transition.goal_changed
    assert env.metrics()["subtask_success"] == [0, 0, 0]
    private = json.loads((tmp_path / "private/state.json").read_text())
    assert len(private["subtasks"]) == 3
    assert all(goal["reason"] == "max_steps" for goal in private["subtasks"])


def test_finalize_preserves_failed_goals_and_closes_simulator(tmp_path):
    env = FakeEnvironment({}, GOATTask({}), tmp_path)
    env.reset(episode([{ "modality": "object", "instruction": "chair", "positions": [[0, 0, 0]]}] * 2, "goat"))
    simulator = env.sim
    env.finalize("model_error")
    assert env.metrics()["partial_success"] == 0
    private = json.loads((tmp_path / "private/state.json").read_text())
    assert private["truncated"]
    assert [goal["reason"] for goal in private["subtasks"]] == ["model_error", "model_error"]
    env.close()
    assert simulator.closed
    env.close()  # Idempotent.


def test_r2r_rejects_pitch_without_action_side_effect(tmp_path):
    env = FakeEnvironment({}, VLNCETask({}), tmp_path)
    env.reset(episode(benchmark="r2r"))
    with pytest.raises(NavigationInputError):
        env.step("look_up")
    assert env.task.steps == 0
    assert env.sim.pitch == 0


def test_sparse_waypoints_never_substitute_missing_official_dense_path(tmp_path):
    env = FakeEnvironment({}, VLNCETask({}), tmp_path)
    env.reset(episode(benchmark="r2r", reference_path=[[0, 0, 0], [0, 0, -0.25]]))
    env.step("move_forward")
    env.finish_goal()
    assert env.metrics()["ndtw"] is None
    assert env.metrics()["ndtw_reference"] == "unavailable"
    assert ndtw([[0, 0, 0], [0, 0, -0.25]], [[0, 0, 0], [0, 0, -0.25]], 3) == 1
    assert spl(True, 0, 0) == 1


def test_reset_capture_failure_closes_native_boundary(tmp_path):
    simulators = []

    class BadCapture(FakeSimulator):
        def __init__(self, episode, config):
            super().__init__(episode, config)
            simulators.append(self)

        def capture(self, workspace, frame):
            raise RuntimeError("renderer failed")

    env = HabitatEnvironment({}, ObjectNavTask({}), tmp_path, simulator_factory=BadCapture)
    with pytest.raises(RuntimeError, match="renderer failed"):
        env.reset(episode())
    assert simulators[0].closed
    assert env.sim is None


def test_unreachable_distance_is_serialized_without_nonfinite_json(tmp_path):
    env = FakeEnvironment({}, ObjectNavTask({}), tmp_path)
    spec = episode()
    spec.goals[0]["positions"] = []
    env.reset(spec)
    env.finish_goal()
    private = json.loads((tmp_path / "private/state.json").read_text())
    assert private["subtasks"][0]["distance"] is None
    assert env.metrics()["success"] == 0
