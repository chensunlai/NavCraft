"""Panoramas are observations, never task actions or persistent pose changes."""

from copy import deepcopy
import math

import numpy as np
from PIL import Image
import pytest

from src.dataset.episodes.base import EpisodeSpec
from src.environment.base import NavigationInputError, Observation
from src.environment.fake import FakeEnvironment, FakeSimulator
from src.environment.tasks.base import GOATTask, ObjectNavTask


def episode(benchmark="objnav", goals=None):
    return EpisodeSpec(
        id="pano:1",
        source_id="1",
        benchmark=benchmark,
        scene_id="fake.glb",
        start_position=[0, 0, 0],
        start_rotation=[0, 0, 0, 1],
        source_file="fixture",
        goals=goals
        or [{"modality": "object", "instruction": "chair", "positions": [[0, 0, -1]]}],
    )


def test_panorama_restores_pitch_and_does_not_touch_task_or_cached_view(tmp_path):
    env = FakeEnvironment({"depth": False}, ObjectNavTask({}), tmp_path)
    env.reset(episode())
    env.step("look_up")
    cached = env.observe()
    point, yaw, pitch = env.sim.point.copy(), env.sim.yaw, env.sim.pitch
    private = deepcopy(env.task.private_state())
    metrics = deepcopy(env.metrics())

    views = env.panorama()

    assert len(views) == 6 and len({view.frame_id for view in views}) == 6
    assert env.observe() is cached
    assert env.task.private_state() == private and env.metrics() == metrics
    np.testing.assert_array_equal(env.sim.point, point)
    assert env.sim.yaw == yaw and env.sim.pitch == pitch
    assert all(view.depth is None for view in views)
    red_values = [np.asarray(Image.open(view.rgb))[0, 0, 0] for view in views]
    assert len(set(red_values)) == 6
    for index, view in enumerate(views):
        forward = np.asarray(view.pose["world_from_optical"])[:3, 2]
        angle = yaw - index * math.pi / 3
        np.testing.assert_allclose(
            forward, [-math.sin(angle), 0, -math.cos(angle)], atol=1e-12
        )
    env.close()


def test_panorama_action_and_stop_captures_have_distinct_ids(tmp_path):
    goal = {"modality": "object", "instruction": "chair", "positions": [[0, 0, 0]]}
    env = FakeEnvironment({}, GOATTask({}), tmp_path)
    initial = env.reset(episode("goat", [goal, goal]))
    first_views = env.panorama()
    move = env.step("turn_right")
    second_views = env.panorama()
    next_goal = env.finish_goal()
    end = env.finish_goal()
    observations = [
        initial.observation,
        *first_views,
        move.observation,
        *second_views,
        next_goal.observation,
        end.observation,
    ]
    assert [view.frame_id for view in observations] == [
        f"frame_{index:06d}" for index in range(16)
    ]
    assert len({view.rgb for view in observations}) == 16
    assert next_goal.goal_changed and end.terminated
    assert env.task.steps == 3
    with pytest.raises(NavigationInputError, match="finished"):
        env.panorama()
    env.close()


def test_panorama_rejects_uninitialized_environment(tmp_path):
    env = FakeEnvironment({}, ObjectNavTask({}), tmp_path)
    with pytest.raises(RuntimeError, match="reset"):
        env.panorama()


def test_fake_panorama_restores_state_after_capture_failure(tmp_path):
    class FailingSimulator(FakeSimulator):
        def capture(self, workspace, frame):
            if frame == 3:
                self.point[:] = 99
                raise RuntimeError("capture failed")
            return super().capture(workspace, frame)

    sim = FailingSimulator(episode(), {})
    sim.yaw, sim.pitch = 0.6, 0.4
    point = sim.point.copy()
    with pytest.raises(RuntimeError, match="capture failed"):
        sim.panorama(tmp_path, list(range(6)))
    np.testing.assert_array_equal(sim.point, point)
    assert sim.yaw == 0.6 and sim.pitch == 0.4


@pytest.mark.parametrize("fail", [False, True])
def test_native_panorama_restores_all_sensor_extrinsics_without_renderer(
    tmp_path, fail
):
    habitat_sim = pytest.importorskip("habitat_sim")
    quaternion = pytest.importorskip("quaternion")
    from src.environment.simulator import HabitatSimulator

    yaw = 0.6
    original = habitat_sim.AgentState()
    original.position = np.array([1.0, 2.0, 3.0])
    original.rotation = quaternion.from_rotation_vector([0, yaw, 0])
    pitched = original.rotation * quaternion.from_rotation_vector([0.4, 0, 0])
    original.sensor_states = {
        "rgb": habitat_sim.SixDOFPose(
            position=np.array([1.2, 3.5, 3.1]), rotation=pitched
        ),
        "depth": habitat_sim.SixDOFPose(
            position=np.array([1.3, 3.4, 3.2]), rotation=pitched
        ),
    }

    class Agent:
        def __init__(self):
            self.state = deepcopy(original)

        def get_state(self):
            return deepcopy(self.state)

        def set_state(self, state, *, reset_sensors, infer_sensor_states):
            assert reset_sensors is False and infer_sensor_states is False
            self.state = deepcopy(state)

    sim = HabitatSimulator.__new__(HabitatSimulator)
    sim.agent = Agent()
    states = []

    def capture(workspace, frame):
        states.append(sim.agent.get_state())
        if fail and frame == 12:
            raise RuntimeError("renderer failed")
        return Observation(
            f"frame_{frame:06d}", workspace / f"{frame}.png", None, {}, {}
        )

    sim.capture = capture
    if fail:
        with pytest.raises(RuntimeError, match="renderer failed"):
            sim.panorama(tmp_path, list(range(10, 16)))
    else:
        assert len(sim.panorama(tmp_path, list(range(10, 16)))) == 6

    for index, state in enumerate(states):
        np.testing.assert_array_equal(state.position, original.position)
        assert state.rotation == original.rotation
        for name, sensor in state.sensor_states.items():
            np.testing.assert_array_equal(
                sensor.position, original.sensor_states[name].position
            )
            matrix = quaternion.as_rotation_matrix(sensor.rotation)
            angle = yaw - index * math.pi / 3
            np.testing.assert_allclose(
                matrix @ [0, 0, -1], [-math.sin(angle), 0, -math.cos(angle)], atol=1e-12
            )
    restored = sim.agent.get_state()
    np.testing.assert_array_equal(restored.position, original.position)
    assert restored.rotation == original.rotation
    for name, sensor in restored.sensor_states.items():
        np.testing.assert_array_equal(
            sensor.position, original.sensor_states[name].position
        )
        assert sensor.rotation == original.sensor_states[name].rotation
