"""Real Habitat controls on a CPU scene graph, without a renderer or model."""

import math
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def make_motion():
    habitat = pytest.importorskip("habitat_sim")
    import magnum as mn
    from src.environment.simulator import HabitatSimulator, SensorConfig

    graphs = []

    def make(move_filter=lambda start, end: end, *, distance=.25, yaw=0):
        graph = habitat.SceneGraph()
        graphs.append(graph)
        config = habitat.agent.AgentConfiguration()
        config.sensor_specifications = []
        agent = habitat.agent.Agent(graph.get_root_node().create_child(), config)
        agent.scene_node.rotate_y(mn.Rad(math.radians(yaw)))
        agent.controls.move_filter_fn = move_filter
        simulator = HabitatSimulator.__new__(HabitatSimulator)
        simulator.agent = agent
        simulator.config = SensorConfig(forward_step=distance)
        simulator.sim = SimpleNamespace(step_physics=lambda dt: None,
            pathfinder=SimpleNamespace(try_step_no_sliding=lambda start, end: end))
        return simulator

    return make


@pytest.mark.parametrize("distance,count", [(.25, 5), (.1, 2)])
@pytest.mark.parametrize("yaw", [0, 90])
@pytest.mark.parametrize("action,local", [
    ("move_forward", [0, 0, -1]), ("move_backward", [0, 0, 1]),
    ("move_left", [-1, 0, 0]), ("move_right", [1, 0, 0]),
])
def test_open_space_keeps_requested_direction_and_heading(make_motion, distance, count, yaw, action, local):
    import quaternion

    simulator = make_motion(distance=distance, yaw=yaw)
    before = simulator.agent.get_state()
    motion = simulator.step(action)
    expected = quaternion.as_rotation_matrix(before.rotation) @ np.asarray(local) * distance
    assert simulator.position() == pytest.approx(expected, abs=1e-6)
    assert motion.auto_yaw_delta_degrees == 0
    assert not motion.collision and not motion.blocked
    assert len(motion.samples) == count + 1
    assert motion.samples[-1]["rotation_xyzw"] == pytest.approx(motion.samples[0]["rotation_xyzw"])


@pytest.mark.parametrize("action", ["move_forward", "move_backward", "move_left", "move_right"])
def test_contact_adjustment_is_gradual_and_not_confused_by_action_offset(make_motion, action):
    theta = math.radians(10)

    def deflect(start, end):
        delta = end - start
        return start + np.asarray([math.cos(theta) * delta[0] + math.sin(theta) * delta[2],
                                   delta[1], -math.sin(theta) * delta[0] + math.cos(theta) * delta[2]])

    simulator = make_motion(deflect)
    motion = simulator.step(action)
    # Movement can change direction without reducing length (collision=False).
    assert not motion.collision and not motion.blocked
    assert motion.auto_yaw_delta_degrees == pytest.approx(15)
    turns = [s["auto_yaw_delta_degrees"] for s in motion.samples]
    assert np.diff(turns) == pytest.approx([3] * 5)
    import quaternion
    forward = quaternion.as_rotation_matrix(simulator.agent.get_state().rotation) @ [0, 0, -1]
    assert math.degrees(math.atan2(-forward[0], -forward[2])) == pytest.approx(15, abs=1e-5)


def test_vertical_snap_is_not_horizontal_progress_or_a_turn_signal(make_motion):
    calls = []

    def vertical_only(start, end):
        calls.append(1)
        return start + np.asarray([0, .1, 0])

    motion = make_motion(vertical_only).step("move_forward")
    assert motion.blocked and not motion.collision
    assert len(calls) == 1 and len(motion.samples) == 2
    assert motion.auto_yaw_delta_degrees == 0


def test_hitting_wall_at_last_microstep_returns_blocked_after_partial_progress(make_motion):
    calls = []

    def wall(start, end):
        calls.append(1)
        return start if len(calls) == 5 else end

    simulator = make_motion(wall)
    motion = simulator.step("move_forward")
    assert simulator.position() == pytest.approx([0, 0, -.2])
    assert motion.collision and motion.blocked
    assert len(motion.samples) == 6


def test_low_total_progress_also_returns_blocked(make_motion):
    motion = make_motion(lambda start, end: start + .04 * (end - start)).step("move_forward")
    assert len(motion.samples) == 6
    assert motion.positions[-1] == pytest.approx([0, 0, -.01])
    assert motion.blocked and motion.auto_yaw_delta_degrees == 0


def test_body_turn_preserves_attached_camera_pitch(make_motion):
    import magnum as mn

    simulator = make_motion(lambda start, end: start + np.asarray([.02, 0, -.04]))
    camera = simulator.agent.scene_node.create_child()
    camera.rotate_x(mn.Rad(math.radians(30)))
    local_rotation = np.asarray(camera.rotation.to_matrix()).copy()
    simulator.step("move_forward")
    assert np.asarray(camera.rotation.to_matrix()) == pytest.approx(local_rotation)
    assert float(simulator.agent.scene_node.rotation.angle()) != 0


def test_corner_cut_is_subdivided_without_discarding_the_requested_distance(make_motion):
    # Two connected half-planes form an L: x<=0 or z>=0. The long
    # projected segment has valid endpoints but crosses the excluded corner.
    simulator = make_motion()
    start = np.asarray([-.02, 0, -.02])

    def project(a, b):
        return np.asarray([.02, 0, .01]) if np.linalg.norm(b - a) > .02 else b

    def straight(a, b):
        # Exact entry/exit interval for the excluded quadrant x>0,z<0.
        low, high = 0., 1.
        for origin, delta in [(a[0], b[0] - a[0]), (-a[2], a[2] - b[2])]:
            if abs(delta) < 1e-12:
                if origin <= 0:
                    return b
            elif delta > 0:
                low = max(low, -origin / delta)
            else:
                high = min(high, -origin / delta)
        return a if low < high and high > 0 and low < 1 else b

    simulator.agent.controls.move_filter_fn = project
    simulator.sim.pathfinder.try_step_no_sliding = straight
    pending = []
    amount, projected = simulator._slide_projection(start, np.asarray([1, 0, 0]), .05, pending)
    assert amount < .02
    assert projected == pytest.approx(start + [amount, 0, 0])
    assert amount + sum(pending) == pytest.approx(.05)
