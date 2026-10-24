import gzip
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
from PIL import Image

from src.bench.metrics import score_episode, score_goal
from src.bench.runner import execute_goals, validate_single_floor
from src.bench.samples import plan, read_config
from src.dataset.episodes import EpisodeSpec, create_source
from src.environment import Environment, MotionResult
from src.environment.fake import FakeSimulator
from src.environment.base import save_depth


def episode(benchmark="objnav", goals=None):
    return EpisodeSpec("fixture", "0", benchmark, "fake.glb", [0, 0, 0], [0, 0, 0, 1],
                       goals or [{"modality": "object", "positions": [[0, 0, -1]], "instruction": "Find a chair."}], "fixture")


def environment(tmp_path, specification=None, factory=FakeSimulator):
    return Environment(specification or episode(), {"gpu": 0}, tmp_path, factory)


def test_gpu_boundary_and_unified_embodiment(tmp_path):
    called = []
    for gpu in (None, 2, 3, -1, True, "0"):
        with pytest.raises(ValueError, match="0 or 1"):
            Environment(episode(), {"gpu": gpu}, tmp_path, lambda **kw: called.append(kw))
    assert not called
    env = Environment(episode(), {"gpu": 1, "width": 512, "turn_angle": 15}, tmp_path, FakeSimulator)
    assert env.config["width"] == 640
    assert env.config["turn_angle"] == 30
    assert not env.config["depth"]
    env.close()
    env.close()


def test_fresh_frames_panorama_and_motion_accounting(tmp_path):
    env = environment(tmp_path)
    first, second = env.observe(), env.observe()
    assert first.frame_id == "f_0001" and second.frame_id == "f_0002"
    before = env.pose()
    views = env.panorama()
    assert len({v.frame_id for v in views}) == 6
    assert all(v.rgb.is_file() for v in views)
    assert env.pose() == before and env.steps == 0
    _, after = env.step("move_forward")
    assert after.frame_id == "f_0009"
    assert env.path_length == pytest.approx(.25)
    assert len(env.motion_trajectory) == 6
    assert len(env.trajectory) == 2 and env.steps == 1
    _, _ = env.step("turn_right")
    assert env.path_length == pytest.approx(.25)
    assert env.steps == 2
    with pytest.raises(ValueError):
        env.step("teleport")
    assert env.steps == 2
    env.close()


def test_super_depth_is_private_metric_png_and_basic_stays_rgb(tmp_path):
    basic = environment(tmp_path / "basic")
    assert basic.observe().depth is None
    basic.close()
    env = Environment(episode(), {"gpu": 0, "tool_tier": "super"}, tmp_path / "super", FakeSimulator)
    observation = env.observe()
    values = np.asarray(Image.open(observation.depth))
    assert observation.depth.parent == env.directory / "observations"
    assert observation.depth.suffix == ".png" and values.dtype == np.uint16
    assert (values == 2000).all()
    assert observation.camera["depth_scale"] == .001
    assert observation.camera["depth_units"] == "metres"
    assert observation.camera["depth_max"] == 10
    assert np.asarray(observation.camera["K"]).shape == (3, 3)
    assert np.asarray(observation.pose["world_from_optical"]).shape == (4, 4)
    assert all(view.depth.is_file() for view in env.panorama())
    env.close()


def test_depth_encoding_marks_far_and_nonfinite_pixels_invalid(tmp_path):
    path = tmp_path / "depth.png"
    save_depth(path, [[.5, 1.234, 0, 10, 70, float("nan"), float("inf")]])
    assert np.asarray(Image.open(path)).tolist() == [[500, 1234, 0, 0, 0, 0, 0]]


def test_small_turns_count_actions_without_changing_position(tmp_path):
    env = environment(tmp_path)
    initial = env.position()
    motion, frame = env.turn(12.5)
    assert not motion.blocked and env.steps == 1
    assert frame.frame_id == "f_0001" and env.position() == initial
    assert env.path_length == 0
    assert env.sim.yaw == pytest.approx(-math.radians(12.5))
    env.turn(-7.5)
    assert env.steps == 2 and env.sim.yaw == pytest.approx(-math.radians(5))
    records = [json.loads(line) for line in (env.directory / "private/motions.jsonl").read_text().splitlines()]
    assert [(row["action"], row["turn_degrees"]) for row in records] == [("turn_right", 12.5), ("turn_left", -7.5)]
    for invalid in (0, 31, -31, True, float("nan"), float("inf"), "15"):
        with pytest.raises(ValueError, match="Turn degrees"):
            env.turn(invalid)
    assert env.steps == 2
    env.close()


def test_native_small_turn_uses_controls_and_restores_filter():
    from src.environment.simulator import HabitatSimulator

    native = HabitatSimulator.__new__(HabitatSimulator)
    events = []
    def original_filter(start, end):
        return end

    controls = SimpleNamespace(move_filter_fn=original_filter)
    def action(node, name, specification):
        events.append((name, specification.amount))
        assert controls.move_filter_fn([1, 0, 0], [2, 0, 0]) == [1, 0, 0]
        return False
    controls.action = action
    native.agent = SimpleNamespace(controls=controls, scene_node=object())
    native.sim = SimpleNamespace(step_physics=lambda seconds: events.append(seconds))
    native._motion_sample = lambda **kwargs: {"position": [0, 0, 0], **kwargs}
    result = native.turn(-4.25)
    assert events == [("turn_left", 4.25), 1 / 60]
    assert controls.move_filter_fn is original_filter
    assert not result.collision and len(result.samples) == 2


def test_near_point_is_reachable_and_never_changes_environment(tmp_path):
    env = environment(tmp_path)
    initial = env.pose()
    candidate = env.near_point([1, 2, -1], .4)
    assert candidate[1] == env.position()[1]
    assert math.hypot(candidate[0] - 1, candidate[2] + 1) <= .4 + 1e-5
    assert env.shortest_path([candidate]) == [env.position(), candidate]
    assert env.pose() == initial and env.steps == 0
    env.sim.shortest_path = lambda start, targets: []
    assert env.near_point([1, 0, -1], .4) is None
    env.close()


def test_curved_motion_uses_microsteps_not_endpoint_chord(tmp_path):
    class CurvedSimulator(FakeSimulator):
        def step(self, action):
            points = [[0, 0, 0], [.1, 0, 0], [.1, 0, -.1]]
            self.point[:] = points[-1]
            return MotionResult(collision=True, samples=[{"position": p} for p in points])

    env = environment(tmp_path, factory=CurvedSimulator)
    motion, _ = env.step("move_forward")
    assert motion.collision and not motion.blocked
    assert env.path_length == pytest.approx(.2)
    assert math.dist(*env.trajectory) == pytest.approx(math.sqrt(.02))
    assert env.motion_trajectory == [[0, 0, 0], [.1, 0, 0], [.1, 0, -.1]]
    env.close()


def test_restore_only_before_movement(tmp_path):
    env = environment(tmp_path)
    pose = {"position": [2, 0, 3], "rotation_xyzw": [0, 0, 0, 1]}
    env.restore_pose(pose)
    assert env.position() == pose["position"]
    assert env.trajectory == [pose["position"]]
    env.step("move_left")
    with pytest.raises(ValueError, match="before movement"):
        env.restore_pose(pose)
    env.close()


def test_score_requires_submit_and_reports_both_thresholds():
    metrics = score_goal("objnav", submitted=True, distance=.4, shortest=2, path_length=4, steps=16)
    assert metrics["success"] == 1 and metrics["spl"] == .5
    assert metrics["official_success"] == 0
    metrics = score_goal("r2r", submitted=False, distance=0, shortest=2, path_length=4, steps=16)
    assert metrics["success"] == 0
    assert metrics["success_distance"] == 3


def test_goat_keeps_environment_and_session_and_target_registration(tmp_path):
    goals = [
        {"modality": "object", "instruction": "Go to the chair.", "positions": [[0, 0, -.25]]},
        {"modality": "image", "instruction": "Find image target.", "positions": [[0, 0, -.5]],
         "image_goal": {"image_dimensions": [48, 64]}},
    ]
    spec = episode("goat", goals)
    env = environment(tmp_path / "env", spec)
    class Session:
        def __init__(self):
            self.calls = []
            self.targets = []

        def register_target(self, path):
            assert Path(path).is_file()
            self.targets.append(path)
            return "target_frame"

        def prompt(self, text, images, limits):
            self.calls.append((text, images, env.steps))
            env.step("move_forward")
            return SimpleNamespace(submitted=True, answer=None, reason="submitted", requests=1, actions=1)

    session = Session()
    rows, reason, failed = execute_goals(env, session, spec, object(), tmp_path)
    assert not failed and reason == "submitted"
    assert [c[2] for c in session.calls] == [0, 1]
    assert session.calls[1][1] == ("target_frame",)
    assert len(session.targets) == 1
    assert env.steps == 2 and env.path_length == pytest.approx(.5)
    assert [r["steps"] for r in rows] == [1, 1]
    assert score_episode("goat", rows, 2)["partial_success"] == 1
    env.close()


def test_goat_service_failure_retains_unattempted_denominator(tmp_path):
    goal = {"modality": "object", "instruction": "Go to chair.", "positions": [[0, 0, -.25]]}
    spec = episode("goat", [goal, goal])
    env = environment(tmp_path / "env", spec)
    class FailedSession:
        def prompt(self, *args, **kwargs):
            return SimpleNamespace(submitted=False, answer=None, reason="model_error", requests=1, actions=0)
    rows, reason, failed = execute_goals(env, FailedSession(), spec, object(), tmp_path)
    assert failed and reason == "model_error"
    assert len(rows) == 2 and rows[1]["reason"] == "not_attempted"
    assert score_episode("goat", rows, 2)["object_success"] == 0
    env.close()


def test_single_floor_preflight_keeps_official_cross_floor_candidates(tmp_path):
    spec = episode(goals=[{"modality": "object", "positions": [[0, 0, -1], [0, 3, -1]], "instruction": "Chair"}])
    env = environment(tmp_path, spec)
    report = validate_single_floor(env, spec)
    assert report["passed"]
    assert report["goals"][0]["same_height_viewpoints"] == 1
    assert len(spec.goals[0]["positions"]) == 2
    spec.reference_path = [[0, 0, 0], [0, 2, 0], [0, 0, -1]]
    assert not validate_single_floor(env, spec)["passed"]
    env.close()


def test_official_objectnav_loader_keeps_all_success_viewpoints(tmp_path):
    source = tmp_path / "obj/val/content/scene.json.gz"
    source.parent.mkdir(parents=True)
    document = {"episodes": [{"scene_id": "hm3d_v0.2/val/scene/scene.basis.glb", "object_category": "chair",
                               "episode_id": "0", "start_position": [0, 0, 0], "start_rotation": [0, 0, 0, 1]}],
                "goals_by_category": {"scene.basis.glb_chair": [
                    {"view_points": [{"agent_state": {"position": [1, 0, 1]}}]},
                    {"view_points": [{"agent_state": {"position": [1, 3, 1]}}]},
                ]}}
    with gzip.open(source, "wt") as stream:
        json.dump(document, stream)
    reader = create_source({"name": "objnav", "root": "obj", "split": "val"}, tmp_path)
    loaded = reader.load(["objnav:val:scene:chair:0"])
    assert loaded[0].goals[0]["positions"] == [[1, 0, 1], [1, 3, 1]]
    assert loaded[0].scene_id.startswith("hm3d/")
    with pytest.raises(ValueError, match="not a canonical ID"):
        reader.load(["0"])


def test_pilot_exact_ids_and_alternation():
    repo = Path(__file__).resolve().parents[1]
    config = read_config(repo / "config/pilot.yaml")
    ids = config["episodes"]
    assert len(ids) == len(set(ids)) == 10
    assert [x.split(":")[0] for x in ids] == ["objnav", "r2r"] * 5
    assert config["environment"]["gpus"] == [0, 1]
    configs = [read_config(repo / path) for path in config["models"].values()]
    assert [c["concurrency"] for c in configs] == [1, 4, 2]
    assert all(c["credentials"] == ".env.harness" for c in configs)
