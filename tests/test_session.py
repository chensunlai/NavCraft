"""Navigation tool authority, durable decisions and runner cleanup contracts."""

import json

import numpy as np
import pytest

from src.bench import run as run_module
from src.common.config import ExperimentConfig
from src.dataset.episodes import EpisodeSpec
from src.environment.fake import FakeEnvironment, FakeSimulator
from src.environment.tasks.base import GOATTask, ObjectNavTask
from src.harness.agent.base import AgentLimits, ToolCall
from src.harness.agent.loop import AgentRuntime
from src.harness.agent.model import ScriptedModel
from src.harness.session import NavigationSession
from src.harness.trace import TraceReader, TraceWriter

TOOLS = ["nav_observe", "nav_depth", "nav_step", "nav_finish"]


def episode(benchmark="objnav", goals=None, source_id="1"):
    return EpisodeSpec(
        id=f"fixture:{source_id}", source_id=source_id, benchmark=benchmark,
        scene_id="fake.glb", start_position=[0, 0, 0], start_rotation=[0, 0, 0, 1],
        goals=goals or [{"modality": "object", "instruction": "Find chair", "positions": [[0, 0, -0.25]]}],
        source_file="fixture", reference_path=[[991, 992, 993]],
    )


@pytest.fixture
def session(tmp_path):
    env = FakeEnvironment({"width": 8, "height": 6}, ObjectNavTask({}), tmp_path)
    trace = TraceWriter(tmp_path, "fixture:1")
    result = NavigationSession(env, episode(), trace, TOOLS)
    result.start()
    yield result
    result.close()
    trace.close()


@pytest.mark.parametrize("arguments", [
    {"actions": ["move_forward", "invalid"]},
    {"actions": ["move_forward", "stop"]},
    {"actions": ["move_forward"] * 7},
    {"actions": []},
    {"actions": ["move_forward"], "extra": 1},
    "{invalid JSON",
])
def test_invalid_step_validated_before_any_motion(session, arguments):
    result = session.execute(ToolCall("invalid", "nav_step", arguments))
    assert not result.ok
    assert session.environment.sim.position() == [0, 0, 0]
    assert session.environment.task.steps == 0
    assert session.transition.observation.frame_id == "frame_000000"


def test_duplicate_call_is_cached_but_changed_arguments_are_rejected(session):
    result = session.execute(ToolCall("move", "nav_step", {"actions": ["move_forward"]}))
    repeated = session.execute(ToolCall("move", "nav_step", '{"actions": ["move_forward"]}'))
    assert repeated is result
    assert session.environment.task.steps == 1
    changed = session.execute(ToolCall("move", "nav_step", {"actions": ["turn_left"]}))
    assert not changed.ok and "different arguments" in changed.data["error"]
    assert session.environment.task.steps == 1
    assert session.environment.sim.position() == [0, 0, -0.25]
    assert session.environment.sim.yaw == 0
    events = list(TraceReader(session.trace.path).events())
    assert len([row for row in events if row["event"] == "action_executed"]) == 1
    assert len([row for row in events if row["event"] == "tool_replayed"]) == 2


def test_failed_capture_after_motion_cannot_execute_same_call_again(session, monkeypatch):
    original_step = session.environment.step

    def capture_failure(action):
        original_step(action)
        raise RuntimeError("capture failed after movement")

    monkeypatch.setattr(session.environment, "step", capture_failure)
    call = ToolCall("move", "nav_step", {"actions": ["move_forward"]})
    with pytest.raises(RuntimeError, match="capture failed after movement"):
        session.execute(call)
    assert session.environment.task.steps == 1
    assert session.environment.sim.position() == [0, 0, -0.25]
    replay = session.execute(call)
    assert not replay.ok and replay.truncated
    assert "will not be executed again" in replay.data["error"]
    assert session.environment.task.steps == 1
    assert session.environment.sim.position() == [0, 0, -0.25]
    events = list(TraceReader(session.trace.path).events())
    assert len([row for row in events if row["event"] == "tool_execution_failed"]) == 1
    assert len([row for row in events if row["event"] == "tool_replayed"]) == 1


def test_depth_queries_require_observed_frames_and_valid_pixels(session):
    depth = np.load(session.transition.observation.depth)
    depth[0, 0], depth[0, 1] = np.nan, 0
    np.save(session.transition.observation.depth, depth)
    valid = session.execute(ToolCall("depth", "nav_depth", {
        "frame_id": "frame_000000", "points": [[0, 0], [1, 0], [7, 5]],
    }))
    assert valid.ok and valid.images == []
    assert valid.data["points"] == [
        {"pixel": [0, 0], "depth_m": None, "valid": False},
        {"pixel": [1, 0], "depth_m": None, "valid": False},
        {"pixel": [7, 5], "depth_m": 2.0, "valid": True},
    ]
    for number, (frame, point) in enumerate([
        ("frame_000099", [0, 0]), ("frame_000000", [-1, 0]),
        ("frame_000000", [8, 0]), ("frame_000000", [0, 6]),
        ("frame_000000", [True, 0]),
    ]):
        invalid = session.execute(ToolCall(f"bad-{number}", "nav_depth", {"frame_id": frame, "points": [point]}))
        assert not invalid.ok
    session.execute(ToolCall("move", "nav_step", {"actions": ["move_forward"]}))
    old = session.execute(ToolCall("old", "nav_depth", {"frame_id": "frame_000000", "points": [[7, 5]]}))
    assert old.ok and old.data["points"][0]["depth_m"] == 2


def test_goal_switch_interrupts_batch_and_exposes_image_without_private_camera(tmp_path):
    goals = [
        {"modality": "object", "instruction": "chair", "positions": [[0, 0, -1]]},
        {"modality": "image", "instruction": "Find image target", "positions": [[0, 0, -0.5]],
         "image_goal": {"position": [771, 882, 993], "rotation": [0, 0, 0, 1],
                        "hfov": 90, "image_dimensions": [6, 8]}},
    ]
    env = FakeEnvironment({"width": 8, "height": 6}, GOATTask({"max_subtask_steps": 2}), tmp_path)
    trace = TraceWriter(tmp_path, "fixture:1")
    session = NavigationSession(env, episode("goat", goals), trace, TOOLS)
    try:
        session.start()
        result = session.execute(ToolCall("move", "nav_step", {"actions": ["move_forward"] * 4}))
        assert result.ok and not result.terminated
        assert len(result.data["executed_actions"]) == 2
        assert result.data["goal_changed"]
        assert result.data["task"]["goal_id"] == 1
        assert env.sim.position() == [0, 0, -0.5]
        assert [path.name for path in result.images] == ["rgb_000002.png", "target_001.png"]
        assert all(path.exists() for path in result.images)
        assert "771" not in json.dumps(result.data)
        assert "image_goal" not in json.dumps(result.data)
        finish = session.execute(ToolCall("finish", "nav_finish", {}))
        assert finish.terminated and env.sim.position() == [0, 0, -0.5]
        assert env.metrics()["subtask_success"] == [0, 1]
    finally:
        session.close()
        trace.close()


def function(call_id, name, arguments):
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": json.dumps(arguments)}


def test_trace_decisions_pair_exact_conditions_and_keep_image_references(tmp_path):
    env = FakeEnvironment({"width": 8, "height": 6}, ObjectNavTask({}), tmp_path)
    trace = TraceWriter(tmp_path, "fixture:1", ("test-secret",))
    session = NavigationSession(env, episode(), trace, TOOLS)
    initial = session.start()
    reason = {"type": "reasoning", "encrypted_content": "opaque-state", "summary": []}
    model = ScriptedModel({"turns": [
        {"output": [reason, function("move", "nav_step", {"actions": ["move_forward"]})]},
        {"output": [function("finish", "nav_finish", {})]},
    ]})
    try:
        result = AgentRuntime(model, {}).run(initial, session.toolset, trace, AgentLimits())
        assert result.reason == "terminated"
        trace.emit("diagnostic", api_key="another-secret", error="failure test-secret")
    finally:
        session.close()
        trace.close()
    decisions = list(TraceReader(trace.path).decisions())
    assert len(decisions) == 2
    assert decisions[0]["request"]["items"] == initial
    assert reason in decisions[1]["request"]["items"]
    assert decisions[0]["request"]["request_id"] == decisions[0]["response"]["request_id"]
    output = next(item for item in decisions[1]["request"]["items"] if item.get("type") == "function_call_output")
    assert output["call_id"] == "move"
    assert json.loads(output["output"])["observation"]["pose"]["position"] == [0, 0, -0.25]
    text = trace.path.read_text()
    assert "image_path" in text and "base64" not in text
    assert "991" not in text and "reference_path" not in text
    assert "test-secret" not in text and "another-secret" not in text
    assert "[REDACTED]" in text


def experiment_config(tmp_path, turns):
    return ExperimentConfig(
        episodes={"name": "objnav", "root": "unused"},
        task={"name": "objectnav", "protocol": "fixture-objectnav", "success_distance": 0.1,
              "max_steps": 10, "max_subtask_steps": 10},
        environment={"name": "fake", "width": 8, "height": 6},
        model={"name": "scripted", "api_key": "test-secret", "turns": turns},
        harness={"runtime": "agent", "tools": TOOLS},
        run={"name": "fixture", "output_dir": str(tmp_path / "runs"), "max_requests": 4,
             "timeout_seconds": 30, "max_context_chars": 100000}, sources={},
    )


def test_runner_keeps_failures_in_denominator_and_closes_every_environment(tmp_path, monkeypatch):
    from src.common.register import episode_registry

    run_module.load_builtins()
    specimens = [episode(source_id="1"), episode(source_id="2")]

    class Source:
        def __init__(self, config):
            pass

        def load(self):
            return specimens

    monkeypatch.setitem(episode_registry._factories, "objectnav", Source)
    simulators = []
    original_init = FakeSimulator.__init__
    original_step = FakeSimulator.step

    def initialize(self, episode, config):
        original_init(self, episode, config)
        self.fixture_id = episode.source_id
        simulators.append(self)

    def step(self, action):
        if self.fixture_id == "2":
            raise RuntimeError("renderer failure test-secret")
        return original_step(self, action)

    monkeypatch.setattr(FakeSimulator, "__init__", initialize)
    monkeypatch.setattr(FakeSimulator, "step", step)
    config = experiment_config(tmp_path, [
        {"output": [function("move", "nav_step", {"actions": ["move_forward"]})]},
        {"output": [function("finish", "nav_finish", {})]},
    ])
    directory, summary = run_module.run_experiment(config)
    assert summary["selected"] == summary["finished"] == 2
    assert summary["statuses"] == {"terminated": 1, "environment_error": 1}
    assert summary["score_over_selected"] == 0.5
    assert len(simulators) == 2 and all(simulator.closed for simulator in simulators)
    results = json.loads((directory / "results.json").read_text())
    assert results[1]["agent"]["error"].endswith("[REDACTED]")
    assert results[1]["metrics"]["success"] == 0
    assert json.loads((directory / "selection.json").read_text()) == [spec.id for spec in specimens]
    assert "test-secret" not in (directory / "config.json").read_text()
    assert "test-secret" not in (directory / "episodes/0001/trace.jsonl").read_text()


def test_runner_records_model_error_and_releases_environment(tmp_path, monkeypatch):
    from src.common.register import episode_registry

    run_module.load_builtins()

    class Source:
        def __init__(self, config):
            pass

        def load(self):
            return [episode()]

    monkeypatch.setitem(episode_registry._factories, "objectnav", Source)
    closed = []
    original_close = FakeSimulator.close

    def close(self):
        original_close(self)
        closed.append(self)

    monkeypatch.setattr(FakeSimulator, "close", close)
    malformed = {"type": "function_call", "name": "nav_step", "arguments": "{}"}
    directory, summary = run_module.run_experiment(experiment_config(tmp_path, [{"output": [malformed]}]))
    assert summary["statuses"] == {"model_error": 1}
    assert summary["score_over_selected"] == 0
    assert len(closed) == 1 and closed[0].closed
    result = json.loads((directory / "results.json").read_text())[0]
    assert result["metrics"]["steps"] == 0
    assert result["metrics"]["reason"] == "model_error"
