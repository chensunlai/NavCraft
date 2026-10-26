import json
from pathlib import Path

import pytest

from src.dataset.episodes import EpisodeSpec
from src.environment import Environment
from src.environment.fake import FakeSimulator
from src.harness.session import PromptLimits, open_session, read_snapshot


class Client:
    def __init__(self, *args):
        self.calls = []

    def request(self, op, args=None, **kwargs):
        self.calls.append((op, args))
        return {"session_id": "fixture"} if op == "init" else {}

    def run_prompt(self, text, images, handler, seconds, **kwargs):
        self.last = text, images, kwargs
        if kwargs.get("mode") == "continue":
            handler("read", "memory_read", {})
            return {"reason": "window_end"}
        self.denied = handler("forbidden", "nav_step", {"actions": ["move_forward"]})
        handler("answer", "submit", {"answer": "front"})
        return {"reason": "host_end"}

    def close(self):
        pass

    def drain_events(self, callback):
        pass


@pytest.fixture
def setup(tmp_path):
    scene = tmp_path / "map.json"
    scene.write_text(json.dumps({"origin": [0, 0], "size_m": 10, "floor_height": 0,
                                 "walls": [[[0, 0], [0, 10]]]}))
    env_file = tmp_path / "credentials"
    env_file.write_text("NAVCRAFT_TEST_BASE_URL=http://fixture.invalid\nNAVCRAFT_TEST_API_KEY=fixture\n")
    config = {"model": "fixture", "tool_tier": "super", "credentials": str(env_file)}
    episode = EpisodeSpec("fixture", "fixture", "generated", "fixture", [2, 0, 3],
                          [0, 0, 0, 1], [], "fixture")
    environments, sessions = [], []

    def make(name, state=None):
        env = Environment(episode, {"gpu": 0, "tool_tier": "super"}, tmp_path / name / "env", FakeSimulator)
        environments.append(env)
        session = open_session(Path(__file__).resolve().parents[1], config, tmp_path / name / "run", env,
                               scene, state, start_core=False, client_factory=Client)
        sessions.append(session)
        return session

    yield make
    for session in sessions:
        session.close()
    for env in environments:
        env.close()


def test_script_snapshot_restores_current_frame_depth_map_and_issued_ids(setup, tmp_path):
    session = setup("source")
    session.begin_script("Find a chair.")
    session.script_tool("nav_step", {"actions": ["move_forward", "turn_left"]})
    session.script_tool("memory_edit", {"operations": [
        {"op": "create", "id": "O01", "type": "object", "fields": {"name": "chair"}},
        {"op": "delete", "id": "O01"}]})
    session.script_tool("map_view", {"view": "full"})
    current = session.current.copy()
    projected = session.super_tools.geometry.resolve_frame(current["frame"], [320, 240])
    session.snapshot(tmp_path / "snapshot")
    restored = setup("restored", read_snapshot(tmp_path / "snapshot"))
    assert restored.client is None
    assert restored.current == current
    assert restored.environment.steps == 2
    assert restored.environment.path_length == pytest.approx(.25)
    assert restored.super_tools.geometry.resolve_frame(current["frame"], [320, 240]) == projected
    assert restored._history[-1]["role"] == "toolResult"
    result = restored.script_tool("memory_edit", {"operations": [
        {"op": "create", "id": "O01", "type": "object", "fields": {"name": "chair"}}]})
    assert result["result"]["isError"]
    assert not list((tmp_path / "snapshot").rglob("requests.json"))


def test_qa_preserves_navigation_target_and_rejects_movement(setup):
    session = setup("qa")
    session.begin_script("Find the pictured chair.", images=[session.environment.current.rgb])
    targets = session._target_frames.copy()
    frame = session.current["frame"]
    result = session.ask("Which direction is map-up?")
    assert result.submitted and result.answer == "front"
    assert session.client.denied["result"]["isError"]
    assert session.environment.steps == 0 and session.current["frame"] == frame
    assert session._current_request == "Find the pictured chair."
    assert session._target_frames == targets and session._prompts == []


def test_continue_adds_no_user_and_runs_without_new_capture(setup):
    session = setup("continue")
    session.begin_script("Find a chair.")
    session.script_tool("nav_observe", {"mode": "current"})
    frame = session.current["frame"]
    before = session._history.copy()
    result = session.continue_run(PromptLimits(max_turns=6, max_path_length=3))
    assert result.reason == "window_end"
    assert session.client.last[0] is None and session.client.last[1] == []
    assert session.client.last[2]["mode"] == "continue"
    assert session._history == before and session.current["frame"] == frame


def test_snapshot_rejects_busy_sessions(setup, tmp_path):
    session = setup("busy")
    session._busy = True
    with pytest.raises(RuntimeError, match="busy"):
        session.snapshot(tmp_path / "bad")
