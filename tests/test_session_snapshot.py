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
        self.denied = handler("forbidden", "move", {"mode": "steps", "actions": ["move_forward"]})
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

    def make(name, state=None, tool_tier="super"):
        env = Environment(episode, {"gpu": 0, "tool_tier": tool_tier, "private_depth": True}, tmp_path / name / "env", FakeSimulator)
        environments.append(env)
        session = open_session(Path(__file__).resolve().parents[1], {**config, "tool_tier": tool_tier}, tmp_path / name / "run", env,
                               scene, state, start_core=False, client_factory=Client)
        sessions.append(session)
        return session

    yield make
    for session in sessions:
        session.close()
    for env in environments:
        env.close()


def test_snapshot_restores_current_frame_depth_map_and_issued_ids(setup, tmp_path):
    session = setup("source")
    session._current_request = "Find a chair."
    session._tool("fixture-move", "move", {"mode": "steps", "actions": ["move_forward", "turn_left"]})
    session._tool("fixture-edit", "memory_edit", {"operations": [
        {"op": "create", "id": "O01", "type": "object", "fields": {"name": "chair"}},
        {"op": "delete", "id": "O01"}]})
    session._tool("fixture-map", "map_view", {"view": "full"})
    current = session.current.copy()
    projected = session.spatial.geometry.resolve_frame(current["frame"], [320, 240])
    session._history = [
        {"role": "assistant", "content": [{"type": "toolCall", "id": "fixture-map", "name": "map_view", "arguments": {"view": "full"}}]},
        {"role": "toolResult", "toolCallId": "fixture-map", "toolName": "map_view",
         "content": session._calls["fixture-map"][1]["result"]["content"], "isError": False},
    ]
    notes = session.workspace / "scratch/plans/route.md"
    notes.parent.mkdir()
    notes.write_text("Inspect the left entrance before returning.")
    derived = session.observations.crop(current["frame"], [.25, .25, .75, .75])
    session.snapshot(tmp_path / "snapshot")
    restored = setup("restored", read_snapshot(tmp_path / "snapshot"))
    assert restored.client is None
    assert restored.current == current
    assert restored.environment.steps == 2
    assert restored.environment.path_length == pytest.approx(.25)
    assert restored.spatial.geometry.resolve_frame(current["frame"], [320, 240]) == projected
    assert restored._history == session._history
    assert restored.observations.resolve_pixel(derived["image"], [40, 60]) == (current["frame"], [200, 180])
    assert (restored.workspace / "scratch/plans/route.md").read_text() == notes.read_text()
    assert not (restored.workspace / "memory/task.json").exists()
    (restored.workspace / "scratch/plans/route.md").write_text("New plan.")
    assert notes.read_text() == "Inspect the left entrance before returning."
    result = restored._tool("fixture-create", "memory_edit", {"operations": [
        {"op": "create", "id": "O01", "type": "object", "fields": {"name": "chair"}}]})
    assert result["result"]["isError"]
    assert not list((tmp_path / "snapshot").rglob("requests.json"))


def test_qa_preserves_navigation_target_and_rejects_movement(setup):
    session = setup("qa")
    session._current_request = "Find the pictured chair."
    session._target_frames = [session.register_target(session.environment.current.rgb)]
    targets = session._target_frames.copy()
    frame = session.current["frame"]
    result = session.ask("Which direction is map-up?")
    assert result.submitted and result.answer == "front"
    assert session.client.denied["result"]["isError"]
    assert session.environment.steps == 0 and session.current["frame"] == frame
    assert session._current_request == "Find the pictured chair."
    assert session._target_frames == targets and session._prompts == []
    configured = [args for op, args in session.client.calls if op == "configure"]
    assert all("allowedTools" not in args or {tool["name"] for tool in session.tool_specs} <= set(args["allowedTools"])
               for args in configured)


def test_continue_adds_no_user_and_runs_without_new_capture(setup):
    session = setup("continue")
    session._current_request = "Find a chair."
    session._tool("fixture-observe", "observe", {"mode": "current"})
    frame = session.current["frame"]
    session._history = [
        {"role": "assistant", "content": [{"type": "toolCall", "id": "fixture-observe", "name": "observe", "arguments": {"mode": "current"}}]},
        {"role": "toolResult", "toolCallId": "fixture-observe", "toolName": "observe",
         "content": session._calls["fixture-observe"][1]["result"]["content"], "isError": False},
    ]
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


def test_snapshot_never_dereferences_scratch_links_into_host_files(setup, tmp_path):
    session = setup("links")
    secret = tmp_path / "host-private.txt"
    secret.write_text("host data must not become a workspace file")
    (session.workspace / "scratch/host-link").symlink_to(secret)
    session.snapshot(tmp_path / "linked-cut")
    assert (tmp_path / "linked-cut/workspace/scratch/host-link").is_symlink()
    restored = setup("links-restored", read_snapshot(tmp_path / "linked-cut"))
    assert (restored.workspace / "scratch/host-link").is_symlink()


def test_restored_receipt_does_not_repeat_an_already_executed_action(setup, tmp_path):
    session = setup("receipt-source")
    session._current_request = "Find a chair."
    result = session._tool("fixture-move", "move", {"mode": "steps", "actions": ["move_forward"]})
    call_id = "fixture-move"
    session.snapshot(tmp_path / "receipt-cut")
    restored = setup("receipt-restored", read_snapshot(tmp_path / "receipt-cut"))
    previous = restored.environment.steps
    replayed = restored._tool(call_id, "move", {"mode": "steps", "actions": ["move_forward"]})
    assert replayed["result"] == result["result"]
    assert restored.environment.steps == previous


def test_basic_label_capture_keeps_depth_private_without_enabling_super_tools(setup):
    from src.harness.geometry import Geometry
    session = setup("basic-labels", tool_tier="basic")
    assert not session.spatial.enabled
    assert "locate" not in {tool["name"] for tool in session.tool_specs}
    depth, camera = Geometry(session.workspace, session.private, session.transform).load_frame(session.current["frame"])
    assert depth.shape == (480, 640) and depth[240, 320] == 2
    assert "depth" not in json.dumps(session.observations.get(session.current["frame"]))
    projected = session._tool("forbidden-projection", "map_view", {
        "project": {"source": session.current["frame"], "points": [[320, 240]]}})
    value = json.loads(projected["result"]["content"][0]["text"])
    assert projected["result"]["isError"] or "error" in value["points"][0]
    edited = session._tool("forbidden-location", "memory_edit", {"operations": [
        {"op": "create", "id": "O01", "type": "object", "fields": {
            "name": "chair", "location": {"image": session.current["frame"], "pixel": [320, 240]}}}]})
    assert edited["result"]["isError"] and session.memory.entities == []


def test_motion_window_allows_one_closing_round_without_repeating_motion(setup):
    class MovingClient(Client):
        def run_prompt(self, text, images, handler, seconds, **kwargs):
            kwargs["on_event"]({"type": "event", "event": {"type": "message_end", "data": {
                "message": {"role": "assistant"}}}})
            handler("one-step", "move", {"mode": "steps", "actions": ["move_forward"]})
            self.denied = handler("extra-step", "move", {"mode": "steps", "actions": ["move_forward"]})
            self.saved = handler("closing-memory", "memory_edit", {"operations": [
                {"op": "create", "id": "O01", "type": "object", "fields": {"name": "chair"}}]})
            return {"reason": "window_end"}

    session = setup("window")
    session._client_factory = MovingClient
    session._current_request = "Find a chair."
    session._tool("fixture-observe", "observe", {"mode": "current"})
    result = session.continue_run(PromptLimits(max_actions=1, max_turns=6))
    assert result.reason == "window_end" and result.actions == 1
    assert session.client.denied["result"]["isError"]
    assert not session.client.saved["result"]["isError"]
    assert ("configure", {"maxTurns": 2}) in session.client.calls
