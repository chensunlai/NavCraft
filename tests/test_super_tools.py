"""Public tool integration using deterministic geometry and real session boundaries."""

import json
import math
from pathlib import Path

import pytest

from src.dataset.episodes import EpisodeSpec
from src.environment import Environment, MotionResult
from src.environment.fake import FakeSimulator
from src.harness.session import HarnessSession
from src.harness.tools.specs import specifications


class NoModelClient:
    def __init__(self, *args):
        pass

    def request(self, op, *args, **kwargs):
        return {"session_id": "fixture"} if op == "init" else {}

    def close(self):
        pass

    def drain_events(self, callback):
        pass


@pytest.fixture
def session(tmp_path):
    scene = tmp_path / "map.json"
    scene.write_text(json.dumps({"origin": [-10, -10], "size_m": 20,
                                "floor_height": 0, "walls": [[[-2, -2], [-2, 2]]]}))
    credentials = tmp_path / "credentials.env"
    credentials.write_text("NAVCRAFT_TEST_BASE_URL=http://fixture.invalid\nNAVCRAFT_TEST_API_KEY=fixture\n")
    episode = EpisodeSpec("fixture", "0", "objnav", "unused", [0, 0, 0], [0, 0, 0, 1], [], "unused")
    env = Environment(episode, {"gpu": 0, "tool_tier": "super"}, tmp_path / "private-env", FakeSimulator)
    with HarnessSession(Path(__file__).resolve().parents[1],
                        {"model": "fixture", "tool_tier": "super", "credentials": str(credentials)},
                        tmp_path / "session", env, scene, client_factory=NoModelClient) as result:
        yield result
    env.close()


def call(session, name, args):
    receipt = session._tool(str(len(session._calls)), name, args)
    return receipt, json.loads(receipt["result"]["content"][0]["text"])


def test_super_schemas_are_opt_in_and_basic_remains_seven_tools():
    basic = specifications()
    assert len(basic) == 7
    super_tools = {t["name"]: t for t in specifications("super")}
    assert len(super_tools) == 10
    assert {"locate", "nav_face", "nav_goto"} <= super_tools.keys()
    assert "grid" not in basic[0]["parameters"]["properties"]
    assert "grid" in super_tools["nav_observe"]["parameters"]["properties"]
    with pytest.raises(ValueError):
        specifications("unknown")


def test_private_depth_and_panorama_capture_bearing(session):
    receipt, panorama = call(session, "nav_observe", {"mode": "panorama", "grid": True, "project": True})
    assert not receipt["result"]["isError"]
    assert len(receipt["result"]["images"]) == 2
    point = session.super_tools.geometry.resolve_frame(panorama["views"][1]["frame"], [319.5, 239.5])
    assert point["bearing"] == pytest.approx(60, abs=.1)
    assert session.environment.steps == 0
    public_files = list(session.workspace.rglob("*"))
    assert not any("depth" in p.name for p in public_files)
    public_index = session.observations.index_path.read_text()
    assert "world_from_optical" not in public_index and "depth_scale" not in public_index
    assert all((session.workspace / p).is_file() for p in receipt["result"]["images"])
    target = session.register_target(session.environment.observe().rgb)
    _, located = call(session, "locate", {"source": target, "points": [[320, 240]]})
    assert located["points"][0] == {"pixel": [320, 240], "valid": False, "reason": "depth_unavailable"}


def test_at_edit_is_atomic_and_provenance_cannot_be_reused_for_other_coordinates(session):
    frame = session.current["frame"]
    create = {"op": "create", "id": "O01", "type": "object", "fields": {
        "name": "chair", "at": {"frame": frame, "pixel": [320, 240]}}}
    _, result = call(session, "memory_edit", {"operations": [create]})
    assert "error" not in result
    entity = session.memory.entities[0]
    assert "at" not in entity and entity["evidence"] == [frame]
    assert session.super_tools.target({"entity": "O01"})["observed"]
    before = session.memory.entities
    _, result = call(session, "memory_edit", {"operations": [
        {"op": "update", "id": "O01", "fields": {"name": "changed"}},
        {"op": "create", "id": "O02", "type": "object", "fields": {
            "name": "invalid", "at": {"frame": frame, "pixel": [-1, 0]}}}]})
    assert "error" in result and session.memory.entities == before
    # Later geometry in the same atomic batch cannot inherit the old pixel's origin.
    create["id"] = "O02"
    _, result = call(session, "memory_edit", {"operations": [create,
        {"op": "update", "id": "O02", "fields": {"point": [.6, .6]}}]})
    assert "error" not in result
    assert not session.super_tools.target({"entity": "O02"})["observed"]
    create["id"] = "O03"
    _, result = call(session, "memory_edit", {"operations": [create, {"op": "delete", "id": "O03"}]})
    assert "error" not in result


def test_map_pixels_and_near_query_use_global_memory_and_current_body(session):
    _, view = call(session, "map_view", {"view": {"ego": 2}})
    _, result = call(session, "memory_edit", {"operations": [{
        "op": "create", "id": "O01", "type": "object", "coordinate_ref": view["view_id"],
        "fields": {"name": "front", "point": [384, 207]}}]})
    assert "error" not in result
    assert session.memory.entities[0]["point"] == pytest.approx([.5, .45])
    assert not session.super_tools.target({"entity": "O01"})["observed"]
    _, near = call(session, "memory_read", {"near": {"self": 2}})
    assert near["total"] == 1
    assert near["entities"][0]["distance"] == 1
    assert near["entities"][0]["bearing"] == 0
    session._perform_motion("turn_right", 30)
    _, near = call(session, "memory_read", {"near": {"entity": "O01", "radius": .1}})
    assert near["entities"][0]["bearing"] == -30
    _, far = call(session, "memory_read", {"near": {"self": .5}})
    assert far["total"] == 0


def test_face_uses_actual_budgeted_micro_turns_and_idempotent_calls(session):
    target = {"uv": [.55, .45]}
    reply = session._tool("face-once", "nav_face", {"target": target})
    result = json.loads(reply["result"]["content"][0]["text"])
    assert result["stop"] == "aligned" and abs(result["bearing_after"]) <= 1
    assert session.environment.steps == 2
    assert session.environment.path_length == 0
    turns = [motion["turn_degrees"] for motion in session.environment.motions]
    assert turns == pytest.approx([30, 15])
    assert session._tool("face-once", "nav_face", {"target": target}) == reply
    assert session.environment.steps == 2


def test_goto_local_and_straight_do_not_submit_or_read_task_goals(session):
    _, result = call(session, "nav_goto", {"target": {"frame": session.current["frame"], "pixel": [320, 240]}})
    assert result["planner"] == "local" and result["reached"]
    assert result["remaining"] <= .6
    assert session.environment.steps > 0 and session.environment.path_length > 0
    assert not session._submitted
    assert "success" not in result
    # UV has no observed provenance and therefore cannot invoke NavMesh planning.
    def forbidden(*args, **kwargs):
        raise AssertionError("UV must not use NavMesh")
    session.environment.near_point = forbidden
    _, straight = call(session, "nav_goto", {"target": {"uv": [.5, .3]}, "max_actions": 1})
    assert straight["planner"] == "straight" and straight["stop"] == "max_actions"


def test_goto_stops_on_blocked_and_shared_action_budget(session):
    def blocked(action):
        session.environment.steps += 1
        return MotionResult(collision=True, blocked=True), session.environment.observe()
    original = session.environment.step
    session.environment.step = blocked
    _, result = call(session, "nav_goto", {"target": {"uv": [.5, .3]}})
    assert result["stop"] == "blocked" and not result["reached"]
    session.environment.step = original
    session._action_end = session.environment.steps + 2
    receipt, result = call(session, "nav_goto", {"target": {"uv": [.7, .5]}})
    assert receipt["control"]["end_run"]
    assert session.environment.steps == session._action_end
    assert result["stop"] == "interrupted"


def test_goto_stops_before_core_timeout_without_extra_motion(session, monkeypatch):
    ticks = iter([0, 0, 91])
    monkeypatch.setattr("src.harness.super_tools.time.monotonic", lambda: next(ticks))
    _, result = call(session, "nav_goto", {"target": {"uv": [.5, .3]}})
    assert result["stop"] == "time_limit" and not result["reached"]
    assert session.environment.steps == 1
    assert not session._submitted


def test_local_planning_refuses_remote_and_cross_floor_routes(session):
    frame = session.current["frame"]
    _, result = call(session, "memory_edit", {"operations": [{"op": "create", "id": "O01", "type": "object",
        "fields": {"name": "point", "at": {"frame": frame, "pixel": [320, 240]}}}]})
    assert "error" not in result
    session.environment.shortest_path = lambda *args: [[0, 0, 0], [0, 1, -2]]
    _, result = call(session, "nav_goto", {"target": {"entity": "O01"}})
    assert result["stop"] == "outside_local_scope" and session.environment.steps == 0
    assert not result["reached"]


def test_motion_limits_are_validated_before_action(session):
    for args in ({"max_actions": True}, {"max_actions": 41}, {"stop_distance": math.nan}):
        receipt, result = call(session, "nav_goto", {"target": {"uv": [.6, .6]}, **args})
        assert receipt["result"]["isError"] and "error" in result
        assert session.environment.steps == 0
