"""Exercise both memory modes through the actual session and tool dispatcher."""

import json
from pathlib import Path

import pytest

from src.common.config import read_config
from src.dataset.episodes import EpisodeSpec
from src.environment import Environment, MotionResult
from src.environment.fake import FakeSimulator
from src.harness.session import HarnessSession, read_snapshot


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def factory(tmp_path):
    resources = []

    def create(mode="node", state=None):
        root = tmp_path / str(len(resources))
        root.mkdir()
        episode = EpisodeSpec("fixture", "0", "objnav", "unused", [0, 0, 0], [0, 0, 0, 1], [], "unused")
        env = Environment(episode, {"gpu": 0, "tool_tier": "super"}, root / "env", FakeSimulator)
        resources.append(env)
        wall = root / "map.json"
        wall.write_text(json.dumps({"origin": [-5, -5], "size_m": 10,
                                    "floor_height": 0, "walls": [[[-2, -2], [-2, 2]]]}))
        config = {"model": "fixture", "tool_tier": "super", "memory_type": mode}
        session = HarnessSession(REPO, config, root / "run", env, wall if mode == "entity" else None,
                                 initial_state=state, start_core=False)
        resources.append(session)
        return session

    yield create
    for resource in reversed(resources):
        resource.close()


def call(session, name, args):
    reply = session._tool(f"call-{len(session._calls)}", name, args)
    return reply, json.loads(reply["result"]["content"][0]["text"])


def node(session, description="Sofa facing the television"):
    reply, panorama = call(session, "observe", {"mode": "panorama"})
    assert not reply["result"]["isError"]
    reply, result = call(session, "memory_edit", {"operations": [
        {"op": "create", "room": "living_room_1", "description": description}]})
    assert not reply["result"]["isError"], result
    return result["changes"][0]["id"], panorama["group"]


def test_modes_have_separate_public_memory_and_fixed_tool_schemas(factory):
    entity, place = factory("entity"), factory("node")
    assert len(entity.tool_specs) == 8 and len(place.tool_specs) == 6
    assert set(t["name"] for t in place.tool_specs) == {"observe", "move", "memory_read", "memory_edit", "task_read", "submit"}
    assert entity.transform["walls"] and not place.transform["walls"]
    assert place.current["pose"]["pos"] == [.5, .5]
    assert (entity.workspace / "memory/entities.json").exists()
    assert not (place.workspace / "memory/entities.json").exists()
    assert (place.workspace / "memory/nodes.json").exists()
    assert "wall map" not in (REPO / "config/prompts/navigation_node.md").read_text()


def test_edit_does_not_capture_and_queries_preserve_image_identity(factory):
    session = factory()
    before = (session.environment.steps, len(session.observations.items), session._observations)
    reply, result = call(session, "memory_edit", {"operations": [{"op": "create", "description": "Entrance"}]})
    assert reply["result"]["isError"] and not result["committed"]
    assert before == (session.environment.steps, len(session.observations.items), session._observations)
    identifier, panorama = node(session)
    before = (session.environment.steps, len(session.observations.items), session._observations)
    for mode in ({"mode": "text", "id": identifier}, {"mode": "image", "refs": [identifier]},
                 {"mode": "map", "view": "ego", "highlight": [identifier]}):
        reply, result = call(session, "memory_read", mode)
        assert not reply["result"]["isError"], result
        assert bool(reply["result"]["images"]) == (mode["mode"] != "text")
        for image in reply["result"]["images"]:
            assert (session.workspace / image).is_file()
    assert before == (session.environment.steps, len(session.observations.items), session._observations)
    reply, events = call(session, "memory_read", {"mode": "text", "source": "observations"})
    assert next(event for event in events["events"] if event["image"] == panorama)["nodes"] == [identifier]


def test_refresh_uses_current_panorama_and_keeps_searchable_summary(factory):
    session = factory()
    identifier, original = node(session)
    call(session, "move", {"mode": "steps", "actions": ["move_forward"]})
    reply, _ = call(session, "memory_edit", {"operations": [{"op": "update", "id": identifier, "refresh": True}]})
    assert reply["result"]["isError"]
    call(session, "observe", {"mode": "panorama"})
    reply, _ = call(session, "memory_edit", {"operations": [{"op": "update", "id": identifier, "refresh": True}]})
    assert not reply["result"]["isError"]
    saved = session.memory.get(identifier)
    assert saved["anchor"] == original and saved["panorama"] != original
    assert session.memory.read(query="television")["total"] == 1
    assert session.spatial.target({"node": identifier})["observed"] is False


def test_panorama_pixels_use_the_displayed_grid_not_an_original_frame(factory):
    session = factory()
    _, panorama = call(session, "observe", {"mode": "panorama"})
    image = panorama["image"]
    assert image.startswith("i_")
    assert session.observations.image(panorama["group"])["image"] == image
    # Pick a real pixel in the second tile using the actual displayed image ID.
    reference, pixel = session.observations.resolve_pixel(image, [1100, 400])
    assert reference == panorama["views"][1]["frame"]
    assert pixel == [460, 374]
    with pytest.raises(ValueError, match="pixel_out_of_bounds"):
        session.observations.resolve_pixel(reference, [1100, 400])


@pytest.mark.parametrize("blocked", [False, True])
def test_node_approach_reports_partial_progress_without_new_route_planning(factory, blocked):
    session = factory()
    identifier, _ = node(session)
    call(session, "move", {"mode": "steps", "actions": ["move_backward"] * 4})
    before = session.environment.steps
    # A visited anchor does not allow the tool to ask NavMesh for a route.
    session.environment.shortest_path = lambda *_: pytest.fail("node must use straight motion")
    if blocked:
        def stop(action, degrees=None):
            session.environment.steps += 1
            return MotionResult(collision=True, blocked=True)
        session._perform_motion = stop
    reply, result = call(session, "move", {"mode": "approach", "target": {"node": identifier},
                                           "max_actions": 1})
    assert not reply["result"]["isError"]
    assert result["target_node"] == identifier and result["planner"] == "straight"
    assert result["stop"] == ("blocked" if blocked else "max_actions")
    assert not result["reached"] and not session._submitted
    assert session.environment.steps == before + 1
    assert "waypoint" in result["hint"] if blocked else "Action limit" in result["hint"]


@pytest.mark.parametrize("name,arguments", [
    ("memory_read", {"mode": "image", "refs": ["p_missing"], "room": "living_room_1"}),
    ("memory_read", {"mode": "map", "view": "explored", "radius_m": 6}),
    ("memory_edit", {"operations": [{"op": "create", "description": "Entrance", "unknown": True}]}),
])
def test_session_rejects_invalid_node_arguments_before_dispatch(factory, monkeypatch, name, arguments):
    session = factory()
    monkeypatch.setattr(session.tools, "execute", lambda *_: pytest.fail("Invalid arguments reached tool execution"))
    reply, result = call(session, name, arguments)
    assert reply["result"]["isError"] and "error" in result
    assert session.memory.nodes == []


def test_readonly_question_cannot_write_or_capture(factory):
    session = factory()
    identifier, _ = node(session)
    session._qa = True
    before = session.memory.path.read_bytes()
    for name, args in (("observe", {"mode": "panorama"}), ("memory_edit", {"operations": [
            {"op": "update", "id": identifier, "description": "Changed"}]}),
            ("memory_edit", {"operations": [{"op": "update", "room": "living_room_1",
                                             "description": "Changed room"}]})):
        reply, _ = call(session, name, args)
        assert reply["result"]["isError"]
    reply, _ = call(session, "memory_read", {"mode": "image", "refs": [identifier]})
    assert not reply["result"]["isError"] and session.memory.path.read_bytes() == before


def test_snapshot_restores_node_refs_calibration_and_ids(factory, tmp_path):
    first = factory()
    identifier, panorama = node(first)
    reply, _ = call(first, "memory_edit", {"operations": [{"op": "update", "room": "living_room_1",
                                                          "description": "Sofa faces the TV; dining entrance by the shelf."}]})
    assert not reply["result"]["isError"]
    call(first, "move", {"mode": "steps", "actions": ["move_forward"]})
    snapshot = tmp_path / "snapshot"
    first.snapshot(snapshot)
    state = read_snapshot(snapshot)
    assert state["memory_type"] == "node"
    restored = factory(state=state)
    assert restored.transform == first.transform
    assert restored.current == first.current
    assert restored.memory.state == first.memory.state
    assert restored.observations.image(panorama)["path"] == first.observations.image(panorama)["path"]
    reply, directory = call(restored, "memory_read", {"mode": "text", "source": "rooms"})
    assert not reply["result"]["isError"] and not reply["result"]["images"]
    assert directory["rooms"] == [{"room": "living_room_1", "nodes": [identifier],
                                    "description": first.memory.rooms["living_room_1"]}]
    reply, mapped = call(restored, "memory_read", {"mode": "map", "room": "living_room_1"})
    assert not reply["result"]["isError"] and reply["result"]["images"]
    assert mapped["room"] == directory["rooms"][0]
    assert mapped["node_index"] == [{"id": identifier, "room": "living_room_1"}]
    with pytest.raises(ValueError, match="memory_type"):
        factory("entity", state)
    call(restored, "memory_edit", {"operations": [{"op": "delete", "id": identifier}]})
    new, _ = node(restored)
    assert new == "N002"


def test_relocation_breaks_microstep_path_and_snapshot(factory):
    session = factory()
    node(session)
    call(session, "move", {"mode": "steps", "actions": ["move_forward"]})
    at = len(session.environment.motion_trajectory)
    session.environment.relocate({"position": [15, 0, 0], "rotation_xyzw": [0, 0, 0, 1]})
    session.refresh_observation()
    assert session.environment.motion_trajectory_breaks == [at]
    assert session.environment.state()["motion_trajectory_breaks"] == [at]
    reply, _ = call(session, "memory_read", {"mode": "map", "view": "explored"})
    assert not reply["result"]["isError"]
    assert session.current["pose"]["pos"][0] > 1


def test_blank_bench_plan_does_not_require_published_walls(monkeypatch, tmp_path):
    from src.bench import samples
    config = read_config([REPO / "config/models/pilot.yaml", REPO / "config/environments/habitat.yaml",
                          REPO / "config/tasks/goat_single_floor.yaml"])
    config["model_overrides"] = {"memory_type": "node"}
    config["episodes"] = config["episodes"][:1]
    monkeypatch.setattr(samples, "resolve_scene", lambda *_: tmp_path)

    def forbidden(*args):
        raise AssertionError("Node mode must not load a wall map")

    monkeypatch.setattr(samples, "load_wall_map", forbidden)
    # Frozen episode reader is patched too: routine tests need no dataset installation.
    episode = EpisodeSpec(config["episodes"][0], "0", "goat", "fixture.glb", [0, 0, 0], [0, 0, 0, 1], [], "unused")
    monkeypatch.setattr(samples, "load_episodes", lambda *_: [episode])
    monkeypatch.setattr(samples, "SOURCES", {"goat": {"selection_file": "selected.json"}})
    (tmp_path / "selected.json").write_text("[]")
    config["models"] = {"sol": str(REPO / "config/models/sol.yaml")}
    result = samples.plan(tmp_path, config)
    assert result["jobs"][0]["map"] is None
    assert result["jobs"][0]["map_error"] is None


def test_custom_entity_sequence_rejects_node_scoring_before_running(tmp_path):
    from src.bench.sequence import run_sequence
    with pytest.raises(ValueError, match="evaluates entity memory"):
        run_sequence(REPO, {"memory_type": "node"}, tmp_path / "missing.json", None, tmp_path, {})
