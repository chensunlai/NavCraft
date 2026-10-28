from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

from jsonschema import Draft7Validator
from PIL import Image
import pytest

from src.harness.node_memory import NodeMemory
from src.harness.observations import Observations
from src.harness.tools.node import NodeMemoryTools
from src.harness.tools.node_specs import memory_specifications, node_reference
from src.harness.tools.runtime import NavigationTools


@pytest.fixture
def schemas():
    return {spec["name"]: Draft7Validator(spec["parameters"]) for spec in memory_specifications()}


@pytest.mark.parametrize("arguments", [
    {}, {"id": "N001"}, {"mode": "text", "room": "living_room_1", "query": "sofa"},
    {"mode": "text", "source": "rooms"},
    {"mode": "text", "source": "rooms", "room": "living_room_1", "query": "TV", "limit": 2},
    {"mode": "text", "source": "observations", "offset": 3, "limit": 10},
    {"mode": "image", "refs": ["N001", "p_old"]},
    {"mode": "image", "refs": ["f_0001"], "crop": [0, 0, 1, 1]},
    {"mode": "map"}, {"mode": "map", "view": "ego", "radius_m": 6, "highlight": ["N001"]},
    {"mode": "map", "view": "explored"},
    {"mode": "map", "room": "living_room_1"},
    {"mode": "map", "room": "living_room_1", "view": "ego", "radius_m": 3},
])
def test_read_schema_accepts_separate_modes(schemas, arguments):
    assert schemas["memory_read"].is_valid(arguments)


@pytest.mark.parametrize("arguments", [
    {"mode": "image", "refs": ["N001"], "room": "kitchen"},
    {"mode": "text", "source": "observations", "query": "sofa"},
    {"mode": "text", "source": "rooms", "id": "N001"},
    {"mode": "text", "source": "rooms", "room": None},
    {"id": "N001", "room": "kitchen"},
    {"mode": "image", "refs": ["p_one", "p_two"], "crop": [0, 0, 1, 1]},
    {"mode": "image", "refs": ["p_one", "p_two", "p_three"]},
    {"mode": "map", "view": "explored", "radius_m": 6},
    {"mode": "map", "query": "sofa"}, {"mode": "text", "limit": 0},
    {"mode": "map", "radius_m": 0}, {"mode": "image", "refs": []},
    {"mode": "map", "room": " "}, {"mode": "map", "room": "living_room_1", "radius_m": 3},
])
def test_read_schema_rejects_mixed_or_unbounded_requests(schemas, arguments):
    assert not schemas["memory_read"].is_valid(arguments)


@pytest.mark.parametrize("operation", [
    {"op": "create", "description": "Entrance", "room": None},
    {"op": "update", "id": "N001", "refresh": True},
    {"op": "update", "id": "N001", "room": None},
    {"op": "update", "room": "living_room_1", "description": "Sofa facing a TV"},
    {"op": "update", "room": "living_room_1", "description": None},
    {"op": "delete", "id": "N001"},
    {"op": "merge", "keep": "N001", "remove": ["N002"], "description": "Entrance"},
])
def test_edit_schema_accepts_flat_node_operations(schemas, operation):
    assert schemas["memory_edit"].is_valid({"operations": [operation]})


@pytest.mark.parametrize("operation", [
    {"op": "create", "id": "N001", "description": "Entrance"},
    {"op": "create", "description": "  "},
    {"op": "create", "description": "Entrance", "room": " "},
    {"op": "update", "id": "N001", "refresh": False},
    {"op": "update", "id": "N001", "fields": {"description": "Entrance"}},
    {"op": "merge", "keep": "N001", "remove": ["N002"]},
    {"op": "delete", "id": "N001", "description": "Entrance"},
    {"op": "update", "room": "living_room_1"},
    {"op": "update", "room": "living_room_1", "description": " "},
    {"op": "update", "room": "living_room_1", "description": "a" * 1001},
    {"op": "update", "room": "living_room_1", "description": "Layout", "refresh": True},
])
def test_edit_schema_rejects_old_or_incomplete_operations(schemas, operation):
    assert not schemas["memory_edit"].is_valid({"operations": [operation]})


def test_node_targets_allow_unbounded_calibrated_coordinates():
    schema = Draft7Validator(node_reference())
    assert schema.is_valid({"node": "N001"})
    assert schema.is_valid({"uv": [-.4, 1.8]})
    assert schema.is_valid({"image": "p_first", "pixel": [100, 200]})
    assert not schema.is_valid({"entity": "O01"})
    assert not schema.is_valid({"node": "N001", "uv": [.5, .5]})


class RecordingTrace:
    def __init__(self):
        self.blobs = {}
        self.events = []

    def store(self, value):
        digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        self.blobs[digest] = deepcopy(value)
        return {"$trace_blob": digest, "encoding": "json"}

    def emit(self, name, value):
        self.events.append((name, deepcopy(value)))


@pytest.fixture
def node_tools(tmp_path):
    transform = {"origin": [0, 0], "size_m": 20, "floor_height": 0, "walls": []}
    image = tmp_path / "source.png"
    Image.new("RGB", (8, 8), "gray").save(image)
    observations = Observations(tmp_path, transform)
    observations.add(image, kind="target")
    frames = [observations.add(image, step=3, pose={"pos": [.5, .5], "heading": yaw},
                               group="p_first", yaw=yaw) for yaw in range(0, 360, 60)]
    observations.add(image, step=4, pose={"pos": [.5, .6], "heading": 0})
    memory = NodeMemory(tmp_path, observations, transform, private_dir=tmp_path / "private")
    session = SimpleNamespace(memory=memory, observations=observations, current=frames[0],
                              trace=RecordingTrace(), _qa=False,
                              environment=SimpleNamespace(observe=Mock(), panorama=Mock()))
    session._reply = lambda value, refs=(), labels=(), error=False: {
        "value": value, "refs": list(refs), "labels": list(labels), "error": error}
    navigation = SimpleNamespace(
        images=Mock(return_value={"opened": True}),
        image_reply=Mock(return_value={"opened": True}),
        map_value=Mock(return_value={"map": "observations/maps/m_0001.png", "view_id": "m_0001"}))
    return NodeMemoryTools(session, navigation)


def create_node(tools):
    reply = tools.edit([{"op": "create", "description": "Entrance with a sofa", "room": "living_room_1"}])
    assert not reply["error"]
    return reply["value"]["changes"][0]["id"]


def test_create_binds_existing_panorama_and_records_before_after_without_images(node_tools):
    identifier = create_node(node_tools)
    session = node_tools.session
    assert identifier == "N001"
    assert session.memory.get(identifier)["anchor"] == "p_first"
    assert session.memory.get(identifier)["panorama"] == "p_first"
    session.environment.observe.assert_not_called()
    session.environment.panorama.assert_not_called()
    node_tools.navigation.images.assert_not_called()
    kind, event = session.trace.events[0]
    assert kind == "memory_changed"
    assert session.trace.blobs[event["before"]["$trace_blob"]] == {"nodes": [], "rooms": {}}
    assert session.trace.blobs[event["after"]["$trace_blob"]]["nodes"][0]["id"] == identifier


def test_missing_current_panorama_does_not_capture_or_mutate_memory(node_tools):
    session = node_tools.session
    session.current = {"step": 8, "pose": {"pos": [.7, .7], "heading": 0}}
    reply = node_tools.edit([{"op": "create", "description": "A room"}])
    assert reply["error"] and reply["value"]["committed"] is False
    assert "observe" in reply["value"]["error"]
    assert session.memory.nodes == [] and session.trace.events == []
    session.environment.observe.assert_not_called()
    session.environment.panorama.assert_not_called()


def test_text_queries_and_history_attach_no_images(node_tools):
    identifier = create_node(node_tools)
    detail = node_tools.read(id=identifier)
    assert detail["value"]["nodes"][0]["history"] == ["p_first"]
    assert detail["refs"] == [] and not detail["error"]
    history = node_tools.read(source="observations", limit=1)
    assert history["value"]["total"] == 2
    assert history["value"]["next_offset"] == 1
    event = history["value"]["events"][0]
    assert event["image"] == "p_first" and event["nodes"] == [identifier]
    second = node_tools.read(source="observations", offset=1)
    assert second["value"]["events"][0]["nodes"] == []
    assert history["refs"] == [] and second["refs"] == []
    node_tools.navigation.images.assert_not_called()


def test_node_image_and_crop_delegate_to_existing_image_reader(node_tools):
    identifier = create_node(node_tools)
    reply = node_tools.read(mode="image", refs=[identifier], crop=[0, 0, .5, 1])
    assert reply == {"opened": True}
    rows = node_tools.navigation.image_reply.call_args.args[0]
    nodes = node_tools.navigation.image_reply.call_args.kwargs["nodes"]
    assert len(rows) == len(nodes) == 1
    assert rows[0]["image"].startswith("i_")
    assert rows[0]["group"] == "p_first"
    assert rows[0]["pose"] == nodes[0]["observation_pose"]
    assert nodes[0]["id"] == identifier
    # The selected crop pixel remains tied to the original observed frame.
    assert node_tools.session.observations.resolve_pixel(rows[0]["image"], [2, 28]) == ("f_0002", [2, 2])
    node_tools.session.environment.observe.assert_not_called()
    node_tools.session.environment.panorama.assert_not_called()


def test_refreshed_node_keeps_anchor_and_capture_pose_distinct(node_tools):
    identifier = create_node(node_tools)
    session = node_tools.session
    image = session.observations.workspace / "source.png"
    refreshed = [session.observations.add(image, step=8, pose={"pos": [.52, .5], "heading": yaw},
                                         group="p_refreshed", yaw=yaw) for yaw in range(0, 360, 60)]
    session.current = refreshed[0]
    assert not node_tools.edit([{"op": "update", "id": identifier, "refresh": True}])["error"]
    before = (len(session.observations.items), session.memory.path.read_bytes())
    node_tools.read(mode="image", refs=[identifier, "f_0001"])
    rows = node_tools.navigation.image_reply.call_args.args[0]
    nodes = node_tools.navigation.image_reply.call_args.kwargs["nodes"]
    assert rows[0]["image"].startswith("i_") and rows[0]["group"] == "p_refreshed"
    assert rows[0]["pose"]["pos"] == [.52, .5]
    assert nodes[0]["id"] == identifier and nodes[0]["pose"]["pos"] == [.5, .5]
    assert rows[1]["image"] == "f_0001" and nodes[1] is None
    assert before == (len(session.observations.items), session.memory.path.read_bytes())
    node_tools.read(mode="image", refs=[identifier], crop=[0, 0, .5, 1])
    row = node_tools.navigation.image_reply.call_args.args[0][0]
    assert row["pose"]["pos"] == [.52, .5] and row["step"] == 8
    assert row["group"] == "p_refreshed"
    assert before == (len(session.observations.items), session.memory.path.read_bytes())
    session.environment.observe.assert_not_called()
    session.environment.panorama.assert_not_called()


def test_node_identity_and_actual_capture_pose_reach_image_receipt(node_tools):
    identifier = create_node(node_tools)
    session = node_tools.session
    image = session.observations.workspace / "source.png"
    frames = [session.observations.add(image, step=9, pose={"pos": [.52, .5], "heading": yaw},
                                      group="p_new", yaw=yaw) for yaw in range(0, 360, 60)]
    session.current = frames[0]
    node_tools.edit([{"op": "update", "id": identifier, "refresh": True}])
    node_tools.navigation.image_reply = lambda rows, nodes=None: NavigationTools.image_reply(
        SimpleNamespace(session=session), rows, nodes=nodes)

    reply = node_tools.read(mode="image", refs=[identifier, "f_0001"])
    first, second = reply["value"]["images"]
    assert first["node"] == identifier and first["image"].startswith("i_") and first["group"] == "p_new"
    assert first["pose"]["pos"] == [.52, .5]
    assert first["anchor_pose"]["pos"] == [.5, .5]
    assert identifier in reply["labels"][0] and first["image"] in reply["labels"][0]
    assert "node" not in second and "anchor_pose" not in second
    assert "TARGET" in reply["labels"][1]

    cropped = node_tools.read(mode="image", refs=[identifier], crop=[0, 0, .5, 1])
    row = cropped["value"]["images"][0]
    assert row["image"].startswith("i_") and row["node"] == identifier
    assert row["pose"] == first["pose"] and row["anchor_pose"] == first["anchor_pose"]
    assert row["image"] in cropped["labels"][0]


@pytest.mark.parametrize("arguments,expected", [
    ({}, {"around_self": 6}), ({"view": "ego", "radius_m": 3}, {"ego": 3}),
    ({"view": "explored"}, "explored"),
])
def test_map_modes_delegate_with_exact_view(node_tools, arguments, expected):
    reply = node_tools.read(mode="map", **arguments)
    node_tools.navigation.map_value.assert_called_once_with(view=expected, highlight=None)
    assert reply["refs"] == ["observations/maps/m_0001.png"]
    assert reply["labels"] == ["[MAP m_0001 | model-written nodes]"]


def test_remote_description_edit_never_requests_current_panorama(node_tools):
    identifier = create_node(node_tools)
    node_tools.session.observations.current_panorama = Mock(side_effect=AssertionError("Unexpected capture lookup"))
    reply = node_tools.edit([{"op": "update", "id": identifier, "description": "Corrected room description"}])
    assert reply["value"] == {"changes": [{"op": "update", "id": identifier}], "committed": True}
    assert reply["refs"] == []
    assert node_tools.session.memory.get(identifier)["description"] == "Corrected room description"


def test_room_description_read_edit_and_clear_need_no_capture(node_tools):
    identifier = create_node(node_tools)
    session = node_tools.session
    session.observations.current_panorama = Mock(side_effect=AssertionError("Unexpected capture lookup"))
    before = len(session.observations.items)
    reply = node_tools.edit([{"op": "update", "room": "living_room_1",
                              "description": "Sofa opposite television; dining table beside window."}])
    assert reply["value"] == {"changes": [{"op": "update", "room": "living_room_1"}], "committed": True}
    directory = node_tools.read(source="rooms")
    room = directory["value"]["rooms"][0]
    assert room == {"room": "living_room_1", "description": session.memory.rooms["living_room_1"],
                    "nodes": [identifier]}
    assert directory["refs"] == [] and not directory["error"]
    assert node_tools.read(room="living_room_1")["value"]["room"] == room
    assert node_tools.read(source="rooms", query="television")["value"]["total"] == 1
    event = session.trace.events[-1][1]
    assert session.trace.blobs[event["before"]["$trace_blob"]]["rooms"] == {}
    assert session.trace.blobs[event["after"]["$trace_blob"]]["rooms"] == session.memory.rooms
    node_tools.edit([{"op": "update", "room": "living_room_1", "description": None}])
    assert node_tools.read(source="rooms")["value"]["rooms"] == [
        {"room": "living_room_1", "description": None, "nodes": [identifier]}]
    assert session.memory.get(identifier)["description"] == "Entrance with a sofa"
    assert len(session.observations.items) == before
    session.environment.observe.assert_not_called()
    session.environment.panorama.assert_not_called()


def test_mixed_node_and_room_edits_have_paired_receipts_and_full_trace(node_tools):
    reply = node_tools.edit([
        {"op": "create", "room": "living_room_1", "description": "Entrance with a sofa"},
        {"op": "update", "room": "living_room_1", "description": "Sofa facing television"}])
    assert not reply["error"] and reply["value"]["committed"]
    first, second = reply["value"]["changes"]
    assert first == {"op": "create", "id": "N001", "room": "living_room_1", "panorama": "p_first"}
    assert second == {"op": "update", "room": "living_room_1"}
    session = node_tools.session
    assert session.trace.blobs[session.trace.events[-1][1]["after"]["$trace_blob"]] == session.memory.state


def test_room_map_highlights_members_and_preserves_explicit_view(node_tools):
    first = create_node(node_tools)
    second = create_node(node_tools)
    node_tools.edit([{"op": "update", "room": "living_room_1", "description": "Shared living area"}])
    reply = node_tools.read(mode="map", room="living_room_1", highlight=[second])
    node_tools.navigation.map_value.assert_called_once_with(view="explored", highlight=[second, first])
    assert reply["value"]["room"] == {
        "room": "living_room_1", "description": "Shared living area", "nodes": [first, second]}
    assert len(reply["refs"]) == 1
    node_tools.navigation.map_value.reset_mock()
    node_tools.read(mode="map", room="living_room_1", view="ego", radius_m=3)
    node_tools.navigation.map_value.assert_called_once_with(view={"ego": 3}, highlight=[first, second])
    node_tools.edit([{"op": "update", "id": first, "room": "kitchen"}])
    node_tools.navigation.map_value.reset_mock()
    node_tools.read(mode="map", room="living_room_1")
    node_tools.navigation.map_value.assert_called_once_with(view="explored", highlight=[second])


def test_room_map_rejects_unknown_group_and_keeps_described_empty_room(node_tools):
    identifier = create_node(node_tools)
    missing = node_tools.read(mode="map", room="missing")
    assert missing["error"] and missing["refs"] == []
    node_tools.navigation.map_value.assert_not_called()
    node_tools.edit([{"op": "update", "room": "living_room_1", "description": "Known room layout"},
                     {"op": "update", "id": identifier, "room": None}])
    reply = node_tools.read(mode="map", room="living_room_1")
    assert not reply["error"] and reply["value"]["room"]["nodes"] == []
    node_tools.navigation.map_value.assert_called_once_with(view="explored", highlight=[])


def test_room_edit_respects_qa_readonly_boundary(node_tools):
    create_node(node_tools)
    session = node_tools.session
    session._qa = True
    before = session.memory.path.read_bytes()
    with pytest.raises(ValueError, match="existing-evidence"):
        node_tools.edit([{"op": "update", "room": "living_room_1", "description": "Changed layout"}])
    assert session.memory.path.read_bytes() == before
    assert not node_tools.read(source="rooms")["error"]


def test_qa_and_multiple_captures_are_rejected_before_writes(node_tools):
    node_tools.session._qa = True
    with pytest.raises(ValueError, match="existing-evidence"):
        node_tools.edit([{"op": "create", "description": "Entrance"}])
    node_tools.session._qa = False
    with pytest.raises(ValueError, match="at most one"):
        node_tools.edit([{"op": "create", "description": "Entrance"},
                         {"op": "create", "description": "Same entrance"}])
    assert node_tools.session.memory.nodes == [] and node_tools.session.trace.events == []


def test_direct_reader_validates_modes_before_dispatch(node_tools):
    with pytest.raises(ValueError, match="Invalid memory_read"):
        node_tools.read(mode="image", refs=["p_first"], room="living_room_1")
    node_tools.navigation.images.assert_not_called()
