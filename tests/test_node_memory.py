import json

import pytest
from PIL import Image

from src.harness.memory import atomic_json
from src.harness.node_memory import NodeMemory
from src.harness.observations import Observations


@pytest.fixture
def store(tmp_path):
    workspace = tmp_path / "workspace"
    observations = Observations(workspace, {"origin": [0, 0], "size_m": 10})
    Image.new("RGB", (32, 24), (80, 100, 120)).save(workspace / "source.png")
    return NodeMemory(workspace, observations, observations.transform)


def panorama(store, group="p_one", pos=None, step=0):
    for yaw in range(0, 360, 60):
        store.observations.add(store.workspace / "source.png", group=group, yaw=yaw, step=step,
                               pose={"pos": pos or [.2, .2], "heading": yaw})
    return group


def create(store, group="p_one", description="窗旁的沙发与矮桌", room="客厅"):
    result = store.edit([{"op": "create", "description": description, "room": room}], group)
    assert "error" not in result, result
    return result["changes"][0]["id"]


def test_create_records_only_system_anchor_and_summary(store):
    panorama(store)
    identifier = create(store)
    assert identifier == "N001"
    assert store.nodes == [{"id": "N001", "anchor": "p_one", "panorama": "p_one", "room": "客厅",
                            "description": "窗旁的沙发与矮桌", "history": ["p_one"]}]
    assert store.get(identifier)["pose"] == {"pos": [.2, .2], "heading": 0}
    assert store.read(id=identifier)["nodes"][0]["history"] == ["p_one"]
    assert store.summary()["nodes"][0]["history_count"] == 1
    assert "error" in store.edit([{"op": "create", "id": "N099", "description": "手填位置"}], "p_one")
    assert "error" in store.edit([{"op": "update", "id": identifier, "anchor": "p_one"}])


def test_failed_batch_does_not_write_or_reserve_ids(store):
    panorama(store)
    before = store.path.read_bytes()
    failed = store.edit([{"op": "create", "description": "入口"}, {"op": "delete", "id": "N999"}], "p_one")
    assert "error" in failed
    assert store.path.read_bytes() == before
    assert not store.used_path.exists()
    assert create(store) == "N001"
    before, ledger = store.path.read_bytes(), store.used_path.read_bytes()
    failed = store.edit([{"op": "update", "id": "N001", "description": "改后摘要"},
                         {"op": "create", "description": " "}], "p_one")
    assert "error" in failed
    assert store.path.read_bytes() == before
    assert store.used_path.read_bytes() == ledger


def test_ids_increase_after_deletion_and_reopen_and_share_existing_ledger(store):
    panorama(store)
    atomic_json(store.used_path, ["R01", "O99", "N008"])
    assert create(store) == "N009"
    assert "error" not in store.edit([{"op": "delete", "id": "N009"}])
    restored = NodeMemory(store.workspace, store.observations, store.transform)
    assert create(restored) == "N010"
    assert {"R01", "O99", "N008", "N009", "N010"} == set(json.loads(store.used_path.read_text()))
    assert store.observations.panorama("p_one")["group"] == "p_one"


def test_refresh_keeps_anchor_and_description_and_cannot_creep(store):
    panorama(store)
    create(store)
    panorama(store, "p_near", [.29, .2], step=4)
    result = store.edit([{"op": "update", "id": "N001", "refresh": True}], "p_near")
    assert "error" not in result
    node = store.get("N001")
    assert node["anchor"] == "p_one"
    assert node["panorama"] == "p_near"
    assert node["description"] == "窗旁的沙发与矮桌"
    assert node["pose"]["pos"] == [.2, .2]
    assert node["observation_pose"]["pos"] == [.29, .2]
    assert node["history"] == ["p_one", "p_near"]
    panorama(store, "p_far", [.38, .2], step=8)
    before = store.path.read_bytes()
    result = store.edit([{"op": "update", "id": "N001", "refresh": True}], "p_far")
    assert "maximum is 1m" in result["error"]
    assert store.path.read_bytes() == before
    assert "error" not in store.edit([{"op": "update", "id": "N001", "description": "远程修订摘要"}])
    assert store.get("N001")["panorama"] == "p_near"


def test_refresh_accepts_boundary_and_does_not_duplicate_history(store):
    panorama(store)
    create(store)
    panorama(store, "p_boundary", [.3, .2], step=4)
    for _ in range(2):
        assert "error" not in store.edit([{"op": "update", "id": "N001", "refresh": True}], "p_boundary")
    assert store.get("N001")["history"] == ["p_one", "p_boundary"]


def test_create_and_refresh_require_current_complete_scene_panorama(store):
    result = store.edit([{"op": "create", "description": "入口"}])
    assert "current complete panorama" in result["error"]
    panorama(store)
    frame = store.observations.panorama("p_one")["frames"][0]["frame"]
    assert "error" in store.edit([{"op": "create", "description": "入口"}], frame)
    create(store)
    assert "error" in store.edit([{"op": "update", "id": "N001", "refresh": True}])
    store.observations.items[frame]["kind"] = "target"
    assert "error" in store.edit([{"op": "create", "description": "入口"}], "p_one")


def test_merge_retains_keep_pose_and_panorama_with_distant_historical_views(store):
    panorama(store)
    first = create(store)
    panorama(store, "p_near", [.29, .2], step=4)
    second = create(store, "p_near", "客厅中央")
    panorama(store, "p_far", [.38, .2], step=8)
    assert "error" not in store.edit([{"op": "update", "id": second, "refresh": True}], "p_far")
    merged = store.edit([{"op": "merge", "keep": first, "remove": [second], "description": "合并后的客厅摘要"}])
    assert "error" not in merged
    node = store.get(first)
    assert node["anchor"] == node["panorama"] == "p_one"
    assert node["history"] == ["p_one", "p_near", "p_far"]
    assert store.observations.panorama("p_far")["pose"]["pos"] == [.38, .2]
    with pytest.raises(ValueError, match="Unknown node"):
        store.get(second)
    assert len(store.observations.items) == 18


def test_merge_requires_description_and_room_resolution_and_rejects_far_anchors(store):
    panorama(store)
    first = create(store)
    panorama(store, "p_near", [.29, .2], step=4)
    second = create(store, "p_near", room="走廊")
    operation = {"op": "merge", "keep": first, "remove": [second], "description": "门口"}
    before = store.path.read_bytes()
    assert "Conflicting room" in store.edit([operation])["error"]
    assert store.path.read_bytes() == before
    assert "description" in store.edit([{key: value for key, value in {**operation, "room": "门厅"}.items()
                                        if key != "description"}])["error"]
    assert "error" not in store.edit([{**operation, "room": "门厅"}])
    assert store.get(first)["room"] == "门厅"
    panorama(store, "p_far", [.4, .2], step=8)
    third = create(store, "p_far")
    assert "maximum is 1m" in store.edit([{"op": "merge", "keep": first, "remove": [third],
                                            "description": "不应合并", "room": "客厅"}])["error"]


def test_read_filters_summaries_and_paginates(store):
    panorama(store)
    create(store, description="Blue sofa", room="Living Room")
    create(store, description="Door beside chair", room="Living Room")
    create(store, description="Sink", room="Kitchen")
    assert store.read(query="SOFA")["total"] == 1
    assert store.read(query="living room")["total"] == 2
    assert store.read(room="Living Room", query="chair")["total"] == 1
    page = store.read(limit=2)
    assert page["total"] == 3 and page["next_offset"] == 2
    assert store.read(offset=2)["nodes"][0]["id"] == "N003"
    assert "history" not in page["nodes"][0]
    assert "error" in store.read(id="N999")
    assert "error" in store.read(id="N001", query="sofa")


def test_single_node_history_paginates_without_changing_internal_detail(store):
    panorama(store)
    identifier = create(store)
    for step in range(1, 5):
        group = panorama(store, f"p_visit_{step}", step=step)
        assert "error" not in store.edit([{"op": "update", "id": identifier, "refresh": True}], group)
    expected = ["p_one", "p_visit_1", "p_visit_2", "p_visit_3", "p_visit_4"]
    first = store.read(id=identifier, limit=2)
    second = store.read(id=identifier, limit=2, offset=first["next_offset"])
    last = store.read(id=identifier, limit=2, offset=second["next_offset"])
    assert first["nodes"][0]["history"] == expected[:2]
    assert second["nodes"][0]["history"] == expected[2:4]
    assert last["nodes"][0]["history"] == expected[4:]
    assert all(page["nodes"][0]["history_total"] == 5 for page in (first, second, last))
    assert first["next_offset"] == 2 and second["next_offset"] == 4 and last["next_offset"] is None
    assert store.read(id=identifier, offset=10)["nodes"][0]["history"] == []
    assert store.get(identifier)["history"] == expected
    assert store.nodes[0]["history"] == expected


def test_summary_never_returns_more_than_ten_nodes(store):
    panorama(store)
    for _ in range(12):
        create(store)
    result = store.summary(limit=20)
    assert len(result["nodes"]) == 10
    assert result["total"] == 12


@pytest.mark.parametrize("change", [
    {"anchor": "p_missing"}, {"panorama": "p_missing"}, {"history": []},
    {"history": ["p_one", "p_one"]}, {"description": " "}, {"room": 1}, {"id": "O001"},
])
def test_restored_snapshot_rejects_invalid_records(store, change):
    panorama(store)
    create(store)
    atomic_json(store.path, [{**store.nodes[0], **change}])
    with pytest.raises(ValueError):
        NodeMemory(store.workspace, store.observations, store.transform)


def test_snapshot_ids_are_reserved_before_deleted(store):
    panorama(store)
    create(store)
    node = {**store.nodes[0], "id": "N057"}
    atomic_json(store.path, [node])
    restored = NodeMemory(store.workspace, store.observations, store.transform)
    restored.edit([{"op": "delete", "id": "N057"}])
    assert create(restored) == "N058"
