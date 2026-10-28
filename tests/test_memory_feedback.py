"""Memory edit receipts remain inspectable, bounded, and safe to retry."""

import json

import numpy as np
from PIL import Image
import pytest

from test_super_tools import call, session


def create(identifier, image, pixel):
    return {"op": "create", "id": identifier, "type": "object", "fields": {
        "name": "chair", "location": {"image": image, "pixel": pixel}}}


def displayed_pixel(tile, source_pixel):
    matrix = np.array([*tile["pixel_to_source"], [0, 0, 1]])
    return np.linalg.solve(matrix, [*source_pixel, 1])[:2].tolist()


@pytest.mark.parametrize("derived", [False, True])
def test_location_preview_marks_source_and_round_trips_both_panels(session, derived):
    frame = session.current["frame"]
    reference, pixel = frame, [280, 200]
    if derived:
        crop = session.observations.crop(frame, [.25, .25, .75, .75])
        grid = session.observations.grid([frame, crop["image"]])
        reference, pixel = grid["image"], [760, 106]
    receipt, value = call(session, "memory_edit", {
        "operations": [create("O01", reference, pixel)]})

    assert not receipt["result"]["isError"]
    preview = value["preview"]
    assert preview["selected"] == [{"id": "O01", "image": frame, "pixel": [280, 200]}]
    assert len(receipt["result"]["images"]) == 1
    view = session.observations.image(preview["image"])
    assert receipt["result"]["images"] == [view["path"]]
    assert preview["size"] == [960, 520]
    source_tile, map_tile = view["sources"]
    point = displayed_pixel(source_tile, [280, 200])
    resolved, original = session.observations.resolve_pixel(preview["image"], point)
    assert resolved == frame and original == pytest.approx([280, 200])

    # A visible red ring surrounds the selected pixel, not just a metadata entry.
    with Image.open(session.workspace / view["path"]) as image:
        x, y = map(round, point)
        patch = np.asarray(image.crop((x - 10, y - 10, x + 11, y + 11)))
    assert np.any((patch[:, :, 0] > 160) & (patch[:, :, 1] < 100) & (patch[:, :, 2] < 100))

    entity = session.memory.entities[0]
    expected = session.spatial.geometry.resolve_frame(frame, [280, 200])["uv"]
    assert entity["point"] == expected
    maps = [json.loads(line) for line in
            (session.workspace / "observations/maps/index.jsonl").read_text().splitlines()]
    mapped = next(row for row in maps if row["view_id"] == preview["map"])
    map_pixel = np.linalg.solve(np.array([*mapped["pixel_to_uv"], [0, 0, 1]]),
                                [*expected, 1])[:2]
    mapped_reference, mapped_pixel = session.observations.resolve_pixel(
        preview["image"], displayed_pixel(map_tile, map_pixel))
    assert mapped_reference == preview["map"]
    assert session.spatial.geometry.resolve_map(mapped_reference, mapped_pixel)["uv"] == expected
    assert entity["evidence"] == [frame]
    assert session.environment.steps == 0


def test_text_update_and_cleared_location_do_not_generate_images(session):
    call(session, "memory_edit", {"operations": [create("O01", session.current["frame"], [320, 240])]})
    views_before = session.observations.views_path.read_bytes()
    for fields in ({"description": "Beside the doorway"}, {"point": None}):
        receipt, value = call(session, "memory_edit", {
            "operations": [{"op": "update", "id": "O01", "fields": fields}]})
        assert not receipt["result"]["isError"]
        assert not receipt["result"]["images"] and "preview" not in value
        assert session.observations.views_path.read_bytes() == views_before
    assert session.memory.entities[0]["point"] is None


def test_failed_batch_does_not_write_memory_or_generate_preview(session):
    frame = session.current["frame"]
    before = session.memory.path.read_bytes()
    receipt, value = call(session, "memory_edit", {"operations": [
        create("O01", frame, [320, 240]), create("O02", frame, [640, 480])]})
    assert receipt["result"]["isError"] and "No memory changes applied" in value["error"]
    assert session.memory.path.read_bytes() == before
    assert not receipt["result"]["images"] and "preview" not in value
    assert not session.observations.views


def test_batch_preview_uses_final_selection_and_excludes_deleted_entities(session):
    frame = session.current["frame"]
    receipt, value = call(session, "memory_edit", {"operations": [
        create("O01", frame, [280, 200]),
        {"op": "update", "id": "O01", "fields": {
            "location": {"image": frame, "pixel": [360, 260]}}},
        create("O02", frame, [320, 240]), {"op": "delete", "id": "O02"},
    ]})
    assert not receipt["result"]["isError"]
    assert value["preview"]["ids"] == ["O01"]
    assert value["preview"]["selected"] == [{"id": "O01", "image": frame, "pixel": [360, 260]}]
    assert session.memory.entities[0]["point"] == session.spatial.geometry.resolve_frame(frame, [360, 260])["uv"]
    assert {label["id"] for label in session.renderer.last_label_layout} == {"O01"}


@pytest.mark.parametrize("same_batch", [False, True])
def test_merge_inherited_position_has_map_feedback_for_surviving_id(session, same_batch):
    frame = session.current["frame"]
    operations = [create("O01", frame, [320, 240]),
                  {"op": "create", "id": "O02", "type": "object", "fields": {"name": "chair"}}]
    if not same_batch:
        receipt, _ = call(session, "memory_edit", {"operations": operations})
        assert not receipt["result"]["isError"]
        operations = []
    operations.append({"op": "merge", "keep": "O02", "remove": ["O01"]})
    receipt, value = call(session, "memory_edit", {"operations": operations})
    assert not receipt["result"]["isError"]
    assert [entity["id"] for entity in session.memory.entities] == ["O02"]
    assert session.memory.entities[0]["point"] == session.spatial.geometry.resolve_frame(frame, [320, 240])["uv"]
    assert value["preview"]["ids"] == ["O02"]
    assert value["preview"]["size"] == [480, 520]
    assert value["preview"]["selected"] == []
    assert len(receipt["result"]["images"]) == 1
    assert {label["id"] for label in session.renderer.last_label_layout} == {"O02"}


def test_many_sources_have_bounded_preview_but_all_positions_on_map(session):
    frames = [session.current["frame"]]
    for _ in range(2):
        _, observed = call(session, "observe", {})
        frames.append(observed["frame"])
    operations = [create(f"O{i + 1:02}", frame, [240 + 80 * i, 240])
                  for i, frame in enumerate(frames)]
    receipt, value = call(session, "memory_edit", {"operations": operations})
    preview = value["preview"]
    assert not receipt["result"]["isError"] and len(receipt["result"]["images"]) == 1
    assert preview["omitted_source_images"] == 1
    assert [mark["image"] for mark in preview["selected"]] == frames[:2]
    assert preview["ids"] == ["O01", "O02", "O03"]
    assert {label["id"] for label in session.renderer.last_label_layout} == set(preview["ids"])
    assert len(session.observations.image(preview["image"])["sources"]) == 3
    assert preview["size"] == [960, 520]


def test_duplicate_call_does_not_repeat_edit_or_preview(session):
    args = {"operations": [create("O01", session.current["frame"], [320, 240])]}
    first = session._tool("edit-once", "memory_edit", args)
    entities = session.memory.path.read_bytes()
    views = session.observations.views_path.read_bytes()
    assert session._tool("edit-once", "memory_edit", args) == first
    assert session.memory.path.read_bytes() == entities
    assert session.observations.views_path.read_bytes() == views


def test_preview_failure_does_not_report_a_failed_memory_transaction(session, monkeypatch):
    def fail_render(*args, **kwargs):
        raise OSError("private diagnostic details")

    monkeypatch.setattr(session.renderer, "render", fail_render)
    receipt, value = call(session, "memory_edit", {
        "operations": [create("O01", session.current["frame"], [320, 240])]})
    assert not receipt["result"]["isError"]
    assert [entity["id"] for entity in session.memory.entities] == ["O01"]
    assert "Memory saved" in value["preview_error"]
    assert "do not repeat create" in value["preview_error"]
    assert "private diagnostic details" not in value["preview_error"]
    assert "error" not in value and not receipt["result"]["images"]
