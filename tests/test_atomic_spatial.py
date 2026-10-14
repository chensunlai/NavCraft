"""Fixture-only integrity checks; never imports Habitat or sends model requests."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.dataset.atomic.schema import AtomicTask, COVERAGE, ALLOWED_TOOLS, checked_resource, score_answer
from src.dataset.atomic.geometry import crossed_opening, horizontal, local_event, map_point
from src.dataset.atomic.export import check_dialogue, portable, visible_messages
from src.dataset.atomic.runtime import AtomicTools, parse_answer
from src.legacy.harness.agent.base import ToolCall


def test_coverage_and_serialization(tmp_path):
    assert sum(len(row[2]) * 2 for row in COVERAGE.values()) == 28
    task = AtomicTask("x", "A01", "simple_qa", "single", "where", [], {}, "t0", "hidden",
                      {"choice": "A", "start_pose": {"position": [1, 2, 3]}})
    task.save(tmp_path)
    public = json.loads((tmp_path / "task.json").read_text())
    assert "private" not in public and "start_pose" not in public
    assert json.loads((tmp_path / "private/labels.json").read_text())["choice"] == "A"


def test_resource_bank_blocks_private_and_traversal(tmp_path):
    (tmp_path / "public.md").write_text("allowed")
    (tmp_path / "private").mkdir()
    (tmp_path / "private/key.json").write_text("secret")
    bank = {"notes": "public.md", "bad": "private/key.json", "escape": "../outside.md"}
    assert checked_resource("notes", bank, tmp_path).name == "public.md"
    for ref in ("../../etc/passwd", "new_frame", "bad", "escape"):
        with pytest.raises(ValueError):
            checked_resource(ref, bank, tmp_path)


def test_form_permission_checked_even_for_forged_call(tmp_path):
    world = SimpleNamespace(sim=SimpleNamespace(pose=lambda: {"position": [0, 0, 0]}))
    task = AtomicTask("x", "A03", "retrieval_qa", "single", "q", [], {}, "t0", "start", {})
    tools = AtomicTools(task, tmp_path, tmp_path / "attempt", world)
    result = tools.call(ToolCall("one", "nav_step", {"actions": ["move_forward"]}))
    assert not result.ok and len(tools.history) == 1
    assert "nav_observe" not in ALLOWED_TOOLS["retrieval_qa"]


def test_answer_time_is_part_of_label():
    private = {"choice": "B", "anchor": "before_motion", "label_method": "geometry"}
    assert not score_answer({"choice": "B", "anchor": "after_motion"}, private)["passed"]
    assert score_answer({"choice": "B", "anchor": "before_motion"}, private)["passed"]


def test_never_score_a_reasoning_draft_as_final_answer():
    draft = {"type": "reasoning", "content": [{"type": "output_text", "text": '{"choice":"A"}'}]}
    answer = {"type": "message", "content": [{"type": "output_text", "text": '{"choice":"B"}'}]}
    assert parse_answer([draft, answer]) == {"choice": "B"}
    assert parse_answer([draft]) == {}


def test_coordinate_orientation_and_calibration():
    calibration = SimpleNamespace(viewbox=(0, 0, 100, 100), panels=[
        SimpleNamespace(x_scale=10, x_offset=50, y_scale=-10, y_offset=50)])
    assert horizontal([2, .1, -3]) == [2, 3]
    assert map_point(calibration, horizontal([2, .1, -3])) == [.7, .2]


def test_opening_crossing_excludes_parallel_and_wrong_door():
    door = [[0, 0], [1, 0]]
    assert crossed_opening([.5, -1], [.5, 1], door)
    assert not crossed_opening([2, -1], [2, 1], door)
    assert not crossed_opening([.2, .1], [.8, .1], door)
    assert not crossed_opening([0, -1], [0, 1], door)


def test_ia_arrival_alone_is_not_success():
    labels = {"door_segment": [[0, 0], [1, 0]], "destination": [.5, .5], "radius": .1}
    h = lambda x, y: {"pose": {"position": [x, 0, -y]}}
    assert not local_event(labels, [h(.4, .5), h(.5, .5)])["passed"]
    assert local_event(labels, [h(.5, -.5), h(.5, 0), h(.5, .5)])["passed"]
    assert not local_event(labels, [h(.5, -4), h(.5, .5)])["passed"]


def test_exports_have_no_reasoning_and_no_future_context(tmp_path):
    initial = {"role": "user", "content": [{"type": "input_text", "text": "q"}]}
    call = {"type": "function_call", "call_id": "a", "name": "read_text", "arguments": "{}"}
    result = {"type": "function_call_output", "call_id": "a", "output": "ok"}
    record = {"messages": [initial, call, result], "decisions": [
        {"input": [initial], "output": [call]}]}
    check_dialogue(record, tmp_path)
    record["decisions"][0]["input"] = [initial, result]
    with pytest.raises(ValueError, match="future"):
        check_dialogue(record, tmp_path)
    assert visible_messages([{"type": "reasoning", "encrypted_content": "secret"}, call]) == [call]
    assert portable({"image_path": str(tmp_path / "x.png")}, tmp_path) == {"image_path": "x.png"}
