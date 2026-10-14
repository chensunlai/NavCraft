"""Python callbacks preserve real environment state without a bridge subprocess."""

import importlib.util
import json
from pathlib import Path
import sys
from threading import Event, get_ident

import pytest

from test_pi_bridge import Bridge, init_args, initialized
from test_pi_memory import workspace


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "src/harness-pi/python"
SPEC = importlib.util.spec_from_file_location("pi_python_tools", PYTHON / "tools/__init__.py",
                                             submodule_search_locations=[str(PYTHON / "tools")])
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
sys.path.insert(0, str(PYTHON))
try:
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path.remove(str(PYTHON))
ToolSet = MODULE.ToolSet


def payload(response):
    return json.loads(response["result"]["content"][0]["text"])


def test_query_tools_read_public_images_and_reject_private_paths(workspace):
    tools = ToolSet(ROOT, workspace)
    assert {spec["name"] for spec in tools.specs()} == {"view_images", "render_map"}
    answer = tools.call("view", "view_images", {"refs": ["f_001"]})
    assert answer["result"]["images"] == ["observations/f_001.png"]
    assert payload(answer)["images"][0]["view"]["right_degrees"] == 60
    refused = tools.call("private", "view_images", {"refs": ["../private/answer.png"]})
    assert refused["result"]["isError"]
    assert "checkpoint" not in refused


def test_render_returns_current_memory_and_retries_do_not_create_artifacts(workspace):
    tools = ToolSet(ROOT, workspace)
    args = {"map_id": "map_L1", "highlight_ids": ["O01"]}
    first = tools.call("render", "render_map", args)
    assert first["checkpoint"]["revision"] == 1
    assert first["checkpoint"]["images"] == first["result"]["images"]
    assert payload(first)["pose_at_frame"] == "f_001"
    assert tools.call("render", "render_map", args) == first
    assert tools.state["revision"] == 1
    changed = tools.call("render", "render_map", {"map_id": "missing"})
    assert changed["result"]["isError"] and tools.state["revision"] == 1


def test_navigation_is_direct_synchronous_and_exactly_once(initialized, monkeypatch):
    bridge, initial, root = initialized
    tools = ToolSet(ROOT, root / "workspace", bridge, {"current": initial})
    owner = get_ident()
    original = bridge.env.step

    def same_thread(action):
        assert get_ident() == owner
        return original(action)

    monkeypatch.setattr(bridge.env, "step", same_thread)
    args = {"actions": ["move_forward", "turn_left"]}
    first = tools.call("move", "nav_step", args)
    assert payload(first)["steps"] == 2
    assert tools.call("move", "nav_step", args) == first
    assert bridge.env.task.steps == 2
    assert tools.state["current"]["task"] == initial["task"]
    assert not first["control"]["end_run"]
    assert "world_from_optical" not in json.dumps(first)
    assert tools.call("move", "nav_step", {"actions": ["turn_left"]})["result"]["isError"]
    assert bridge.env.task.steps == 2


def test_panorama_preserves_goal_and_replaces_old_observation(initialized):
    bridge, initial, root = initialized
    tools = ToolSet(ROOT, root / "workspace", bridge, {"current": initial})
    answer = tools.call("pano", "nav_observe", {"mode": "panorama"})
    assert len(answer["result"]["images"]) == 1
    assert len(payload(answer)["panorama_views"]) == 6
    assert payload(answer)["restored_camera"]
    assert tools.state["current"]["task"] == initial["task"]
    assert "observation" not in tools.state["current"]
    assert answer["checkpoint"]["images"] == answer["result"]["images"]
    current = tools.call("single", "nav_observe", {"mode": "single"})
    assert len(tools.state["current"]["views"]) == 1
    assert "restored_camera" not in tools.state["current"]
    assert payload(current)["observation"]["id"] != initial["observation"]["id"]
    tools.call("move", "nav_step", {"actions": ["turn_left"]})
    assert "views" not in tools.state["current"]
    assert "mode" not in tools.state["current"]


def test_goal_change_stops_remaining_calls_until_next_turn(tmp_path):
    goals = [{"modality": "description", "instruction": str(i), "positions": [[0, 0, -1]]}
             for i in range(2)]
    bridge = Bridge()
    initial = bridge.init(**init_args(tmp_path, family="goat", goals=goals,
                                     task_overrides={"max_subtask_steps": 1}))
    try:
        tools = ToolSet(ROOT, tmp_path / "workspace", bridge, {"current": initial})
        switched = tools.call("move", "nav_step", {"actions": ["turn_left", "turn_left"]})
        assert switched["control"] == {"stop_remaining_tools": True, "end_run": False}
        assert payload(switched)["task"]["goal_id"] == 1
        denied = tools.call("too-soon", "nav_step", {"actions": ["turn_left"]})
        assert denied["result"]["isError"] and bridge.env.task.steps == 1
        tools.turn_start()
        ended = tools.call("finish", "nav_finish", {"goal_id": 1})
        assert ended["control"] == {"stop_remaining_tools": True, "end_run": True}
        assert tools.state["done"]
        assert tools.call("after", "nav_observe", {"mode": "single"})["result"]["isError"]
    finally:
        bridge.close()


def test_cancel_interrupts_between_actual_actions_and_retains_feedback(initialized, monkeypatch):
    bridge, initial, root = initialized
    cancel = Event()
    tools = ToolSet(ROOT, root / "workspace", bridge, {"current": initial}, cancel_event=cancel)
    original = bridge.env.step

    def cancel_after_first(action):
        transition = original(action)
        cancel.set()
        return transition

    monkeypatch.setattr(bridge.env, "step", cancel_after_first)
    answer = tools.call("move", "nav_step", {"actions": ["turn_left"] * 6})
    data = payload(answer)
    assert data["interruption"] == "cancelled"
    assert len(data["executed_actions"]) == data["steps"] == 1
    assert not data["terminated"] and not data["truncated"]
    assert answer["control"] == {"stop_remaining_tools": True, "end_run": True}
    assert tools.call("after", "nav_step", {"actions": ["turn_left"]})["result"]["isError"]
    assert bridge.env.task.steps == 1


def test_bridge_cancel_before_batch_retains_current_goal(initialized):
    bridge, initial, _ = initialized
    bridge.cancel_event = Event()
    bridge.cancel_event.set()
    data = bridge.step(["move_forward"])
    assert data["interruption"] == "cancelled" and not data["executed_actions"]
    assert data["task"] == initial["task"]
    assert data["reason"] is None and not data["truncated"]
    assert bridge.env.task.steps == 0


def test_local_tools_do_not_offer_navigation_finish(initialized):
    bridge, initial, root = initialized
    tools = ToolSet(ROOT, root / "workspace", bridge, {"current": initial}, allow_finish=False)
    assert "nav_finish" not in {spec["name"] for spec in tools.specs()}
    rejected = tools.call("unknown", "nav_finish", {"goal_id": 0})
    assert rejected["result"]["isError"] and bridge.env.task.steps == 0
    assert tools.call("unknown", "nav_finish", {"goal_id": 0}) == rejected
