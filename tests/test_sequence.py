from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.bench import sequence
from src.environment import Environment, MotionResult
from src.environment.fake import FakeSimulator
from src.harness.session import PromptResult


ROOMS = {key: {"polygon": polygon} for key, polygon in {
    "A": [[-1, 0], [1, 0], [1, -1], [-1, -1]],
    "B": [[-1, 0], [1, 0], [1, 1], [-1, 1]],
    "C": [[-1, 1], [1, 1], [1, 2], [-1, 2]],
}.items()}
FIRST = {"kind": "portal", "id": "D01", "segment_xz": [[-.5, 0], [.5, 0]],
         "from_room": "A", "to_room": "B"}
SECOND = {"kind": "portal", "id": "D02", "segment_xz": [[-.5, -1], [.5, -1]],
          "from_room": "B", "to_room": "C"}


def scenario(rounds=12):
    result = {"protocol": "fixture", "scene_id": "fake.glb", "start_position": [0, 0, 0],
              "start_rotation": [0, 0, 0, 1], "environment": {"gpu": 2, "scenes": "old-host"},
              "private_rooms": [], "rounds": []}
    position = result["start_position"]
    for index in range(1, rounds + 1):
        row = {"id": f"round{index}", "kind": "explore" if index == 1 else "navigation",
               "instruction": f"Only the public request {index}", "max_actions": 20 + index,
               "max_requests": 5 + index, "seconds": 60 + index, "route_events": []}
        if index >= 10:
            position = [index, 0, index]
            row["teleport"] = {"position": position, "rotation_xyzw": [0, 0, 0, 1]}
        if index > 1:
            row.update(positions=[position], success_distance=1.)
        result["rounds"].append(row)
    return result


class ScriptedSession:
    def __init__(self, directory, environment, callback):
        self.directory, self.environment, self.callback = directory, environment, callback
        self.workspace, self.private = directory / "workspace", directory / "private"
        self.memory = SimpleNamespace(entities=[])
        self.observations = SimpleNamespace(items={})
        self.trace = SimpleNamespace(calls=[])
        self.prompts, self.refreshes, self.closed = [], [], False
        self.refresh_observation()

    def refresh_observation(self):
        observation = self.environment.observe()
        frame = observation.frame_id
        relative = f"observations/{frame}.png"
        path = self.workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(observation.rgb.read_bytes())
        self.observations.items[frame] = {"path": relative}
        self.current = {"frame": frame}
        self.refreshes.append(self.environment.position())

    def memory_operations(self, operations):
        self.trace.calls.append({"name": "memory_edit", "isError": False})
        event = {"type": "public_tool_reply", "data": {"name": "memory_edit",
                 "args": {"operations": operations}, "result": {"isError": False}}}
        with (self.private / "trace.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")

    def prompt(self, text, limits):
        before = self.environment.steps
        self.prompts.append((text, limits, self.environment.position()))
        options = self.callback(self, len(self.prompts)) or {}
        submitted, reason = options.get("submitted", True), options.get("reason", "submitted")
        if submitted:
            self.trace.calls.append({"name": "submit", "isError": False})
        return PromptResult(str(len(self.prompts)), submitted, options.get("answer", ""), reason,
                            1, self.environment.steps - before, "private/prompt.json", .1,
                            options.get("error"))

    def close(self):
        self.closed = True


def execute(tmp_path, monkeypatch, specification, callback, simulator=FakeSimulator):
    inputs, environments, sessions = [], [], []

    def environment_factory(episode, config, directory):
        inputs.append(config)
        value = Environment(episode, config, directory, simulator)
        environments.append(value)
        return value

    def session_factory(repo, config, directory, environment, scene_map):
        value = ScriptedSession(directory, environment, callback)
        sessions.append(value)
        return value

    monkeypatch.setattr(sequence, "Environment", environment_factory)
    monkeypatch.setattr(sequence, "open_session", session_factory)
    source, map_path = tmp_path / "scenario.json", tmp_path / "map.json"
    source.write_text(json.dumps(specification))
    map_path.write_text('{"origin":[-5,-5],"size_m":20,"walls":[]}')
    original = source.read_bytes()
    result = sequence.run_sequence(tmp_path, {"model": "fixture", "reasoning_effort": "high"},
                                   source, map_path, tmp_path / "run", {"gpu": 1})
    assert source.read_bytes() == original
    return result, environments, sessions, inputs


def test_twelve_prompts_share_environment_memory_and_three_explicit_relocations(tmp_path, monkeypatch):
    spec = scenario()

    def callback(session, index):
        if index == 1:
            session.memory.entities.append({"id": "R01", "type": "room", "name": "remember me",
                                            "evidence": [session.current["frame"]]})
            session.memory_operations([{"op": "create", "id": "R01"}])
        else:
            assert session.memory.entities[0]["name"].startswith("remember me")
            if index == 2:
                session.memory.entities[0]["name"] += " updated"
                session.memory_operations([{"op": "update", "id": "R01"}])
        if index in (3, 4, 5):
            return {"submitted": False, "reason": {3: "action_budget", 4: "request_budget", 5: "unsubmitted"}[index]}

    result, environments, sessions, configs = execute(tmp_path, monkeypatch, spec, callback)
    assert result["status"] == "completed" and result["completed_rounds"] == 12
    assert len(environments) == len(sessions) == 1
    session, environment = sessions[0], environments[0]
    assert len(session.prompts) == 12 and len(session.refreshes) == 4
    assert session.refreshes[1:] == [[10, 0, 10], [11, 0, 11], [12, 0, 12]]
    assert environment.steps == 0 and environment.path_length == 0
    assert len(result["environment"]["trajectory_breaks"]) == 3
    assert configs[0]["gpu"] == 1 and "scenes" not in configs[0]
    assert all(text.startswith("你已被重新放置") for text, _, _ in session.prompts[9:])
    assert all("positions" not in text and "success_distance" not in text for text, _, _ in session.prompts)
    for (_, limits, _), row in zip(session.prompts, spec["rounds"]):
        assert (limits.max_actions, limits.max_requests, limits.seconds) == (row["max_actions"], row["max_requests"], row["seconds"])
    first, second = result["rounds"][:2]
    assert "success" not in first
    assert first["memory"]["operations"]["create"] == 1
    assert first["memory"]["entity_delta"]["created"] == ["R01"]
    assert first["memory"]["types"] == {"room": 1}
    assert first["memory"]["valid_evidence_references"] == 1
    assert second["memory"]["operations"]["update"] == 1
    assert second["memory"]["entity_delta"]["revised"] == ["R01"]
    assert result["metrics"]["navigation_planned"] == 11
    assert result["metrics"]["navigation_successes"] == 8
    assert result["metrics"]["success_rate"] == pytest.approx(8 / 11)
    assert session.closed and environment.sim is None
    assert len(list((tmp_path / "run/private/rounds").glob("*.json"))) == 12


def test_round_motion_slices_do_not_repeat_previous_actions(tmp_path, monkeypatch):
    spec = scenario(3)
    for index, row in enumerate(spec["rounds"][1:], 1):
        row["positions"] = [[0, 0, -.25 * index]]

    def callback(session, index):
        if index > 1:
            session.environment.step("move_forward")

    result, environments, _, _ = execute(tmp_path, monkeypatch, spec, callback)
    assert result["status"] == "completed"
    assert [row["actions"] for row in result["rounds"]] == [0, 1, 1]
    assert [row["path_length"] for row in result["rounds"]] == pytest.approx([0, .25, .25])
    paths = [json.loads((tmp_path / f"run/private/rounds/{i:02d}-round{i}.json").read_text())["trajectory"] for i in (2, 3)]
    assert paths[0][-1] == paths[1][0] == [0, 0, -.25]
    assert len(paths[0]) == len(paths[1]) == 6
    assert environments[0].steps == 2


def test_infrastructure_failure_stops_sequence_and_keeps_partial_results(tmp_path, monkeypatch):
    def callback(session, index):
        return {"submitted": False, "reason": "model_error", "error": "fixture service failure"} if index == 2 else None

    result, _, sessions, _ = execute(tmp_path, monkeypatch, scenario(), callback)
    assert result["status"] == "infrastructure_error" and result["completed_rounds"] == 2
    assert result["finish_reason"] == "model_error"
    assert result["metrics"]["success_rate"] is None
    assert len(sessions[0].prompts) == 2 and sessions[0].closed
    assert result["rounds"][1]["error"] == "fixture service failure"


def test_relocation_preflight_blocks_drift_before_opening_model_session(tmp_path, monkeypatch):
    class OffMesh(FakeSimulator):
        def snap_point(self, point):
            return [point[0] + .2, self.point[1], point[2]]

    result, environments, sessions, _ = execute(tmp_path, monkeypatch, scenario(), lambda *_: None, OffMesh)
    assert result["status"] == "infrastructure_error" and result["completed_rounds"] == 0
    assert not sessions and environments[0].sim is None
    report = json.loads((tmp_path / "run/private/preflight.json").read_text())
    assert not report["passed"] and len(report["relocations"]) == 3
    assert all(not item["passed"] and not item["frozen_pose_changed"] for item in report["relocations"])


def test_relocation_preflight_reports_small_drift_without_changing_pose(tmp_path, monkeypatch):
    class NearMesh(FakeSimulator):
        def snap_point(self, point):
            return [point[0] + .05, self.point[1], point[2]]

    spec = scenario()
    original = deepcopy(spec)
    result, _, sessions, _ = execute(tmp_path, monkeypatch, spec, lambda *_: None, NearMesh)
    assert result["status"] == "completed" and spec == original
    assert sessions[0].refreshes[-1] == spec["rounds"][-1]["teleport"]["position"]
    report = json.loads((tmp_path / "run/private/preflight.json").read_text())
    assert report["passed"]
    assert all(item["snap_drift_m"] == pytest.approx(.05) for item in report["relocations"])


def test_directed_crossings_require_the_actual_opening_and_order():
    assert sequence.crosses([0, 0, .2], [0, 0, -.2], FIRST, ROOMS)
    assert not sequence.crosses([0, 0, -.2], [0, 0, .2], FIRST, ROOMS)
    assert not sequence.crosses([.8, 0, .2], [.8, 0, -.2], FIRST, ROOMS)
    ordered = [[0, 0, .3], [0, 0, -.3], [0, 0, -.8], [0, 0, -1.2]]
    assert sequence.score_route({"route_events": [FIRST, SECOND]}, ordered, ROOMS)["passed"]
    wrong = [[0, 0, -.8], [0, 0, -1.2], [0, 0, .3], [0, 0, -.3]]
    events = sequence.score_route({"route_events": [FIRST, SECOND]}, wrong, ROOMS)["events"]
    assert events[0]["passed"] and not events[1]["passed"]


def test_concave_qa_bedroom_door_uses_local_sides():
    rooms = {
        "R01": {"polygon": [[.9, 1.2], [2.13, 1.2], [2.13, -3.8], [6.05, -3.8],
                              [6.05, -2.1], [6.65, -2.1], [6.65, 1.65], [6.05, 1.65],
                              [6.05, 2.55], [.9, 2.55]]},
        "R05": {"polygon": [[-6.55, 1.2], [2.13, 1.2], [2.13, -3.7], [-1.42, -3.7],
                              [-1.42, -2.84], [-2.56, -2.84], [-2.56, -3.48],
                              [-6.2, -3.48], [-6.2, -2.32], [-6.55, -2.32]]},
    }
    event = {"segment_xz": [[1.07, -1.2], [2., -1.2]], "from_room": "R05", "to_room": "R01"}
    before, after = [1.4165, .03574, -1.13615], [1.4165, .03574, -1.38615]
    assert sequence.crosses(before, after, event, rooms)
    assert not sequence.crosses(after, before, event, rooms)


def test_curved_microstep_route_does_not_use_the_endpoint_chord(tmp_path, monkeypatch):
    class Curved(FakeSimulator):
        def step(self, action):
            points = [[0, 0, .5], [.8, 0, .5], [.8, 0, -.5], [0, 0, -.5]]
            self.point[:] = points[-1]
            return MotionResult(samples=[{"position": p} for p in points])

    spec = scenario(2)
    spec["start_position"] = [0, 0, .5]
    spec["private_rooms"] = [{"id": name, **value} for name, value in ROOMS.items()]
    spec["rounds"][1].update(positions=[[0, 0, -.5]], route_events=[FIRST])

    def callback(session, index):
        if index == 2:
            session.environment.step("move_forward")

    result, _, _, _ = execute(tmp_path, monkeypatch, spec, callback, Curved)
    row = result["rounds"][1]
    assert row["arrived"] and row["submitted"] and not row["success"]
    assert not row["route"]["passed"] and row["path_length"] == pytest.approx(2.6)


def test_waypoint_checks_the_travelled_segment_and_memory_audit_distinguishes_net_delta(tmp_path, monkeypatch):
    waypoint = {"kind": "waypoint", "id": "table", "positions": [[0, 0, 0]], "radius": .1}
    assert sequence.score_route({"route_events": [waypoint]}, [[0, 0, .5], [0, 0, -.5]], {})["passed"]

    def callback(session, index):
        session.memory_operations([{"op": "create"}, {"op": "update"}, {"op": "merge"}, {"op": "delete"}])

    result, _, _, _ = execute(tmp_path, monkeypatch, scenario(1), callback)
    memory = result["rounds"][0]["memory"]
    assert memory["operations"] == {"create": 1, "update": 1, "delete": 1, "merge": 1}
    assert memory["entity_delta"] == {"created": [], "revised": [], "deleted": []}


def test_frozen_qa_sequence_still_contains_exactly_twelve_original_rounds():
    path = Path(__file__).resolve().parents[1] / "data/datasets/navcraft/qa_campaign_v2/scenario.json"
    frozen = json.loads(path.read_text())
    sequence._validate(deepcopy(frozen))
    assert len(frozen["rounds"]) == 12
    assert frozen["rounds"][0]["kind"] == "explore"
    assert sum(bool(row.get("teleport")) for row in frozen["rounds"]) == 3
    assert all(row["success_distance"] == 1. for row in frozen["rounds"][1:])
