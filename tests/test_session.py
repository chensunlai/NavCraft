import json
from pathlib import Path

from PIL import Image
import pytest

from src.environment.base import MotionResult, Observation
from src.harness.observations import map_pose
from src.harness.session import HarnessSession, PromptLimits


class FakeEnvironment:
    def __init__(self, root):
        self.root = root
        self.image = root / "source.png"
        Image.new("RGB", (640, 480), "gray").save(self.image)
        self.steps, self.path_length = 0, 0
        self.trajectory = [[5., 0., 5.]]
        self.trajectory_breaks = []

    def pose(self):
        return {"position": self.trajectory[-1], "rotation_xyzw": [0., 0., 0., 1.]}

    def observe(self):
        return Observation("unused", self.image, None, self.pose(), {})

    def panorama(self):
        return [self.observe() for _ in range(6)]

    def step(self, action):
        self.steps += 1
        point = self.trajectory[-1].copy()
        point[2] -= .25
        self.trajectory.append(point)
        return MotionResult(samples=[{"position": point}]), self.observe()


class FakeClient:
    def __init__(self, repo, private):
        self.initialized = None
        self.prompts = 0
        self.closed = False
        self.requests = []

    def request(self, op, args=None, **kwargs):
        self.requests.append((op, args))
        if op == "init":
            self.initialized = args
            return {"session_id": "one-session"}
        return {}

    def run_prompt(self, text, images, handler, seconds, **kwargs):
        self.prompts += 1
        handler(f"move-{self.prompts}", "move", {"mode": "steps", "actions": ["move_forward"]})
        handler(f"submit-{self.prompts}", "submit", {"answer": str(self.prompts)})
        self.after_submit = handler(f"late-{self.prompts}", "move", {"mode": "steps", "actions": ["move_forward"]})
        return {"reason": "host_end", "requests": self.prompts}

    def close(self):
        self.closed = True

    def drain_events(self, callback):
        pass


@pytest.fixture
def session(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    scene = tmp_path / "map.json"
    scene.write_text(json.dumps({"origin": [0, 0], "size_m": 10,
                                "walls": [[[1, 1], [9, 1]]], "floor_height": 0}))
    credentials = tmp_path / "fixture.env"
    credentials.write_text("NAVCRAFT_TEST_BASE_URL=http://fixture.invalid\nNAVCRAFT_TEST_API_KEY=fixture-key\n")
    result = HarnessSession(repo, {"model": "fixture", "reasoning_effort": "medium", "credentials": str(credentials)},
                            tmp_path / "run", FakeEnvironment(tmp_path), scene,
                            client_factory=FakeClient)
    yield result
    result.close()


def test_persistent_prompts_and_submit_stops_remaining_calls(session):
    first = session.prompt("First request")
    second = session.prompt("Second request")
    assert first.submitted and second.submitted
    assert first.answer == "1" and second.answer == "2"
    assert session.client.prompts == 2 and session.environment.steps == 2
    assert session.client.after_submit["control"]["end_run"]
    assert session._prompts == ["First request", "Second request"]
    assert "checkpoint" not in session.client.initialized
    assert all(op != "set_checkpoint" for op, _ in session.client.requests)
    assert "checkpoint" not in session.client.after_submit


def test_refresh_after_host_change_preserves_memory_and_session(session):
    session._tool("record-room", "memory_edit", {"operations": [
        {"op": "create", "id": "R01", "type": "room", "fields": {"name": "room"}}]})
    previous = session.current["frame"]
    session.environment.trajectory.append([8., 0., 2.])
    current = session.refresh_observation()
    assert current["frame"] != previous and current["pose"]["pos"] == [.8, .2]
    assert session.memory.entities[0]["id"] == "R01"
    assert session.session_id == "one-session" and session.environment.steps == 0
    session._busy = True
    with pytest.raises(RuntimeError, match="busy"):
        session.refresh_observation()
    session._busy = False


def test_duplicate_tool_id_does_not_repeat_motion(session):
    args = {"mode": "steps", "actions": ["move_forward"]}
    first = session._tool("same", "move", args)
    assert session._tool("same", "move", args) == first
    assert session.environment.steps == 1
    assert session._tool("same", "move", {"mode": "steps", "actions": ["turn_left"]})["result"]["isError"]
    assert session.environment.steps == 1


def test_batch_validated_before_moving_and_budget_enforced(session):
    invalid = session._tool("bad", "move", {"mode": "steps", "actions": ["move_forward", "fly"]})
    assert invalid["result"]["isError"] and session.environment.steps == 0
    session._action_end = 1
    reply = session._tool("limited", "move", {"mode": "steps", "actions": ["move_forward"] * 6})
    assert session.environment.steps == 1 and reply["control"]["end_run"]
    assert "budget" not in reply["result"]["content"][0]["text"]


def test_target_has_no_private_pose_and_panorama_restores_current(session):
    fid = session.register_target(session.environment.image)
    target = session.observations.get(fid)
    assert target["pose"] is None and target["step"] is None
    reply = session._tool("pano", "observe", {"mode": "panorama"})
    metadata = json.loads(reply["result"]["content"][0]["text"])
    assert len(metadata["views"]) == 6
    assert metadata["size"] == [1920, 1012]
    assert all(view["size"] == [640, 480] for view in metadata["views"])
    assert session.current["frame"] == metadata["views"][0]["frame"]
    assert session.environment.steps == 0


def test_failed_prompt_does_not_replace_queryable_task(session):
    first = session.register_target(session.environment.image)
    second = session.register_target(session.environment.image)
    instruction = "Find the first pictured object. Use the side entrance."
    session.prompt(instruction, images=[first])
    before = session.tools.task_read()["result"]
    session.refresh_observation()
    assert session.tools.task_read()["result"] == before
    with pytest.raises(FileNotFoundError):
        session.prompt("Invalid next request", images=[second, "missing.jpg"])
    assert session.tools.task_read()["result"] == before
    assert session.observations.get(first) and session.observations.get(second)


def test_tools_and_public_data_boundary(session):
    session._tool("edit", "memory_edit", {"operations": [
        {"op": "create", "id": "R01", "type": "room", "fields": {"name": "room", "anchor": [.5, .5], "evidence": [session.current["frame"]]}}]})
    assert session.memory.entities[0]["id"] == "R01"
    for identifier, name, args in [
        ("read", "memory_read", {"ids": ["R01"]}),
        ("map", "map_view", {"view": "full"}),
        ("image", "view_images", {"images": [{"image": session.current["frame"], "crop": [0, 0, .5, .5]}]}),
    ]:
        reply = session._tool(identifier, name, args)
        assert not reply["result"]["isError"]
        for ref in reply["result"]["images"]:
            assert (session.workspace / ref).is_file()
        assert not any(word in json.dumps(reply["result"]) for word in ("rotation_xyzw", "world_from_optical", "success", "goal_id", "budget_remaining"))


def test_free_workspace_notes_persist_without_fixed_task_memory(session):
    notes = session.workspace / "scratch/route-notes.md"
    notes.write_text("Left approach blocked. Try the right side.")
    session.prompt("First target")
    session.prompt("Next target")
    assert notes.read_text() == "Left approach blocked. Try the right side."
    assert not (session.workspace / "memory/task.json").exists()
    assert "task_memory" not in {tool["name"] for tool in session.tool_specs}
    task = session.tools.task_read()["result"]
    assert "scratch_files" not in json.dumps(task)
    assert notes.read_text() not in json.dumps(task)


def test_move_with_panorama_returns_one_reusable_image_and_keeps_motion_receipt(session):
    arguments = {"mode": "steps", "actions": ["move_forward", "move_right"], "after": "panorama"}
    receipt = session._tool("move-and-look", "move", arguments)
    value = json.loads(receipt["result"]["content"][0]["text"])
    assert not receipt["result"]["isError"]
    assert value["done"] == [["move_forward", 1], ["move_right", 1]]
    assert session.environment.steps == 2
    assert value["after"]["executed"] and len(value["after"]["views"]) == 6
    assert value["after"]["pose"] == session.current["pose"]
    assert value["frame"] == value["after"]["views"][0]["frame"]
    assert value["size"] == [640, 480]
    assert value["after"]["size"] == [1920, 1012]
    assert len(receipt["result"]["images"]) == 1
    assert session.observations.image(value["after"]["image"])["path"] == receipt["result"]["images"][0]
    assert session.observations.resolve_pixel(value["after"]["image"], [320, 260]) == (
        value["after"]["views"][0]["frame"], [320, 234])
    assert session._observations == 1
    assert session._tool("move-and-look", "move", arguments) == receipt
    assert session.environment.steps == 2 and session._observations == 1


def test_observation_limit_is_per_prompt_and_shared_with_combined_move(session, monkeypatch):
    receipts = []

    def run_prompt(text, images, handler, seconds, **kwargs):
        turn = len(receipts)
        combined = handler(f"combined-{turn}", "move", {
            "mode": "steps", "actions": ["move_forward"], "after": "panorama"})
        extra = handler(f"extra-{turn}", "observe", {"mode": "current"})
        receipts.append((combined, extra))
        return {"reason": "completed"}

    monkeypatch.setattr(session.client, "run_prompt", run_prompt)
    for index in range(2):
        result = session.prompt(f"Request {index}", limits=PromptLimits(max_observations=1))
        assert result.actions == 1
        combined, extra = receipts[-1]
        assert not combined["result"]["isError"]
        assert json.loads(combined["result"]["content"][0]["text"])["after"]["executed"]
        assert extra["result"]["isError"]
    assert session.environment.steps == 2 and session._observations == 2


def test_observation_limit_does_not_erase_an_executed_move(session):
    session._observation_end = 0
    arguments = {"mode": "steps", "actions": ["move_forward"], "after": "panorama"}
    receipt = session._tool("move-no-observation", "move", arguments)
    value = json.loads(receipt["result"]["content"][0]["text"])
    assert not receipt["result"]["isError"] and value["done"] == [["move_forward", 1]]
    assert value["after"]["executed"] is False
    assert "Observation limit" in value["after"]["reason"]
    assert receipt["result"]["images"] == [session.current["path"]]
    assert session.environment.steps == 1 and session._observations == 0
    assert session._tool("move-no-observation", "move", arguments) == receipt
    assert session.environment.steps == 1


def test_failed_combined_observation_preserves_motion_and_ends_execution(session, monkeypatch):
    def broken_panorama():
        raise RuntimeError("fixture capture failure")

    monkeypatch.setattr(session.environment, "panorama", broken_panorama)
    receipt = session._tool("move-capture-failed", "move", {
        "mode": "steps", "actions": ["move_forward"], "after": "panorama"})
    value = json.loads(receipt["result"]["content"][0]["text"])
    assert not receipt["result"]["isError"] and value["done"] == [["move_forward", 1]]
    assert not value["after"]["executed"]
    assert session.environment.steps == 1 and session._broken
    assert receipt["control"]["end_run"] and session._stop_reason == "environment_error"
    assert receipt["result"]["images"] == [session.current["path"]]


def test_map_heading_and_bounds():
    transform = {"origin": [0, 0], "size_m": 10}
    assert map_pose({"position": [5, 1, 5], "rotation_xyzw": [0, 0, 0, 1]}, transform) == {"pos": [.5, .5], "heading": 0}
    assert map_pose({"position": [5, 1, 5], "rotation_xyzw": [0, -.70710678, 0, .70710678]}, transform)["heading"] == 90
    precise = {"position": [4.123456, 1, 5.876543], "rotation_xyzw": [0, 0, 0, 1]}
    assert map_pose(precise, transform) == {"pos": [.4123, .5877], "heading": 0}
    assert precise["position"] == [4.123456, 1, 5.876543]


def test_prompt_limits_reject_invalid_numbers():
    with pytest.raises(ValueError):
        PromptLimits(max_actions=True)
    with pytest.raises(ValueError):
        PromptLimits(seconds=float("inf"))


def test_invalid_input_image_does_not_leave_session_busy(session):
    with pytest.raises(FileNotFoundError):
        session.prompt("Missing input", images=["missing.jpg"])
    assert session.prompt("Valid request").submitted


def test_memory_read_validation_is_tool_error(session):
    assert session._tool("invalid-read", "memory_read", {"ids": [], "filter": {}})["result"]["isError"]


def test_failed_memory_edit_preserves_change_history(session):
    from src.harness.trace import read_trace

    invalid = session._tool("bad-edit", "memory_edit", {"operations": [
        {"op": "create", "id": "R01", "type": "room", "fields": {"name": "room"}},
        {"op": "create", "id": "O01", "type": "object", "fields": {"name": "chair", "room": "R99"}},
    ]})
    assert invalid["result"]["isError"]
    assert "No memory changes applied" in invalid["result"]["content"][0]["text"]
    assert session.memory.entities == []
    trace = session.private / "trace.jsonl"
    assert not any(event["type"] == "memory_changed" for event in read_trace(trace))

    valid = session._tool("good-edit", "memory_edit", {"operations": [
        {"op": "create", "id": "R01", "type": "room", "fields": {"name": "room"}},
    ]})
    assert not valid["result"]["isError"]
    changes = [event["data"] for event in read_trace(trace) if event["type"] == "memory_changed"]
    assert len(changes) == 1
    assert changes[0]["before"] != changes[0]["after"]


def test_memory_location_validation_reports_unchanged_batch(session):
    operations = [
        {"op": "create", "id": "R01", "type": "room", "fields": {"name": "room"}},
        {"op": "create", "id": "O01", "type": "object", "fields": {
            "name": "chair", "point": [.5, .5], "location": {"uv": [.6, .6]}}},
    ]
    invalid = session._tool("location-conflict", "memory_edit", {"operations": operations})
    text = invalid["result"]["content"][0]["text"]
    assert invalid["result"]["isError"]
    assert "location cannot be combined" in text and "No memory changes applied" in text
    assert session.memory.entities == []
    del operations[1]["fields"]["location"]
    valid = session._tool("location-fixed", "memory_edit", {"operations": operations})
    assert not valid["result"]["isError"]
    assert {entity["id"] for entity in session.memory.entities} == {"R01", "O01"}


def test_basic_memory_location_recovery_uses_map_without_rgb_projection(session):
    operation = {"op": "create", "id": "O01", "type": "object", "fields": {
        "name": "chair", "evidence": [session.current["frame"]],
        "location": {"image": session.current["frame"], "pixel": [320, 240]}}}
    reply = session._tool("rgb-point", "memory_edit", {"operations": [operation]})
    error = json.loads(reply["result"]["content"][0]["text"])["error"]
    assert reply["result"]["isError"] and "map image (m_)" in error
    assert "No memory changes applied" in error and session.memory.entities == []
    mapped = session._tool("map-point", "map_view", {"view": {"ego": 2}})
    view = json.loads(mapped["result"]["content"][0]["text"])["view_id"]
    record = json.loads((session.workspace / "observations/maps/index.jsonl").read_text().splitlines()[-1])
    left, top, right, bottom = record["plot_bounds"]
    operation["fields"]["location"] = {"image": view, "pixel": [(left + right) / 2, (top + bottom) / 2]}
    fixed = session._tool("map-location-fixed", "memory_edit", {"operations": [operation]})
    assert not fixed["result"]["isError"]
    assert session.memory.entities[0]["point"] == pytest.approx(session.current["pose"]["pos"])
    assert session.memory.entities[0]["evidence"] == [session.current["frame"]]
    assert session.environment.steps == 0 and not session.spatial.enabled


@pytest.mark.parametrize("stage", ["write", "provenance"])
def test_memory_error_after_write_does_not_claim_unchanged(session, monkeypatch, stage):
    if stage == "write":
        from src.harness import memory
        original = memory.atomic_json

        def fail_after_write(path, value):
            original(path, value)
            if path == session.memory.path:
                raise ValueError("Failure after memory write")

        monkeypatch.setattr(memory, "atomic_json", fail_after_write)
    else:
        def fail_provenance(operations, sources):
            raise ValueError("Failure after memory write")

        monkeypatch.setattr(session.spatial, "remember_sources", fail_provenance)
    reply = session._tool("post-write-error", "memory_edit", {"operations": [
        {"op": "create", "id": "R01", "type": "room", "fields": {"name": "room"}},
    ]})
    assert reply["result"]["isError"]
    assert "No memory changes applied" not in reply["result"]["content"][0]["text"]
    assert [entity["id"] for entity in session.memory.entities] == ["R01"]


def test_imported_observation_cannot_point_to_private_file(session):
    from src.harness.observations import Observations

    hidden = session.private / "secret.json"
    hidden.write_text("private")
    item = {**session.current, "path": "../private/secret.json"}
    session.observations.index_path.write_text(json.dumps(item) + "\n")
    with pytest.raises(ValueError, match="outside"):
        Observations(session.workspace, session.transform)


def test_memory_schema_exposes_legal_values_and_references():
    from src.harness.tools.specs import specifications

    tool = next(t for t in specifications() if t["name"] == "memory_edit")
    fields = tool["parameters"]["properties"]["operations"]["items"]["properties"]["fields"]["properties"]
    assert fields["passage"]["enum"] == ["unknown", "seen", "passed", "blocked", None]
    assert fields["connects"]["minItems"] == fields["connects"]["maxItems"] == 2
    assert fields["room"]["pattern"] == "^R[0-9]{2,4}$"
    assert "name" in fields and "evidence" in fields


def test_small_memory_query_returns_details_and_bounded_evidence(session):
    frames = [session.current["frame"], session.refresh_observation()["frame"],
              session.refresh_observation()["frame"]]
    edit = session._tool("two-candidates", "memory_edit", {"operations": [
        {"op": "create", "id": identifier, "type": "object", "fields": {
            "name": name, "category": "chair", "description": description,
            "point": [.5, .5], "evidence": frames}}
        for identifier, name, description in (
            ("O01", "Wooden chair", "Wooden chair beside the window."),
            ("O02", "Office chair", "Black chair beside the desk."))]})
    written = json.loads(edit["result"]["content"][0]["text"])
    assert {entity["id"] for entity in written["entities"]} == {"O01", "O02"}
    assert all(entity["point"] == [.5, .5] for entity in written["entities"])
    receipt = session._tool("candidates", "memory_read", {"filter": {"category": "chair"}})
    records = json.loads(receipt["result"]["content"][0]["text"])["entities"]
    assert [entity["description"] for entity in records] == [
        "Wooden chair beside the window.", "Black chair beside the desk."]
    assert all([item["frame"] for item in entity["evidence"]] == frames for entity in records)
    assert 1 <= len(receipt["result"]["images"]) <= 2
    assert len(set(receipt["result"]["images"])) == len(receipt["result"]["images"])
    assert session.environment.steps == 0


def test_map_returns_highlighted_memory_connections_without_changing_records(session):
    session._tool("rooms-and-door", "memory_edit", {"operations": [
        {"op": "create", "id": "R01", "type": "room", "fields": {"name": "Hall", "anchor": [.4, .5]}},
        {"op": "create", "id": "R02", "type": "room", "fields": {"name": "Kitchen", "anchor": [.6, .5]}},
        {"op": "create", "id": "D01", "type": "door", "fields": {
            "name": "Kitchen entrance", "point": [.5, .5], "connects": ["R01", "R02"], "passage": "passed"}},
    ]})
    before = session.memory.path.read_bytes()
    receipt = session._tool("mapped-door", "map_view", {"view": "full", "highlight": ["D01"]})
    value = json.loads(receipt["result"]["content"][0]["text"])
    assert len(value["records"]) <= 12
    door = next(record for record in value["records"] if record["id"] == "D01")
    assert door["connects"] == ["R01", "R02"] and door["passage"] == "passed"
    assert session.memory.path.read_bytes() == before and session.environment.steps == 0
