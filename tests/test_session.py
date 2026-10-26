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

    def request(self, op, args=None, **kwargs):
        if op == "init":
            self.initialized = args
            return {"session_id": "one-session"}
        return {}

    def run_prompt(self, text, images, handler, seconds, **kwargs):
        self.prompts += 1
        handler(f"move-{self.prompts}", "nav_step", {"actions": ["move_forward"]})
        handler(f"submit-{self.prompts}", "submit", {"answer": str(self.prompts)})
        self.after_submit = handler(f"late-{self.prompts}", "nav_step", {"actions": ["move_forward"]})
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
    assert session.checkpoint()["content"].count("First request") == 1


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
    args = {"actions": ["move_forward"]}
    first = session._tool("same", "nav_step", args)
    assert session._tool("same", "nav_step", args) == first
    assert session.environment.steps == 1
    assert session._tool("same", "nav_step", {"actions": ["turn_left"]})["result"]["isError"]
    assert session.environment.steps == 1


def test_batch_validated_before_moving_and_budget_enforced(session):
    invalid = session._tool("bad", "nav_step", {"actions": ["move_forward", "fly"]})
    assert invalid["result"]["isError"] and session.environment.steps == 0
    session._action_end = 1
    reply = session._tool("limited", "nav_step", {"actions": ["move_forward"] * 6})
    assert session.environment.steps == 1 and reply["control"]["end_run"]
    assert "budget" not in reply["result"]["content"][0]["text"]


def test_target_has_no_private_pose_and_panorama_restores_current(session):
    fid = session.register_target(session.environment.image)
    target = session.observations.get(fid)
    assert target["pose"] is None and target["step"] is None
    reply = session._tool("pano", "nav_observe", {"mode": "panorama"})
    metadata = json.loads(reply["result"]["content"][0]["text"])
    assert len(metadata["views"]) == 6
    assert session.current["frame"] == metadata["views"][0]["frame"]
    assert session.environment.steps == 0


def test_checkpoint_tracks_only_the_current_requests_target_images(session):
    first = session.register_target(session.environment.image)
    second = session.register_target(session.environment.image)
    instruction = "Find the first pictured object.\nUse the side entrance, then stop beside it."
    session.prompt(instruction, images=[first])
    checkpoint = session.checkpoint()
    facts = json.loads(checkpoint["content"].split("\n", 1)[1])
    assert facts["target_frames"] == [first]
    assert facts["current_request"] == instruction
    assert checkpoint["image_labels"] == [f"[CURRENT {session.current['frame']}]", f"[TARGET {first}]"]
    assert checkpoint["images"][1] == session.observations.get(first)["path"]

    session.refresh_observation()
    assert session.checkpoint()["images"][1] == checkpoint["images"][1]
    assert json.loads(session.checkpoint()["content"].split("\n", 1)[1])["current_request"] == instruction
    with pytest.raises(FileNotFoundError):
        session.prompt("Invalid next request", images=[second, "missing.jpg"])
    assert session.checkpoint()["image_labels"][-1] == f"[TARGET {first}]"

    session.prompt("Find the second pictured object", images=[second])
    assert session.checkpoint()["image_labels"][-1] == f"[TARGET {second}]"
    assert session.observations.get(first)["path"] not in session.checkpoint()["images"]
    session.prompt("Find a chair")
    checkpoint = session.checkpoint()
    assert json.loads(checkpoint["content"].split("\n", 1)[1])["target_frames"] == []
    assert checkpoint["images"] == [session.current["path"]]
    assert session.observations.get(first) and session.observations.get(second)


def test_tools_and_public_data_boundary(session):
    session._tool("edit", "memory_edit", {"operations": [
        {"op": "create", "id": "R01", "type": "room", "fields": {"name": "room", "anchor": [.5, .5], "evidence": [session.current["frame"]]}}]})
    assert session.memory.entities[0]["id"] == "R01"
    for identifier, name, args in [
        ("read", "memory_read", {"ids": ["R01"]}),
        ("map", "map_view", {"view": "full"}),
        ("image", "view_images", {"frames": [{"frame": session.current["frame"], "crop": [0, 0, .5, .5]}]}),
    ]:
        reply = session._tool(identifier, name, args)
        assert not reply["result"]["isError"]
        for ref in reply["result"]["images"]:
            assert (session.workspace / ref).is_file()
        assert not any(word in json.dumps(reply["result"]) for word in ("rotation_xyzw", "world_from_optical", "success", "goal_id", "budget_remaining"))


def test_task_memory_replaces_reads_clears_and_preserves_other_memory(session):
    from src.harness.trace import read_trace

    def call(call_id, args):
        reply = session._tool(call_id, "task_memory", args)
        assert not reply["result"]["isError"]
        return reply, json.loads(reply["result"]["content"][0]["text"])

    assert call("empty-notes", {"action": "read"})[1] == {"content": ""}
    session._tool("spatial-record", "memory_edit", {"operations": [
        {"op": "create", "id": "R01", "type": "room", "fields": {"name": "Hall"}}]})
    original_entities = session.memory.entities
    call("first-notes", {"action": "edit", "content": "Inspect the left exit."})
    notes = "Left approach blocked.\nTry the right side; destination still uncertain."
    reply, value = call("replace-notes", {"action": "edit", "content": notes})
    assert value == {"updated": True}
    assert json.loads(reply["checkpoint"]["content"].split("\n", 1)[1])["task_memory"] == notes
    assert call("read-notes", {"action": "read"})[1] == {"content": notes}
    assert session.memory.entities == original_entities
    assert session.environment.steps == 0

    session.prompt("First target")
    session.prompt("Next target")
    assert json.loads(session.task_memory_path.read_text()) == {"content": notes}
    assert json.loads(session.checkpoint()["content"].split("\n", 1)[1])["task_memory"] == notes
    # prompt has submitted; use a new prompt to exercise tools through the normal lifecycle.
    def clear_notes(*args, **kwargs):
        call("clear-notes", {"action": "edit", "content": ""})
        assert call("cleared-notes", {"action": "read"})[1] == {"content": ""}
        return {"reason": "model_finished", "requests": 0}
    session.client.run_prompt = clear_notes
    session.prompt("Clear obsolete plans")
    changes = [event for event in read_trace(session.private / "trace.jsonl") if event["type"] == "task_memory_changed"]
    assert len(changes) == 3
    assert changes[1]["data"] == {"before": {"content": "Inspect the left exit."}, "after": {"content": notes}}


@pytest.mark.parametrize("args", [{"action": "append", "content": "bad"}, {"action": "edit"},
                                  {"action": "edit", "content": 7}, {"action": "read", "content": "bad"},
                                  {"content": "missing action"}])
def test_invalid_task_memory_call_preserves_notes(session, args):
    session._tool("seed-notes", "task_memory", {"action": "edit", "content": "Keep this plan."})
    before = session.task_memory_path.read_bytes()
    revision = session.checkpoint()["revision"]
    reply = session._tool("bad-notes", "task_memory", args)
    assert reply["result"]["isError"]
    assert session.task_memory_path.read_bytes() == before
    assert session.checkpoint()["revision"] == revision
    assert not session._broken


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


def test_failed_memory_edit_preserves_revision_and_change_history(session):
    from src.harness.trace import read_trace

    revision = session.checkpoint()["revision"]
    invalid = session._tool("bad-edit", "memory_edit", {"operations": [
        {"op": "create", "id": "R01", "type": "room", "fields": {"name": "room"}},
        {"op": "create", "id": "O01", "type": "object", "fields": {"name": "chair", "room": "R99"}},
    ]})
    assert invalid["result"]["isError"]
    assert session.memory.entities == []
    assert session.checkpoint()["revision"] == revision
    trace = session.private / "trace.jsonl"
    assert not any(event["type"] == "memory_changed" for event in read_trace(trace))

    valid = session._tool("good-edit", "memory_edit", {"operations": [
        {"op": "create", "id": "R01", "type": "room", "fields": {"name": "room"}},
    ]})
    assert not valid["result"]["isError"]
    assert session.checkpoint()["revision"] == revision + 1
    changes = [event["data"] for event in read_trace(trace) if event["type"] == "memory_changed"]
    assert len(changes) == 1
    assert changes[0]["before"] != changes[0]["after"]


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
