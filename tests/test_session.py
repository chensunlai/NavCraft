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


def test_all_seven_tools_and_public_data_boundary(session):
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
