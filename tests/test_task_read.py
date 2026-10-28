"""The explicit task query exposes only current public instructions and images."""

from copy import deepcopy
import json
from pathlib import Path

from jsonschema import Draft7Validator
from PIL import Image
import pytest

from src.dataset.episodes import EpisodeSpec
from src.environment import Environment
from src.environment.fake import FakeSimulator
from src.harness.session import HarnessSession
from src.harness.tools.specs import specifications


REPO = Path(__file__).resolve().parents[1]


class TaskQueryClient:
    def __init__(self):
        self.calls = 0
        self.receipts = []

    def request(self, *args, **kwargs):
        return {}

    def run_prompt(self, text, images, handler, seconds, **kwargs):
        self.calls += 1
        self.receipts.append(handler(f"read-{self.calls}", "task_read", {}))
        handler(f"submit-{self.calls}", "submit", {"answer": "fixture answer"})
        return {"reason": "host_end", "requests": self.calls}

    def close(self):
        pass

    def drain_events(self, callback):
        pass


@pytest.fixture(params=[("entity", "basic"), ("entity", "super"),
                       ("node", "basic"), ("node", "super")])
def session(request, tmp_path):
    mode, tier = request.param
    episode = EpisodeSpec("fixture", "0", "objnav", "unused", [0, 0, 0], [0, 0, 0, 1], [], "unused")
    environment = Environment(episode, {"gpu": 0, "tool_tier": tier}, tmp_path / "env", FakeSimulator)
    wall = tmp_path / "map.json"
    wall.write_text(json.dumps({"origin": [-5, -5], "size_m": 10,
                                "floor_height": 0, "walls": [[[-2, -2], [-2, 2]]]}))
    instance = HarnessSession(REPO, {"model": "fixture", "memory_type": mode, "tool_tier": tier},
                              tmp_path / "run", environment, wall if mode == "entity" else None,
                              start_core=False)
    instance.client = TaskQueryClient()
    yield instance
    instance.close()
    environment.close()


def value(reply):
    return json.loads(reply["result"]["content"][0]["text"])


def picture(tmp_path, name):
    path = tmp_path / name
    Image.new("RGB", (32, 24), "gray").save(path)
    return path


def test_schema_is_identical_across_modes_and_rejects_arguments(session):
    tool = next(t for t in session.tool_specs if t["name"] == "task_read")
    for mode in ("entity", "node"):
        for tier in ("basic", "super"):
            assert tool == next(t for t in specifications(tier, mode) if t["name"] == "task_read")
    validator = Draft7Validator(tool["parameters"])
    assert validator.is_valid({})
    assert not validator.is_valid({"history": True})
    with pytest.raises(ValueError, match="Additional properties"):
        session._execute("task_read", {"history": True})


def test_query_is_readonly_and_only_returns_current_public_task(session, tmp_path):
    target = session.register_target(picture(tmp_path, "target.png"))
    session._current_request = "Find the pictured chair."
    session._target_frames = [target]
    session._prompts = ["Earlier request must not be returned"]
    session._submitted = True
    before = (deepcopy(session.current), deepcopy(session.environment.state()),
              len(session.observations.items), session.memory.path.read_bytes(),
              session._observations)
    first = session._execute("task_read", {})
    assert session._execute("task_read", {}) == first
    assert value(first) == {"request": "Find the pictured chair.",
                            "images": [{"kind": "target", "frame": target, "size": [32, 24]}]}
    assert first["result"]["images"] == [session.observations.get(target)["path"]]
    assert first["result"]["image_labels"] == [f"[TARGET {target} | 32x24]"]
    assert before == (session.current, session.environment.state(), len(session.observations.items),
                      session.memory.path.read_bytes(), session._observations)
    assert session._submitted is True


def test_new_prompt_switches_targets_and_text_only_prompt_clears_images(session, tmp_path):
    first = session.register_target(picture(tmp_path, "first.png"))
    second = session.register_target(picture(tmp_path, "second.png"))
    session.prompt("Find the first object", images=[first])
    session.prompt("Find the second object", images=[second])
    reply = session.client.receipts[-1]
    assert value(reply) == {"request": "Find the second object",
                            "images": [{"kind": "target", "frame": second, "size": [32, 24]}]}
    assert first not in json.dumps(reply["result"])
    session.prompt("Find a chair")
    reply = session.client.receipts[-1]
    assert value(reply) == {"request": "Find a chair"}
    assert reply["result"]["images"] == []


def test_question_query_preserves_navigation_and_restores_original_task(session, tmp_path):
    target = session.register_target(picture(tmp_path, "target.png"))
    session.prompt("Go next to the pictured chair", images=[target])
    result = session.ask("Which side of the chair is visible?", images=[picture(tmp_path, "question.png")])
    assert result.submitted
    reply = session.client.receipts[-1]
    data = value(reply)
    assert data["request"] == "Go next to the pictured chair"
    assert data["question"] == "Which side of the chair is visible?"
    assert [row["kind"] for row in data["images"]] == ["target", "question"]
    question = data["images"][1]["frame"]
    assert data["images"][0]["frame"] == target
    assert reply["result"]["image_labels"] == [f"[TARGET {target} | 32x24]", f"[QUESTION {question} | 32x24]"]
    restored = session._execute("task_read", {})
    assert value(restored) == {"request": "Go next to the pictured chair",
                               "images": [{"kind": "target", "frame": target, "size": [32, 24]}]}
    assert question not in json.dumps(restored["result"])
    assert session.environment.steps == 0
