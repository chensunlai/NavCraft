"""Persistent sessions keep task state while each prompt gets its own boundary."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from src.harness import runtime
from src.harness.trace import TraceWriter


class FakeTools:
    def __init__(self, workspace):
        self.workspace = workspace
        self.state = {"current": {"task": {"goal_id": 7}, "step": 0},
                      "done": False, "goal_changed": False, "revision": 0}
        self.bridge_closed = 0
        self.context = SimpleNamespace(bridge=SimpleNamespace(close=self.close_bridge),
                                       cancel_event=None, state=self.state)
        self.calls = []
        self.turns = 0

    def close_bridge(self):
        self.bridge_closed += 1

    def specs(self):
        return [{"name": "inspect", "description": "Inspect the current view.",
                 "parameters": {"type": "object", "properties": {}}}]

    def turn_start(self):
        self.turns += 1
        self.state["goal_changed"] = False

    def checkpoint(self):
        return {"content": json.dumps(self.state["current"]), "images": [],
                "metadata": deepcopy(self.state["current"]),
                "revision": self.state["revision"]}

    def call(self, call_id, name, args):
        self.calls.append((call_id, name, args))
        self.state["current"]["step"] += 1
        self.state["revision"] += 1
        return {"result": {"content": [{"type": "text", "text": "new view"}]},
                "checkpoint": self.checkpoint(), "control": {"end_run": False}}


@pytest.fixture
def harness(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "credentials.env").write_text(
        "NAVCRAFT_TEST_BASE_URL=https://fixture.invalid/v1\n"
        "NAVCRAFT_TEST_MODEL=fixture\n"
        "NAVCRAFT_TEST_API_KEY=runtime-fixture-secret\n")
    config = {"credentials": "credentials.env", "model": "fixture",
              "api": "openai-responses", "python": sys.executable}
    directory = tmp_path / "run"
    workspace = directory / "workspace"
    for folder in ("scene", "observations", "memory", "scratch"):
        (workspace / folder).mkdir(parents=True)
    tools = FakeTools(workspace)
    cores = []
    scripts = []

    class FakeCore:
        def __init__(self, repo, private):
            self.private = Path(private)
            self.operations = []
            self.history = []
            self.tool_replies = []
            self.started = threading.Event()
            self.close_count = 0
            self.drain_count = 0
            self.close_status = {"acknowledged": True}
            cores.append(self)

        def request(self, op, args=None, **kwargs):
            args = args or {}
            self.operations.append((op, deepcopy(args)))
            if op == "init":
                self.options = args
                self.ledger = Path(args["budgetFile"])
                self.ledger.parent.mkdir(parents=True, exist_ok=True)
                self.ledger.write_text(json.dumps({"limit": None, "attempts": []}))
                return {"initialized": True, "session_id": "persistent-fixture-session"}
            return {"updated": True}

        def attempt(self, kind):
            ledger = json.loads(self.ledger.read_text())
            ledger["attempts"].append({"number": len(ledger["attempts"]) + 1,
                                       "run": str(self.private.parent), "kind": kind})
            self.ledger.write_text(json.dumps(ledger))

        def run_prompt(self, text, images, tool_handler, seconds, on_event=None,
                       cancel_event=None, prompt_id=None):
            script = scripts.pop(0) if scripts else {}
            self.history.append({"text": text, "images": list(images), "id": prompt_id})
            self.started.set()
            for kind in script.get("attempts", ["normal"]):
                self.attempt(kind)
            if on_event:
                on_event({"type": "event", "event": {"type": "turn_start", "data": {}}})
            if script.get("tool"):
                self.tool_replies.append(deepcopy(
                    tool_handler("call-" + str(len(self.history)), "inspect", {})))
            if script.get("compaction") and on_event:
                on_event({"type": "event", "event": {"type": "compaction_end",
                          "data": {"result": {"summary": "kept the same goal"}}}})
            if script.get("wait"):
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    if cancel_event is not None and cancel_event.is_set():
                        break
                    time.sleep(0.01)
                else:
                    raise AssertionError("abort did not reach the active prompt")
                script["reason"] = "aborted"
            if script.get("error"):
                raise RuntimeError(script["error"])
            return {"type": "settled", "prompt_id": prompt_id,
                    "reason": script.get("reason", "model_finished"),
                    "answer": script.get("answer", text + " answered"),
                    "requests": len(json.loads(self.ledger.read_text())["attempts"]),
                    "last": {"stopReason": "stop"}}

        def close(self):
            self.close_count += 1

        def drain_events(self, on_event):
            self.drain_count += 1

    monkeypatch.setattr(runtime, "CoreClient", FakeCore)

    def open_runtime(**kwargs):
        return runtime.open_session(repo, config, directory, workspace,
                                    "Keep the existing navigation behavior.", tools,
                                    max_requests=5, **kwargs)

    return SimpleNamespace(open=open_runtime, directory=directory, workspace=workspace,
                           tools=tools, cores=cores, scripts=scripts)


def test_followup_prompts_keep_core_goal_workspace_and_tool_identity(harness):
    harness.scripts.extend([{"tool": True}, {}, {"tool": True}])
    with harness.open() as session:
        first = session.prompt("Observe the room")
        note = harness.workspace / "memory/notes.md"
        note.write_text("Door at the left; still uncertain.")
        second = session.prompt("Explain the exit")
        third = session.prompt("Check the uncertain doorway")
        assert session.session_id == "persistent-fixture-session"
        assert session.requests == 3
        assert harness.tools.state["current"] == {"task": {"goal_id": 7}, "step": 2}
        assert note.read_text() == "Door at the left; still uncertain."
    core = harness.cores[0]
    assert len(harness.cores) == 1
    assert [op for op, _ in core.operations].count("init") == 1
    assert [item["text"] for item in core.history] == [
        "Observe the room", "Explain the exit", "Check the uncertain doorway"]
    assert len({first.prompt_id, second.prompt_id, third.prompt_id}) == 3
    assert [result.requests for result in (first, second, third)] == [1, 1, 1]


def test_prompt_budgets_count_summary_and_failed_requests_without_reset(harness):
    harness.scripts.extend([
        {"attempts": ["normal", "summary", "failed"], "compaction": True},
        {"attempts": ["normal", "failed"], "reason": "request_budget"},
    ])
    with harness.open() as session:
        first = session.prompt("Inspect", limits=runtime.PromptLimits(max_requests=3, seconds=2))
        second = session.prompt("Continue", limits=runtime.PromptLimits(max_requests=2, seconds=2))
        assert (first.requests, first.total_requests) == (3, 3)
        assert (second.requests, second.total_requests, session.requests) == (2, 5, 5)
        assert second.reason == "request_budget"
    configured = [args["maxRequests"] for op, args in harness.cores[0].operations
                  if op == "configure" and "maxRequests" in args]
    assert configured == [3, 5]
    ledger = json.loads((harness.directory / "private/requests.json").read_text())
    assert [item["kind"] for item in ledger["attempts"]] == [
        "normal", "summary", "failed", "normal", "failed"]


def test_host_end_returns_control_without_closing_or_advancing_goal(harness):
    harness.scripts.extend([{"reason": "host_end", "answer": "Submitted"},
                            {"answer": "The doorway is in the saved note."}, {}])
    with harness.open() as session:
        harness.tools.state["done"] = True
        first = session.prompt("Submit the current target")
        assert first.reason == "host_end"
        assert harness.cores[0].close_count == harness.tools.bridge_closed == 0
        assert session.prompt("Explain the remembered doorway").answer.startswith("The doorway")
        # Only the application explicitly prepares another goal.
        harness.tools.state.update(done=False, goal_changed=False)
        harness.tools.state["current"] = {"task": {"goal_id": 8}, "step": 0}
        session.update_checkpoint()
        session.prompt("Approach the new target")
        assert harness.tools.state["current"]["task"]["goal_id"] == 8
    checkpoints = [args for op, args in harness.cores[0].operations if op == "set_checkpoint"]
    assert checkpoints[-1]["checkpoint"]["metadata"]["task"]["goal_id"] == 8


def test_unknown_motion_outcome_stops_execution_preserves_receipt_and_blocks_followup(harness):
    original_call = harness.tools.call

    def uncertain_call(*args):
        reply = original_call(*args)
        harness.tools.context.bridge.execution_uncertain = True
        return reply

    harness.tools.call = uncertain_call
    harness.scripts.append({"tool": True, "reason": "host_end"})
    with harness.open() as session:
        result = session.prompt("Move to the doorway")
        assert result.reason == "runtime_error" and result.to_dict()["error"]
        reply = harness.cores[0].tool_replies[0]
        assert reply["result"] == {"content": [{"type": "text", "text": "new view"}]}
        assert reply["control"] == {"end_run": True, "stop_remaining_tools": True}
        with pytest.raises(RuntimeError):
            session.prompt("Try the action again")
        assert len(harness.cores[0].history) == 1
        assert len(harness.tools.calls) == 1


def test_rejected_tool_parameters_do_not_poison_a_healthy_session(harness):
    harness.tools.context.bridge.execution_uncertain = False

    def invalid_call(*args):
        return {"result": {"content": [{"type": "text", "text": "Invalid actions argument"}],
                           "isError": True},
                "control": {"end_run": False, "stop_remaining_tools": False}}

    harness.tools.call = invalid_call
    harness.scripts.extend([{"tool": True}, {"answer": "Continuing with a valid action"}])
    with harness.open() as session:
        result = session.prompt("An invalid action request")
        assert result.reason == "model_finished"
        reply = harness.cores[0].tool_replies[0]
        assert reply["result"]["isError"] is True
        assert reply["control"]["end_run"] is False
        assert session.prompt("Continue safely").answer == "Continuing with a valid action"
        assert harness.tools.state["current"]["step"] == 0


def test_busy_abort_and_followup_preserve_a_serial_session(harness):
    harness.scripts.extend([{"wait": True}, {"answer": "Resumed"}])
    with harness.open() as session:
        results = []
        worker = threading.Thread(target=lambda: results.append(session.prompt("Keep observing")))
        worker.start()
        assert harness.cores[0].started.wait(1)
        try:
            with pytest.raises(runtime.SessionBusyError):
                session.prompt("This must not overlap")
            with pytest.raises(runtime.SessionBusyError):
                session.close()
            session.abort()
        finally:
            worker.join(timeout=4)
        assert not worker.is_alive()
        assert len(results) == 1 and results[0].reason == "aborted"
        assert session.prompt("Resume the same task").answer == "Resumed"
        assert harness.tools.state["current"]["task"]["goal_id"] == 7
    assert len(harness.cores[0].history) == 2


def test_unsettled_transport_error_has_own_record_and_no_previous_answer(harness):
    harness.scripts.extend([{"answer": "Previous successful answer"},
                            {"error": "transport lost before settled", "attempts": ["failed"]}])
    with harness.open() as session:
        first = session.prompt("First question")
        failed = session.prompt("Second question")
        assert failed.prompt_id != first.prompt_id
        assert failed.reason == "runtime_error" and failed.answer == ""
        assert (failed.requests, failed.total_requests) == (1, 2)
        assert "transport lost" in failed.to_dict()["error"]
        with pytest.raises(RuntimeError):
            session.prompt("Cannot continue without settled")
    raw = (harness.directory / failed.record).read_text()
    assert "Previous successful answer" not in raw
    assert "transport lost before settled" in raw


def test_per_prompt_callbacks_and_records_do_not_mix_rounds(harness):
    first_events, second_events, shared_events = [], [], []
    with harness.open(on_event=shared_events.append) as session:
        first = session.prompt("First", on_event=first_events.append)
        second = session.prompt("Second", on_event=second_events.append)
        assert first.to_dict()["prompt_id"] != second.to_dict()["prompt_id"]
    assert first_events and second_events
    assert len(shared_events) == len(first_events) + len(second_events)
    for result in (first, second):
        record = json.loads((harness.directory / result.record).read_text())
        assert record["prompt_id"] == result.prompt_id
        assert record["answer"] == result.answer
    trace = [json.loads(line) for line in
             (harness.directory / "private/trace.jsonl").read_text().splitlines()]
    finished = [row["data"] for row in trace if row["type"] == "prompt_end"]
    assert [row["prompt_id"] for row in finished] == [first.prompt_id, second.prompt_id]


def test_close_releases_core_and_bridge_once_but_not_external_trace(harness):
    trace = TraceWriter(harness.directory / "external")
    session = harness.open(trace=trace)
    try:
        session.prompt("Question")
        session.close()
        session.close()
        assert harness.cores[0].close_count == 1
        assert harness.tools.bridge_closed == 1
        assert trace.closed is False
        trace.emit("caller_still_owns_trace")
        with pytest.raises(RuntimeError):
            session.prompt("After close")
    finally:
        trace.close()


@pytest.mark.parametrize("requests,seconds", [(0, 2), (-1, 2), (True, 2), (1, 0), (1, -1)])
def test_invalid_prompt_limits_do_not_send_model_requests(harness, requests, seconds):
    with harness.open() as session:
        with pytest.raises(ValueError):
            session.prompt("Invalid budget", limits=runtime.PromptLimits(
                max_requests=requests, seconds=seconds))
        assert harness.cores[0].history == []
        assert session.requests == 0
