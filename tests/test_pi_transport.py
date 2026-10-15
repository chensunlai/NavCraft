"""Real local pipes and processes exercise the Python-owned Core lifecycle."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

from src.harness import core_client as transport
from src.harness import runner, runtime, worker


ROOT = Path(__file__).resolve().parents[1]


def test_core_stderr_redacts_explicit_credentials(tmp_path):
    fake = r'''
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    if request['op'] == 'init':
        print('fixture diagnostic ' + request['args']['auth']['apiKey'], file=sys.stderr, flush=True)
    print(json.dumps({'id': request['id'], 'ok': True, 'data': {}}), flush=True)
    if request['op'] == 'close':
        break
'''
    client = transport.CoreClient(ROOT, tmp_path, command=[sys.executable, "-u", "-c", fake])
    try:
        client.request("init", {"auth": {"apiKey": "private-fixture-key"}})
    finally:
        client.close()
    diagnostic = (tmp_path / "core.log").read_text()
    assert "private-fixture-key" not in diagnostic
    assert "[REDACTED]" in diagnostic
    assert not client._stderr_reader.is_alive()


FAKE_CORE = r'''
import json, sys
def emit(value):
    print(json.dumps(value), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    op = request["op"]
    emit({"id": request["id"], "ok": True, "data": {"accepted": True}})
    if op == "prompt":
        prompt_id = request["id"]
        emit({"type": "event", "event": {"type": "turn_start", "data": {}}})
        emit({"type": "tool_request", "prompt_id": prompt_id, "call_id": "actual-call", "name": "nav_step",
              "args": {"actions": ["turn_left"]}})
    elif op == "tool_reply":
        assert request["args"]["call_id"] == "actual-call"
        assert request["args"]["result"]["content"][0]["text"] == "actual feedback"
        emit({"type": "settled", "prompt_id": prompt_id, "reason": "host_end", "answer": "completed", "requests": 1})
    elif op == "close":
        break
'''


def test_tool_roundtrip_pumps_requests_and_executes_on_callers_thread(tmp_path):
    client = transport.CoreClient(ROOT, tmp_path, command=[sys.executable, "-u", "-c", FAKE_CORE])
    owner, calls, events = threading.get_ident(), [], []

    def execute(call_id, name, args):
        assert threading.get_ident() == owner
        calls.append((call_id, name, args))
        return {"result": {"content": [{"type": "text", "text": "actual feedback"}]},
                "control": {"end_run": True, "stop_remaining_tools": True}}

    try:
        client.request("init", {"tools": []})
        result = client.run_prompt("go", [], execute, seconds=2, on_event=events.append)
        assert result["reason"] == "host_end" and result["answer"] == "completed"
        assert calls == [("actual-call", "nav_step", {"actions": ["turn_left"]})]
        assert [event["type"] for event in events] == ["event", "tool_request", "settled"]
    finally:
        client.close()
    client.close()
    assert client.process.poll() == 0
    assert not client._reader.is_alive()


def test_tool_callback_failure_is_returned_to_core_instead_of_deadlocking(tmp_path):
    script = FAKE_CORE.replace(
        'assert request["args"]["result"]["content"][0]["text"] == "actual feedback"',
        'assert request["args"]["result"]["isError"] is True')
    client = transport.CoreClient(ROOT, tmp_path, command=[sys.executable, "-u", "-c", script])

    def execute(*_):
        raise ValueError("fixture rejected action")

    try:
        result = client.run_prompt("go", [], execute, seconds=2)
        assert result["reason"] == "host_end"
    finally:
        client.close()


def test_successive_prompts_ignore_a_stale_settled_from_another_call(tmp_path):
    script = r'''
import json, sys
previous = None
for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({"id": request["id"], "ok": True, "data": {}}), flush=True)
    if request["op"] == "prompt":
        if previous:
            print(json.dumps({"type": "settled", "prompt_id": previous,
                              "answer": "OLD ANSWER", "reason": "model_finished"}), flush=True)
        previous = request["id"]
        print(json.dumps({"type": "settled", "prompt_id": previous,
                          "answer": request["args"]["text"],
                          "reason": "model_finished", "requests": 1}), flush=True)
    elif request["op"] == "close":
        break
'''
    client = transport.CoreClient(ROOT, tmp_path, command=[sys.executable, "-u", "-c", script])
    pid = client.process.pid
    events = []
    try:
        first = client.run_prompt("first answer", [], lambda *_: {}, seconds=2)
        second = client.run_prompt("second answer", [], lambda *_: {}, seconds=2,
                                   on_event=events.append)
        assert first["answer"] == "first answer"
        assert second["answer"] == "second answer"
        assert first["prompt_id"] != second["prompt_id"]
        assert client.process.pid == pid and client.process.poll() is None
        assert len(events) == 1 and events[0]["prompt_id"] == second["prompt_id"]
    finally:
        client.close()


@pytest.mark.parametrize("wrong_id", ["previous-prompt", None])
def test_tool_request_outside_the_active_prompt_never_executes(tmp_path, wrong_id):
    script = FAKE_CORE.replace('"prompt_id": prompt_id, "call_id": "actual-call"',
                               '"prompt_id": ' + repr(wrong_id) + ', "call_id": "actual-call"')
    client = transport.CoreClient(ROOT, tmp_path, command=[sys.executable, "-u", "-c", script])
    calls = []
    try:
        with pytest.raises(transport.CoreError, match="does not belong to the active prompt"):
            client.run_prompt("go", [], lambda *args: calls.append(args), seconds=2)
        assert calls == []
    finally:
        client.close()


def test_late_events_cannot_enter_the_next_prompt_callback(tmp_path):
    script = r'''
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({"id": request["id"], "ok": True, "data": {}}), flush=True)
    if request["op"] == "prompt":
        for event_type in ["turn_start", "compaction_end"]:
            print(json.dumps({"type": "event", "prompt_id": "previous-prompt",
                              "event": {"type": event_type, "data": {}}}), flush=True)
        print(json.dumps({"type": "event", "event": {"type": "runtime_configured", "data": {}}}), flush=True)
        print(json.dumps({"type": "event", "prompt_id": request["id"],
                          "event": {"type": "turn_start", "data": {}}}), flush=True)
        print(json.dumps({"type": "settled", "prompt_id": request["id"],
                          "reason": "model_finished", "answer": "current answer"}), flush=True)
    elif request["op"] == "close":
        break
'''
    client = transport.CoreClient(ROOT, tmp_path, command=[sys.executable, "-u", "-c", script])
    events = []
    try:
        result = client.run_prompt("Current question", [], lambda *_: {}, seconds=2,
                                   on_event=events.append, prompt_id="current-prompt")
        assert result["answer"] == "current answer"
        assert [event.get("prompt_id") for event in events] == [None, "current-prompt", "current-prompt"]
        assert [event["event"]["type"] for event in events if event["type"] == "event"] == [
            "runtime_configured", "turn_start"]
    finally:
        client.close()


def test_core_native_exit_during_prompt_reports_transport_failure(tmp_path):
    script = r'''
import json, os, sys
for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({"id": request["id"], "ok": True, "data": {"accepted": True}}), flush=True)
    os._exit(7)
'''
    client = transport.CoreClient(ROOT, tmp_path, command=[sys.executable, "-u", "-c", script])
    started = time.monotonic()
    try:
        with pytest.raises(transport.CoreError):
            client.run_prompt("go", [], lambda *_: {}, seconds=2)
        assert time.monotonic() - started < 3
    finally:
        client.close()
    assert client.process.returncode == 7


def test_cancellation_receives_settled_after_actual_tool_feedback(tmp_path):
    script = FAKE_CORE.replace(
        'emit({"type": "settled", "prompt_id": prompt_id, "reason": "host_end", "answer": "completed", "requests": 1})',
        'pass\n    elif op == "abort":\n        emit({"type": "settled", "prompt_id": prompt_id, "reason": "aborted", "requests": 1})')
    client = transport.CoreClient(ROOT, tmp_path, command=[sys.executable, "-u", "-c", script])
    cancel, calls = threading.Event(), []

    def execute(*args):
        calls.append(args)
        cancel.set()
        return {"result": {"content": [{"type": "text", "text": "actual feedback"}]}}

    try:
        result = client.run_prompt("go", [], execute, seconds=2, cancel_event=cancel)
        assert result["reason"] == "aborted" and len(calls) == 1
    finally:
        client.close()


def test_runner_archives_even_when_core_close_and_scoring_fail(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "src/harness").mkdir(parents=True)
    (repo / "src/harness/navigation_prompt.md").write_text("Fixture shared navigation prompt")
    (repo / "src/harness/navigation_experience.md").write_text("Fixture navigation experience")
    (repo / "config/harness-pi").mkdir(parents=True)
    (repo / "config/harness-pi/smoke.yaml").write_text(
        "cases:\n  query:\n    sample: sample\n    requests: 2\n    seconds: 2\n")
    (repo / "unused").write_text(
        "NAVCRAFT_TEST_BASE_URL=http://fixture.invalid\n"
        "NAVCRAFT_TEST_MODEL=fixture\n"
        "NAVCRAFT_TEST_API_KEY=fixture-not-a-secret\n")
    monkeypatch.setattr(runner, "_prepare_query", lambda *_: (
        {"task": {"goal_id": 0, "instruction": "fixture"}}, [], "fixture prompt"))

    class Core:
        def __init__(self, repo, directory):
            (directory / "environment").mkdir()
            (directory / "environment/result.json").write_text("{interrupted")

        def request(self, *args, **kwargs):
            return {"accepted": True, "session_id": "fixture-session"}

        def run_prompt(self, *args, **kwargs):
            return {"reason": "model_finished", "answer": "partial answer", "requests": 1}

        def close(self):
            raise RuntimeError("fixture close failure")

    def score(**kwargs):
        raise ValueError("fixture scoring failure")

    monkeypatch.setattr(runtime, "CoreClient", Core)
    monkeypatch.setattr(runner.verify, "qa", score)
    directory = tmp_path / "run"
    runner.run_case(repo, {"output_dir": "runs", "model": "fixture", "api": "fixture",
                          "credentials": "unused", "python": sys.executable},
                    "query", directory=directory)
    result = json.loads((directory / "result.json").read_text())
    assert result["answer"] == "partial answer" and result["requests"] == 1
    assert "fixture close failure" in result["cleanup_error"]
    assert "fixture scoring failure" in result["scoring_error"]
    assert "Environment result" in result["archive_error"]
    assert (directory / "private/trace.jsonl").exists()


def _running(pid):
    try:
        # A killed orphan can remain briefly as a zombie until its reaper runs.
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def _wait_stopped(pid):
    deadline = time.monotonic() + 2
    while _running(pid) and time.monotonic() < deadline:
        time.sleep(.02)
    return not _running(pid)


@pytest.mark.parametrize("exit_code", [0, 7])
def test_supervisor_reaps_core_descendant_after_worker_exit(tmp_path, exit_code):
    pidfile = tmp_path / "core.pid"
    script = r'''
import os, pathlib, signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
pathlib.Path(sys.argv[1]).write_text(str(child.pid))
os._exit(int(sys.argv[2]))
'''
    child_pid = None
    try:
        result = worker.supervise([sys.executable, "-c", script, str(pidfile), str(exit_code)],
                                  tmp_path, seconds=1, cwd=ROOT)
        child_pid = int(pidfile.read_text())
        assert _wait_stopped(child_pid), "Core descendant survived worker exit"
        status = json.loads((tmp_path / "worker_status.json").read_text())
        assert status["exit_code"] == exit_code
        assert result["finish_reason"] == "worker_error"
    finally:
        if child_pid is None and pidfile.exists():
            child_pid = int(pidfile.read_text())
        if child_pid and _running(child_pid):
            os.kill(child_pid, signal.SIGKILL)


def test_supervisor_preserves_valid_result_on_clean_exit(tmp_path):
    script = "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('{\"finish_reason\":\"stop\",\"answer\":\"done\"}')"
    result = worker.supervise([sys.executable, "-c", script, str(tmp_path / "result.json")],
                              tmp_path, seconds=1, cwd=ROOT)
    assert result["finish_reason"] == "stop" and result["answer"] == "done"


def test_supervisor_hard_timeout_kills_unresponsive_worker_and_core(tmp_path, monkeypatch):
    original = worker.subprocess.Popen

    class ShortGraceProcess(original):
        def wait(self, timeout=None):
            return super().wait(timeout=min(timeout, .3) if timeout is not None else None)

    monkeypatch.setattr(worker.subprocess, "Popen", ShortGraceProcess)
    pidfile = tmp_path / "core.pid"
    script = r'''
import pathlib, signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(300)"])
pathlib.Path(sys.argv[1]).write_text(str(child.pid))
time.sleep(300)
'''
    child_pid = None
    try:
        result = worker.supervise([sys.executable, "-c", script, str(pidfile)],
                                  tmp_path, seconds=1, cwd=ROOT)
        child_pid = int(pidfile.read_text())
        assert result["finish_reason"] == "timeout"
        assert result["worker"]["exit_code"] == -signal.SIGKILL
        assert _wait_stopped(child_pid), "Core descendant survived a forced timeout"
    finally:
        if child_pid is None and pidfile.exists():
            child_pid = int(pidfile.read_text())
        if child_pid and _running(child_pid):
            os.kill(child_pid, signal.SIGKILL)


def test_supervisor_recovers_interrupted_result_json(tmp_path):
    script = "import pathlib,sys,os; pathlib.Path(sys.argv[1]).write_text('{interrupted'); os._exit(9)"
    result = worker.supervise([sys.executable, "-c", script, str(tmp_path / "result.json")],
                              tmp_path, seconds=1, cwd=ROOT)
    assert result["finish_reason"] == "worker_error"
    assert json.loads((tmp_path / "worker_status.json").read_text())["exit_code"] == 9
    assert json.loads((tmp_path / "result.json").read_text())["finish_reason"] == "worker_error"


def test_worker_crash_cannot_be_reported_as_success_from_existing_result(tmp_path):
    script = "import pathlib,sys,os; pathlib.Path(sys.argv[1]).write_text('{\"finish_reason\":\"stop\",\"answer\":\"partial\"}'); os._exit(9)"
    result = worker.supervise([sys.executable, "-c", script, str(tmp_path / "result.json")],
                              tmp_path, seconds=1, cwd=ROOT)
    assert result["finish_reason"] == "worker_error"
    assert result["answer"] == "partial"


def test_new_harness_imports_do_not_load_legacy_or_native_simulator():
    script = r'''
import sys
from src.harness import core_client, runner, session, memory, tools, worker, coding
assert not any(name == "src.legacy" or name.startswith("src.legacy.") for name in sys.modules)
assert "habitat_sim" not in sys.modules
assert "src.harness.agent.model" not in sys.modules
from src.common.register import model_registry, runtime_registry, tool_registry
assert model_registry.list() == []
assert runtime_registry.list() == []
assert tool_registry.list() == []
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
