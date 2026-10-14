"""Application boundaries: private configuration, failure archives and audit events."""

import copy
import gc
import json
from pathlib import Path
import sys
import weakref
from unittest.mock import Mock

from PIL import Image
import pytest

from src.harness import coding, runner, runtime
from src.harness.trace import TraceWriter


@pytest.fixture
def app(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "config/harness-pi").mkdir(parents=True)
    (repo / "config/harness-pi/smoke.yaml").write_text(
        "cases:\n  query:\n    sample: sample\n    requests: 2\n    seconds: 2\n")
    sample = repo / "sample"
    (sample / "observations").mkdir(parents=True)
    Image.new("RGB", (16, 12), "white").save(sample / "observations/start.png")
    (sample / "task.json").write_text(json.dumps({
        "question": "Identify the opening in the current image.",
        "resources": {"initial_observation": "observations/start.png"},
        "initial_images": ["observations/start.png"],
    }))
    secret = "fixture-api-key-NEVER-PUBLISH"
    (repo / "credentials.env").write_text(
        "NAVCRAFT_TEST_BASE_URL=https://private-model-service.invalid\n"
        "NAVCRAFT_TEST_MODEL=private-model-name\n"
        f"NAVCRAFT_TEST_API_KEY={secret}\n")
    config = {"output_dir": "runs", "model": "private-model-name",
              "api": "openai-responses", "credentials": "credentials.env",
              "python": sys.executable}
    monkeypatch.setattr(runner.verify, "qa", lambda **_: {"passed": False})
    return repo, config, secret


def run_app(kind, app, directory):
    repo, config, _ = app
    if kind == "coding":
        coding.run_coding(repo, config, max_requests=2, seconds=2, directory=directory)
    else:
        runner.run_case(repo, config, "query", max_requests=2, seconds=2, directory=directory)
    return json.loads((directory / "result.json").read_text())


@pytest.mark.parametrize("kind", ["query", "coding"])
def test_missing_credentials_archive_failure_without_starting_core(app, tmp_path, monkeypatch, kind):
    repo, _, _ = app
    (repo / "credentials.env").unlink()
    starts = []

    def forbidden_core(*args, **kwargs):
        starts.append((args, kwargs))
        raise AssertionError("Core must not start without credentials")

    monkeypatch.setattr(runtime, "CoreClient", forbidden_core)
    directory = tmp_path / kind
    result = run_app(kind, app, directory)
    assert starts == []
    assert "FileNotFoundError" in result["error"]
    assert result.get("requests", 0) == 0
    rows = [json.loads(line) for line in (directory / "private/trace.jsonl").read_text().splitlines()]
    assert rows[-1]["type"] == "run_finished"
    assert rows[-1]["data"]["error"] == result["error"]


@pytest.mark.parametrize("kind", ["query", "coding"])
def test_model_configuration_is_sent_only_to_core_not_public_workspace(app, tmp_path, monkeypatch, kind):
    requests = []

    class Core:
        close_status = {"acknowledged": True}

        def __init__(self, *_):
            pass

        def request(self, op, args):
            requests.append((op, copy.deepcopy(args)))
            return {"accepted": True, "session_id": "fixture-session"}

        def run_prompt(self, *args, **kwargs):
            return {"reason": "model_finished", "answer": "fixture answer", "requests": 1,
                    "last": {"stopReason": "stop"}}

        def close(self):
            pass

        def drain_events(self, _):
            pass

    monkeypatch.setattr(runtime, "CoreClient", Core)
    directory = tmp_path / kind
    run_app(kind, app, directory)
    assert [op for op, _ in requests] == ["init", "configure", "set_checkpoint"]
    options = requests[0][1]
    assert options["auth"]["apiKey"] == app[2]
    assert options["maxRequests"] == 2
    assert options["tracePersist"] is False
    assert Path(options["budgetFile"]).is_relative_to(directory / "private")
    public = b"\n".join(path.read_bytes() for path in (directory / "workspace").rglob("*") if path.is_file())
    for private_value in (app[2], app[1]["model"], "private-model-service.invalid", "credentials.env"):
        assert private_value.encode() not in public


@pytest.mark.parametrize("kind", ["query", "compaction"])
def test_compaction_diagnostic_restores_custom_budget_without_resetting_regular_cases(
        app, tmp_path, monkeypatch, kind):
    repo, config, _ = app
    config.update(context_window=131072, compaction_threshold=98000,
                  keep_recent_tokens=18000, scene_archive="unused-scene")
    core = Mock()
    core.request.return_value = {"session_id": "fixture-session"}
    monkeypatch.setattr(runtime, "CoreClient", Mock(return_value=core))
    monkeypatch.setattr(runner, "case_config", lambda *_: {"sample": "sample", "requests": 4, "seconds": 2})
    if kind == "compaction":
        (repo / "src/harness").mkdir(parents=True)
        (repo / "src/harness/navigation_prompt.md").write_text("Fixture navigation task")
        monkeypatch.setattr(runner.memory, "prepare", lambda _, workspace, **kwargs:
                            (Path(workspace) / "memory/state.json").write_text("{}"))
        monkeypatch.setattr(runner.memory, "render", lambda *_: {"images": ["scene/map.png"], "data": {}})
        session = Mock()
        session.init.return_value = {"task": {"goal_id": 0}, "initial_hint": {"map_id": "L1"}, "images": []}
        session.finalize.return_value = {"truncated": True}
        monkeypatch.setattr(runner, "NavigationSession", Mock(return_value=session))

    phases = []

    def prompt(text, images, execute, seconds, on_event, cancel_event, prompt_id=None):
        phases.append(text)
        initial = core.request.call_args_list[0].args[1]
        assert initial["contextWindow"] - initial["compaction"]["reserveTokens"] == 98000
        assert initial["compaction"]["keepRecentTokens"] == 18000
        if kind == "query" or len(phases) == 2:
            before = core.request.call_count
            if kind == "compaction":
                forced = [call.args[1]["compaction"] for call in core.request.call_args_list
                          if call.args[0] == "configure" and "compaction" in call.args[1]][-1]
                assert initial["contextWindow"] - forced["reserveTokens"] < 2000, \
                    "The short diagnostic must trigger even with a 128k context window"
            on_event({"type": "event", "event": {"type": "compaction_end",
                      "data": {"result": {"summary": "fixture summary"}}}})
            if kind == "query":
                assert core.request.call_count == before, "Normal compaction must not override runtime settings"
            else:
                assert core.request.call_args.args == ("configure", {
                    "compaction": {"enabled": True, **initial["compaction"]}})
        return {"reason": "model_finished", "answer": "continued", "requests": len(phases),
                "last": {"stopReason": "stop"}}

    core.run_prompt.side_effect = prompt
    directory = tmp_path / kind
    runner.run_case(repo, config, kind, directory=directory)
    result = json.loads((directory / "result.json").read_text())
    assert result["error"] is None
    assert len(phases) == (2 if kind == "compaction" else 1)
    if kind == "compaction":
        assert result["compaction"] == {"compacted": True, "count": 1}


@pytest.mark.parametrize("kind", ["query", "coding"])
def test_provider_failure_cannot_publish_credentials_in_result_archive(app, tmp_path, monkeypatch, kind):
    class FailingCore:
        close_status = {"acknowledged": True}

        def __init__(self, *_):
            pass

        def request(self, op, args):
            raise RuntimeError(f"Provider rejected supplied token {args['auth']['apiKey']}")

        def close(self):
            pass

        def drain_events(self, _):
            pass

    monkeypatch.setattr(runtime, "CoreClient", FailingCore)
    directory = tmp_path / kind
    result = run_app(kind, app, directory)
    assert "Provider rejected supplied token" in result["error"]
    for path in (directory / "result.json", directory / "private/trace.jsonl"):
        assert app[2] not in path.read_text()
        assert "[REDACTED]" in path.read_text()


@pytest.mark.parametrize("settled", [
    {"reason": "request_budget"},
    {"reason": "output_limit", "last": {"stopReason": "length"}},
    {"reason": "tool_timeout"},
])
def test_environment_completion_has_priority_over_core_stop_reason(settled):
    assert runner.finish_reason({"done": True, "current": {"reason": "local_event_complete"}},
                                settled, "cleanup failed") == "local_event_complete"


@pytest.mark.parametrize("settled,expected", [
    ({"reason": "request_budget"}, "request_budget"),
    ({"reason": "model_finished", "last": {"stopReason": "length"}}, "output_limit"),
    ({"reason": "tool_timeout"}, "tool_timeout"),
    ({"reason": "timeout"}, "timeout"),
    ({"reason": "aborted"}, "timeout"),
    ({"reason": "model_error"}, "model_error"),
])
def test_core_failure_reasons_are_not_reported_as_model_completion(settled, expected):
    assert runner.finish_reason({"done": False}, settled) == expected


def test_invalid_tool_result_is_not_reported_as_completion():
    assert runner.finish_reason({"done": False}, {"reason": "tool_result_invalid"}) == "tool_result_invalid"


def test_trace_preserves_core_sequence_and_file_metadata_without_retaining_context(tmp_path):
    class Context(dict):
        pass

    secret = "fixture-trace-secret"
    writer = TraceWriter(tmp_path, secrets=[secret])
    context = Context(messages=[{"role": "user", "content": f"current evidence {secret}"}])
    reference = weakref.ref(context)
    writer.accept({"type": "event", "seq": 21, "event": {
        "timestamp": "2026-10-14T00:00:00Z", "type": "model_input", "data": {"context": context}}})
    del context
    gc.collect()
    assert reference() is None, "Audit logging retained a model context in Python memory"
    metadata = {"label": "after_tool", "changes": [{"path": "memory/state.json", "before": None,
                                                        "after": "content-sha", "bytes": 123}]}
    writer.accept({"type": "event", "seq": 22, "event": {"type": "workspace_changes", "data": metadata}})
    tool = {"toolName": "view_images", "toolCallId": "call-actual", "isError": False,
            "result": {"images": ["scratch/images/crop_001.png"], "metadata": {"step": 2}}}
    writer.accept({"type": "event", "seq": 23, "event": {"type": "tool_execution_end", "data": tool}})
    writer.accept({"type": "tool_request", "call_id": "call-actual", "name": "view_images", "args": {}})
    writer.close()
    writer.close()
    raw = (tmp_path / "trace.jsonl").read_text()
    rows = [json.loads(line) for line in raw.splitlines()]
    assert [row["type"] for row in rows] == ["model_input", "workspace_changes", "tool_execution_end"]
    assert [row["core_seq"] for row in rows] == [21, 22, 23]
    assert rows[0]["timestamp"] == "2026-10-14T00:00:00Z"
    assert rows[1]["data"] == metadata and rows[2]["data"] == tool
    assert secret not in raw and "[REDACTED]" in raw
    assert writer.calls == [{"name": "view_images", "isError": False}]


@pytest.mark.parametrize("argument", ["--max-requests", "--seconds"])
def test_cli_rejects_explicit_zero_limits(monkeypatch, argument):
    from src.harness import __main__ as entry

    monkeypatch.setattr(sys, "argv", ["harness", "run", "--case", "local", argument, "0"])
    monkeypatch.setattr(entry, "load_config", lambda *_: {})
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2
