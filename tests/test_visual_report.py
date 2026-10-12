"""Offline reports keep pending work and repeated diagnostic attempts distinct."""

import json

from src.bench.visual_report import (
    collect_report,
    collect_run,
    render_report,
    trace_costs,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def run_fixture(
    tmp_path, *, task="objectnav", identifiers=None, name="visual-smoke-fixture"
):
    identifiers = identifiers or ["task-0"]
    directory = tmp_path / name
    write(
        directory / "config.json",
        {
            "run": {"name": name},
            "task": {
                "name": task,
                "success_distance": 0.1 if task == "objectnav" else 3,
            },
            "model": {"model": "fixture-model", "api_key": "must-not-appear"},
            "harness": {"version": "abc"},
        },
    )
    write(directory / "selection.json", identifiers)
    return directory


def finish(directory, index, identifier, **overrides):
    row = {
        "episode_id": identifier,
        "benchmark": "objnav",
        "status": "terminated",
        "metrics": {"success": 0, "navigation_error": 0.108, "success_distance": 0.1},
        "elapsed_seconds": 10,
        **overrides,
    }
    write(directory / "episodes" / f"{index:04d}" / "result.json", row)


def test_pending_not_failure_and_precision_stop_keeps_strict_score(tmp_path):
    run = run_fixture(tmp_path, identifiers=["task-0", "task-1", "task-2"])
    finish(run, 0, "task-0")
    (run / "episodes" / "0001").mkdir()
    report = collect_run(run)
    assert report["completed"] == 1 and report["selected"] == 3
    assert report["sr_completed"] == 0
    assert report["sr_selected"] is None
    assert report["episodes"][0]["category"] == "precision_stop"
    assert [p["state"] for p in report["pending"]] == [
        "running_or_incomplete",
        "not_started",
    ]
    assert "must-not-appear" not in json.dumps(report)


def test_pending_trace_costs_count_without_fabricating_completion(tmp_path):
    run = run_fixture(tmp_path, identifiers=["task-0", "task-1"])
    finish(run, 0, "task-0", agent={"requests": 1, "usage": {"input_tokens": 30}})
    trace = run / "episodes" / "0001" / "trace.jsonl"
    trace.parent.mkdir()
    rows = [
        {"event": "model_request", "request_id": "one", "timestamp": 10},
        {
            "event": "model_response",
            "request_id": "one",
            "timestamp": 12,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "input_tokens_details": {"cached_tokens": 40},
            },
        },
        {"event": "model_request", "request_id": "in-flight", "timestamp": 15},
    ]
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    write(run / "diagnostic.json", {"reason": "实施过程中代码调整，仅诊断不可比较"})
    report = collect_run(run)
    assert report["completed"] == 1
    assert report["selected"] == 2
    assert report["sr_selected"] is None
    assert report["costs"]["requests"] == 3
    assert report["costs"]["input_tokens"] == 130
    assert report["costs"]["output_tokens"] == 20
    assert report["costs"]["cached_tokens"] == 40
    assert report["costs"]["response_latency_seconds"] == 2
    assert report["pending"][0]["costs"]["requests"] == 2
    assert report["diagnostic_note"] == "实施过程中代码调整，仅诊断不可比较"
    whole = collect_report([run], cache=tmp_path / "cache", suites=[])
    assert "仅诊断：实施过程中代码调整，仅诊断不可比较" in render_report(
        whole, tmp_path / "Results.md"
    )


def test_goat_micro_denominator_includes_pending_and_unattempted_subgoals(tmp_path):
    run = run_fixture(
        tmp_path,
        task="goat",
        identifiers=["chain-0", "chain-1"],
        name="visual-develop-fixture",
    )
    manifest = tmp_path / "manifest.json"
    write(
        manifest,
        {
            "episodes": [
                {"id": "chain-0", "goals": [{}, {}, {}]},
                {"id": "chain-1", "goals": [{}, {}]},
            ]
        },
    )
    config = json.loads((run / "config.json").read_text())
    config["episodes"] = {"manifest": str(manifest)}
    write(run / "config.json", config)
    finish(
        run,
        0,
        "chain-0",
        benchmark="goat",
        metrics={
            "partial_success": 1 / 3,
            "composite_success": 0,
            "subtask_success": [1, 0, 0],
        },
    )
    report = collect_run(run)
    assert report["planned_subgoals"] == 5
    assert report["successful_subgoals"] == 1
    assert report["goat_micro_selected"] == 0.2
    assert report["goat_chain_sr"] == 0
    finish(
        run,
        1,
        "chain-1",
        benchmark="goat",
        metrics={"composite_success": 1, "subtask_success": [1, 1]},
    )
    report = collect_run(run)
    assert report["goat_micro_selected"] == 0.6
    assert report["goat_chain_sr"] == 0.5
    assert report["sr_selected"] == 0.5
    assert report["episodes"][1]["category"] == "goal_finish"


def test_trace_costs_sum_top_fields_and_fallback_timestamp(tmp_path):
    path = tmp_path / "trace.jsonl"
    rows = [
        {"event": "model_request", "request_id": "one", "timestamp": 10},
        {
            "event": "model_response",
            "request_id": "one",
            "timestamp": 13,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 10,
                "input_tokens_details": {"cached_tokens": 40},
                "attribution": {"items": {"fake": {"input_tokens": 99999}}},
            },
        },
        {"event": "model_request", "request_id": "two", "timestamp": 20},
        {
            "event": "model_response",
            "request_id": "two",
            "timestamp": 100,
            "latency_seconds": 2,
            "usage": {"input_tokens": 50, "output_tokens": 5},
        },
        {"event": "model_request", "request_id": "timeout", "timestamp": 110},
        {"event": "context_compacted"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    report = trace_costs(path)
    assert report["requests"] == 3 and report["responses"] == 2
    assert report["input_tokens"] == 150 and report["cached_tokens"] == 40
    assert report["response_latency_seconds"] == 5
    assert report["compactions"] == 1


def test_overlapping_smoke_develop_kept_separate_and_acceptance_flag_checked(tmp_path):
    smoke = run_fixture(tmp_path)
    develop = run_fixture(tmp_path, name="visual-develop-fixture")
    finish(smoke, 0, "task-0", metrics={"success": 1})
    finish(develop, 0, "task-0", status="model_error")
    suite = tmp_path / "suite.json"
    write(suite, {"stage": "validate", "acceptance_passed": True, "runs": []})
    report = collect_report([smoke, develop], cache=tmp_path / "cache", suites=[suite])
    assert not report["acceptance_passed"]
    assert len(report["overlapping_attempts"]) == 1
    assert [row["stage"] for row in report["runs"]] == ["smoke", "develop"]
    markdown = render_report(report, tmp_path / "docs" / "Results.md")
    assert "不能声称达到 80%" in markdown
    assert "service_error: 1" in markdown
    assert "must-not-appear" not in markdown
