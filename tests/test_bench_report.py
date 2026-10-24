import json

import pytest

from src.bench.batch import summarize
from src.bench.report import collect_report, write_report
from src.common.io import write_json


def result(benchmark, success, *, goals=1, status="completed"):
    metrics = {"partial_success" if benchmark == "goat" else "success": success,
               "spl": success / 2, "official_success": success / 2, "navigation_error": 2.0}
    return {"benchmark": benchmark, "planned_goals": goals, "status": status,
            "finish_reason": "submitted", "environment": {"metrics": metrics, "steps": 7}}


def test_summary_weights_goat_by_planned_goals():
    rows = [result("goat", 1, goals=2), result("goat", 0.25, goals=4)]
    summary = summarize(rows, 2)
    assert summary["evaluated_goals"] == 6
    assert summary["successes"] == 3
    assert summary["success_rate"] == 0.5
    assert summary["complete"]


def test_summary_distinguishes_pending_and_infrastructure_from_failure():
    summary = summarize([result("r2r", 0), result("r2r", 1, status="infrastructure_error")], 3)
    assert summary["finished"] == 1
    assert summary["infrastructure_errors"] == 1
    assert summary["pending"] == 1
    assert summary["observed_success_rate"] == 0
    assert summary["success_rate"] is None
    assert not summary["complete"]
    assert summarize([{"status": "completed"}], 1)["success_rate"] is None


def test_report_offline_mixed_status_and_missing_usage(tmp_path):
    jobs = [{"index": i, "episode_id": f"task:{i}", "benchmark": "r2r", "planned_goals": 1}
            for i in range(4)]
    write_json(tmp_path / "plan.json", {"jobs": jobs})
    write_json(tmp_path / "model/000-r2r/result.json", result("r2r", 1))
    write_json(tmp_path / "model/001-r2r/result.json", result("r2r", 0, status="infrastructure_error"))
    write_json(tmp_path / "model/002-r2r/result.json", {"status": "running", "benchmark": "r2r"})
    write_json(tmp_path / "model/000-r2r/private/requests.json", {
        "attempts": [{"number": 1}, {"number": 2, "service_error": True}, {"number": 3}]})
    write_json(tmp_path / "model/002-r2r/private/requests.json", {"attempts": [{"number": 1}]})
    report = collect_report(tmp_path)
    group, = report["groups"]
    assert (group["selected"], group["completed"], group["pending"], group["infrastructure_errors"]) == (4, 1, 2, 1)
    assert group["observed_completed_metrics"] == {"sr": 1, "spl": 0.5, "strict_sr": 0.5, "ne_m": 2}
    assert group["final_metrics"] is None
    assert group["totals"]["requests"] == 4
    assert group["totals"]["charged_requests"] == 3
    assert group["totals"]["actions"] == 14
    assert group["totals"]["episodes_with_actions"] == 2
    assert group["usage"]["status"] == "missing"
    assert group["usage"]["cost"] is None
    output = write_report(tmp_path)
    assert "not final" in output.read_text()
    assert "missing" in output.read_text()
    assert json.loads((tmp_path / "report.json").read_text())["complete"] is False
    assert json.loads((tmp_path / "model/002-r2r/result.json").read_text())["status"] == "running"


def test_report_goat_goal_weighting(tmp_path):
    write_json(tmp_path / "plan.json", {"jobs": [
        {"index": 0, "episode_id": "goat:0", "benchmark": "goat", "planned_goals": 2},
        {"index": 1, "episode_id": "goat:1", "benchmark": "goat", "planned_goals": 4}]})
    write_json(tmp_path / "model/000-goat/result.json", result("goat", 1, goals=2))
    write_json(tmp_path / "model/001-goat/result.json", result("goat", 0.25, goals=4))
    group, = collect_report(tmp_path)["groups"]
    assert group["selected_goals"] == 6
    assert group["final_metrics"] == {"sr": 0.5, "spl": 0.25, "strict_sr": 0.25, "ne_m": None}
    assert group["totals"]["requests"] is None


def test_report_cli_does_not_load_environment_or_config(tmp_path, monkeypatch):
    from src.bench import __main__
    write_json(tmp_path / "plan.json", {"jobs": []})
    monkeypatch.setattr(__main__, "plan", lambda *_: pytest.fail("report regenerated tasks"))
    monkeypatch.setattr(__main__, "read_config", lambda *_: pytest.fail("report read live config"))
    assert __main__.main(["report", "--output", str(tmp_path)]) == 0
    assert (tmp_path / "report.md").exists()
