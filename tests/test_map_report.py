"""Saved-evidence fixtures; no model, network, Habitat or GPU work."""

import hashlib
import json
from pathlib import Path
import urllib.request

import pytest

from src.data.map_report import summarize


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def unexpected_request(*args, **kwargs):
        pytest.fail("A saved-evidence report must never make an API request")

    monkeypatch.setattr(urllib.request, "urlopen", unexpected_request)


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    return root


def audit(root, name="one", *, seconds=12.5, usage=None, payload=None, ok=True,
          error=None, schema_in_text=False, response_model=None):
    directory = root / "audit" / name
    record = {"request_id": name, "model": "deepseek-flash", "ok": ok,
              "seconds": seconds, "http_status": 200, "usage": usage, "route": "responses"}
    if error:
        record["error"] = error
    if response_model:
        record["response_model"] = response_model
    schema = {"type": "object"}
    prompt = "Inspect the image."
    if schema_in_text:
        prompt += " Complete JSON Schema: " + json.dumps(schema)
    write_json(directory / "record.json", record)
    write_json(directory / "request.json", {
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        "text": {"format": {"schema": schema}},
    })
    if payload is None:
        payload = {"model": "deepseek-v4.1-flash-1017", "status": "completed", "output": []}
    if isinstance(payload, str):
        (directory / "response.txt").write_text(payload, encoding="utf-8")
    else:
        write_json(directory / "response.txt", payload)
    result = {"ok": ok, "record_path": str(directory / "record.json"),
              "attempts": [str(directory / "record.json")], "seconds": 99999}
    if error:
        result["error"] = error
    return result


def suite(root, methods, *, elapsed=0, folder="scene/deepseek-flash"):
    directory = root / "model_methods" / folder
    write_json(directory / "model_results.json", {"model": "deepseek-flash", "methods": methods})
    write_json(directory / "summary.json", {"seconds": elapsed, "methods": {}})
    return directory


def group(report, method, phase="initial_suite"):
    return next(item for item in report["model_method_groups"]
                if item["method"] == method and item["phase"] == phase)


def test_same_audit_is_counted_once_across_methods_repeats_and_probe(run):
    value = audit(run, usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    # Repeated references represent the same HTTP attempt, not three new calls.
    value["attempts"] *= 2
    suite(run, {"baseline": value, "repeatvote": {"ok": True, "reviews": [value, value]}})
    write_json(run / "probes" / "vision-result.json", {"result": value})
    report = summarize(run)
    assert len(report["calls"]) == 1
    assert report["total_unique_calls"]["calls"] == 1
    assert report["total_unique_calls"]["reported_tokens"]["total_tokens"] == 15
    assert group(report, "repeatvote")["calls"] == 1
    assert len(report["calls"][0]["references"]) == 3


def test_cost_uses_original_request_audit_seconds_not_result_or_rescore_zero(run):
    value = audit(run, seconds=12.5)
    suite(run, {"baseline": value}, elapsed=0)
    report = summarize(run)
    assert report["suites"][0]["summary_elapsed_not_used_for_call_cost"] == 0
    assert report["total_unique_calls"]["sum_seconds"] == 12.5
    assert group(report, "baseline")["sum_seconds"] == 12.5


def test_selected_active_only_suite_does_not_mark_unrequested_modes_pending(run):
    value = audit(run)
    suite(run, {"active": value})
    report = summarize(run)
    assert not any(item.get("method") in {"baseline", "point_depth", "analyst", "repeatvote", "hybrid"}
                   for item in report["readiness"])
    assert [entry["method"] for entry in report["entries"]] == ["active"]


@pytest.mark.parametrize("usage,expected,missing", [
    (None, {"input_tokens": None, "output_tokens": None, "total_tokens": None},
     {"input_tokens": 1, "output_tokens": 1, "total_tokens": 1}),
    ({}, {"input_tokens": None, "output_tokens": None, "total_tokens": None},
     {"input_tokens": 1, "output_tokens": 1, "total_tokens": 1}),
    ({"input_tokens": 10, "output_tokens": 5},
     {"input_tokens": 10, "output_tokens": 5, "total_tokens": None},
     {"input_tokens": 0, "output_tokens": 0, "total_tokens": 1}),
    ({"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
     {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
     {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}),
])
def test_missing_usage_remains_unknown_and_only_explicit_zero_is_zero(run, usage, expected, missing):
    suite(run, {"baseline": audit(run, usage=usage)})
    cost = summarize(run)["total_unique_calls"]
    assert cost["reported_tokens"] == expected
    assert cost["usage_missing_call_counts"] == missing


def test_returned_model_is_recovered_from_raw_response_without_rewriting_audit(run):
    value = audit(run)
    path = Path(value["record_path"])
    original = path.read_bytes()
    assert "response_model" not in json.loads(original)
    suite(run, {"baseline": value})
    report = summarize(run)
    call = report["calls"][0]
    assert call["requested_model"] == "deepseek-flash"
    assert call["response_model"] == "deepseek-v4.1-flash-1017"
    assert call["model_mapping_source"] == "raw_response"
    assert path.read_bytes() == original
    assert group(report, "baseline")["returned_models"] == {"deepseek-v4.1-flash-1017": 1}


@pytest.mark.parametrize("payload", [
    {"status": "completed", "output": [{"type": "message", "content": [
        {"type": "output_text", "text": "AI provider temporarily unavailable (upstream error: 502)"}]}]},
    "AI provider temporarily unavailable (upstream error: 502)",
])
def test_http_200_provider_notice_is_classified_without_overwriting_original_json_error(run, payload):
    value = audit(run, ok=False, error={"type": "JSONDecodeError", "message": "Expecting value"}, payload=payload)
    suite(run, {"active": value})
    report = summarize(run)
    call = report["calls"][0]
    assert call["http_status"] == 200
    assert call["reported_error_type"] == "JSONDecodeError"
    assert call["error_category"] == "ProviderUnavailable"
    assert report["total_unique_calls"]["errors"] == {"ProviderUnavailable": 1}
    assert group(report, "active")["result_errors"] == {"JSONDecodeError": 1}


def test_incomplete_response_is_separate_from_provider_unavailable(run):
    value = audit(run, ok=False, error={"type": "ValueError"},
                  payload={"status": "incomplete", "output": []})
    suite(run, {"analyst": value})
    report = summarize(run)
    assert report["total_unique_calls"]["errors"] == {"Incomplete": 1}


def test_unfinished_suite_and_partially_written_files_are_not_model_failures(run):
    missing = run / "model_methods" / "scene1" / "Qwen3.8-27B"
    partial = run / "model_methods" / "scene2" / "gpt-6.1-sol"
    missing.mkdir(parents=True)
    partial.mkdir(parents=True)
    (partial / "model_results.json").write_text('{"methods":', encoding="utf-8")
    report = summarize(run)
    assert len(report["suites"]) == 2
    assert all(item["model_results_state"] == "not_ready" for item in report["suites"])
    assert all(item["status"] == "not_ready" for item in report["readiness"])
    assert report["total_unique_calls"]["calls"] == 0
    assert report["total_unique_calls"]["errors"] == {}
    assert report["entries"] == []


def test_initial_empty_points_and_schema_visible_targeted_retry_are_kept_separate(run):
    initial = audit(run, "initial", schema_in_text=False)
    initial["parsed"] = {"points": []}
    directory = suite(run, {"point_depth": initial})
    retry = audit(run, "retry", schema_in_text=True)
    retry["parsed"] = {"points": [{"image": "p0.jpg", "pixel": [2, 3]}]}
    points = [{"image": "p0.jpg", "ok": True, "world_xyz": [1, 2, 3]}]
    write_json(directory / "targeted_point_retry.json", {"result": retry, "projected_points": points})
    report = summarize(run)
    first = group(report, "point_depth")
    second = group(report, "point_depth", "targeted_retry")
    assert first["protocol"] == "without_schema_text" and first["empty_point_results"] == 1
    assert second["protocol"] == "schema_in_text" and second["empty_point_results"] == 0
    assert second["point_count"] == 1 and second["valid_depth_points"] == 1
    assert first["calls"] == second["calls"] == 1


def test_final_image_review_keeps_its_own_phase_schema_and_audit_deduplication(run):
    old = {**audit(run, "old", schema_in_text=False), "model": "Qwen3.8-27B", "index": 3}
    final = {**audit(run, "final", schema_in_text=True), "requested_model": "Qwen3.8-27B", "source_id": 3}
    write_json(run / "model_initial_reviews.json", {"reviews": [old]})
    write_json(run / "final_model_initial_reviews.json", {"phase": "initial_review_final", "reviews": [final]})
    # Another saved reference to the final call must not increase global cost.
    write_json(run / "probes" / "reused-result.json", {"result": final})
    report = summarize(run)
    first = group(report, "initial_review", "initial_review")
    second = group(report, "initial_review", "initial_review_final")
    assert first["protocol"] == "without_schema_text"
    assert second["protocol"] == "schema_in_text"
    assert first["calls"] == second["calls"] == 1
    assert report["total_unique_calls"]["calls"] == 2
    assert report["total_unique_calls"]["sum_seconds"] == 25
    entry = next(item for item in report["entries"] if item["phase"] == "initial_review_final")
    assert entry["source_id"] == 3
    assert entry["source"].endswith("final_model_initial_reviews.json")


def test_summarize_is_repeatable_and_leaves_all_saved_evidence_unchanged(run):
    suite(run, {"baseline": audit(run)})
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in run.rglob("*") if path.is_file()}
    first, second = summarize(run), summarize(run)
    first.pop("generated_utc")
    second.pop("generated_utc")
    assert first == second
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in run.rglob("*") if path.is_file()}
    assert before == after
