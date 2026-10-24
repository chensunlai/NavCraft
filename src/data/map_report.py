"""Summarize saved map experiment evidence without API, GPU or source mutations.

Run ``python -m src.data.map_report --run runs/<experiment>``. Request time and
usage come from each unique audit record, never from rescoring elapsed time.
Initial attempts and targeted retries remain separate observations.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re


METHODS = ("baseline", "point_depth", "active", "repeatvote", "analyst", "hybrid")


def _read(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _resolve(value, root):
    path = Path(value)
    if path.is_absolute():
        return path
    # Audit paths were written relative to the repository, not the run folder.
    for base in (Path.cwd(), Path(__file__).resolve().parents[2], root, root.parent):
        candidate = base / path
        if candidate.exists():
            return candidate.resolve()
    return (Path(__file__).resolve().parents[2] / path).resolve()


def _leaves(value):
    if isinstance(value, dict):
        if value.get("record_path"):
            yield value
        elif isinstance(value.get("reviews"), list):
            for child in value["reviews"]:
                yield from _leaves(child)
        elif isinstance(value.get("result"), dict):
            yield from _leaves(value["result"])


def _text_parts(payload):
    result = []
    if not isinstance(payload, dict):
        return result
    for item in payload.get("output", []):
        if item.get("type") == "message":
            result.extend(part.get("text", "") for part in item.get("content", [])
                          if part.get("type") == "output_text")
    for choice in payload.get("choices", []):
        content = choice.get("message", {}).get("content")
        if isinstance(content, str):
            result.append(content)
    return result


def _error_category(record, payload, text):
    error = record.get("error") or {}
    notice = " ".join(_text_parts(payload)) or text
    if (record.get("http_status") in {502, 503, 504}
            or re.match(r"\s*(?:error:\s*)?(?:ai\s+)?provider\s+(?:is\s+)?temporarily\s+unavailable\b",
                        notice, re.IGNORECASE)):
        return "ProviderUnavailable"
    if isinstance(payload, dict):
        if payload.get("status") == "incomplete" or any(
                choice.get("finish_reason") in {"length", "content_filter"} for choice in payload.get("choices", [])):
            return "Incomplete"
        if payload.get("status") in {"failed", "cancelled", "in_progress", "queued"}:
            return "ResponseNotComplete"
        provider_error = payload.get("error")
        if isinstance(provider_error, dict) and str(provider_error.get("code")) in {"502", "503", "504"}:
            return "ProviderUnavailable"
    return error.get("type") if not record.get("ok") else None


def _visible_text(request):
    texts = []
    if not isinstance(request, dict):
        return texts
    for item in request.get("input", request.get("messages", [])):
        if not isinstance(item, dict):
            continue
        content = item.get("content", [])
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(part.get("text", "") for part in content
                         if part.get("type") in {"input_text", "text"})
    return texts


def _schema_in_text(request):
    if not isinstance(request, dict):
        return None
    schema = request.get("text", {}).get("format", {}).get("schema")
    if schema is None:
        schema = request.get("response_format", {}).get("json_schema", {}).get("schema")
    if schema is None:
        return None
    expected = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    # Whitespace-normalized matching avoids depending on a prompt marker/version.
    compact = lambda text: re.sub(r"\s+", "", text)
    return any(compact(expected) in compact(text) for text in _visible_text(request))


def _usage(payload):
    if not isinstance(payload, dict):
        return None
    result = {}
    names = {"input_tokens": ("input_tokens", "prompt_tokens"),
             "output_tokens": ("output_tokens", "completion_tokens"), "total_tokens": ("total_tokens",)}
    for name, alternatives in names.items():
        result[name] = next((payload[key] for key in alternatives
                             if isinstance(payload.get(key), (int, float)) and not isinstance(payload[key], bool)), None)
    for name in ("input_tokens_details", "output_tokens_details", "prompt_tokens_details", "completion_tokens_details"):
        if isinstance(payload.get(name), dict):
            result[name] = {key: value for key, value in payload[name].items()
                            if isinstance(value, (int, float)) and not isinstance(value, bool)}
    return result


def _call(record_path, root):
    path = _resolve(record_path, root)
    record = _read(path)
    if not isinstance(record, dict):
        return {"record_path": str(path), "state": "not_ready", "reason": "audit record missing or incomplete"}
    response_path = path.parent / "response.txt"
    try:
        text = response_path.read_text(encoding="utf-8")
    except OSError:
        text = ""
    try:
        payload = json.loads(text)
    except ValueError:
        payload = None
    request = _read(path.parent / "request.json")
    category = _error_category(record, payload, text)
    actual_model = record.get("response_model") or (payload.get("model") if isinstance(payload, dict) else None)
    requested_model = record.get("requested_model") or record.get("model")
    if record.get("route") == "models":
        requested_model = None
    tool_calls = []
    if isinstance(payload, dict):
        tool_calls.extend({"name": item.get("name"), "call_id": item.get("call_id")}
                          for item in payload.get("output", []) if item.get("type") == "function_call")
        for choice in payload.get("choices", []):
            tool_calls.extend({"name": item.get("function", {}).get("name"), "call_id": item.get("id")}
                              for item in choice.get("message", {}).get("tool_calls", []))
    seconds = record.get("seconds")
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
        seconds = None
    complete_record = bool(record.get("ok") or record.get("error"))
    return {"record_path": str(path), "response_path": str(response_path),
            "state": "ready" if complete_record else "not_ready",
            "request_id": record.get("request_id"), "created_utc": record.get("created_utc"),
            "requested_model": requested_model, "response_model": actual_model or "unknown",
            "model_mapping_source": "audit" if record.get("response_model") else "raw_response" if actual_model else "unknown",
            "reasoning_effort": record.get("reasoning_effort"), "route": record.get("route"),
            "http_status": record.get("http_status"), "http_attempted": record.get("http_status") is not None or "latency_seconds" in record,
            "audit_ok": record.get("ok", False), "reported_error_type": (record.get("error") or {}).get("type"),
            "error_category": category, "seconds": seconds, "latency_seconds": record.get("latency_seconds"),
            "usage": _usage(record.get("usage") or (payload.get("usage") if isinstance(payload, dict) else None)),
            "schema_in_text": _schema_in_text(request), "native_tool_calls": tool_calls,
            "references": []}


def _quality(value):
    if not isinstance(value, dict):
        return None
    result = {"pass": value.get("pass")}
    for name in ("traversability", "leaks", "alignment"):
        section = value.get(name)
        if not isinstance(section, dict):
            continue
        result[name] = {key: item for key, item in section.items()
                        if item is None or isinstance(item, (str, int, float, bool)) or key == "reasons"}
        if name == "traversability":
            result[name]["split_components"] = sum(bool(item.get("split")) for item in section.get("components", []))
    return result


def _outcome(value, path):
    if not isinstance(value, dict):
        return {"state": "not_ready", "path": str(path)}
    return {"state": "ready", "path": str(path), "model_output_valid": value.get("model_output_valid"),
            "changed_pixels": value.get("changed_pixels"), "before": _quality(value.get("before")),
            "after": _quality(value.get("after")), "point_count": value.get("point_count"),
            "valid_depth_points": value.get("valid_depth_points"),
            "multiview_world_agreements": value.get("multiview_world_agreements"),
            "geometry_decisions": [{"accepted": item.get("accepted"), "changed_pixels": item.get("changed_pixels"),
                                    "reason": item.get("reason")} for item in value.get("geometry_decisions", [])]}


def summarize(run_folder):
    root = Path(run_folder).resolve(strict=True)
    calls, entries, suites, readiness = {}, [], [], []

    def add(value, source, method, phase, model=None, scene=None, outcome=None, outer=None):
        leaves = list(_leaves(value))
        refs = []
        for leaf in leaves:
            paths = leaf.get("attempts") or [leaf["record_path"]]
            for raw in paths:
                path = str(_resolve(raw, root))
                call = calls.setdefault(path, _call(raw, root))
                reference = {"source": str(source), "method": method, "phase": phase, "scene": scene}
                if reference not in call["references"]:
                    call["references"].append(reference)
                if path not in refs:
                    refs.append(path)
        points = value.get("projected_points", (outer or {}).get("projected_points", []))
        parsed_points = value.get("parsed", {}).get("points", []) if isinstance(value.get("parsed"), dict) else []
        errors = [leaf["error"].get("type", "Unknown") for leaf in leaves if leaf.get("error")]
        events = value.get("tool_events", [])
        schema_values = {calls[path].get("schema_in_text") for path in refs}
        protocol = ("schema_in_text" if schema_values == {True} else "without_schema_text"
                    if schema_values == {False} else "unknown" if schema_values <= {None}
                    else "mixed_schema_transport")
        entry = {"source": str(source), "method": method, "phase": phase, "scene": scene,
                 "source_id": value.get("source_id", value.get("index")),
                 "protocol": protocol,
                 "requested_model": model or next((calls[path].get("requested_model") for path in refs), None),
                 "status": "not_ready" if not leaves else "valid" if value.get("ok") else "failed",
                 "result_ok": value.get("ok"), "result_error_types": errors,
                 "audit_records": refs, "tool_callback_events": len(events),
                 "executed_tool_events": sum("result" in event for event in events),
                 "failed_tool_events": sum("result" not in event for event in events),
                 "executed_tool_names": dict(Counter(event.get("name", "unknown") for event in events if "result" in event)),
                 "point_count": len(points) if points else len(parsed_points),
                 "valid_depth_points": sum(bool(item.get("ok")) for item in points) if points else None,
                 "distinct_annotated_images": len({item.get("image") for item in points or parsed_points}) if points or parsed_points else None,
                 "empty_point_result": method == "point_depth" and value.get("ok") is True and not points and not parsed_points}
        if (outer or {}).get("score"):
            entry["vision_correct"] = outer["score"].get("vision_correct")
        if outcome is not None:
            entry["outcome"] = outcome
        entries.append(entry)

    for filename, phase in (("model_initial_reviews.json", "initial_review"),
                            ("final_model_initial_reviews.json", "initial_review_final")):
        initial = root / filename
        data = _read(initial)
        if isinstance(data, dict):
            for value in data.get("reviews", []):
                add(value, initial, "initial_review", phase,
                    value.get("model") or value.get("requested_model"), value.get("scene"))
        else:
            readiness.append({"source": str(initial), "status": "not_ready"})
    for source in sorted((root / "probes").glob("*.json")):
        data = _read(source)
        if not isinstance(data, dict):
            readiness.append({"source": str(source), "status": "not_ready"})
            continue
        value = data.get("result", data)
        if list(_leaves(value)):
            method = "model_list" if value.get("route") == "models" else "vision_probe"
            phase = "probe_retry" if "retry" in source.name else "probe"
            add(value, source, method, phase, outer=data)
    for folder in sorted((root / "model_methods").glob("*/*")):
        if not folder.is_dir():
            continue
        source = folder / "model_results.json"
        data = _read(source)
        summary_path = folder / "summary.json"
        summary = _read(summary_path)
        suite = {"folder": str(folder), "scene": folder.parent.name, "requested_model": folder.name,
                 "model_results_state": "ready" if isinstance(data, dict) else "not_ready",
                 "summary_state": "ready" if isinstance(summary, dict) else "not_ready",
                 "summary_elapsed_not_used_for_call_cost": summary.get("seconds") if isinstance(summary, dict) else None,
                 "summary_tool_calls": summary.get("tool_calls") if isinstance(summary, dict) else None}
        suites.append(suite)
        if isinstance(data, dict):
            for method in data.get("methods", {}):
                value = data.get("methods", {}).get(method)
                if not isinstance(value, dict):
                    readiness.append({"source": str(source), "method": method, "status": "not_ready"})
                    continue
                path = folder / method / "outcome.json"
                outcome = _read(path) or (summary.get("methods", {}).get(method) if isinstance(summary, dict) else None)
                add(value, source, method, "initial_suite", data.get("model"), folder.parent.name,
                    outcome=_outcome(outcome, path))
        else:
            readiness.append({"source": str(source), "status": "not_ready"})
        for targeted in sorted(folder.glob("*targeted*.json")):
            data = _read(targeted)
            if not isinstance(data, dict):
                readiness.append({"source": str(targeted), "status": "not_ready"})
                continue
            value = data.get("result", data)
            method = "point_depth" if "point" in targeted.name else "active" if "active" in targeted.name else "targeted"
            add(value, targeted, method, "targeted_retry", data.get("requested_model", folder.name),
                folder.parent.name, outer=data)

    aggregates = {}
    for entry in entries:
        key = (entry["requested_model"] or "service_metadata", entry["method"], entry["phase"], entry["protocol"])
        group = aggregates.setdefault(key, {"requested_model": key[0], "method": key[1], "phase": key[2],
                     "protocol": key[3],
                     "entry_count": 0, "valid_results": 0, "failed_results": 0, "not_ready_results": 0,
                     "audit_records": [], "executed_tool_events": 0, "point_count": 0,
                     "failed_tool_events": 0,
                     "valid_depth_points": 0, "entries_with_depth_results": 0, "empty_point_results": 0,
                     "changed_pixels": 0, "outcomes_ready": 0, "quality_transitions": Counter(), "result_errors": Counter()})
        group["entry_count"] += 1
        group[{"valid": "valid_results", "failed": "failed_results", "not_ready": "not_ready_results"}[entry["status"]]] += 1
        group["audit_records"].extend(path for path in entry["audit_records"] if path not in group["audit_records"])
        group["executed_tool_events"] += entry["executed_tool_events"]
        group["failed_tool_events"] += entry["failed_tool_events"]
        group["point_count"] += entry["point_count"]
        if entry["valid_depth_points"] is not None:
            group["entries_with_depth_results"] += 1
            group["valid_depth_points"] += entry["valid_depth_points"]
        group["empty_point_results"] += bool(entry["empty_point_result"])
        group["result_errors"].update(entry["result_error_types"])
        outcome = entry.get("outcome", {})
        if outcome.get("state") == "ready":
            group["outcomes_ready"] += 1
            group["changed_pixels"] += outcome.get("changed_pixels") or 0
            group["quality_transitions"][f"{outcome.get('before', {}).get('pass')}->{outcome.get('after', {}).get('pass')}"] += 1
    for group in aggregates.values():
        audited = [calls[path] for path in group["audit_records"]]
        group.update(_cost(audited))
        group["result_errors"] = dict(group["result_errors"])
        group["quality_transitions"] = dict(group["quality_transitions"])
    unique = list(calls.values())
    return {"generated_utc": datetime.now(timezone.utc).isoformat(), "run_folder": str(root),
            "methodology": ["Audit request times and usage are summed once per unique record path.",
                "Reported tokens are never estimated; unknown values remain null with missing counts.",
                "Initial suites, targeted retries, probes and initial visual reviews are separate groups.",
                "response_model is read from audit or raw response; unknown is not inferred from requested alias.",
                "Errors are classified from saved evidence without changing the original audit records.",
                "Schema-in-text checks the actual saved request body; absent schema is distinct from targeted retry.",
                "Pixel changes and quality transitions are observations, not proof of improvement.",
                "Incomplete files or running suites are not_ready, not model errors."],
            "total_unique_calls": _cost(unique), "model_method_groups": sorted(aggregates.values(), key=lambda row: (row["requested_model"], row["method"], row["phase"], row["protocol"])),
            "suites": suites, "readiness": readiness, "entries": entries,
            "calls": sorted(unique, key=lambda row: row.get("created_utc") or row["record_path"])}


def _cost(calls):
    ready = [call for call in calls if call.get("state") == "ready"]
    seconds = [call["seconds"] for call in ready if call.get("seconds") is not None]
    tokens = {}
    missing = {}
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        values = [(call.get("usage") or {}).get(name) for call in ready]
        known = [value for value in values if value is not None]
        tokens[name] = sum(known) if known else None
        missing[name] = len(values) - len(known)
    return {"audit_record_count": len(calls), "calls": sum(call.get("http_attempted", False) for call in ready),
            "not_ready_audit_records": len(calls) - len(ready), "sum_seconds": sum(seconds) if seconds else None,
            "seconds_missing_count": len(ready) - len(seconds), "reported_tokens": tokens,
            "usage_missing_call_counts": missing, "http_errors": dict(Counter(
                str(call["http_status"]) for call in ready if isinstance(call.get("http_status"), int) and call["http_status"] >= 400)),
            "errors": dict(Counter(call["error_category"] for call in ready if call.get("error_category"))),
            "native_tool_calls": sum(len(call.get("native_tool_calls", [])) for call in ready),
            "native_tool_names": dict(Counter(tool["name"] or "unknown" for call in ready for tool in call.get("native_tool_calls", []))),
            "returned_models": dict(Counter(call.get("response_model", "unknown") for call in ready)),
            "schema_in_text": dict(Counter("yes" if call.get("schema_in_text") is True else "no" if call.get("schema_in_text") is False else "unknown" for call in ready))}


def markdown(report):
    lines = ["# Saved model experiment evidence", "", f"Run: `{report['run_folder']}`", "",
             "All costs come from saved unique request audits. Token totals include only API-reported usage. "
             "The original records are unchanged; initial attempts and targeted retries remain separate.", "",
             "| Requested model | Method | Phase | Valid/entries | Calls | Seconds | Input/output tokens | Call/result errors | Tools | Points/valid depth | Changed pixels | Before→after pass |",
             "|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|"]
    fmt = lambda value: "unknown" if value is None else str(value)
    for group in report["model_method_groups"]:
        usage = group["reported_tokens"]
        depth = group["valid_depth_points"] if group["entries_with_depth_results"] else None
        seconds = "unknown" if group["sum_seconds"] is None else f"{group['sum_seconds']:.2f}"
        lines.append(f"| {group['requested_model']} | {group['method']} | {group['phase']} / {group['protocol']} | "
                     f"{group['valid_results']}/{group['entry_count']} | {group['calls']} | {seconds} | "
                     f"{fmt(usage['input_tokens'])}/{fmt(usage['output_tokens'])} | "
                     f"{json.dumps(group['errors'], ensure_ascii=False)} / {json.dumps(group['result_errors'], ensure_ascii=False)} | {group['native_tool_calls']} | "
                     f"{group['point_count']}/{fmt(depth)} | "
                     f"{group['changed_pixels'] if group['outcomes_ready'] else 'unscored'} | "
                     f"{json.dumps(group['quality_transitions'])} |")
    lines.extend(["", "Requested and returned model IDs (raw-response IDs supplement old audit fields):", "",
                  "| Requested model | Method / phase | Returned model counts | Schema visible in text |", "|---|---|---|---|"])
    for group in report["model_method_groups"]:
        lines.append(f"| {group['requested_model']} | {group['method']} / {group['phase']} / {group['protocol']} | "
                     f"{json.dumps(group['returned_models'], ensure_ascii=False)} | {json.dumps(group['schema_in_text'])} |")
    unfinished = [item for item in report["suites"] if "not_ready" in (item["model_results_state"], item["summary_state"])]
    lines.extend(["", f"Suites not ready: {len(unfinished)}. Missing or incomplete artifact records: {len(report['readiness'])}.", "",
                  "Changes do not establish improved wall quality. Generic visual findings, empty point lists, "
                  "provider failures and geometry rejection must remain visible. See model_comparison.json for "
                  "per-entry before/after metrics, actual error categories, tool calls, point counts and audit paths.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_folder", nargs="?")
    parser.add_argument("--run", dest="run_option")
    args = parser.parse_args()
    selected = args.run_option or args.run_folder
    if not selected:
        parser.error("supply a run folder or --run")
    report = summarize(selected)
    root = Path(report["run_folder"])
    (root / "model_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (root / "model_comparison.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"output": str(root / "model_comparison.json"), "groups": len(report["model_method_groups"]),
                      "unique_calls": report["total_unique_calls"]["calls"],
                      "not_ready": len(report["readiness"]), "errors": report["total_unique_calls"]["errors"]}))


if __name__ == "__main__":
    main()
