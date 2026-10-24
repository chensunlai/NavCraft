"""Offline snapshots from frozen plans, results and actual request ledgers only."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path

from src.common.io import write_json
from .batch import summarize


def _read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _weighted_metric(results, key):
    if not results:
        return None
    numerator, denominator = 0.0, 0
    for result in results:
        value = result.get("environment", {}).get("metrics", {}).get(key)
        if value is None or not math.isfinite(value):
            return None
        weight = int(result.get("planned_goals", 1)) if result.get("benchmark") == "goat" else 1
        numerator += value * weight
        denominator += weight
    return numerator / denominator if denominator else None


def collect_report(directory):
    directory = Path(directory).resolve()
    plan = _read(directory / "plan.json")
    if plan is None:
        raise ValueError(f"Run has no plan.json: {directory}")
    jobs = plan["jobs"]
    groups, episodes = [], []
    # A model directory is created when its first worker starts. No credentials or
    # current source configuration are needed to read an existing run.
    for model_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
        if not any(model_dir.glob("*/result.json")):
            continue
        model_rows = []
        for job in jobs:
            case = model_dir / f"{job['index']:03d}-{job['benchmark']}"
            result = _read(case / "result.json")
            ledger = _read(case / "private/requests.json")
            attempts = ledger.get("attempts", []) if ledger is not None else None
            row = {
                "model": model_dir.name, "benchmark": job["benchmark"],
                "episode_id": job["episode_id"], "planned_goals": job.get("planned_goals", 1),
                "status": result.get("status", "pending") if result else "pending",
                "finish_reason": result.get("finish_reason") if result else None,
                "requests": len(attempts) if attempts is not None else None,
                "service_errors": sum(bool(a.get("service_error")) for a in attempts) if attempts is not None else None,
                "actions": result.get("environment", {}).get("steps") if result else None,
                "result": str(case.relative_to(directory) / "result.json") if result else None,
            }
            row["charged_requests"] = (row["requests"] - row["service_errors"]
                                       if attempts is not None else None)
            model_rows.append((job, result, row))
            episodes.append(row)
        for benchmark in sorted({job["benchmark"] for job in jobs}):
            members = [item for item in model_rows if item[0]["benchmark"] == benchmark]
            results = [result for _, result, _ in members if result is not None]
            summary = summarize(results, len(members))
            completed = [result for result in results if result.get("status") == "completed"]
            totals = {}
            for field in ("requests", "charged_requests", "service_errors", "actions"):
                present = [row[field] for _, _, row in members if row[field] is not None]
                totals[field] = sum(present) if present else None
                totals[f"episodes_with_{field}"] = len(present)
            observed = {"sr": summary["observed_success_rate"],
                        "spl": _weighted_metric(completed, "spl"),
                        "strict_sr": _weighted_metric(completed, "official_success"),
                        "ne_m": _weighted_metric(completed, "navigation_error") if benchmark in ("r2r", "rxr") else None}
            groups.append({
                "model": model_dir.name, "benchmark": benchmark,
                **summary, "completed": summary["finished"],
                "selected_goals": sum(job.get("planned_goals", 1) for job, _, _ in members),
                "observed_completed_metrics": observed,
                "final_metrics": observed if summary["complete"] else None,
                "totals": totals,
                "usage": {"input_tokens": None, "output_tokens": None, "cached_input_tokens": None,
                          "cost": None, "status": "missing",
                          "reason": "Token/cache/cost usage is not recorded in these plan/result/request files."},
            })
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "directory": str(directory),
            "complete": bool(groups) and all(g["complete"] for g in groups),
            "groups": groups, "episodes": episodes}


def _number(value, percent=False):
    if value is None:
        return "missing"
    return f"{value:.1%}" if percent else f"{value:.3f}"


def write_report(directory):
    directory = Path(directory).resolve()
    report = collect_report(directory)
    write_json(directory / "report.json", report)
    lines = ["# Pilot report", "", f"Snapshot: {report['generated_at']}", "",
             "Metrics below cover completed episodes only. Pending and infrastructure failures are excluded; "
             "final metrics are available only when every selected episode in that group completed.", "",
             "GOAT SR/SPL are weighted by planned goals, including unattempted goals in completed episodes. "
             "ObjectNav uses <1 m (strict <0.1 m), GOAT <1 m (strict <0.25 m), R2R <3 m; all require submission.", "",
             "| Model | Benchmark | Selected | Completed | Pending | Infra | Observed SR | SPL | Strict SR | R2R NE / m | Final SR |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group in report["groups"]:
        metric = group["observed_completed_metrics"]
        final = _number(group["success_rate"], True) if group["complete"] else "not final"
        lines.append(f"| {group['model']} | {group['benchmark']} | {group['selected']} | "
                     f"{group['completed']} | {group['pending']} | {group['infrastructure_errors']} | "
                     f"{_number(metric['sr'], True)} | {_number(metric['spl'])} | "
                     f"{_number(metric['strict_sr'], True)} | {_number(metric['ne_m'])} | {final} |")
    lines += ["", "Request totals include ongoing episodes and service failures; charged requests exclude recorded service failures. "
              "Action totals cover only episodes whose result already records actions. No prices or token counts are inferred.", "",
              "| Model | Benchmark | Actual requests | Charged requests | Service failures | Recorded actions | Action records / selected | Tokens / cache / cost |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
    for group in report["groups"]:
        totals = group["totals"]
        display = lambda key: totals[key] if totals[key] is not None else "missing"
        lines.append(f"| {group['model']} | {group['benchmark']} | {display('requests')} | "
                     f"{display('charged_requests')} | {display('service_errors')} | {display('actions')} | "
                     f"{totals['episodes_with_actions']} / {group['selected']} | missing |")
    lines += ["", "Per-episode status, finish reason and result links are in report.json. "
              "Re-run this command to refresh the snapshot; it does not contact models or change worker state.", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return directory / "report.md"
