"""Run selected frozen tasks through the same evaluator used for students."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from src.common.io import read_json, write_json
from src.common.progress import Progress

from .records import load_records, save_records, update_manifest


def model_started(directory):
    path = Path(directory) / "private/requests.json"
    return path.exists() and bool(read_json(path).get("attempts"))


def save_summary(root, selection, results, repo, active=None, ended=False):
    teacher = [row for row in results if model_started(Path(repo) / row["run_dir"])]
    summary = {"selected": len(selection), "attempt_records": len(results), "teacher_attempts": len(teacher),
               "startup_failures": len(results) - len(teacher),
               "successful_samples": len({row["sample_id"] for row in teacher if row["score"]["success"]}),
               "by_status": dict(Counter(row["status"] for row in teacher)),
               "by_reason": dict(Counter(row["score"]["reason"] for row in teacher)),
               "status": "ended" if ended else "running", "active_sample": active}
    write_json(Path(root) / "reports/teacher.json", summary)
    update_manifest(root, teacher=summary)
    return summary


def solve(root, config, repo, config_files):
    root, repo = Path(root).resolve(), Path(repo).resolve()
    selection = read_json(root / "teacher_selection.json")
    from .check import check
    checked = check(root, config, repo)
    if not checked["valid"]:
        raise ValueError("Frozen task validation failed; see reports/check.json")
    results_path = root / "solutions.jsonl"
    results = load_records(results_path) if results_path.exists() else []
    run_file = root / "teacher_run.json"
    identity = {"model": config["model"], "effort": config.get("reasoning_effort", "high"),
                "selection": selection, "tool_tier": config.get("tool_tier", "super"),
                "samples_sha256": hashlib.sha256((root / "samples.jsonl").read_bytes()).hexdigest(),
                "content_sha256": checked["content_sha256"]}
    if run_file.exists():
        previous_run = read_json(run_file)
        if any(previous_run.get(key) != value for key, value in identity.items()):
            raise ValueError("Teacher model, effort, selection or frozen samples changed")
        run_root = repo / previous_run["directory"]
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_root = repo / "runs" / f"{stamp}-dataset-sol"
        write_json(run_file, {"directory": run_root.relative_to(repo).as_posix(), **identity})
    if config.get("concurrency", 1) != 1:
        raise ValueError("This acceptance collection uses one teacher worker")
    attempts = int(config.get("generation", {}).get("attempts", 2))
    if not 1 <= attempts <= 2:
        raise ValueError("Acceptance allows at most two independent attempts per task")
    progress = Progress("Teacher tasks", selection)
    for identifier in selection:
        previous = [row for row in results if row["sample_id"] == identifier]
        # A native renderer crash before any model call is a startup diagnostic,
        # not one of the two independent teacher demonstrations.
        used = sum(model_started(repo / row["run_dir"]) for row in previous)
        if any(row["score"]["success"] for row in previous) or used >= attempts:
            progress.set(identifier, "finished" if any(row["score"]["success"] for row in previous) else "failed")
            continue
        progress.set(identifier, "running")
        save_summary(root, selection, results, repo, active=identifier)
        for offset in range(attempts - used):
            attempt = len(previous) + offset + 1
            directory = run_root / identifier / f"attempt-{attempt}"
            existed = directory.exists()
            directory.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-m", "src.bench.subtasks", "--collection", str(root),
                       "--sample", identifier, "--output", str(directory)]
            for path in config_files:
                command.extend(["--config", str(path)])
            path = directory / "result.json"
            if not existed:
                environment = os.environ.copy()
                environment.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
                with (directory / "worker.log").open("w") as log:
                    subprocess.run(command, cwd=repo, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
            if not path.exists():
                tasks = {sample["id"]: sample for sample in load_records(root / "samples.jsonl")}
                write_json(path, {"sample_id": identifier, "task_id": tasks[identifier]["task_id"],
                                  "status": "infrastructure_error",
                                  "score": {"success": False, "reason": "worker_exit", "metrics": {}},
                                  "error": "Worker ended before saving a result; its actions were not replayed."})
            value = read_json(path)
            row = {"sample_id": identifier, "task_id": value["task_id"], "attempt": attempt,
                   "run_dir": directory.relative_to(repo).as_posix(), "status": value["status"],
                   "score": value["score"],
                   "model_started": model_started(directory)}
            results.append(row)
            save_records(results_path, results)
            save_summary(root, selection, results, repo, active=identifier)
            if value["score"]["success"]:
                break
        progress.set(identifier, "finished" if row["score"]["success"] else "failed")
    return save_summary(root, selection, results, repo, ended=True)
