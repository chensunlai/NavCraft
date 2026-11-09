"""Independent episode processes with a bounded queue for each model."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

from src.common.config import merge_config

from src.common.io import redact, write_json
from src.common.progress import Progress
from src.common.config import read_config


def summarize(results, selected):
    complete = [r for r in results if r.get("status") == "completed"]
    infra = sum(r.get("status") == "infrastructure_error" for r in results)
    weights = [int(r.get("planned_goals", 1)) if r.get("benchmark") == "goat" else 1
               for r in complete]
    values = [r.get("environment", {}).get("metrics", {}).get(
        "partial_success" if r.get("benchmark") == "goat" else "success") for r in complete]
    evaluated_goals = sum(weights)
    successes = (sum(value * weight for value, weight in zip(values, weights))
                 if all(value is not None for value in values) else None)
    observed = successes / evaluated_goals if successes is not None and evaluated_goals else None
    finished = len(complete) == selected and selected > 0
    return {"selected": selected, "finished": len(complete),
            "infrastructure_errors": infra, "pending": selected - len(complete) - infra,
            "evaluated_goals": evaluated_goals, "successes": successes,
            "observed_success_rate": observed,
            "success_rate": observed if finished else None, "complete": finished}


def run_batch(repo, pilot, prepared_plan, directory, model_names):
    """Submission order alternates tasks; concurrently finishing order is unconstrained."""
    repo, directory = Path(repo).resolve(), Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "plan.json", prepared_plan)
    write_json(directory / "pilot.json", pilot)
    results = {name: [] for name in model_names}
    lock = threading.Lock()
    progress = Progress("Episodes", [(name, job["index"]) for name in model_names for job in prepared_plan["jobs"]])

    def save():
        write_json(directory / "summary.json", {
            name: summarize(rows, len(prepared_plan["jobs"])) for name, rows in results.items()})

    def run_model(name, model_index):
        config_path = repo / pilot["models"][name]
        config = merge_config(read_config(config_path), pilot.get("model_overrides", {}))

        def execute(job):
            progress.set((name, job["index"]), "running")
            output = directory / name / f"{job['index']:03d}-{job['benchmark']}"
            try:
                output.mkdir(parents=True)
                specification = {**job, "directory": str(output), "limits": pilot["limits"],
                                 "model_config": str(config_path),
                                 "model_overrides": pilot.get("model_overrides", {}),
                                 "environment": {"scenes": str((repo / pilot["environment"]["scenes"]).resolve()),
                                                 "gpu": (job["index"] + model_index) % 2,
                                                 "seed": pilot["environment"].get("seed", 0)}}
                write_json(output / "job.json", specification)
                command = [config.get("python", sys.executable), "-m", "src.bench", "worker", "--job", str(output / "job.json")]
                with (output / "worker.log").open("w", encoding="utf-8") as stream:
                    environment = os.environ.copy()
                    environment.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
                    process = subprocess.Popen(command, cwd=repo, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                    write_json(output / "worker.json", {"pid": process.pid, "started_at": datetime.now(timezone.utc).isoformat()})
                    # PromptLimits owns time accounting, including provider retry exclusions.
                    code = process.wait()
                result_path = output / "result.json"
                if result_path.exists():
                    value = json.loads(result_path.read_text())
                else:
                    value = {"episode_id": job["episode_id"], "benchmark": job["benchmark"],
                             "status": "infrastructure_error", "finish_reason": "worker_exit", "exit_code": code}
                    write_json(result_path, value)
                if value.get("status") == "running":
                    value.update(status="infrastructure_error", finish_reason="worker_exit", exit_code=code)
                    write_json(result_path, value)
            except Exception as error:
                value = {"episode_id": job["episode_id"], "benchmark": job["benchmark"],
                         "status": "infrastructure_error", "finish_reason": "worker_error",
                         "error": redact(str(error)), "error_type": type(error).__name__}
                output.mkdir(parents=True, exist_ok=True)
                write_json(output / "result.json", value)
            with lock:
                results[name].append(value)
                save()
            progress.set((name, job["index"]), "finished" if value.get("status") == "completed" else "failed")
            return value

        with ThreadPoolExecutor(max_workers=int(config["concurrency"])) as pool:
            pending = [pool.submit(execute, job) for job in prepared_plan["jobs"]]
            for future in as_completed(pending):
                future.result()

    save()
    with ThreadPoolExecutor(max_workers=len(model_names)) as pool:
        futures = [pool.submit(run_model, name, i) for i, name in enumerate(model_names)]
        for future in as_completed(futures):
            future.result()
    return directory
