"""Independent episode processes with a bounded queue for each model."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import threading

from src.common.io import write_json
from .samples import read_config


def summarize(results, selected):
    complete = [r for r in results if r.get("status") == "completed"]
    successes = sum(r.get("environment", {}).get("metrics", {}).get("success", 0.0) for r in complete)
    return {"selected": selected, "finished": len(complete),
            "infrastructure_errors": sum(r.get("status") == "infrastructure_error" for r in results),
            "pending": selected - len(complete), "successes": successes,
            "success_rate": successes / len(complete) if complete else None,
            "success_over_selected": successes / selected if selected else None,
            "complete": len(complete) == selected}


def run_batch(repo, pilot, prepared_plan, directory, model_names):
    """Submission order alternates tasks; concurrently finishing order is unconstrained."""
    repo, directory = Path(repo).resolve(), Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "plan.json", prepared_plan)
    write_json(directory / "pilot.json", pilot)
    results = {name: [] for name in model_names}
    lock = threading.Lock()

    def save():
        write_json(directory / "summary.json", {
            name: summarize(rows, len(prepared_plan["jobs"])) for name, rows in results.items()})

    def run_model(name, model_index):
        config_path = repo / pilot["models"][name]
        config = read_config(config_path)

        def execute(job):
            output = directory / name / f"{job['index']:03d}-{job['benchmark']}"
            output.mkdir(parents=True)
            specification = {**job, "directory": str(output), "limits": pilot["limits"],
                             "model_config": str(config_path),
                             "environment": {"scenes": str((repo / pilot["environment"]["scenes"]).resolve()),
                                             "gpu": (job["index"] + model_index) % 2,
                                             "seed": pilot["environment"].get("seed", 0)}}
            write_json(output / "job.json", specification)
            command = [config["python"], "-m", "src.bench", "worker", "--job", str(output / "job.json")]
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
            with lock:
                results[name].append(value)
                save()
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
