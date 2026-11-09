"""Repeat frozen short tasks in isolated processes with one shared concurrency limit."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import threading

from src.common.config import read_config
from src.common.io import read_json, write_json
from src.common.progress import Progress


REPO = Path(__file__).resolve().parents[2]


def summary(rows, jobs):
    groups = {}
    for name in dict.fromkeys(job["model"] for job in jobs):
        planned = [job for job in jobs if job["model"] == name]
        finished = [row for row in rows if row["batch_model"] == name]
        counts = Counter(row.get("score", {}).get("status", row["status"]) for row in finished)
        correct, incorrect = counts["correct"], counts["incorrect"]
        groups[name] = {"planned": len(planned), "ended": len(finished),
                        "remaining": len(planned) - len(finished), "outcomes": dict(counts),
                        "confirmed_successes": correct,
                        "conclusive_success_rate": correct / (correct + incorrect) if correct + incorrect else None,
                        "conclusive_count": correct + incorrect,
                        "confirmed_success_fraction_of_planned": correct / len(planned),
                        "by_kind": {kind: dict(Counter(
                            row.get("score", {}).get("status", row["status"]) for row in finished
                            if row.get("kind") == kind)) for kind in dict.fromkeys(job["kind"] for job in planned)}}
    return {"planned": len(jobs), "ended": len(rows), "remaining": len(jobs) - len(rows),
            "complete": len(rows) == len(jobs), "models": groups,
            "note": "needs_review and infrastructure errors remain explicit; conclusive rate uses only correct/incorrect."}


def run(collection, output, model_files, repeats, harness_config, concurrency):
    collection, output = Path(collection).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "plan.json").exists():
        raise FileExistsError("Choose a new batch output; existing runs are not overwritten")
    tasks = read_json(collection / "manifest.json")["tasks"]
    if len(tasks) != 16 or Counter(task["kind"] for task in tasks) != dict.fromkeys(
            ("ground", "recognize", "revisit", "recover"), 4):
        raise ValueError("Expected the frozen set of four tasks per kind")
    configured = []
    for model_file, repeat in zip(model_files, repeats):
        config = read_config([model_file, harness_config])
        name = config["model"]
        path = output / "configs" / (Path(model_file).stem + ".json")
        write_json(path, config)
        configured.append((name, path, repeat))
    jobs = []
    # Each round visits all kinds; model queues share the same eight worker slots.
    for repeat in range(1, max(repeats) + 1):
        for task in tasks:
            for name, config_path, count in configured:
                if repeat > count:
                    continue
                model_key = config_path.stem
                directory = output / model_key / f"repeat-{repeat}" / task["id"]
                jobs.append({"index": len(jobs), "model": name, "kind": task["kind"], "task_id": task["id"],
                             "repeat": repeat, "task": str(collection / task["path"]),
                             "config": str(config_path), "directory": str(directory)})
    write_json(output / "plan.json", {"collection": str(collection), "concurrency": concurrency,
               "gpus": [0, 1], "jobs": jobs, "started_at": datetime.now(timezone.utc).isoformat()})
    progress = Progress("Short task runs", range(len(jobs)))
    pending, lock, results = Queue(), threading.Lock(), []
    for job in jobs:
        pending.put(job)

    def save():
        write_json(output / "summary.json", summary(results, jobs))

    def worker(slot):
        while True:
            try:
                job = pending.get_nowait()
            except Empty:
                return
            gpu = slot % 2
            progress.set(job["index"], "running")
            log = output / "logs" / f"{job['index']:03d}.log"
            log.parent.mkdir(exist_ok=True)
            command = [sys.executable, "-u", "-m", "src.bench.short_tasks", "--task", job["task"],
                       "--output", job["directory"], "--config", job["config"], "--gpu", str(gpu)]
            environment = os.environ.copy()
            environment.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
            environment.setdefault("MAGNUM_LOG", "quiet")
            environment.setdefault("HABITAT_SIM_LOG", "quiet")
            try:
                with log.open("w") as stream:
                    process = subprocess.Popen(command, cwd=REPO, env=environment,
                                               stdout=stream, stderr=subprocess.STDOUT)
                    write_json(output / "workers" / f"{job['index']:03d}.json",
                               {"pid": process.pid, "gpu": gpu, **job})
                    code = process.wait()
                result_path = Path(job["directory"]) / "result.json"
                value = read_json(result_path) if result_path.exists() else {}
                if value.get("status") not in {"completed", "infrastructure_error"}:
                    value.update(status="infrastructure_error", finish_reason="worker_exit", exit_code=code)
                value.update(batch_model=job["model"], repeat=job["repeat"], episode_id=job["task_id"],
                             kind=job["kind"], gpu=gpu)
                write_json(result_path, value)
            except Exception as error:
                value = {"batch_model": job["model"], "repeat": job["repeat"], "episode_id": job["task_id"],
                         "kind": job["kind"], "status": "infrastructure_error", "error_type": type(error).__name__}
                write_json(Path(job["directory"]) / "result.json", value)
            with lock:
                results.append(value)
                save()
            progress.set(job["index"], "finished" if value["status"] == "completed" else "failed")
            pending.task_done()

    save()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker, slot) for slot in range(concurrency)]
        for future in futures:
            future.result()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, action="append", required=True)
    parser.add_argument("--repeats", type=int, nargs="+", required=True)
    parser.add_argument("--harness-config", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    if len(args.model_config) != len(args.repeats) or any(count < 1 for count in args.repeats):
        parser.error("Provide one positive repeat count per model config")
    if not 1 <= args.concurrency <= 8:
        parser.error("Use 1-8 workers across GPUs 0 and 1")
    print(run(args.collection, args.output, args.model_config, args.repeats,
              args.harness_config, args.concurrency), flush=True)


if __name__ == "__main__":
    main()
