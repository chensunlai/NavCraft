"""Episode execution and isolation from native renderer process exits."""

from dataclasses import asdict
import multiprocessing
import os
from pathlib import Path
import time

from src.bench.evaluator import evaluate
from src.common.config import ExperimentConfig
from src.common.io import read_json, redact, write_json
from src.dataset.episodes.base import EpisodeSpec
from src.harness.build import build
from src.harness.trace import TraceWriter


def run_episode(config: ExperimentConfig, episode: EpisodeSpec, directory: Path) -> dict:
    secrets = (config.model.get("api_key", ""),)
    trace = TraceWriter(directory, episode.id, secrets)
    experiment = None
    started = time.monotonic()
    result = {"episode_id": episode.id, "benchmark": episode.benchmark, "status": "setup_error",
              "protocol": config.task["protocol"], "metrics": {}}
    phase = "setup"
    try:
        experiment = build(config, episode, directory, trace)
        initial = experiment.session.start()
        phase = "runtime"
        agent_result = experiment.runtime.run(initial, experiment.session.toolset, trace, experiment.limits)
        result.update(status=agent_result.reason, agent=asdict(agent_result))
        phase = "evaluation"
        result["metrics"] = evaluate(experiment.session.environment, agent_result.reason)
    except Exception as error:
        result["status"] = f"{phase}_error"
        result["error"] = redact(f"{type(error).__name__}: {error}", secrets)
        trace.emit("episode_error", error=result["error"])
        if experiment is not None:
            try:
                result["metrics"] = evaluate(experiment.session.environment, result["status"])
            except Exception as scoring_error:
                result["scoring_error"] = redact(f"{type(scoring_error).__name__}: {scoring_error}", secrets)
    finally:
        try:
            if experiment is not None:
                experiment.close()
        except Exception as close_error:
            result["close_error"] = redact(f"{type(close_error).__name__}: {close_error}", secrets)
            result["status"] = "cleanup_error"
        result["elapsed_seconds"] = time.monotonic() - started
        result = redact(result, secrets)
        trace.emit("episode_end", result=result)
        trace.close()
    write_json(directory / "result.json", result)
    return result


def _child(config: ExperimentConfig, episode: EpisodeSpec, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    vendor = config.environment.get("egl_vendor_file")
    if vendor:
        if not Path(vendor).is_file():
            raise FileNotFoundError(f"EGL vendor file does not exist: {vendor}")
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(vendor)
    write_json(directory / "worker.json", {
        "pid": os.getpid(), "gpu": config.environment.get("gpu"), "egl_vendor_file": vendor,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    })
    with (directory / "worker.log").open("w") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        run_episode(config, episode, directory)


def execute_episode(config: ExperimentConfig, episode: EpisodeSpec, directory: Path) -> dict:
    if config.environment["name"] != "habitat":
        return run_episode(config, episode, directory)
    started = time.monotonic()
    process = multiprocessing.get_context("spawn").Process(target=_child, args=(config, episode, directory))
    process.start()
    process.join(timeout=config.run["timeout_seconds"] + config.environment.get("startup_timeout_seconds", 120) + 30)
    timed_out = process.is_alive()
    if timed_out:
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
    result_file = directory / "result.json"
    if not timed_out and process.exitcode == 0 and result_file.exists():
        result = read_json(result_file)
    else:
        state_file = directory / "private" / "state.json"
        metrics = read_json(state_file).get("metrics", {}) if state_file.exists() else {}
        result = {
            "episode_id": episode.id, "benchmark": episode.benchmark,
            "status": "worker_timeout" if timed_out else "worker_error",
            "protocol": config.task["protocol"], "metrics": metrics,
            "error": "Habitat worker exceeded wall-clock budget" if timed_out else f"Habitat worker exited with code {process.exitcode}; see worker.log",
            "elapsed_seconds": time.monotonic() - started,
        }
    result["worker_exit_code"] = process.exitcode
    process.close()
    write_json(result_file, result)
    return result
