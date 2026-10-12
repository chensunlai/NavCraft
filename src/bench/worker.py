"""Episode execution and isolation from native renderer process exits."""

from dataclasses import asdict
import math
import multiprocessing
import os
from pathlib import Path
import signal
import time

from src.bench.evaluator import evaluate
from src.common.config import ExperimentConfig
from src.common.io import read_json, redact, write_json
from src.dataset.episodes.base import EpisodeSpec
from src.harness.build import build
from src.harness.trace import TraceWriter


def run_episode(
    config: ExperimentConfig, episode: EpisodeSpec, directory: Path
) -> dict:
    secrets = (config.model.get("api_key", ""),)
    trace = TraceWriter(directory, episode.id, secrets)
    experiment = None
    started = time.monotonic()
    result = {
        "episode_id": episode.id,
        "benchmark": episode.benchmark,
        "status": "setup_error",
        "protocol": config.task["protocol"],
        "metrics": {},
        "planned_goals": len(episode.goals),
        "provenance": getattr(episode, "provenance", "official"),
    }
    phase = "setup"
    try:
        experiment = build(config, episode, directory, trace)
        initial = experiment.session.start()
        phase = "runtime"
        agent_result = experiment.runtime.run(
            initial, experiment.session.toolset, trace, experiment.limits
        )
        result.update(status=agent_result.reason, agent=asdict(agent_result))
        phase = "evaluation"
        result["metrics"] = evaluate(
            experiment.session.environment, agent_result.reason
        )
    except Exception as error:
        result["status"] = f"{phase}_error"
        result["error"] = redact(f"{type(error).__name__}: {error}", secrets)
        trace.emit("episode_error", error=result["error"])
        if experiment is not None:
            try:
                result["metrics"] = evaluate(
                    experiment.session.environment, result["status"]
                )
            except Exception as scoring_error:
                result["scoring_error"] = redact(
                    f"{type(scoring_error).__name__}: {scoring_error}", secrets
                )
    finally:
        try:
            if experiment is not None:
                experiment.close()
        except Exception as close_error:
            result["close_error"] = redact(
                f"{type(close_error).__name__}: {close_error}", secrets
            )
            result["status"] = "cleanup_error"
        result["elapsed_seconds"] = time.monotonic() - started
        result = redact(result, secrets)
        trace.emit("episode_end", result=result)
        trace.close()
    write_json(directory / "result.json", result)
    return result


def _child(config: ExperimentConfig, episode: EpisodeSpec, directory: Path) -> None:
    # The parent owns interruption and must reap every active episode.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    directory.mkdir(parents=True, exist_ok=True)
    vendor = config.environment.get("egl_vendor_file")
    if vendor:
        if not Path(vendor).is_file():
            raise FileNotFoundError(f"EGL vendor file does not exist: {vendor}")
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(vendor)
    write_json(
        directory / "worker.json",
        {
            "pid": os.getpid(),
            "gpu": config.environment.get("gpu"),
            "egl_vendor_file": vendor,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )
    with (directory / "worker.log").open("w") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        run_episode(config, episode, directory)


def _read_object(path: Path) -> dict | None:
    try:
        value = read_json(path)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _finite_number(value) -> bool:
    try:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    except OverflowError:
        return False


class EpisodeWorker:
    """One spawned episode, with independently checked deadline and cleanup."""

    def __init__(self, config: ExperimentConfig, episode: EpisodeSpec, directory: Path):
        self.config, self.episode, self.directory = config, episode, Path(directory)
        self.process = None
        self.started: float | None = None
        self.budget_seconds = (
            config.run["timeout_seconds"]
            + config.environment.get("startup_timeout_seconds", 120)
            + 30
        )
        self._start_error: str | None = None
        self._result: dict | None = None
        self._closed = False

    @property
    def remaining_seconds(self) -> float:
        if self.started is None:
            return self.budget_seconds
        return max(0.0, self.budget_seconds - (time.monotonic() - self.started))

    def start(self) -> None:
        if self.started is not None:
            raise RuntimeError("Episode worker already started")
        self.started = time.monotonic()
        try:
            self.process = multiprocessing.get_context("spawn").Process(
                target=_child, args=(self.config, self.episode, self.directory)
            )
            self.process.start()
        except Exception as error:
            self._start_error = redact(
                f"Worker failed to start: {type(error).__name__}: {error}",
                (self.config.model.get("api_key", ""),),
            )

    def poll(self) -> dict | None:
        if self._result is not None:
            return self._result
        if self.started is None:
            raise RuntimeError("Episode worker has not started")
        if self._start_error is not None:
            return self._collect(error=self._start_error)
        if not self.process.is_alive():
            self.process.join(timeout=0)
            return self._collect()
        if self.remaining_seconds <= 0:
            return self._collect(timed_out=True)
        return None

    def wait(self) -> dict:
        """Blocking compatibility path; concurrent runs use poll instead."""
        if self._result is not None:
            return self._result
        if self.started is None:
            raise RuntimeError("Episode worker has not started")
        if self._start_error is not None:
            return self._collect(error=self._start_error)
        self.process.join(timeout=self.remaining_seconds)
        return self._collect(timed_out=self.process.is_alive())

    def _collect(self, timed_out: bool = False, error: str | None = None) -> dict:
        if timed_out:
            self._stop()
        exit_code = self.process.exitcode if self.process is not None else None
        result = None
        if not timed_out and error is None and exit_code == 0:
            result = _read_object(self.directory / "result.json")
            if not self._valid_result(result):
                result = None
                error = "Worker exited with code 0 but result.json is missing or invalid; see worker.log"
        if result is None:
            state = _read_object(self.directory / "private" / "state.json") or {}
            metrics = state.get("metrics", {})
            metrics = metrics.copy() if isinstance(metrics, dict) else {}
            for key in ("success", "partial_success"):
                if metrics.get(key) is not None and not _finite_number(metrics[key]):
                    metrics.pop(key)
            result = {
                "episode_id": self.episode.id,
                "benchmark": self.episode.benchmark,
                "planned_goals": len(self.episode.goals),
                "provenance": getattr(self.episode, "provenance", "official"),
                "status": "worker_timeout" if timed_out else "worker_error",
                "protocol": self.config.task["protocol"],
                "metrics": metrics,
                "error": (
                    "Episode worker exceeded wall-clock budget"
                    if timed_out
                    else error or f"Worker exited with code {exit_code}; see worker.log"
                ),
                "elapsed_seconds": time.monotonic() - self.started,
            }
        result["worker_exit_code"] = exit_code
        self._result = result
        self.close()
        write_json(self.directory / "result.json", result)
        return result

    def _valid_result(self, result: dict | None) -> bool:
        if result is None:
            return False
        elapsed = result.get("elapsed_seconds")
        return (
            result.get("episode_id") == self.episode.id
            and result.get("benchmark") == self.episode.benchmark
            and isinstance(result.get("status"), str)
            and isinstance(result.get("metrics"), dict)
            and all(
                result["metrics"].get(key) is None
                or _finite_number(result["metrics"][key])
                for key in ("success", "partial_success")
            )
            and _finite_number(elapsed)
        )

    def _stop(self) -> None:
        if self.process is not None and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=5)

    def close(self) -> None:
        if not self._closed:
            self._stop()
            if self.process is not None:
                self.process.close()
            self._closed = True


def execute_episode(
    config: ExperimentConfig, episode: EpisodeSpec, directory: Path
) -> dict:
    if config.environment["name"] != "habitat":
        return run_episode(config, episode, directory)
    worker = EpisodeWorker(config, episode, directory)
    try:
        worker.start()
        return worker.wait()
    finally:
        worker.close()
