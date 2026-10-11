"""Native-child exits and watchdog termination must leave durable failed rows."""

import json

import pytest

from src.bench import run as run_module
from src.bench import worker
from src.common.config import ExperimentConfig
from src.common.io import write_json
from src.common.register import episode_registry
from src.dataset.episodes import EpisodeSpec
from src.environment.fake import FakeSimulator


def episode(number):
    return EpisodeSpec(
        id=f"goat:fixture:scene:{number}", source_id=str(number), benchmark="goat",
        scene_id="not-loaded.glb", start_position=[0, 0, 0], start_rotation=[0, 0, 0, 1],
        goals=[{"modality": "object", "instruction": "Find chair", "positions": [[0, 0, 0]]}],
        source_file="fixture",
    )


def config(tmp_path):
    return ExperimentConfig(
        episodes={"name": "goat", "root": "unused"},
        task={"name": "goat", "protocol": "fixture-goat", "max_steps": 20,
              "max_subtask_steps": 10, "success_distance": 3},
        environment={"name": "habitat", "startup_timeout_seconds": 11},
        model={"name": "scripted"}, harness={"runtime": "agent"},
        run={"name": "worker-fixture", "output_dir": str(tmp_path / "runs"),
             "timeout_seconds": 7, "max_requests": 2, "max_context_chars": 10000}, sources={},
    )


class ProcessDouble:
    """Lifecycle double: emulate abrupt native exit or a stuck native call."""

    def __init__(self, *, target, args, stuck=False):
        self.args = args
        self.stuck = self.alive = stuck
        self.exitcode = None
        self.joins = []
        self.terminated = self.killed = self.closed = False

    def start(self):
        _, specimen, directory = self.args
        if self.stuck:
            write_json(directory / "private/state.json", {"metrics": {"partial_success": 0.5, "steps": 4}})
        elif specimen.source_id == "1":
            self.exitcode = 0
            write_json(directory / "result.json", {
                "episode_id": specimen.id, "benchmark": "goat", "status": "terminated",
                "metrics": {"partial_success": 1}, "elapsed_seconds": 0.1,
            })
        else:
            self.exitcode = -11
            write_json(directory / "private/state.json", {"metrics": {"partial_success": 0.5, "steps": 4}})
            # An interrupted final artifact must not hide a nonzero child exit.
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "result.json").write_text("{interrupted")

    def join(self, timeout):
        self.joins.append(timeout)

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True
        # Simulate a native call that remains stuck after SIGTERM.

    def kill(self):
        self.killed, self.alive, self.exitcode = True, False, -9

    def close(self):
        self.closed = True


def install_process_double(monkeypatch, stuck=False):
    processes = []

    class Context:
        def Process(self, **kwargs):
            result = ProcessDouble(**kwargs, stuck=stuck)
            processes.append(result)
            return result

    monkeypatch.setattr(worker.multiprocessing, "get_context", lambda method: Context())
    return processes


def test_native_exit_keeps_progress_and_failed_episode_in_selected_denominator(tmp_path, monkeypatch):
    run_module.load_builtins()
    specimens = [episode(1), episode(2)]

    class Source:
        def __init__(self, config):
            pass

        def load(self):
            return specimens

    monkeypatch.setitem(episode_registry._factories, "goat", Source)
    processes = install_process_double(monkeypatch)
    directory, summary = run_module.run_experiment(config(tmp_path))
    assert summary["selected"] == summary["finished"] == 2
    assert summary["statuses"] == {"terminated": 1, "worker_error": 1}
    assert summary["score_over_selected"] == 0.75
    rows = json.loads((directory / "results.json").read_text())
    failure = rows[1]
    assert failure["worker_exit_code"] == -11
    assert failure["metrics"] == {"partial_success": 0.5, "steps": 4}
    assert "exited with code -11" in failure["error"]
    assert all(process.closed for process in processes)
    assert json.loads((directory / "episodes/0001/result.json").read_text()) == failure


def test_watchdog_kills_stuck_worker_and_preserves_last_durable_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(worker.time, "monotonic", lambda: 100.0)
    processes = install_process_double(monkeypatch, stuck=True)
    destination = tmp_path / "episode"
    result = worker.execute_episode(config(tmp_path), episode(2), destination)
    assert result["status"] == "worker_timeout"
    assert result["worker_exit_code"] == -9
    assert result["metrics"] == {"partial_success": 0.5, "steps": 4}
    process = processes[0]
    assert process.terminated and process.killed and process.closed
    assert process.joins == [7 + 11 + 30, 5, 5]
    assert json.loads((destination / "result.json").read_text()) == result


def test_scoring_failure_is_evaluation_error_and_closes_environment(tmp_path, monkeypatch):
    run_module.load_builtins()
    experiment = config(tmp_path)
    experiment.environment = {"name": "fake", "width": 8, "height": 6}
    experiment.harness["tools"] = ["nav_observe", "nav_step", "nav_finish"]
    experiment.model["turns"] = [{"output": [{
        "type": "function_call", "call_id": "finish", "name": "nav_finish", "arguments": "{}",
    }]}]
    closed = []
    original_close = FakeSimulator.close

    def close(simulator):
        original_close(simulator)
        closed.append(simulator)

    monkeypatch.setattr(FakeSimulator, "close", close)

    def fail_score(environment, reason):
        raise RuntimeError("score computation failed")

    monkeypatch.setattr(worker, "evaluate", fail_score)
    destination = tmp_path / "episode"
    result = worker.run_episode(experiment, episode(1), destination)
    assert result["status"] == "evaluation_error"
    assert result["agent"]["reason"] == "terminated"
    assert result["error"] == "RuntimeError: score computation failed"
    assert result["scoring_error"] == result["error"]
    assert result["metrics"] == {}
    assert len(closed) == 1 and closed[0].closed
    assert json.loads((destination / "result.json").read_text()) == result
    monkeypatch.setattr(run_module, "parse_config_paths", lambda: {})
    monkeypatch.setattr(run_module, "load_config", lambda paths: experiment)
    monkeypatch.setattr(run_module, "run_experiment", lambda config: (destination, {"statuses": {"evaluation_error": 1}}))
    with pytest.raises(SystemExit) as error:
        run_module.main()
    assert error.value.code == 1
