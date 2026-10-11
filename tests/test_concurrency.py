"""Bounded episode scheduling, isolated spawn workers and interruption recovery."""

import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from types import SimpleNamespace

import pytest

from src.bench import run as run_module
from src.bench import worker as worker_module
from src.common.config import ExperimentConfig, validate
from src.common.io import write_json
from src.common.register import episode_registry
from src.dataset.episodes import EpisodeSpec


def specimen(number):
    return EpisodeSpec(
        id=f"goat:fixture:scene:{number}",
        source_id=str(number),
        benchmark="goat",
        scene_id="not-loaded.glb",
        start_position=[number, 0, 0],
        start_rotation=[0, 0, 0, 1],
        goals=[
            {
                "modality": "object",
                "instruction": f"Find chair specimen-{number}",
                "positions": [[number, 0, 0]],
            }
        ],
        source_file="fixture",
    )


def experiment(tmp_path, *, workers=None):
    config = ExperimentConfig(
        episodes={"name": "goat", "root": "unused"},
        task={
            "name": "goat",
            "protocol": "fixture-goat",
            "max_steps": 20,
            "max_subtask_steps": 10,
            "success_distance": 3,
        },
        environment={"name": "fake", "width": 8, "height": 6},
        model={
            "name": "scripted",
            "turns": [
                {
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "finish",
                            "name": "nav_finish",
                            "arguments": "{}",
                        }
                    ]
                }
            ],
        },
        harness={
            "runtime": "agent",
            "tools": ["nav_observe", "nav_depth", "nav_step", "nav_finish"],
        },
        run={
            "name": "concurrency-fixture",
            "output_dir": str(tmp_path / "runs"),
            "timeout_seconds": 30,
            "max_requests": 2,
            "max_context_chars": 10000,
        },
        sources={},
    )
    if workers is not None:
        config.run["num_workers"] = workers
    return config


def source(monkeypatch, count):
    run_module.load_builtins()
    specimens = [specimen(number) for number in range(count)]

    class Source:
        def __init__(self, config):
            pass

        def load(self):
            return specimens

    monkeypatch.setitem(episode_registry._factories, "goat", Source)
    return specimens


def row(episode, status="terminated"):
    return {
        "episode_id": episode.id,
        "benchmark": episode.benchmark,
        "status": status,
        "metrics": {"partial_success": 1},
        "elapsed_seconds": 0.1,
    }


def controlled_workers(
    monkeypatch, *, releases, limit, interrupt_after=None, statuses=None
):
    """Release chosen workers only at scheduler waits, without wall-clock sleeps."""
    started = []
    active = {}
    completion_order = []
    release_order = iter(releases)

    class Worker:
        def __init__(self, config, episode, directory):
            self.config, self.episode, self.directory = config, episode, directory
            self.process = SimpleNamespace(sentinel=object())
            self.ready = self.closed = False

        def start(self):
            number = int(self.episode.source_id)
            assert len(active) < limit, "scheduler exceeded its active-episode budget"
            assert (
                self.config.model["nested"]["seen"] == []
            ), "worker config leaked across episodes"
            self.config.model["nested"]["seen"].append(number)
            active[number] = self
            started.append(self)

        @property
        def remaining_seconds(self):
            return 100

        def poll(self):
            if not self.ready:
                return None
            completion_order.append(int(self.episode.source_id))
            result = row(
                self.episode,
                (statuses or {}).get(int(self.episode.source_id), "terminated"),
            )
            if result["status"] != "terminated":
                result["metrics"]["partial_success"] = 0
            return result

        def close(self):
            if not self.closed:
                active.pop(int(self.episode.source_id), None)
                self.closed = True

    def wait(sentinels, timeout=None):
        assert set(sentinels) == {item.process.sentinel for item in active.values()}
        if interrupt_after is not None and len(completion_order) >= interrupt_after:
            raise KeyboardInterrupt
        number = next(release_order)
        assert number in active, "a free slot was not promptly refilled"
        active[number].ready = True
        return [active[number].process.sentinel]

    monkeypatch.setattr(run_module, "EpisodeWorker", Worker)
    monkeypatch.setattr(run_module, "wait", wait)
    return started, active, completion_order


def test_scheduler_refills_slots_preserves_gpu_binding_and_original_order(
    tmp_path, monkeypatch
):
    specimens = source(monkeypatch, 5)
    config = experiment(tmp_path, workers=2)
    config.environment["gpus"] = [2, 3]
    config.model["nested"] = {"seen": []}
    original = copy.deepcopy(config)
    started, active, completed = controlled_workers(
        monkeypatch,
        releases=[1, 2, 0, 3, 4],
        limit=2,
    )
    directory, summary = run_module.run_experiment(config)

    assert completed == [1, 2, 0, 3, 4]
    assert [int(item.episode.source_id) for item in started] == list(range(5))
    assert [item.config.environment["gpu"] for item in started] == [2, 3, 3, 3, 2]
    assert all("gpus" not in item.config.environment for item in started)
    assert all(item.closed for item in started) and not active
    assert config == original
    assert summary["selected"] == summary["finished"] == 5
    results = json.loads((directory / "results.json").read_text())
    assert [result["episode_id"] for result in results] == [
        item.id for item in specimens
    ]
    for index, result in enumerate(results):
        assert (
            json.loads((directory / f"episodes/{index:04d}/result.json").read_text())
            == result
        )


@pytest.mark.parametrize(
    "count,gpus,expected",
    [
        (4, [2, 3], [2, 3, 2, 3]),
        (2, [2, 3, 4], [2, 3]),
        (3, None, [2, 2, 2]),
    ],
)
def test_gpu_assignment_with_more_or_fewer_slots_than_cards(
    tmp_path, monkeypatch, count, gpus, expected
):
    source(monkeypatch, count)
    config = experiment(tmp_path, workers=count)
    config.model["nested"] = {"seen": []}
    if gpus is not None:
        config.environment["gpus"] = gpus
    else:
        config.environment["gpu"] = 2
    started, _, _ = controlled_workers(
        monkeypatch, releases=list(range(count)), limit=count
    )
    run_module.run_experiment(config)
    assert [item.config.environment["gpu"] for item in started] == expected


def test_interrupt_closes_active_workers_and_preserves_only_collected_results(
    tmp_path, monkeypatch
):
    specimens = source(monkeypatch, 5)
    config = experiment(tmp_path, workers=2)
    config.model["nested"] = {"seen": []}
    started, active, _ = controlled_workers(
        monkeypatch,
        releases=[1],
        limit=2,
        interrupt_after=1,
    )
    with pytest.raises(KeyboardInterrupt):
        run_module.run_experiment(config)
    assert all(item.closed for item in started) and not active
    (directory,) = (tmp_path / "runs").iterdir()
    results = json.loads((directory / "results.json").read_text())
    assert [result["episode_id"] for result in results] == [specimens[1].id]
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["selected"] == 5 and summary["finished"] == 1
    assert summary["score_over_selected"] == 0.2


@pytest.mark.parametrize("status", ["worker_error", "worker_timeout"])
def test_worker_failure_frees_slot_without_retry_or_cancelling_other_episodes(
    tmp_path, monkeypatch, status
):
    source(monkeypatch, 3)
    config = experiment(tmp_path, workers=2)
    config.model["nested"] = {"seen": []}
    started, active, _ = controlled_workers(
        monkeypatch,
        releases=[0, 1, 2],
        limit=2,
        statuses={0: status},
    )
    directory, summary = run_module.run_experiment(config)
    assert [int(item.episode.source_id) for item in started] == [0, 1, 2]
    assert all(item.closed for item in started) and not active
    assert summary["selected"] == summary["finished"] == 3
    assert summary["statuses"] == {status: 1, "terminated": 2}
    assert summary["score_over_selected"] == pytest.approx(2 / 3)
    results = json.loads((directory / "results.json").read_text())
    assert results[0]["status"] == status


def test_main_returns_130_for_keyboard_interrupt(tmp_path, monkeypatch):
    config = experiment(tmp_path)
    monkeypatch.setattr(run_module, "parse_config_paths", lambda: {})
    monkeypatch.setattr(run_module, "load_config", lambda paths: config)

    def interrupted(config):
        raise KeyboardInterrupt

    monkeypatch.setattr(run_module, "run_experiment", interrupted)
    with pytest.raises(SystemExit) as error:
        run_module.main()
    assert error.value.code == 130


@pytest.mark.parametrize("workers", [None, 1])
def test_default_and_explicit_single_worker_fake_remain_inline(
    tmp_path, monkeypatch, workers
):
    specimens = source(monkeypatch, 2)
    directory, summary = run_module.run_experiment(
        experiment(tmp_path, workers=workers)
    )
    assert summary["statuses"] == {"terminated": 2}
    assert summary["score_over_selected"] == 1
    results = json.loads((directory / "results.json").read_text())
    assert [result["episode_id"] for result in results] == [
        item.id for item in specimens
    ]
    assert not list((directory / "episodes").glob("*/worker.json"))


def test_spawn_fake_scripted_matches_inline_scores(tmp_path, monkeypatch):
    source(monkeypatch, 2)
    config = experiment(tmp_path, workers=1)
    inline_directory, inline_summary = run_module.run_experiment(config)
    config.run["num_workers"] = 2
    spawned_directory, spawned_summary = run_module.run_experiment(config)
    assert spawned_summary == inline_summary
    inline_rows = json.loads((inline_directory / "results.json").read_text())
    spawned_rows = json.loads((spawned_directory / "results.json").read_text())
    for index, (inline, spawned) in enumerate(zip(inline_rows, spawned_rows)):
        assert spawned["episode_id"] == inline["episode_id"]
        assert spawned["metrics"] == inline["metrics"]
        assert spawned["agent"]["requests"] == inline["agent"]["requests"] == 1
        assert spawned["worker_exit_code"] == 0
        assert (spawned_directory / f"episodes/{index:04d}/worker.json").is_file()


def test_spawn_fake_workers_overlap_requests_and_keep_episode_artifacts_isolated(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    specimens = source(monkeypatch, 4)
    config = experiment(tmp_path, workers=2)
    config.environment["gpus"] = [2, 3]
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    requests, errors = [], []
    active = peak = 0

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            nonlocal active, peak
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                requests.append(payload)
                active += 1
                peak = max(peak, active)
            try:
                # A serial scheduler cannot release either response at this barrier.
                barrier.wait(timeout=15)
                response = {
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "finish",
                            "name": "nav_finish",
                            "arguments": "{}",
                        }
                    ],
                    "status": "completed",
                }
            except threading.BrokenBarrierError as error:
                with lock:
                    errors.append(error)
                response = {"error": "episode requests did not overlap"}
            finally:
                with lock:
                    active -= 1
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config.model = {
        "name": "responses",
        "base_url": f"http://127.0.0.1:{server.server_port}/v1",
        "model": "local-fixture",
        "max_retries": 0,
        "timeout_seconds": 20,
        "max_output_tokens": 100,
    }
    try:
        directory, summary = run_module.run_experiment(config)
    finally:
        barrier.abort()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert not errors
    assert peak == 2 and active == 0 and len(requests) == 4
    assert all(len(request["input"]) == 2 for request in requests)
    assert summary["statuses"] == {"terminated": 4}
    assert summary["score_over_selected"] == 1
    metadata = [
        json.loads((directory / f"episodes/{index:04d}/worker.json").read_text())
        for index in range(4)
    ]
    assert len({item["pid"] for item in metadata}) == 4
    assert sorted(item["gpu"] for item in metadata[:2]) == [2, 3]
    results = json.loads((directory / "results.json").read_text())
    assert [result["episode_id"] for result in results] == [
        item.id for item in specimens
    ]
    for index, item in enumerate(specimens):
        events = [
            json.loads(line)
            for line in (directory / f"episodes/{index:04d}/trace.jsonl")
            .read_text()
            .splitlines()
        ]
        assert all(event["episode_id"] == item.id for event in events)
        assert (
            directory / f"episodes/{index:04d}/observations/rgb_000000.png"
        ).is_file()
        assert results[index]["worker_exit_code"] == 0


class FinishedProcess:
    """A process that exits successfully without a usable final artifact."""

    def __init__(self, *, target, args, content=None, startup_error=False):
        self.args, self.content, self.startup_error = args, content, startup_error
        self.exitcode = None
        self.closed = False

    def start(self):
        if self.startup_error:
            raise OSError("fixture spawn failed")
        _, _, directory = self.args
        write_json(
            directory / "private/state.json", {"metrics": {"partial_success": 0.5}}
        )
        if self.content is not None:
            (directory / "result.json").write_text(self.content)
        self.exitcode = 0

    def is_alive(self):
        return False

    def join(self, timeout=None):
        pass

    def close(self):
        self.closed = True


@pytest.mark.parametrize("content", [None, "{interrupted", "[]", "{}"])
def test_zero_exit_with_missing_or_damaged_result_is_worker_error(
    tmp_path, monkeypatch, content
):
    processes = []

    class Context:
        def Process(self, **kwargs):
            process = FinishedProcess(**kwargs, content=content)
            processes.append(process)
            return process

    monkeypatch.setattr(
        worker_module.multiprocessing, "get_context", lambda method: Context()
    )
    config = experiment(tmp_path)
    config.environment["name"] = "habitat"
    result = worker_module.execute_episode(config, specimen(0), tmp_path / "episode")
    assert result["status"] == "worker_error"
    assert result["worker_exit_code"] == 0
    assert result["metrics"] == {"partial_success": 0.5}
    assert processes[0].closed


def test_worker_spawn_failure_is_durable_and_close_is_idempotent(tmp_path, monkeypatch):
    processes = []

    class Context:
        def Process(self, **kwargs):
            process = FinishedProcess(**kwargs, startup_error=True)
            processes.append(process)
            return process

    monkeypatch.setattr(
        worker_module.multiprocessing, "get_context", lambda method: Context()
    )
    destination = tmp_path / "episode"
    worker = worker_module.EpisodeWorker(experiment(tmp_path), specimen(0), destination)
    worker.start()
    result = worker.poll()
    worker.close()
    worker.close()
    assert result["status"] == "worker_error"
    assert "fixture spawn failed" in result["error"]
    assert processes[0].closed
    assert json.loads((destination / "result.json").read_text()) == result


@pytest.mark.parametrize("artifact", ["result", "state"])
def test_invalid_score_artifact_does_not_abort_later_episodes(
    tmp_path, monkeypatch, artifact
):
    source(monkeypatch, 2)
    processes = []

    class ArtifactProcess(FinishedProcess):
        def start(self):
            _, episode, directory = self.args
            self.exitcode = 0
            if episode.source_id == "1":
                write_json(directory / "result.json", row(episode))
                return
            metrics = {"partial_success": "broken", "steps": 4}
            if artifact == "result":
                write_json(
                    directory / "result.json", {**row(episode), "metrics": metrics}
                )
                write_json(
                    directory / "private/state.json",
                    {"metrics": {"partial_success": 0.5}},
                )
            else:
                write_json(directory / "private/state.json", {"metrics": metrics})

    class Context:
        def Process(self, **kwargs):
            process = ArtifactProcess(**kwargs)
            processes.append(process)
            return process

    monkeypatch.setattr(
        worker_module.multiprocessing, "get_context", lambda method: Context()
    )
    config = experiment(tmp_path, workers=2)
    config.environment["name"] = "habitat"
    directory, summary = run_module.run_experiment(config)
    assert summary["selected"] == summary["finished"] == 2
    assert summary["statuses"] == {"worker_error": 1, "terminated": 1}
    assert summary["score_over_selected"] == (0.75 if artifact == "result" else 0.5)
    results = json.loads((directory / "results.json").read_text())
    assert results[0]["metrics"].get("partial_success") != "broken"
    assert results[1]["status"] == "terminated"
    assert all(process.closed for process in processes)


def test_worker_deadline_starts_at_launch_and_does_not_wait_for_other_workers(
    tmp_path, monkeypatch
):
    clock = SimpleNamespace(now=10)
    monkeypatch.setattr(worker_module.time, "monotonic", lambda: clock.now)
    processes = []

    class LiveProcess(FinishedProcess):
        def start(self):
            self.alive = True

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.alive = False
            self.exitcode = -15

    class Context:
        def Process(self, **kwargs):
            process = LiveProcess(**kwargs)
            processes.append(process)
            return process

    monkeypatch.setattr(
        worker_module.multiprocessing, "get_context", lambda method: Context()
    )
    config = experiment(tmp_path)
    config.environment["startup_timeout_seconds"] = 10
    first = worker_module.EpisodeWorker(config, specimen(0), tmp_path / "first")
    second = worker_module.EpisodeWorker(config, specimen(1), tmp_path / "second")
    clock.now = 500
    assert first.remaining_seconds == second.remaining_seconds == 70
    first.start()
    clock.now = 510
    second.start()
    try:
        clock.now = 570
        assert first.remaining_seconds == 0 and second.remaining_seconds == 10
        result = first.poll()
        assert result["status"] == "worker_timeout"
        assert result["elapsed_seconds"] == 70
        assert processes[0].closed and not processes[0].is_alive()
        assert second.poll() is None and processes[1].is_alive()
    finally:
        first.close()
        second.close()
    assert all(process.closed and not process.is_alive() for process in processes)


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2"])
def test_num_workers_requires_positive_integer(tmp_path, value):
    config = experiment(tmp_path, workers=value)
    with pytest.raises(ValueError, match="num_workers"):
        validate(config)


@pytest.mark.parametrize("gpus", [[], [2, 2], [-1], [True], [1.5], "2,3"])
def test_gpu_list_requires_nonempty_unique_nonnegative_integers(tmp_path, gpus):
    config = experiment(tmp_path)
    config.environment["gpus"] = gpus
    with pytest.raises(ValueError, match="gpus"):
        validate(config)


def test_single_and_multi_gpu_configuration_are_mutually_exclusive(tmp_path):
    config = experiment(tmp_path)
    config.environment.update(gpu=2, gpus=[2, 3])
    with pytest.raises(ValueError, match="gpu"):
        validate(config)
