"""Run selected episodes; explicit CLI paths come from the launch script."""

import argparse
import copy
from datetime import datetime, timezone
from multiprocessing.connection import wait
from pathlib import Path
import re
import sys
import uuid

from src.bench.report import summarize
from src.bench.worker import EpisodeWorker, execute_episode
from src.common.config import ExperimentConfig, load_config, validate
from src.common.io import write_json
from src.common.register import episode_registry
from src.legacy.harness.build import load_builtins


def _slot_config(config: ExperimentConfig, slot: int, episode=None) -> ExperimentConfig:
    """Resolve a slot's GPU without sharing mutable runtime configuration."""
    assigned = copy.deepcopy(config)
    if (
        episode is not None
        and episode.benchmark == "goat"
        and assigned.run.get("budget_per_goal", False)
    ):
        assigned.run["max_requests"] *= len(episode.goals)
        assigned.run["timeout_seconds"] *= len(episode.goals)
    if config.environment["name"] == "habitat" or any(
        name in config.environment for name in ("gpu", "gpus")
    ):
        gpus = assigned.environment.pop("gpus", [assigned.environment.get("gpu", 2)])
        assigned.environment["gpu"] = gpus[slot % len(gpus)]
    return assigned


def _run_concurrent(config, episodes, directory, record) -> None:
    """Refill any available slot, independently of the other episode workers."""
    active: dict[int, tuple[int, EpisodeWorker]] = {}
    next_index = 0
    slots = min(config.run["num_workers"], len(episodes))
    try:
        while next_index < len(episodes) or active:
            for slot in range(slots):
                if slot not in active and next_index < len(episodes):
                    index = next_index
                    worker = EpisodeWorker(
                        _slot_config(config, slot, episodes[index]),
                        episodes[index],
                        directory / "episodes" / f"{index:04d}",
                    )
                    active[slot] = (index, worker)
                    next_index += 1
                    worker.start()
            completed = False
            for slot, (index, worker) in list(active.items()):
                result = worker.poll()
                if result is not None:
                    record(index, result)
                    worker.close()
                    del active[slot]
                    completed = True
            if completed:
                continue
            if active:
                workers = [worker for _, worker in active.values()]
                wait(
                    [worker.process.sentinel for worker in workers],
                    timeout=min(
                        1.0, min(worker.remaining_seconds for worker in workers)
                    ),
                )
    finally:
        for _, worker in active.values():
            worker.close()


def run_experiment(config: ExperimentConfig) -> tuple[Path, dict]:
    validate(config)
    load_builtins()
    source = episode_registry.create(
        "manifest" if config.episodes["name"] == "manifest" else config.task["name"],
        config=config.episodes,
    )
    episodes = source.load()
    label = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "-",
        f"{config.run.get('name', 'run')}-{config.episodes['name']}-{config.model.get('model', config.model['name'])}",
    )
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{label}-{uuid.uuid4().hex[:8]}"
    directory = Path(config.run["output_dir"]) / run_id
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "config.json", config.snapshot())
    write_json(directory / "selection.json", [episode.id for episode in episodes])
    completed: dict[int, dict] = {}

    def persist() -> dict:
        results = [completed[index] for index in sorted(completed)]
        planned_goals = (
            sum(len(episode.goals) for episode in episodes)
            if episodes[0].benchmark == "goat"
            else None
        )
        summary = summarize(results, len(episodes), planned_goals)
        write_json(directory / "results.json", results)
        write_json(directory / "summary.json", summary)
        return summary

    def record(index: int, result: dict) -> None:
        completed[index] = result
        write_json(directory / "episodes" / f"{index:04d}" / "result.json", result)
        persist()
        print(
            f"{episodes[index].id}: {result['status']} ({result['elapsed_seconds']:.1f}s)",
            flush=True,
        )

    persist()
    try:
        if config.run["num_workers"] == 1:
            for index, episode in enumerate(episodes):
                episode_dir = directory / "episodes" / f"{index:04d}"
                record(
                    index,
                    execute_episode(
                        _slot_config(config, 0, episode), episode, episode_dir
                    ),
                )
        else:
            _run_concurrent(config, episodes, directory, record)
    finally:
        summary = persist()
        print(f"Run: {directory}", flush=True)
    return directory, summary


def parse_config_paths(argv: list[str] | None = None) -> dict[str, str]:
    parser = argparse.ArgumentParser(
        description="Run a NavCraft experiment with five YAML configurations"
    )
    for role in ("task", "env", "model", "harness", "run"):
        parser.add_argument(f"--{role}-config", required=True, help=f"{role} YAML path")
    values = vars(parser.parse_args(argv))
    return {name.upper(): value for name, value in values.items()}


def main() -> None:
    try:
        _, summary = run_experiment(load_config(parse_config_paths()))
    except KeyboardInterrupt:
        print(
            "Interrupted; active episode workers stopped.", file=sys.stderr, flush=True
        )
        sys.exit(130)
    failures = {
        "setup_error",
        "runtime_error",
        "evaluation_error",
        "model_error",
        "environment_error",
        "cleanup_error",
        "model_incomplete",
        "worker_error",
        "worker_timeout",
    }
    if any(status in failures for status in summary["statuses"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
