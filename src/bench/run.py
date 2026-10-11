"""Run selected episodes; explicit CLI paths come from the launch script."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
import uuid

from src.bench.report import summarize
from src.bench.worker import execute_episode
from src.common.config import ExperimentConfig, load_config
from src.common.io import write_json
from src.common.register import episode_registry
from src.harness.build import load_builtins


def run_experiment(config: ExperimentConfig) -> tuple[Path, dict]:
    load_builtins()
    source = episode_registry.create(config.task["name"], config=config.episodes)
    episodes = source.load()
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{config.run.get('name', 'run')}-{config.episodes['name']}-{config.model.get('model', config.model['name'])}")
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{label}-{uuid.uuid4().hex[:8]}"
    directory = Path(config.run["output_dir"]) / run_id
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "config.json", config.snapshot())
    write_json(directory / "selection.json", [episode.id for episode in episodes])
    results = []
    for index, episode in enumerate(episodes):
        episode_dir = directory / "episodes" / f"{index:04d}"
        result = execute_episode(config, episode, episode_dir)
        results.append(result)
        write_json(episode_dir / "result.json", result)
        write_json(directory / "results.json", results)
        write_json(directory / "summary.json", summarize(results, len(episodes)))
        print(f"{episode.id}: {result['status']} ({result['elapsed_seconds']:.1f}s)", flush=True)
    summary = summarize(results, len(episodes))
    print(f"Run: {directory}", flush=True)
    return directory, summary


def parse_config_paths(argv: list[str] | None = None) -> dict[str, str]:
    parser = argparse.ArgumentParser(description="Run a NavCraft experiment with five YAML configurations")
    for role in ("task", "env", "model", "harness", "run"):
        parser.add_argument(f"--{role}-config", required=True, help=f"{role} YAML path")
    values = vars(parser.parse_args(argv))
    return {name.upper(): value for name, value in values.items()}


def main() -> None:
    _, summary = run_experiment(load_config(parse_config_paths()))
    failures = {"setup_error", "runtime_error", "evaluation_error", "model_error", "environment_error", "cleanup_error", "model_incomplete", "worker_error", "worker_timeout"}
    if any(status in failures for status in summary["statuses"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
