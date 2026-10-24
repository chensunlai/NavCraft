"""Offline preparation and episode evaluation entry points."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

from src.common.io import write_json
from .samples import load_episodes, plan, read_config

REPO = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["doctor", "prepare", "run", "pilot", "worker"])
    parser.add_argument("--config", type=Path, default=REPO / "config/pilot.yaml")
    parser.add_argument("--model", choices=["all", "deepseek", "qwen27", "sol"], default="all")
    parser.add_argument("--episode", help="Frozen canonical ID for run")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--job", type=Path)
    args = parser.parse_args(argv)
    if args.command == "worker":
        from .runner import run_episode
        job = json.loads(args.job.read_text())
        episode = load_episodes(REPO, [job["episode_id"]])[0]
        config = read_config(job["model_config"])
        result = run_episode(REPO, config, episode, job["environment"], job["map"],
                             job["directory"], job["limits"])
        print(json.dumps({"episode_id": episode.id, "status": result["status"]}))
        return 0 if result["status"] == "completed" else 1
    config = read_config(args.config)
    if args.command == "run":
        if not args.episode or args.model == "all":
            parser.error("run requires --episode and a single --model")
        config["episodes"] = [args.episode]
    prepared = plan(REPO, config)
    if args.command == "doctor":
        print(json.dumps({"selected": len(prepared["jobs"]),
                          "simulator_gpus": config["environment"]["gpus"],
                          "habitat_sim_available": importlib.util.find_spec("habitat_sim") is not None,
                          "models": {name: {"python_exists": Path(read_config(REPO / path)["python"]).is_file(),
                                            "concurrency": read_config(REPO / path)["concurrency"]}
                                     for name, path in config["models"].items()},
                          "maps_ready": sum(Path(j["map"]).is_file() for j in prepared["jobs"]),
                          "dispatch_order": prepared["dispatch_order"]}, indent=2))
        return 0
    if args.command == "prepare":
        from src.dataset.wall_maps import build
        for job in prepared["jobs"]:
            target = Path(job["map"])
            if not target.is_file():
                build(job["scene"], job["floor_height"], target.parent)
            print(json.dumps({"episode_id": job["episode_id"], "map": str(target)}), flush=True)
        output = args.output or REPO / "cache/pilot-plan.json"
        write_json(output, prepared)
        return 0
    missing = [j["map"] for j in prepared["jobs"] if not Path(j["map"]).is_file()]
    if missing:
        parser.error("Wall maps are not prepared; run `python -m src.bench prepare` first")
    from .batch import run_batch
    output = args.output or REPO / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-pilot")
    models = list(config["models"]) if args.model == "all" else [args.model]
    print(run_batch(REPO, config, prepared, output, models))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
