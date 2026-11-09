"""Evaluation entry points for frozen tasks and prepared maps."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import sys
from pathlib import Path

from src.common.config import merge_config
from src.common.progress import Progress

from .samples import plan
from src.dataset.episodes import EpisodeSpec
from src.common.config import read_config

REPO = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["doctor", "run", "pilot", "worker", "report"])
    parser.add_argument("--config", type=Path, action="append")
    parser.add_argument("--model-config", type=Path, action="append", default=[])
    parser.add_argument("--model", default="all", help="Model name registered in config, or all")
    parser.add_argument("--episode", help="Frozen canonical ID for run")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--job", type=Path)
    args = parser.parse_args(argv)
    if args.command == "report":
        if args.output is None:
            parser.error("report requires --output <existing run directory>")
        from .report import write_report
        print(write_report(args.output))
        return 0
    if args.command == "worker":
        from .runner import run_episode
        job = json.loads(args.job.read_text())
        episode = EpisodeSpec(**job["episode"])
        config = job["model_config"]
        result = run_episode(REPO, config, episode, job["environment"], job["map"],
                             job["directory"], job["limits"], map_layer=job.get("map_layer", "walls"))
        print(json.dumps({"episode_id": episode.id, "status": result["status"]}))
        return 0 if result["status"] == "completed" else 1
    config = read_config(args.config or [REPO / "config/pilot.yaml"])
    config["model_overrides"] = merge_config(config.get("model_overrides", {}), read_config(args.model_config))
    if args.model != "all" and args.model not in config["models"]:
        parser.error(f"Model {args.model} is not registered in the selected config")
    if args.command == "run":
        if not args.episode or args.model == "all":
            parser.error("run requires --episode and a single --model")
        config["episodes"] = [args.episode]
    models = list(config["models"]) if args.model == "all" else [args.model]
    progress = Progress("Prepare", ["samples and maps"])
    progress.set("samples and maps", "running")
    prepared = plan(REPO, config, models)
    progress.set("samples and maps", "finished")
    if args.command == "doctor":
        model_configs = prepared["models"]
        print(json.dumps({"selected": len(prepared["jobs"]),
                          "simulator_gpus": config["environment"]["gpus"],
                          "habitat_sim_available": importlib.util.find_spec("habitat_sim") is not None,
                          "models": {name: {"python_exists": Path(model.get("python", sys.executable)).is_file(),
                                            "concurrency": model["concurrency"]}
                                     for name, model in model_configs.items()},
                          "maps_ready": sum(j["map_error"] is None for j in prepared["jobs"]),
                          "map_issues": [{"episode": j["episode_id"], "error": j["map_error"]}
                                         for j in prepared["jobs"] if j["map_error"]],
                          "dispatch_order": prepared["dispatch_order"]}, indent=2))
        return 0
    missing = [j for j in prepared["jobs"] if j["map_error"]]
    if missing:
        parser.error("Published maps are unavailable or invalid: " + "; ".join(
            f"{j['episode_id']}: {j['map_error']}" for j in missing))
    from .batch import run_batch
    output = args.output or REPO / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-pilot")
    run_batch(REPO, config, prepared, output, models)
    from .report import write_report
    print(write_report(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
