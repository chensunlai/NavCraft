"""Python entry point for the Pi-backed navigation application."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading

from .config import REPO, load_config
from .coding import run_coding
from .runner import run_case
from .worker import supervise


def doctor(config):
    script = """import {createCodingTools,sandboxRun} from './src/agent-core/sandbox.ts';
import {realpathSync,mkdirSync} from 'node:fs';
const workspace='cache/harness-pi/doctor/workspace';
for (const name of ['scene','observations','memory','scratch']) mkdirSync(workspace+'/'+name,{recursive:true});
const python=process.argv[1];
const result=await sandboxRun(realpathSync(workspace),python,"python -B -c 'import numpy,PIL; print(numpy.__version__,PIL.__version__)'");
console.log(JSON.stringify({node:process.version,sdk:import.meta.resolve('@earendil-works/pi-coding-agent'),sandbox:result}));
"""
    completed = subprocess.run(["node", "--import", "tsx", "--input-type=module", "-e", script, config["python"]],
                               cwd=REPO, text=True, capture_output=True, check=True)
    result = json.loads(completed.stdout)
    result.update(python=config["python"], scene_present=(REPO / config["scene_archive"]).exists() if config.get("scene_archive") else None,
                  credentials_present=(REPO / config["credentials"]).is_file())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="Python-led Pi navigation Harness")
    parser.add_argument("--config", default="config/harness-pi/default.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("doctor")
    check.add_argument("--online", action="store_true")
    run = subparsers.add_parser("run")
    run.add_argument("--case", default="local")
    run.add_argument("--max-requests", type=int)
    run.add_argument("--seconds", type=float)
    run.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    run.add_argument("--directory", help=argparse.SUPPRESS)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--suite", choices=("quick", "full"), default="quick")
    sequence = subparsers.add_parser("sequence", help="Run multiple tasks in the same environment and Agent session")
    sequence.add_argument("--scenario", required=True, help="Private task sequence JSON")
    sequence.add_argument("--directory")
    args = parser.parse_args()
    if args.command == "sequence":
        from .sequence import main as sequence_main
        sys.argv = [sys.argv[0], "--config", args.config, "--scenario", args.scenario]
        if args.directory:
            sys.argv.extend(["--directory", args.directory])
        sequence_main()
        return
    config = load_config(args.config)
    if args.command == "doctor" and not args.online:
        doctor(config)
        return
    from src.bench.harness_cases import case_config
    names = (("coding", "local", "compaction") if args.command == "smoke" and args.suite == "quick"
             else ("coding", "query", "local", "objnav", "instruction", "goat", "mapping", "compaction")
             if args.command == "smoke" else ("coding",) if args.command == "doctor" else (args.case,))
    for name in names:
        job = case_config(REPO, name)
        seconds = getattr(args, "seconds", None)
        limit = getattr(args, "max_requests", None)
        seconds = job["seconds"] if seconds is None else seconds
        limit = job["requests"] if limit is None else limit
        if limit <= 0 or seconds <= 0:
            parser.error("Request and time limits must be positive")
        if getattr(args, "worker", False):
            cancelled = threading.Event()
            signal.signal(signal.SIGTERM, lambda *_: cancelled.set())
            handler = run_coding if name == "coding" else run_case
            handler(REPO, config, name=name, max_requests=limit, seconds=seconds,
                    directory=args.directory, cancel_event=cancelled)
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        directory = REPO / config["output_dir"] / f"{stamp}-{name}"
        print(json.dumps({"directory": str(directory), "case": name, "phase": "supervised"}), flush=True)
        command = [sys.executable, "-B", "-u", "-m", "src.harness", "--config", args.config,
                   "run", "--worker", "--directory", str(directory), "--case", name,
                   "--max-requests", str(limit), "--seconds", str(seconds)]
        result = supervise(command, directory, seconds, REPO)
        print(json.dumps({"directory": str(directory), **result}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
