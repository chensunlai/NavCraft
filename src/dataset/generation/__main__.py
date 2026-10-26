"""Stages for constructing and inspecting real navigation-task collections."""

import argparse
import hashlib
import json
from pathlib import Path

from src.common.config import parse_config_args, read_config
from src.common.io import write_json

from .records import update_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "collect", "derive", "solve", "check", "export"))
    parser.add_argument("--output", type=Path, required=True)
    args = parse_config_args(parser, argv)
    config = read_config(args.config)
    config["output"] = str(args.output)
    repo = Path(__file__).resolve().parents[3]
    root = (repo / args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        from .scenes import prepare
        safe_config = {key: value for key, value in config.items() if key not in {"api_key", "key", "auth"}}
        digest = hashlib.sha256(json.dumps(safe_config, sort_keys=True).encode()).hexdigest()
        existing = root / "config.json"
        if existing.exists() and json.loads(existing.read_text()) != safe_config:
            raise ValueError("Collection configuration changed; choose a separate output")
        write_json(existing, safe_config)
        update_manifest(root, config_sha256=digest, source="real_habitat", schema="navigation_tasks")
        result = {"scenes": len(prepare(root, config, repo))}
    elif args.stage == "collect":
        from .collect import collect
        result = {"snapshots": len(collect(root, config, repo))}
    elif args.stage == "derive":
        from .derive import derive
        result = {"samples": len(derive(root, config, repo))}
    elif args.stage == "solve":
        from .teacher import solve
        result = solve(root, config, repo, args.config)
    elif args.stage == "check":
        from .check import check
        result = check(root, config, repo)
    else:
        from .export import export
        result = export(root, config, repo)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("valid") is False else 0


raise SystemExit(main())
