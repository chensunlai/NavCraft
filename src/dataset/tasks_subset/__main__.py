"""Collect and freeze the selected training task subset."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path

from tqdm import tqdm

from src.common.config import read_config
from src.common.io import read_json

from .bundle import write_json
from .sampling.collection import collect, collect_trials


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=0)
    parser.add_argument("--scene", action="append", help="Restrict to configured asset ID")
    parser.add_argument("--phase", choices=("collect", "trials", "build", "check", "review", "semantics", "enrich", "prepare", "all"), default="collect")
    parser.add_argument("--trial", action="append", help="Limit the trials phase to this kind")
    parser.add_argument("--bundle", type=Path, help="Bundle path relative to each scene directory, or absolute for one scene")
    parser.add_argument("--invalidations", type=Path, help="Explicit supersedes or invalidations record for check")
    parser.add_argument("--frame", action="append", help="Saved frame key to render for semantic evidence")
    parser.add_argument("--candidate", action="append", help="Review one candidate ID")
    parser.add_argument("--import-reviews", type=Path, help="Apply existing JSONL without calling a model")
    args = parser.parse_args(argv)
    config = read_config(args.config)
    scenes = args.scene or config["scenes"]
    if len(set(scenes)) != len(scenes) or any(scene not in config["scenes"] for scene in scenes):
        parser.error("--scene must select unique configured asset IDs")
    if args.bundle and args.bundle.is_absolute() and len(scenes) != 1:
        parser.error("absolute --bundle requires one scene")
    if args.bundle and args.phase in ("collect", "all"):
        parser.error("collect and all use a new collection, without --bundle")
    if args.phase == "enrich" and not args.bundle:
        parser.error("enrich requires an explicit reviewed --bundle")
    summary = []
    for scene in tqdm(scenes, unit="scene"):
        directory = args.output / "scenes" / scene.replace("/", "__")
        bundle_path = directory / (args.bundle or Path("private/bundle.json"))
        built_path = directory / "private/built_bundle.json"
        if args.phase == "build" and bundle_path.resolve() == built_path.resolve():
            parser.error("build input must differ from private/built_bundle.json")
        if args.phase in ("build", "all") and built_path.exists():
            raise FileExistsError(built_path)
        input_record = None
        if args.phase not in ("collect", "all"):
            bundle = read_json(bundle_path)
            input_record = {"path": str(bundle_path.resolve()),
                            "sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest()}
            if bundle["scene_id"] != scene:
                parser.error(f"bundle scene does not match {scene}")
        if args.phase in ("collect", "all"):
            bundle = collect({**config, "scene": scene}, directory, args.gpu)
        if args.phase in ("trials", "all"):
            bundle = collect_trials(config, directory, args.gpu, args.trial or (), bundle_path)
        if args.phase in ("trials", "prepare", "all"):
            from .navigation.construction import prepare_navigation
            prepare_navigation(bundle, directory, config, args.gpu, args.trial or ())
            write_json(bundle_path, bundle)
        if args.phase == "semantics":
            from .sampling.semantics import collect_instance_evidence
            destination = directory / "private/instance_bundle.json"
            if destination.exists() and destination.resolve() != bundle_path.resolve():
                raise FileExistsError("Select the existing instance bundle to extend or apply bindings")
            bundle = collect_instance_evidence(bundle, directory, config, args.gpu, args.frame)
            write_json(destination, bundle)
        if args.phase == "review":
            from .review.stage import run_review
            bundle = run_review(bundle, directory, bundle_path, config, args.candidate or (), args.import_reviews)
        if args.phase == "enrich":
            from .review.memory import enrich_node_memory
            bundle = enrich_node_memory(bundle, directory)
        previous = read_json(directory / "manifest.json") if (directory / "manifest.json").exists() else {}
        cases = previous.get("cases", [])
        if args.phase in ("build", "all"):
            from .judgment.construction import build_judgments
            from .navigation.construction import build_navigation
            cases = build_judgments(bundle, directory, config) + build_navigation(bundle, directory, config)
            write_json(built_path, bundle)
        if args.phase == "check":
            from .review.check import check_collection
            from .review.inspection import write_inspection
            report = check_collection(directory, bundle_path, args.invalidations)
            write_json(directory / "check.json", report)
            write_inspection(directory, report, bundle_path)
        manifest = {"scene_id": scene, "source": bundle["source"], "frames": len(bundle["frames"]),
                    "states": len(bundle["states"]), "trials": len(bundle["trials"]),
                    "physical_replays": sum(t["replay"]["passed"] for t in bundle["trials"]),
                    "coverage": dict(Counter(c["kind"] for c in cases)), "cases": cases,
                    "rejections": "rejections.jsonl" if (directory / "rejections.jsonl").exists() else None}
        output_path = {"review": directory / "private/reviewed_bundle.json",
                       "semantics": directory / "private/instance_bundle.json",
                       "enrich": directory / "private/enriched_bundle.json",
                       "build": built_path, "all": built_path}.get(args.phase, bundle_path)
        if output_path.is_file():
            manifest["stage"] = {"phase": args.phase, "input_bundle": input_record,
                                 "output_bundle": {"path": str(output_path.resolve()),
                                                  "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest()}}
        write_json(directory / "manifest.json", manifest)
        summary.append({"manifest": str((directory / "manifest.json").relative_to(args.output)),
                        **{k: v for k, v in manifest.items() if k not in {"cases", "source"}}})
    top = args.output / "manifest.json"
    old = read_json(top).get("scenes", []) if top.exists() else []
    updated = {row["scene_id"]: row for row in old}
    updated.update({row["scene_id"]: row for row in summary})
    write_json(top, {"phase": args.phase, "scenes": list(updated.values()),
               "config_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in args.config]})


if __name__ == "__main__":
    main()
