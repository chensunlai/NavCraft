"""Execute a frozen task; construction and label repair never happen here."""

import argparse
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import traceback

import numpy as np

from src.common.config import parse_config_args, read_config
from src.common.io import read_json, write_json
from src.dataset.generation.records import (SceneRecord, Snapshot, TaskSample, load_records,
                                           local_file)
from src.dataset.generation.scenes import open_environment
from src.harness.session import PromptLimits, open_session, read_snapshot

from .scoring import score_sample


def runtime_evidence(label, directory, outcome):
    """Use only observations actually produced by this independent attempt."""
    directory = Path(directory)
    sources = directory / "private/location_sources.json"
    outcome["location_sources"] = read_json(sources) if sources.exists() else {}
    if "frame_visibility" in label:
        visibility = dict(label["frame_visibility"])
        for path in (directory / "private/semantics").glob("*.npy"):
            semantic = np.load(path, allow_pickle=False)
            visibility[path.stem] = [item["id"] for item in label["instances"]
                                     if np.count_nonzero(semantic == item["semantic_id"]) >= 500]
        label["frame_visibility"] = visibility


def score_saved_run(root, directory):
    """Recompute a private score while preserving the original recorded verdict."""
    root, directory = Path(root).resolve(), Path(directory).resolve()
    value = read_json(directory / "result.json")
    sample = read_json(directory / "sample.json")
    label = read_json(local_file(root, sample["label_file"]))
    if "mask_file" in label:
        label["mask_file"] = str(local_file(root, label["mask_file"]))
    outcome = value["outcome"]
    runtime_evidence(label, directory, outcome)
    score = score_sample(sample, label, outcome)
    if score != value["score"]:
        value.setdefault("original_score", value["score"])
        value["score"] = score
    if value["status"] == "requires_review" and score["reason"] != "memory_requires_review":
        value["status"] = "completed"
    value["score_source_sha256"] = hashlib.sha256(Path(__file__).with_name("scoring.py").read_bytes()).hexdigest()
    write_json(directory / "result.json", value)
    return value


def run_sample(root, sample_id, config, directory, repo):
    root, directory, repo = Path(root).resolve(), Path(directory).resolve(), Path(repo).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    samples = {sample.id: sample for sample in load_records(root / "samples.jsonl", TaskSample)}
    sample = samples[sample_id]
    snapshots = {cut.id: cut for cut in load_records(root / "snapshots.jsonl", Snapshot)}
    snapshot = snapshots[sample.snapshot]
    scenes = {scene.id: scene for scene in load_records(root / "scenes.jsonl", SceneRecord)}
    scene = scenes[snapshot.scene_id]
    label = read_json(local_file(root, sample.label_file))
    if "mask_file" in label:
        label["mask_file"] = str(local_file(root, label["mask_file"]))
    state = read_snapshot(local_file(root, snapshot.directory + "/state.json"))
    state["memory"] = read_json(local_file(root, sample.inputs["memory_file"]))
    # Script cuts carry a public factual head just like a recovered runtime.
    # It precedes the real paired prefix, so continuation still ends at a receipt.
    if snapshot.source == "script":
        facts = {"current_request": state.get("current_request", ""),
                 "current_frame": state["current_frame"],
                 "entities": [{key: item.get(key) for key in ("id", "type", "name", "category")}
                              for item in state["memory"]]}
        state["history"] = [{"role": "user", "timestamp": 0, "content": [{"type": "text", "text":
                              "Public facts at this saved state:\n" + json.dumps(facts)}]}, *state["history"]]
    write_json(directory / "sample.json", asdict(sample))
    environment, session = None, None
    outcome = {"submitted": False, "answer": "", "reason": "runtime_error"}
    try:
        environment = open_environment(scene, config, repo, directory / "environment", pose=state["pose"])
        session = open_session(repo, config, directory, environment, local_file(root, scene.map_path),
                               initial_state=state, start_core=False)
        write_json(session.private / "location_sources.json", label.get("location_sources", {}))
        before_steps, before_path = environment.steps, environment.path_length
        limits = PromptLimits(**sample.limits)
        images = [local_file(root, ref) for ref in sample.inputs.get("images", [])]
        if sample.entry == "ask":
            result = session.ask(sample.question, images=images, limits=limits)
        elif sample.entry == "continue":
            result = session.continue_run(limits=limits)
        else:
            result = session.prompt(sample.question, images=images, limits=limits)
        outcome = {**result.to_dict(), "entities": session.memory.entities,
                   "pose": environment.pose(), "actions": environment.steps - before_steps,
                   "path_length": environment.path_length - before_path,
                   "tool_errors": sum(call.get("isError", False) for call in session.trace.calls)}
        runtime_evidence(label, directory, outcome)
        if sample.task_id == "14b":
            outcome["distance_to_approach"] = environment.distance(label["approach_points"])
        score = score_sample(asdict(sample), label, outcome)
        infrastructure = result.reason in {"runtime_error", "execution_uncertain", "model_error",
                                            "transport_error", "tool_timeout", "timeout"}
        status = "infrastructure_error" if infrastructure else "completed"
        if score["reason"] == "memory_requires_review":
            status = "requires_review"
        value = {"sample_id": sample.id, "task_id": sample.task_id, "form": sample.form,
                 "status": status, "outcome": outcome, "score": score,
                 "model": config["model"], "reasoning_effort": config.get("reasoning_effort", "high"),
                 "tool_tier": config.get("tool_tier", "super")}
    except Exception as error:
        value = {"sample_id": sample.id, "task_id": sample.task_id, "form": sample.form,
                 "status": "infrastructure_error", "outcome": outcome,
                 "score": {"success": False, "reason": "execution_error", "metrics": {}},
                 "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
    finally:
        if session:
            session.close()
        if environment:
            environment.close()
    write_json(directory / "result.json", value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parse_config_args(parser, argv)
    repo = Path(__file__).resolve().parents[3]
    result = run_sample(args.collection, args.sample, read_config(args.config), args.output, repo)
    print(json.dumps({key: result[key] for key in ("sample_id", "status", "score")}))
    return 1 if result["status"] == "infrastructure_error" else 0
