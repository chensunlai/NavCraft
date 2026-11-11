"""Bound offline semantic assessment of actual newly recorded node panoramas."""
import hashlib
import json
from pathlib import Path
import shutil
import re

from src.common.io import read_json
from src.harness.runtime.request import PromptLimits
from ..bundle import snapshot_digest, write_json
from .model import run_fixed_evidence


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture_execution_review(task_path, run_dir, executor, initial, score):
    run_dir = Path(run_dir).resolve()
    task_path = Path(task_path).resolve()
    task = read_json(task_path)
    directory = run_dir / "private/execution-review"
    directory.mkdir(parents=True, exist_ok=True)
    nodes = [n for n in executor.memory.nodes if n["id"] in score["details"].get("new_nodes", [])]
    references = [n for n in initial["memory"]["nodes"] if n.get("room")]
    label = read_json(task_path.parent / task["label"])
    expected_rooms = label.get("room_names", [])
    mode = "known_alias" if expected_rooms else "first_room" if not references else "new_room"
    known_rooms = {n["room"] for n in references}
    assignment = (all(n.get("room") for n in nodes) if mode == "first_room" else
                  all(n.get("room") in expected_rooms for n in nodes) if mode == "known_alias" else
                  all(n.get("room") and n["room"] not in known_rooms for n in nodes))
    rows = []
    for role, selected in (("created", nodes), ("room_reference", references)):
        for node in selected:
            panorama = executor.observations.panorama(node["panorama"])
            images = []
            for frame in panorama["frames"]:
                source = executor.workspace / frame["path"]
                destination = directory / (role + "-" + node["id"] + "-" + frame["frame"] + ".jpg")
                shutil.copyfile(source, destination)
                images.append({"path": str(destination.relative_to(run_dir)), "sha256": _hash(destination),
                               "yaw": frame["yaw"]})
            rows.append({"role": role, "node": node, "capture_pose": panorama["pose"], "images": images})
    manifest = {"task_sha256": _hash(task_path), "instruction": task["instruction"],
                "snapshot_sha256": snapshot_digest(task_path.parent / task["snapshot"]),
                "run_node_memory_sha256": _hash(executor.memory.path),
                "room_assignment_mode": mode, "room_assignment_valid": bool(nodes and assignment),
                "structural_score": score, "nodes": rows}
    write_json(directory / "manifest.json", manifest)
    return manifest


def review_execution(repo, config, run_dir, reviewer):
    run_dir = Path(run_dir).resolve()
    path = run_dir / "private/execution-review/manifest.json"
    manifest = read_json(path)
    record_path = run_dir / "private/execution-review" / (reviewer + ".json")
    for saved_path in path.parent.glob("*.json"):
        record = read_json(saved_path)
        if (record.get("manifest_sha256") == _hash(path) and record.get("model") == config["model"]
                and record.get("reasoning_effort") == config.get("reasoning_effort", "high")):
            return record
    if record_path.exists():
        record = read_json(record_path)
        if record["manifest_sha256"] != _hash(path) or record["model"] != config["model"]:
            raise ValueError("Existing reviewer record belongs to a different binding")
        return record
    images, claims = [], []
    for row in manifest["nodes"]:
        for image in row["images"]:
            source = run_dir / image["path"]
            if _hash(source) != image["sha256"]:
                raise ValueError("Execution review RGB has changed")
            images.append({"path": source, "caption": f"{row['role']} node {row['node']['id']}, yaw {image['yaw']}"})
        claims.append({key: row[key] for key in ("role", "node", "capture_pose")})
    mode = manifest["room_assignment_mode"]
    criterion = (
        "The room label is a newly chosen identifier; no reference room is required. Pure aliases such as "
        "Room 1 need no visual category evidence. If the label contains a category such as bedroom, verify "
        "that category from the new panorama. The program checks label validity/uniqueness."
        if mode in {"first_room", "new_room"} else
        "The program checks that the label exactly matches the expected known alias. Do not infer the alias "
        "from its spelling. Check visual continuity using supplied room references when making visual claims.")
    prompt = ("Review each created node description using its actual six-view panorama. " + criterion +
              " Return JSON {description_supported:boolean, room_supported:boolean, decision:"
              "'accept'|'reject'|'uncertain', reason:string}. Mark uncertain if visual evidence is insufficient. "
              "Set room_supported true for a pure alias with no visual-category claim. Claims: " + json.dumps(claims))
    response = run_fixed_evidence(repo, config, run_dir / "private/execution-review" / reviewer,
                                 prompt, images, PromptLimits(max_requests=1, max_turns=1, seconds=180))
    try:
        answer = json.loads(response["answer"])
        if (answer["decision"] not in {"accept", "reject", "uncertain"} or
                any(type(answer[key]) is not bool for key in ("description_supported", "room_supported"))):
            raise ValueError("Malformed semantic assessment")
    except (KeyError, TypeError, ValueError):
        answer = {"decision": "uncertain", "reason": "No valid assessment",
                  "description_supported": False, "room_supported": False}
    record = {"reviewer": reviewer, "manifest_sha256": _hash(path), "answer": answer,
              "response_reason": response["reason"], "model": config["model"],
              "reasoning_effort": config.get("reasoning_effort", "high")}
    write_json(record_path, record)
    return record


def finalize_execution(run_dir, reviews):
    run_dir = Path(run_dir)
    manifest_path = run_dir / "private/execution-review/manifest.json"
    digest = _hash(manifest_path)
    rows = list(reviews)
    if any(row["manifest_sha256"] != digest for row in rows):
        raise ValueError("Review is bound to a different execution")
    if len({r["reviewer"] for r in rows}) != len(rows):
        raise ValueError("Duplicate execution reviewer")
    if len({r["model"] for r in rows}) != len(rows):
        raise ValueError("Independent reviews require distinct models")
    manifest = read_json(manifest_path)
    for row in manifest["nodes"]:
        for image in row["images"]:
            if _hash(run_dir / image["path"]) != image["sha256"]:
                raise ValueError("Execution review RGB has changed")
    created = [row for row in manifest["nodes"] if row["role"] == "created"]
    assignment = manifest["room_assignment_valid"]
    pure_alias = all(re.fullmatch(r"(?:Room|room|R|房间)[ _-]*\d+", row["node"]["room"])
                     for row in created)
    room_needs_visual = manifest.get("room_assignment_mode") in {"first_room", "new_room"} and not pure_alias
    accepted = len(rows) >= 2 and all(r["answer"].get("decision") == "accept"
        and r["answer"].get("description_supported") is True
        and (not room_needs_visual or r["answer"].get("room_supported") is True) for r in rows)
    event = manifest["structural_score"]["details"]["event"]
    success = bool(event and accepted and assignment)
    service_error = any(row.get("response_reason") in {"runtime_error", "service_error", "model_error", "core_error"}
                        for row in rows)
    rejected = any(row["answer"].get("decision") == "reject" for row in rows)
    status = "review_service_error" if service_error else "incorrect" if rejected else "correct" if success else "needs_review"
    result = {"success": None if status in {"review_service_error", "needs_review"} else success, "status": status,
              "manifest_sha256": digest, "reviews": rows}
    write_json(run_dir / "final_assessment.json", result)
    return result


def main(argv=None):
    import argparse
    from src.common.config import read_config
    from src.harness.runtime.config import credentials
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--review-model", type=Path, action="append", required=True)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[4]
    configs = [read_config(path) for path in args.review_model]
    for config in configs:
        config["model"] = credentials(config, repo)["model"]
    if len({config["model"] for config in configs}) != len(configs):
        parser.error("Independent reviewers require distinct actual model IDs")
    digest = _hash(args.run / "private/execution-review/manifest.json")
    rows = []
    for config in configs:
        key = hashlib.sha256((digest + config["model"] + config.get("reasoning_effort", "high")).encode()).hexdigest()[:24]
        rows.append(review_execution(repo, config, args.run, key))
    result = finalize_execution(args.run, rows)
    print(json.dumps({"status": result["status"], "success": result["success"],
                      "assessment": str(args.run / "final_assessment.json")}))


if __name__ == "__main__":
    main()

