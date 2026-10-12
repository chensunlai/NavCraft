"""Audit selected records and independently replay real IA actions in Habitat."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from .build import load_task, setup_world
from .export import check_dialogue, export_build, visible_messages
from .geometry import distance, local_event
from .runtime import parse_answer
from .schema import ALLOWED_TOOLS, score_answer, write_json


def audit_sample(directory, root):
    task = load_task(directory)
    selection = json.loads((directory / "selected.json").read_text())
    record = json.loads((directory / selection["record"]).read_text())
    check_dialogue(record, root)
    expected_tools = set(ALLOWED_TOOLS[task.form])
    assert {t["name"] for t in record["tools"]} == expected_tools
    calls = record["calls"]
    assert all(c["call"]["name"] in expected_tools for c in calls)
    # A successful query may include a genuine recovery from a rejected call.
    # Export masks those invalid calls; keep the actual context that enabled recovery.
    decisions = record["decisions"]
    assert len(decisions) <= {"simple_qa": 1, "retrieval_qa": 4}.get(task.form, 12)
    assert all(visible_messages(d["output"]) for d in decisions), "Empty/opaque-only training target"
    # Check against CURRENT task revision, not a stale attempt's cached label.
    if task.form != "interactive_ia":
        answer = parse_answer(decisions[-1]["output"])
        assert score_answer(answer, task.private)["passed"], "Answer/anchor does not match final task revision"
    for decision in decisions:
        text = json.dumps(decision["input"])
        for forbidden in ('"target_pixels"', '"reference_instance"', '"reference_actions"', '"pose_candidates"', '/private/'):
            assert forbidden not in text, f"Private field in actor context: {forbidden}"
    public = json.loads((directory / "task.json").read_text())
    assert "private" not in public
    # SHA256 pins the exact images in the accepted conversation, including tool views.
    images = set()
    for message in record["messages"]:
        for part in message.get("content", []) if isinstance(message.get("content"), list) else []:
            if part.get("type") == "input_image":
                images.add(Path(part["image_path"]))
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(images)}
    return {"sample": task.id, "passed": True, "tools_paired": len(calls),
            "decisions": len(decisions), "image_sha256": hashes}


def replay_ia(directory, root, world):
    task = load_task(directory)
    selection = json.loads((directory / "selected.json").read_text())
    attempt = directory / selection["record"]
    original = json.loads((attempt.parent / "private/motion.json").read_text())
    world.reset(task.private["start_pose"])
    history = [{"pose": world.sim.pose(), "action": None}]
    for entry in original[1:]:
        collision = world.sim.step(entry["action"])
        history.append({"pose": world.sim.pose(), "action": entry["action"], "collision": collision})
    geometry = local_event(task.private, history)
    maximum_error = max(distance(a["pose"]["position"], b["pose"]["position"]) for a, b in zip(history, original))
    geometry["replay_max_position_error_m"] = maximum_error
    geometry["replay_pose_match"] = maximum_error < 1e-4
    geometry["passed"] &= geometry["replay_pose_match"]
    path = world.capture(directory / "private/replay")
    geometry["replay_final_image"] = str(path.relative_to(root))
    write_json(directory / "private/replay.json", {"validation": geometry, "history": history})
    assert geometry["passed"], f"Replay failed: {task.id}"
    return geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="config/dataset/atomic_demo.yaml")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    world = setup_world(yaml.safe_load(Path(args.config).read_text()), root) if args.replay else None
    report = []
    try:
        for directory in sorted((root / "samples").iterdir()):
            if not (directory / "selected.json").exists():
                continue
            row = audit_sample(directory, root)
            if world and load_task(directory).form == "interactive_ia":
                row["replay"] = replay_ia(directory, root, world)
            report.append(row)
            print(directory.name, "passed", flush=True)
    finally:
        if world:
            world.close()
    write_json(root / "private/audit.json", report)
    print(export_build(root), flush=True)


if __name__ == "__main__":
    main()
