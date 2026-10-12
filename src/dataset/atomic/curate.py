"""Execute explicitly curator-approved repairs, retaining original attempts.

This is NOT a teacher-success generator. A curation JSON specifies sample IDs,
reasons, grounded explanations and (optionally) tool calls. Generated answers are
marked agent_assisted, and navigation is always re-executed in Habitat.
"""

import argparse
import json
from pathlib import Path

import yaml

from .build import load_task, setup_world
from .runtime import collect_attempt
from .schema import write_json


def function(name, args, index):
    return [{"type": "function_call", "call_id": f"curated-{index:02d}",
             "name": name, "arguments": json.dumps(args, ensure_ascii=False)}]


def curated_turns(task, note):
    turns = []
    if task.form == "retrieval_qa":
        turns.append(function("read_text", {"ref": "resources.md"}, len(turns)))
        turns.append(function("view_images", {"refs": ["task_map.png"]}, len(turns)))
    elif task.form == "interactive_qa":
        if task.private.get("require_translation"):
            turns.append(function("nav_step", {"actions": ["move_forward"]}, len(turns)))
        elif task.private.get("require_motion"):
            turns.append(function("nav_step", {"actions": ["turn_left"]}, len(turns)))
        turns.append(function("nav_observe", {"mode": "panorama" if task.observation == "panorama" or task.cluster == "A01" else "single"}, len(turns)))
    elif task.form == "interactive_ia":
        if task.cluster == "A08":
            turns.append(function("view_images", {"refs": ["task_map.png"]}, len(turns)))
        # Small chunks retain actual feedback between orientation and translation.
        actions = list(task.private["reference_actions"])
        while actions:
            chunk = [actions.pop(0)]
            while actions and actions[0] == chunk[0] and len(chunk) < 2:
                chunk.append(actions.pop(0))
            turns.append(function("nav_step", {"actions": chunk}, len(turns)))
        return turns
    choice = note.get("choice", task.private.get("choice"))
    answer = {"choice": choice, "anchor": task.anchor, "explanation": note["explanation"]}
    turns.append([{"type": "message", "role": "assistant", "content": [
        {"type": "output_text", "text": json.dumps(answer, ensure_ascii=False)}]}])
    return turns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--curation", required=True)
    parser.add_argument("--config", default="config/dataset/atomic_demo.yaml")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    notes = json.loads(Path(args.curation).read_text())
    if notes.get("approved_by") != "current_agent_visual_review":
        raise ValueError("Requires an explicit curator review file")
    world = setup_world(yaml.safe_load(Path(args.config).read_text()), root)
    try:
        for identifier, note in notes["samples"].items():
            directory = root / "samples" / identifier
            task = load_task(directory)
            task.resources.update({path: path for path in list(task.resources.values())})
            task.save(directory)
            write_json(directory / "private/curation.json", {**note, "approved_by": notes["approved_by"],
                       "policy": "Rule-derived reference plan chosen by curator; visible answer is curator-authored. Not a teacher success."})
            if (directory / "attempts/agent_revision/record.json").exists():
                raise ValueError(f"Refusing to overwrite existing attempt for {identifier}")
            record = collect_attempt(task, directory, world, suffix="agent_revision",
                                     assisted=curated_turns(task, note))
            if note.get("source") == "mixed_revision":
                record["source"] = "mixed_revision"
                write_json(directory / "attempts/agent_revision/record.json", record)
            print(identifier, record["validation"], flush=True)
            if record["validation"]["passed"]:
                write_json(directory / "selected.json", {"record": "attempts/agent_revision/record.json",
                           "reason": note["reason"], "visual_review": "curator-reviewed; final interaction audit pending",
                           "source_detail": "Agent curated rule-generated answer/plan, tools executed live; not DeepSeek success",
                           "prior_attempts": [str(p.relative_to(directory)) for p in (directory / "attempts").glob("*/record.json") if "agent_revision" not in str(p)]})
    finally:
        world.close()


if __name__ == "__main__":
    main()
