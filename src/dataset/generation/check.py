"""Check frozen inputs and label provenance without changing the collection."""

from collections import Counter
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from src.common.io import read_json, write_json
from src.harness.session import read_snapshot

from .records import SceneRecord, Snapshot, TaskSample, load_records, local_file


def check(root, config=None, repo=None):
    root = Path(root).resolve()
    scenes = {scene.id: scene for scene in load_records(root / "scenes.jsonl", SceneRecord)}
    cuts = {cut.id: cut for cut in load_records(root / "snapshots.jsonl", Snapshot)}
    samples = load_records(root / "samples.jsonl", TaskSample)
    errors = []
    if not samples:
        errors.append({"sample_id": None, "error": "Empty frozen collection"})
    manifest = read_json(root / "manifest.json")
    counts = dict(Counter(sample.task_id for sample in samples))
    if manifest.get("sample_counts") is not None and manifest["sample_counts"] != counts:
        errors.append({"sample_id": None, "error": "Sample counts differ from the frozen manifest"})
    options = (config or {}).get("generation", {})
    if options.get("per_task") is not None and any(counts.get(str(task)) != options["per_task"] for task in options.get("tasks", counts)):
        errors.append({"sample_id": None, "error": "Collection does not meet the configured task quotas"})
    ids, groups = set(), {}
    for sample in samples:
        try:
            if sample.id in ids:
                raise ValueError("Duplicate sample ID")
            ids.add(sample.id)
            if sample.group_id in groups and groups[sample.group_id] != sample.split:
                raise ValueError("Source group crosses splits")
            groups[sample.group_id] = sample.split
            cut = cuts[sample.snapshot]
            scene = scenes[cut.scene_id]
            if scene.split != sample.split:
                raise ValueError("Sample and scene splits disagree")
            state = read_snapshot(local_file(root, cut.directory + "/state.json"))
            index = {row["frame"]: row for row in load_records(Path(state["workspace"]) / "observations/index.jsonl")}
            image = Path(state["workspace"]) / index[state["current_frame"]]["path"]
            with Image.open(image) as opened:
                if opened.size != (640, 480):
                    raise ValueError("Unexpected current RGB dimensions")
                opened.verify()
            if sample.entry == "continue" and (not state["history"] or state["history"][-1]["role"] != "toolResult"):
                raise ValueError("Continuation does not end at a tool receipt")
            calls = set()
            for message in state["history"]:
                if message["role"] == "assistant":
                    calls.update(block["id"] for block in message["content"] if block["type"] == "toolCall")
                elif message["role"] == "toolResult":
                    if message["toolCallId"] not in calls:
                        raise ValueError("Unpaired script receipt")
                    calls.remove(message["toolCallId"])
            if calls:
                raise ValueError("Unanswered script tool calls")
            memory = read_json(local_file(root, sample.inputs["memory_file"]))
            if len({item["id"] for item in memory}) != len(memory):
                raise ValueError("Duplicate initial memory IDs")
            for reference in sample.inputs.get("images", []):
                with Image.open(local_file(root, reference)) as opened:
                    opened.verify()
            label = read_json(local_file(root, sample.label_file))
            if "mask_file" in label:
                mask = np.load(local_file(root, label["mask_file"]), allow_pickle=False)
                if mask.shape != (480, 640):
                    raise ValueError("Mask does not align with the original RGB")
            if sample.task_id == "14b" and (not cut.attributes.get("reference_success")
                                             or label["reference_steps"] > 12
                                             or label["reference_path_length"] > 2 + 1e-6):
                raise ValueError("Local action reference is missing or exceeds its specification")
        except (ValueError, KeyError, OSError) as error:
            errors.append({"sample_id": sample.id, "error": str(error)})
    report = {"valid": not errors, "samples": len(samples), "by_task": dict(Counter(s.task_id for s in samples)),
              "by_form": dict(Counter(s.form for s in samples)), "errors": errors}
    fingerprint = hashlib.sha256()
    paths = {root / name for name in ("scenes.jsonl", "snapshots.jsonl", "samples.jsonl")}
    for cut in cuts.values():
        paths.update(path for path in (root / cut.directory).rglob("*")
                     if path.is_file() and path.name != "review-map.png")
    for sample in samples:
        paths.add(root / sample.label_file)
        paths.add(root / sample.inputs["memory_file"])
        paths.update(root / name for name in sample.inputs.get("images", []))
    for path in sorted(paths):
        if path.is_file():
            fingerprint.update(path.relative_to(root).as_posix().encode() + b"\0")
            fingerprint.update(hashlib.sha256(path.read_bytes()).digest())
    report["content_sha256"] = fingerprint.hexdigest()
    write_json(root / "reports/check.json", report)
    return report
