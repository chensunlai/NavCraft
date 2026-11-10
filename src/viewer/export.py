"""Export saved continuous navigation runs into an independent offline website."""

from src.common.progress import Progress

import argparse
from copy import deepcopy
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

from PIL import Image

from src.harness.runtime.trace import read_trace

TEXT_LIMIT = 16000


def _read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def _short(text):
    if len(text) > TEXT_LIMIT:
        return text[:TEXT_LIMIT] + f"\n[truncated: {len(text) - TEXT_LIMIT} characters omitted]"
    return text


def _texts(content):
    if isinstance(content, str):
        return content
    return "\n".join(item["text"] for item in content or []
                     if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str))


class _Assets:
    def __init__(self, output):
        self.output, self.items, self.paths = output, {}, {}
        (output / "assets").mkdir(parents=True, exist_ok=True)

    def image(self, path, label=""):
        path = Path(path)
        if path in self.paths:
            return {**self.items[self.paths[path]], "label": label}
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.paths[path] = digest
        if digest not in self.items:
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.thumbnail((960, 960))
                url = f"assets/{digest}.jpg"
                if not (self.output / url).is_file():
                    image.save(self.output / url, quality=78, optimize=True)
                self.items[digest] = {"url": url, "width": image.width, "height": image.height}
        return {**self.items[digest], "label": label}


def _workspace_image(workspace, reference):
    if not isinstance(reference, str):
        return None
    path = (workspace / reference).resolve()
    return path if path.is_relative_to(workspace.resolve()) and path.is_file() else None


def _tool_images(value, model_dir, assets):
    workspace = model_dir / "workspace"
    images = []
    for index, reference in enumerate(value.get("images", [])):
        path = _workspace_image(workspace, reference)
        if path:
            labels = value.get("image_labels", [])
            images.append(assets.image(path, labels[index] if index < len(labels) else reference))
    # Native read returns an image blob rather than a public image path. Read
    # the saved raw image bytes, without constructing base64 or request bodies.
    if not images:
        for item in value.get("content", []):
            reference = item.get("data") if isinstance(item, dict) and item.get("type") == "image" else None
            if isinstance(reference, dict) and re.fullmatch(r"[a-f0-9]{64}", reference.get("$trace_blob", "")):
                path = model_dir / "private/blobs" / reference["$trace_blob"]
                if path.is_file() and not path.is_symlink():
                    if hashlib.sha256(path.read_bytes()).hexdigest() != reference["$trace_blob"]:
                        raise ValueError("Image trace blob hash mismatch")
                    images.append(assets.image(path, "Image returned by tool"))
    return images


def _frame_ids(value, frames):
    if isinstance(value, str):
        if value in frames:
            return [value]
        return [frame for frame, row in frames.items() if value == row["path"]]
    if isinstance(value, list):
        return [frame for item in value for frame in _frame_ids(item, frames)]
    if isinstance(value, dict):
        return [frame for item in value.values() for frame in _frame_ids(item, frames)]
    return []


def _memory_delta(before, after):
    left, right = {item["id"]: item for item in before}, {item["id"]: item for item in after}
    return {"created": sorted(right.keys() - left.keys()), "deleted": sorted(left.keys() - right.keys()),
            "revised": sorted(key for key in right.keys() & left.keys() if right[key] != left[key])}


def _memory_records(memory, frames):
    """Project node memory into display records without changing saved state."""
    if isinstance(memory, list):
        return deepcopy(memory)
    groups = {}
    for frame in frames.values():
        if frame.get("kind") == "observation" and frame.get("group"):
            groups.setdefault(frame["group"], []).append(frame)
    for group in groups.values():
        group.sort(key=lambda frame: frame.get("yaw", 0))
    records, members = [], {}
    for node in memory["nodes"]:
        anchor = groups.get(node["anchor"], [])
        evidence = list(dict.fromkeys(frame["frame"] for group in node["history"]
                                     for frame in groups.get(group, [])))
        records.append({"id": node["id"], "type": "node", "name": node["id"],
                        "description": node["description"], "room": node["room"],
                        "point": deepcopy(anchor[0].get("pose", {}).get("pos")) if anchor else None,
                        "evidence": evidence})
        if node["room"] is not None:
            members.setdefault(node["room"], []).append(node)
    for room in dict.fromkeys([*members, *memory["rooms"]]):
        nodes = members.get(room, [])
        evidence = list(dict.fromkeys(group[0]["frame"] for node in nodes
                                     if (group := groups.get(node["panorama"]))))
        records.append({"id": "room:" + room, "type": "room", "name": room,
                        "description": memory["rooms"].get(room), "anchor": None,
                        "nodes": [node["id"] for node in nodes], "evidence": evidence})
    return records


def _round_summary(row, specification, model):
    memory = row.get("memory", {})
    summary = {key: deepcopy(value) for key, value in row.items()
               if key not in {"start_position", "end_position", "record", "error"}}
    summary.update(number=row["round"], instruction=specification["instruction"],
                   success=row.get("success") if row["kind"] == "navigation" else None,
                   entities=memory.get("entities"), types=memory.get("types", {}),
                   tool_errors=memory.get("tool_errors", 0),
                   detail=f"data/rounds/{model}-{row['round']:02d}.json")
    return summary


def _record(model_dir):
    result = _read(model_dir / "result.json")
    scenario = _read(model_dir / "private/scenario.json")
    if not result or scenario:
        return result, scenario
    episode = _read(model_dir / "private/episode.json")
    if not episode or len(episode.get("goals", [])) != 1:
        return result, None
    prompt = next((e["data"]["text"] for e in read_trace(model_dir / "private/trace.jsonl")
                   if e["type"] == "prompt_start"), "")
    metrics = result.get("environment", {}).get("metrics", {})
    row = {**metrics, "round": 1, "id": episode["id"], "kind": "navigation",
           "status": result["status"], "success": bool(metrics["success"]) if result["status"] == "completed" else None,
           "actions": metrics.get("steps", 0), "requests": metrics.get("requests", 0),
           "arrived": metrics.get("navigation_error", float("inf")) < metrics.get("success_distance", 0)}
    scenario = {"scene_id": episode["scene_id"], "protocol": episode["benchmark"], "record_kind": "episode",
                "rounds": [{"id": episode["id"], "instruction": prompt}]}
    return {**result, "rounds": [row], "planned_navigation": 1,
            "metrics": {"navigation_successes": int(bool(row["success"]))}}, scenario


def _model_export(model_dir, output, assets, model_id=None):
    result, scenario = _record(model_dir)
    if not result or not scenario:
        return None
    model = model_id or model_dir.name
    frames = {row["frame"]: row for row in
              (json.loads(line) for line in (model_dir / "workspace/observations/index.jsonl").read_text().splitlines() if line.strip())}
    rows = result.get("rounds", [])
    if len(rows) > len(scenario["rounds"]):
        raise ValueError(f"More result rounds than frozen tasks: {model}")
    details = []
    for row, specification in zip(rows, scenario["rounds"]):
        if row["id"] != specification["id"]:
            raise ValueError(f"Result and frozen task IDs disagree: {model}")
        summary = _round_summary(row, specification, model)
        replay = _read(model_dir / f"private/rounds/{row['round']:02d}-{row['id']}.json", {})
        if scenario.get("record_kind") == "episode":
            replay = {"trajectory": result.get("environment", {}).get("motion_trajectory", [])}
        details.append({"model": model, "number": row["round"], "instruction": specification["instruction"],
                        "summary": summary, "frames": [], "trajectory": replay.get("trajectory", []),
                        "trajectory_source": "Private recorded physical microstep path, for reviewer verification only.",
                        "events": [], "evidence": {}, "memory_before": [], "memory_after": [],
                        "memory_changes": {"created": [], "revised": [], "deleted": []},
                        "memory_status": "No memory snapshot recovered yet."})
    active, prompt_index, memory = None, -1, None
    starts, known_public = {}, set()
    frame_ranges = {}
    for event in read_trace(model_dir / "private/trace.jsonl", images=False):
        kind, value = event["type"], event.get("data", {})
        timestamp = event.get("timestamp")
        if kind == "application_started" and value.get("memory_source") == "empty":
            memory = []
            continue
        if kind == "prompt_start":
            prompt_index += 1
            active = details[prompt_index] if prompt_index < len(details) else None
            if active is None:
                continue
            initial = _frame_ids(value.get("images", []), frames)
            frame_ranges[prompt_index] = set(initial)
            if memory is not None:
                active["memory_before"] = deepcopy(memory)
                active["memory_status"] = "Inherited from recorded initial state and memory changes."
            active["events"].append({"type": "notice", "time": timestamp,
                                     "text": "User request: " + value["text"],
                                     "images": [assets.image(path, reference) for reference in value.get("images", [])
                                                if (path := _workspace_image(model_dir / "workspace", reference))]})
            continue
        if kind == "memory_changed":
            memory = _memory_records(value["after"], frames)
            if active is not None:
                if active["memory_status"] == "No memory snapshot recovered yet.":
                    active["memory_before"] = _memory_records(value["before"], frames)
                active["memory_after"] = deepcopy(memory)
                active["memory_status"] = (
                    "Display records derived from saved node/room snapshots; descriptions and grouping are model claims."
                    if isinstance(value["after"], dict) else
                    "Exact recorded memory snapshots; locations and identities are model claims.")
            continue
        if active is None:
            continue
        if kind == "prompt_end":
            if memory is not None:
                active["memory_after"] = deepcopy(memory)
                active["memory_changes"] = _memory_delta(active["memory_before"], memory)
            active = None
            continue
        if kind == "motion":
            frame_ranges[prompt_index].update(_frame_ids(value.get("frame"), frames))
            continue
        if kind == "message_end":
            message = value.get("message", {})
            text = _texts(message.get("content")) if message.get("role") == "assistant" else ""
            if text.strip():
                active["events"].append({"type": "assistant", "time": timestamp,
                                         "text": _short(text), "images": [], "truncated": len(text) > TEXT_LIMIT})
        elif kind == "tool_execution_start":
            starts[value["toolCallId"]] = value
        elif kind in {"public_tool_reply", "tool_execution_end"}:
            public = kind == "public_tool_reply"
            call_id = value["call_id"] if public else value["toolCallId"]
            if call_id in known_public:
                continue
            if public:
                known_public.add(call_id)
            start = starts.get(call_id, {})
            name = value["name"] if public else value["toolName"]
            body = value["result"]
            text = _texts(body.get("content"))
            reply = {"type": "tool", "time": timestamp, "name": name,
                     "args": value.get("args", start.get("args", {})), "result": _short(text),
                     "images": _tool_images(body, model_dir, assets),
                     "error": bool(body.get("isError") or value.get("isError")), "truncated": len(text) > TEXT_LIMIT}
            current = []
            if name in {"observe", "move"}:
                captured = list(current)
                for block in body.get("content", []):
                    if block.get("type") == "text":
                        try:
                            captured.extend(_frame_ids(json.loads(block["text"]), frames))
                        except (ValueError, TypeError):
                            pass
                current = [fid for fid in captured if frames[fid].get("kind") == "observation"]
                frame_ranges[prompt_index].update(current)
            if current:
                reply["frame"] = current[0]
                reply["step"] = frames[current[0]].get("step")
            active["events"].append(reply)
        elif kind == "compaction_end":
            active["events"].append({"type": "notice", "time": timestamp,
                                     "text": "The recorded conversation was compacted at this point.", "images": []})
    memory_file = "nodes.json" if result.get("memory_type") == "node" else "entities.json"
    final_memory = _memory_records(_read(model_dir / "workspace/memory" / memory_file, []), frames)
    for index, detail in enumerate(details):
        captured = frame_ranges.get(index, set())
        if captured:
            low, high = min(int(fid[2:]) for fid in captured), max(int(fid[2:]) for fid in captured)
            for fid, frame in frames.items():
                if frame.get("kind") == "observation" and low <= int(fid[2:]) <= high:
                    path = _workspace_image(model_dir / "workspace", frame["path"])
                    if path:
                        detail["frames"].append({"id": fid, "step": frame.get("step"), "pose": frame.get("pose"),
                                                 **{key: value for key, value in assets.image(path).items() if key != "label"}})
        if detail["memory_status"] == "No memory snapshot recovered yet.":
            detail["memory_status"] = "No snapshot recorded for this round; empty lists mean unavailable, not confirmed empty."
        if index == len(details) - 1 and memory is None:
            detail["memory_after"] = final_memory
            detail["memory_status"] = "Only the final saved memory is available; earlier round snapshots are unknown."
        for entity in detail["memory_before"] + detail["memory_after"]:
            for fid in entity.get("evidence", []):
                frame = frames.get(fid)
                path = _workspace_image(model_dir / "workspace", frame["path"]) if frame else None
                if path:
                    detail["evidence"][fid] = {"url": assets.image(path)["url"],
                                               "step": frame.get("step"), "pose": frame.get("pose")}
        if scenario.get("record_kind") == "episode":
            tools = [event for event in detail["events"] if event["type"] == "tool"]
            counts = dict(Counter(event["name"] for event in tools))
            memory_info = {"entities": len(detail["memory_after"]), "tools": counts,
                           "tool_errors": sum(event["error"] for event in tools)}
            rows[index]["memory"] = memory_info
            detail["summary"].update(memory=memory_info, entities=memory_info["entities"],
                                     tool_errors=memory_info["tool_errors"])
        _write(output / detail["summary"]["detail"], detail)
    tools = [row.get("memory", {}).get("tools", {}) for row in rows]
    return {"id": model, "label": result["model"], "effort": result["reasoning_effort"], "status": result["status"],
            "successes": result["metrics"].get("navigation_successes", 0),
            "navigation_total": result.get("planned_navigation", result["metrics"].get("navigation_planned")),
            "requests": sum(row.get("requests", 0) for row in rows),
            "actions": sum(row.get("actions", 0) for row in rows),
            "path_length": sum(row.get("path_length", 0) for row in rows),
            "first_entities": rows[0].get("memory", {}).get("entities", 0) if rows else 0,
            "final_entities": len(final_memory), "memory_reads": sum(item.get("memory_read", 0) for item in tools),
            "memory_writes": sum(item.get("memory_edit", 0) for item in tools),
            "rounds": [detail["summary"] for detail in details]}


def export_sequence(run_dir, output_dir) -> Path:
    run_dir, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output == run_dir or run_dir.is_relative_to(output) or output.is_relative_to(run_dir):
        raise ValueError("Export output must be separate from the source run")
    output.mkdir(parents=True, exist_ok=True)
    assets = _Assets(output)
    models, scene_map, scenario = [], None, None
    direct = (run_dir / "result.json").is_file()
    directories = [run_dir] if direct else sorted(path for path in run_dir.iterdir() if path.is_dir() and (path / "result.json").is_file())
    progress = Progress("Export models", [path.name for path in directories])
    for model_dir in directories:
        progress.set(model_dir.name, "running")
        model_id = model_dir.parent.name if direct and model_dir.name[:3].isdigit() else model_dir.name
        model = _model_export(model_dir, output, assets, model_id)
        progress.set(model_dir.name, "finished" if model is not None else "failed")
        if model is not None:
            models.append(model)
            scene_map = scene_map or _read(model_dir / "workspace/scene/map.json")
            scenario = scenario or _record(model_dir)[1]
    if not models:
        raise ValueError("No supported saved task runs found")
    index = {"title": "NavCraft · 导航任务回放", "record_kind": scenario.get("record_kind", "sequence"), "generated_at": datetime.now(timezone.utc).isoformat(),
             "models": models, "map": {key: scene_map[key] for key in ("origin", "size_m", "walls")},
             "scene_id": scenario["scene_id"], "protocol": scenario.get("protocol"),
             "notes": ["Reasoning blocks, HTTP records, credentials and private task labels are excluded from the conversation.",
                       "Memory is what the model recorded, not verified scene truth.",
                       "Map trajectories are private recorded paths shown for reviewer verification."]}
    _write(output / "data/index.json", index)
    web = Path(__file__).with_name("web")
    if web.is_dir():
        shutil.copytree(web, output, dirs_exist_ok=True)
    return output / "index.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(export_sequence(args.run, args.output))


if __name__ == "__main__":
    main()
