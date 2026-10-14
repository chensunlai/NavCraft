"""Private acceptance checks for Pi smoke runs; never expose this entry to the actor.

One JSON request on stdin, one {ok,data} response. Mapping checks use archived
geometry and real frame provenance; geometric view consistency is not a semantic
RGB or occlusion verdict, which remains an explicit visual-review requirement.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _json(path):
    return json.loads(Path(path).read_text())


def _lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def parse_answer(text):
    """Require an actual unambiguous final JSON answer, without filling its anchor."""
    if isinstance(text, dict):
        values = [text]
    elif isinstance(text, str):
        values = []
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                value, _ = decoder.raw_decode(text[match.start():])
                if isinstance(value, dict) and "choice" in value:
                    values.append(value)
            except json.JSONDecodeError:
                pass
    else:
        raise ValueError("answer must be the final assistant text or its JSON object")
    valid = [v for v in values if "choice" in v and "anchor" in v]
    canonical = {json.dumps({"choice": v["choice"], "anchor": v["anchor"]}, sort_keys=True) for v in valid}
    if len(canonical) != 1:
        raise ValueError("Expected one unambiguous JSON object with choice and anchor")
    return json.loads(canonical.pop())


def qa(*, workspace, sample_dir, answer, **unused):
    from src.dataset.atomic.schema import score_answer

    public = _json(Path(sample_dir) / "task.json")
    if public["form"] == "interactive_ia":
        raise ValueError("IA requires private local-event scoring, not a text answer")
    try:
        actual = parse_answer(answer)
    except ValueError as error:
        return {"passed": False, "parse_error": str(error), "sample_id": public["id"]}
    result = score_answer(actual, _json(Path(sample_dir) / "private/labels.json"))
    return {**result, "sample_id": public["id"], "actual_answer": actual}


def _blob(trace_dir, digest):
    if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("Invalid trace blob digest")
    content = (trace_dir / "blobs" / digest).read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("Trace blob hash mismatch")
    return content.decode()


def _field(card, label):
    found = re.search(rf"(?m)^\s*-?\s*{re.escape(label)}\s*[:：]\s*(.*)$", card)
    return found.group(1) if found else ""


def _refs(card, images):
    return {identifier for identifier in images
            if re.search(rf"(?<![\w-]){re.escape(identifier)}(?![\w-])", _field(card, "证据"))}


def _segment_error(candidate, expected):
    a, b = candidate[0], candidate[-1]
    c, d = expected[0], expected[-1]
    middle = lambda x, y: [(x[i] + y[i]) / 2 for i in (0, 1)]
    center_error = math.dist(middle(a, b), middle(c, d))
    endpoint_error = min(max(math.dist(a, c), math.dist(b, d)), max(math.dist(a, d), math.dist(b, c)))
    u, v = [b[i] - a[i] for i in (0, 1)], [d[i] - c[i] for i in (0, 1)]
    length, reference_length = math.hypot(*u), math.hypot(*v)
    angle = 90 if min(length, reference_length) < 1e-9 else math.degrees(math.acos(
        min(1, abs(sum(u[i] * v[i] for i in (0, 1))) / (length * reference_length))))
    return {"center_error_m": center_error, "endpoint_error_m": endpoint_error,
            "angle_error_degrees": angle, "length_ratio": length / max(reference_length, 1e-9)}


def _successful(event):
    if event.get("type") != "tool_execution_end":
        return False
    data = event.get("data", {})
    if data.get("isError"):
        return False
    result = data.get("result", {})
    if isinstance(result, dict):
        if result.get("isError") or result.get("ok") is False:
            return False
        for content in result.get("content", []):
            if content.get("type") == "text":
                try:
                    parsed = json.loads(content["text"])
                    if isinstance(parsed, dict) and parsed.get("ok") is False:
                        return False
                except (ValueError, KeyError):
                    pass
    return True


def _timeline(trace_dir, map_path, entity_id, images):
    events = _lines(trace_dir / "trace.jsonl")
    card_path = f"memory/entities/{entity_id}.md"
    snapshots, files, first_seen = [], {}, {}
    starts = {}
    for index, event in enumerate(events):
        data = event.get("data", {})
        if event.get("type") == "tool_execution_start":
            starts[data.get("toolCallId")] = data
        if _successful(event) and data.get("toolName") in {"nav_observe", "nav_step"}:
            content = json.dumps(data)
            for frame in images:
                if frame in content:
                    first_seen.setdefault(frame, index)
        if event.get("type") != "workspace_changes":
            continue
        for change in data.get("changes", []):
            if change.get("after"):
                files[change["path"]] = change["after"]
            else:
                files.pop(change["path"], None)
        raw_entities = _blob(trace_dir, files[map_path]) if map_path in files else "[]"
        try:
            entities = json.loads(raw_entities)
        except json.JSONDecodeError:
            # File edits may be incremental. Only the final map must be valid;
            # an intermediate invalid JSON version is still retained in the trace.
            continue
        if not isinstance(entities, list) or any(not isinstance(e, dict) for e in entities):
            continue
        entity = next((e for e in entities if e.get("id") == entity_id), None)
        card = _blob(trace_dir, files[card_path]) if card_path in files else None
        snapshots.append({"index": index, "initial": data.get("label") == "initial",
                          "entity": entity, "card": card, "files": deepcopy(files)})
    initial = next((s for s in snapshots if s["initial"]), None)
    created = next((s for s in snapshots if not s["initial"] and s["entity"] is not None and s["card"] is not None), None)
    newly_created = bool(initial and initial["entity"] is None and created)
    revision, new_evidence = None, set()
    if newly_created:
        old_refs = _refs(created["card"], images)
        for snapshot in snapshots:
            if snapshot["index"] <= created["index"] or not snapshot["card"] or not snapshot["entity"]:
                continue
            changed = snapshot["card"] != created["card"] or snapshot["entity"] != created["entity"]
            added = _refs(snapshot["card"], images) - old_refs
            added = {ref for ref in added if created["index"] < first_seen.get(ref, -1) < snapshot["index"]}
            if changed and added:
                revision, new_evidence = snapshot, added
                break
    uses = []
    if revision:
        for index, event in enumerate(events):
            data = event.get("data", {})
            if index <= revision["index"] or not _successful(event):
                continue
            start = starts.get(data.get("toolCallId"), {})
            args = start.get("args", {})
            name = data.get("toolName")
            if name == "read" and args.get("path", args.get("file_path")) in {map_path, card_path}:
                uses.append({"index": index, "tool": name})
            if name == "render_map":
                # Renderer records exactly which current map hash it consumed.
                if revision["files"].get(map_path, "missing") in json.dumps(data):
                    uses.append({"index": index, "tool": name})
    return {"created_during_run": newly_created, "revised_after_new_evidence": revision is not None,
            "reused_after_revision": bool(uses), "creation_event": created["index"] if created else None,
            "revision_event": revision["index"] if revision else None,
            "creation_evidence_ids": sorted(_refs(created["card"], images)) if created else [],
            "new_evidence_ids": sorted(new_evidence), "use_events": uses,
            "final_hashes": {path: files.get(path) for path in (map_path, card_path)}}


def mapping(*, workspace, scene_dir, trace_dir, environment_private_dir=None,
            expected_portal="D03", center_tolerance_m=.5, endpoint_tolerance_m=.6,
            angle_tolerance_degrees=25, **unused):
    from src.common.map import MapCalibration
    import numpy as np

    workspace, scene_dir, trace_dir = map(Path, (workspace, scene_dir, trace_dir))
    environment_private_dir = Path(environment_private_dir) if environment_private_dir else trace_dir / "environment"
    doors = _json(scene_dir / "raw/doors.json")["doors"]
    expected = next(d for d in doors if d["id"] == expected_portal)
    calibration = MapCalibration(scene_dir)
    panel = next(p for p in calibration.panels if p.level_id == expected["level"])
    vx, vy, vw, vh = calibration.viewbox
    to_world = lambda p: [(p[0] * vw + vx - panel.x_offset) / panel.x_scale,
                          (p[1] * vh + vy - panel.y_offset) / panel.y_scale]
    maps = _json(workspace / "scene/maps/index.json")["maps"]
    map_info = next(m for m in maps if m["map_id"] == f"map_{expected['level']}")
    map_path = map_info["current_entities_ref"]
    candidates = []
    for entity in _json(workspace / map_path):
        geometry = entity.get("geometry")
        if entity.get("type") != "portal" or not geometry or "segment" not in geometry:
            continue
        points = [to_world(p) for p in geometry["segment"]]
        error = _segment_error(points, expected["segment"])
        nearest = min((d for d in doors if d["level"] == expected["level"]),
                      key=lambda d: _segment_error(points, d["segment"])["center_error_m"])["id"]
        error["nearest_archived_portal"] = nearest
        error["passed"] = (nearest == expected_portal and error["center_error_m"] <= center_tolerance_m
                           and error["endpoint_error_m"] <= endpoint_tolerance_m
                           and error["angle_error_degrees"] <= angle_tolerance_degrees
                           and .4 <= error["length_ratio"] <= 1.8)
        candidates.append((error["center_error_m"], entity, error))
    if not candidates:
        return {"passed": False, "geometry_ok": False, "reason": "No located portal entity"}
    _, entity, geometry = min(candidates, key=lambda item: item[0])
    card = (workspace / f"memory/entities/{entity['id']}.md").read_text()
    connections = set(re.findall(r"\bR\d+\b", _field(card, "连接")))
    connections_ok = connections == set(expected["connects"]) and "未知" not in _field(card, "连接")
    images = {item["id"]: item for item in _lines(workspace / "observations/index.jsonl")
              if item.get("source") == "observation"}
    frames = {item["id"]: item for item in _lines(environment_private_dir / "frames.jsonl")}
    evidence, valid = [], []
    midpoint = np.mean(np.array(expected["segment"]), axis=0)
    for ref in sorted(_refs(card, images)):
        item, private = images[ref], frames.get(ref)
        original = environment_private_dir / "environment/observations" / f"rgb_{ref.removeprefix('frame_')}.png"
        public = (workspace / item["path"]).resolve()
        authentic = bool(private and public.is_relative_to(workspace.resolve()) and public.is_file()
                         and original.is_file() and public.read_bytes() == original.read_bytes())
        in_view = False
        if authentic:
            matrix = np.asarray(private["pose"]["world_from_optical"])
            target = [midpoint[0], matrix[1, 3], -midpoint[1], 1]
            local = np.linalg.inv(matrix) @ target
            in_view = local[2] > 0 and abs(math.degrees(math.atan2(local[0], local[2]))) <= private["camera"]["hfov_degrees"] / 2
            if in_view:
                valid.append((ref, matrix))
        evidence.append({"id": ref, "authentic_rgb": authentic, "portal_in_horizontal_view": bool(in_view)})
    viewpoint_changed = any(not np.allclose(a, b, atol=.01) for i, (_, a) in enumerate(valid) for _, b in valid[i + 1:])
    lifecycle = _timeline(trace_dir, map_path, entity["id"], images)
    lifecycle["final_snapshot_matches"] = all(
        digest == hashlib.sha256((workspace / path).read_bytes()).hexdigest()
        for path, digest in lifecycle.pop("final_hashes").items())
    old_views = [matrix for ref, matrix in valid if ref in lifecycle["creation_evidence_ids"]]
    new_views = [matrix for ref, matrix in valid if ref in lifecycle["new_evidence_ids"]]
    lifecycle["revision_grounded_in_new_view"] = any(
        not np.allclose(a, b, atol=.01) for a in old_views for b in new_views)
    evidence_ok = len(valid) >= 2 and viewpoint_changed and lifecycle["revision_grounded_in_new_view"]
    unique = sum(error["passed"] for _, _, error in candidates) == 1
    return {"passed": bool(geometry["passed"] and unique and connections_ok and evidence_ok
                            and all(lifecycle[k] for k in ("created_during_run", "revised_after_new_evidence", "reused_after_revision", "final_snapshot_matches"))),
            "entity_id": entity["id"], "expected_portal": expected_portal,
            "geometry": geometry, "unique_match": unique, "connections_ok": connections_ok,
            "evidence_ok": evidence_ok, "viewpoint_changed": viewpoint_changed,
            "evidence": evidence, "lifecycle": lifecycle,
            "thresholds": {"center_m": center_tolerance_m, "endpoint_m": endpoint_tolerance_m,
                           "angle_degrees": angle_tolerance_degrees, "length_ratio": [.4, 1.8]},
            "visual_review_required": True,
            "limitation": "Authentic RGB and camera geometry do not establish semantic visibility or rule out occlusion; inspect the actual cited images."}


def image_goal(*, private_dir, env_config="config/environment/goat_visual.yaml",
               task_config="config/task/goat_dev_visual.yaml", episode_id="goat:val_unseen:QaLdnwvtxbs:2",
               goal_index=2, repo=str(REPO), **unused):
    import yaml
    import numpy as np
    from src.dataset.episodes.base import EpisodeSpec

    repo, output = Path(repo), Path(private_dir)
    output.mkdir(parents=True, exist_ok=True)
    load = lambda value: deepcopy(value) if isinstance(value, dict) else yaml.safe_load((repo / value).read_text())
    config = load(env_config)
    config = config.get("environment", config)
    config["scenes"] = str((repo / config.get("scenes", "data/nas/scene_datasets")).resolve())
    config["gpu"] = config.get("gpu", config.get("gpus", [2])[0])
    config["depth"] = False
    if config.get("egl_vendor_file"):
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = config["egl_vendor_file"]
    task = load(task_config)
    manifest = _json(repo / task["episodes"]["manifest"])
    episode = EpisodeSpec(**next(e for e in manifest["episodes"] if e["id"] == episode_id))
    goal = episode.goals[goal_index]
    if goal["modality"] != "image":
        raise ValueError("Selected original goal is not an image target")
    if config.get("name") == "fake":
        from src.environment.fake import FakeSimulator as Simulator
    else:
        from src.environment.simulator import HabitatSimulator as Simulator
    sim = Simulator(episode, config)
    try:
        before = sim.capture(output, 0)
        sim.target_image(goal["image_goal"], output / "target.png")
        after = sim.capture(output, 1)
        restored = np.allclose(before.pose["world_from_optical"], after.pose["world_from_optical"], atol=1e-6)
        from PIL import Image
        with Image.open(output / "target.png") as image:
            dimensions = list(image.size)
        result = {"passed": bool(restored and dimensions == list(reversed(goal["image_goal"]["image_dimensions"]))),
                  "episode_id": episode_id, "original_goal_index": goal_index,
                  "camera_restored": bool(restored), "image_size": dimensions, "target_image": str(output / "target.png"),
                  "before_pose": before.pose, "after_pose": after.pose,
                  "kind": "environment_only_target_render; no navigation/model score"}
        (output / "image_goal_result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        sim.close()


def main():
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    try:
        request = json.loads(sys.stdin.read())
        mode = request.pop("mode")
        functions = {"qa": qa, "mapping": mapping, "image_goal": image_goal}
        if mode not in functions:
            raise ValueError(f"Unknown verification mode: {mode}")
        result = {"ok": True, "data": functions[mode](**request)}
    except Exception as error:
        result = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    protocol.write(json.dumps(result, ensure_ascii=False) + "\n")
    protocol.close()


if __name__ == "__main__":
    main()
