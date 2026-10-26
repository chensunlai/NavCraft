"""Collect real, short, reproducible tool prefixes without a teacher model."""

import math
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from src.common.io import read_json, write_json
from src.common.progress import Progress
from src.harness.session import open_session

from .records import SceneRecord, Snapshot, load_records, save_records, stable_id, update_manifest
from .scenes import generation_options, open_environment


def facing_pose(position, point):
    angle = -math.atan2(point[0] - position[0], -(point[2] - position[2])) / 2
    rotation = [0, math.sin(angle), 0, math.cos(angle)]
    return {"position": list(position), "rotation_xyzw": rotation,
            "camera_position": [position[0], position[1] + .88, position[2]],
            "camera_rotation_xyzw": rotation}


def reference_pose(position, point):
    """Keep body yaw while pitching the reference camera toward object height."""
    pose = facing_pose(position, point)
    horizontal = math.hypot(point[0] - position[0], point[2] - position[2])
    pitch = math.atan2(point[1] - pose["camera_position"][1], horizontal)
    pitch = max(-math.radians(70), min(math.radians(70), pitch))
    sy, cy = pose["rotation_xyzw"][1], pose["rotation_xyzw"][3]
    sx, cx = math.sin(pitch / 2), math.cos(pitch / 2)
    pose["camera_rotation_xyzw"] = [cy * sx, sy * cx, -sy * sx, cy * cx]
    return pose


def _informative_box(box):
    if not box or len(box) != 4:
        return False
    width, height = box[2] - box[0], box[3] - box[1]
    return min(width, height) >= 48 and min(width, height) / max(width, height) >= .2


def refresh_reference_views(root, config, repo):
    """Repair reference framing without changing cuts, scripts or verified routes."""
    root, repo = Path(root).resolve(), Path(repo).resolve()
    scenes = load_records(root / "scenes.jsonl", SceneRecord)
    snapshots = load_records(root / "snapshots.jsonl", Snapshot)
    wanted = [row for row in snapshots if not row.attributes.get("reference_view_checked")
              or not _informative_box(row.attributes.get("reference_bbox"))]
    if not wanted:
        return snapshots
    progress = Progress("Frame target references", [row.id for row in wanted])
    failures = []
    for scene in scenes:
        cuts = [row for row in wanted if row.scene_id == scene.id]
        if not cuts:
            continue
        instances = {row["id"]: row for row in read_json(root / scene.instances_path)["instances"]}
        environment = open_environment(scene, config, repo, repo / "cache/generation" / root.name / "references" / scene.id)
        try:
            for cut in cuts:
                instance = instances[cut.attributes["target_instance"]]
                center = np.asarray(instance["center"])
                endpoint = np.asarray(cut.attributes["reference_endpoint"])
                direction = endpoint[[0, 2]] - center[[0, 2]]
                radius = float(np.linalg.norm(direction))
                direction /= max(radius, .001)
                state_file = root / cut.directory / "state.json"
                state_before = state_file.read_bytes()
                cut_position = read_json(state_file)["pose"]["position"]
                best = None
                for degrees in (0, 30, -30, 60, -60):
                    angle = math.radians(degrees)
                    c, s = math.cos(angle), math.sin(angle)
                    away = np.array([c * direction[0] - s * direction[1], s * direction[0] + c * direction[1]])
                    for extra in (0, .5, 1., 1.5, 2., 2.5):
                        point = center[[0, 2]] + away * (radius + extra)
                        requested = [point[0], endpoint[1], point[1]]
                        stance = environment.snap_point(requested)
                        if (stance is None or math.dist(stance, requested) > .3
                                or abs(stance[1] - endpoint[1]) > .35 or math.dist(stance, cut_position) < .3):
                            continue
                        if not environment.shortest_path([stance], endpoint.tolist()):
                            continue
                        pose = reference_pose(stance, instance["center"])
                        environment.sim.restore_pose(pose)
                        observation = environment.observe()
                        semantic = np.load(observation.semantic, allow_pickle=False)
                        y, x = np.nonzero(semantic == instance["semantic_id"])
                        if len(x) < 500:
                            continue
                        height, width = semantic.shape
                        box = [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]
                        margin = min(box[0], box[1], width - box[2], height - box[3])
                        if margin < 8 or not _informative_box(box):
                            continue
                        best = observation, semantic, box, margin, pose
                        break
                    if best is not None:
                        break
                if best is None:
                    failures.append({"snapshot": cut.id, "instance": instance["id"],
                                     "reason": "No unclipped, visible reference from a distinct reachable stance"})
                    progress.set(cut.id, "failed")
                    continue
                observation, semantic, box, margin, pose = best
                image_path = root / cut.attributes["target_image"]
                with Image.open(observation.rgb) as image:
                    image.convert("RGB").save(image_path, quality=85)
                np.save(root / cut.attributes["target_semantic_file"], semantic, allow_pickle=False)
                index_file = root / cut.directory / "workspace/observations/index.jsonl"
                rows = [json.loads(line) for line in index_file.read_text().splitlines()]
                target_row = next(row for row in rows if row["frame"] == cut.attributes["target_frame"])
                target_row["sha256"] = hashlib.sha256(image_path.read_bytes()).hexdigest()
                temporary = index_file.with_suffix(".tmp")
                temporary.write_text("".join(json.dumps(row) + "\n" for row in rows))
                temporary.replace(index_file)
                cut.attributes["frame_visibility"][cut.attributes["target_frame"]] = [
                    row["id"] for row in visible_instances(semantic, list(instances.values()))]
                proof = {"pose": observation.pose, "camera": observation.camera, "target_instance": instance["id"],
                         "target_frame": cut.attributes["target_frame"], "bbox": box, "edge_margin_pixels": margin,
                         "scope": "Reference-only camera framing; original cut state and motion replay unchanged"}
                write_json(root / cut.directory / "private/reference_camera.json", proof)
                if state_file.read_bytes() != state_before:
                    raise RuntimeError("Reference framing changed the saved cut state")
                cut.attributes["reference_view_checked"] = True
                cut.attributes["reference_bbox"] = box
                progress.set(cut.id, "finished")
        finally:
            environment.close()
    save_records(root / "snapshots.jsonl", snapshots)
    write_json(root / "reports/reference_view_failures.json", failures)
    if failures:
        raise RuntimeError(f"{len(failures)} reference views need review; see reports/reference_view_failures.json")
    return snapshots


def footprint_distance(position, instance):
    delta = np.maximum(np.abs(np.asarray(position)[[0, 2]] - np.asarray(instance["center"])[[0, 2]])
                       - np.asarray(instance["size"])[[0, 2]] / 2, 0)
    return float(np.linalg.norm(delta))


def visible_instances(semantic, instances, pixels=500):
    identifiers, counts = np.unique(semantic, return_counts=True)
    visible = {int(i): int(n) for i, n in zip(identifiers, counts) if n >= pixels}
    return [{**row, "pixels": visible[row["semantic_id"]]} for row in instances
            if row["semantic_id"] in visible]


def _candidate_views(environment, scene, instances, rng):
    """Actual semantic views establish legal target sides and approach regions."""
    for instance in rng.permutation(instances):
        center = instance["center"]
        half = np.asarray(instance["size"])[[0, 2]] / 2
        if max(half) > 2.5 or min(instance["size"]) <= .02:
            continue
        approaches, pairs = [], []
        for index in range(12):
            angle = index * math.tau / 12
            direction = np.array([math.cos(angle), math.sin(angle)])
            extent = min(half / np.maximum(np.abs(direction), 1e-6))
            point = np.asarray(center)[[0, 2]] + direction * (extent + .55)
            endpoint = environment.snap_point([point[0], scene.floor_height, point[1]])
            if endpoint is None or abs(endpoint[1] - scene.floor_height) > .35:
                continue
            if not .15 <= footprint_distance(endpoint, instance) <= .85:
                continue
            environment.sim.restore_pose(facing_pose(endpoint, center))
            observation = environment.observe()
            if np.count_nonzero(np.load(observation.semantic, allow_pickle=False) == instance["semantic_id"]) < 500:
                continue
            if not any(math.dist(endpoint, previous) < .2 for previous in approaches):
                approaches.append(endpoint)
            start_point = np.asarray(center)[[0, 2]] + direction * (extent + 1.6)
            start = environment.snap_point([start_point[0], scene.floor_height, start_point[1]])
            if start is None or abs(start[1] - scene.floor_height) > .35 or footprint_distance(start, instance) < 1.2:
                continue
            route = environment.shortest_path([endpoint], start)
            length = sum(math.dist(a, b) for a, b in zip(route, route[1:]))
            if not route or not .45 <= length <= 1.8 or any(abs(p[1] - scene.floor_height) > .35 for p in route):
                continue
            environment.sim.restore_pose(facing_pose(start, center))
            initial = environment.observe()
            if np.count_nonzero(np.load(initial.semantic, allow_pickle=False) == instance["semantic_id"]) >= 500:
                pairs.append((start, endpoint))
        if approaches and pairs:
            yield instance, approaches, pairs


def _verify_replay(scene, config, repo, directory, pose, actions, endpoint):
    environment = open_environment(scene, config, repo, directory, pose)
    try:
        for row in actions:
            if row.get("turn_degrees") is not None:
                environment.turn(row["turn_degrees"])
            else:
                environment.step(row["action"])
        distance = environment.distance([endpoint])
        result = {"passed": distance <= .25 and environment.steps <= 12 and environment.path_length <= 2,
                  "distance": distance, "steps": environment.steps, "path_length": environment.path_length,
                  "endpoint": environment.position(), "motion_trajectory": environment.motion_trajectory}
        write_json(directory / "replay.json", result)
        return result
    finally:
        environment.close()


def _execute_reference(session, endpoint):
    """Use the configured real tools; basic never invokes a super-only action."""
    transform = session.transform
    uv = [(endpoint[axis] - transform["origin"][i]) / transform["size_m"] for i, axis in enumerate((0, 2))]
    if session.tool_tier == "super":
        session.script_tool("nav_goto", {"target": {"uv": uv}, "stop_distance": .18, "max_actions": 12})
        return
    for _ in range(12):
        position = session.environment.position()
        dx, dz = endpoint[0] - position[0], endpoint[2] - position[2]
        if math.hypot(dx, dz) <= .18:
            break
        heading = session.current["pose"]["heading"]
        bearing = (math.degrees(math.atan2(dx, -dz)) - heading + 180) % 360 - 180
        action = ("turn_right" if bearing > 0 else "turn_left") if abs(bearing) > 15 else "move_forward"
        reply = session.script_tool("nav_step", {"actions": [action]})
        if reply["result"]["isError"] or json.loads(reply["result"]["content"][0]["text"]).get("stop") == "blocked":
            break


def collect(root, config, repo):
    root, repo = Path(root).resolve(), Path(repo).resolve()
    existing = root / "snapshots.jsonl"
    scenes = load_records(root / "scenes.jsonl", SceneRecord)
    options = generation_options(config)
    wanted = int(options.get("snapshots_per_scene", 6))
    rng = np.random.default_rng(int(options.get("seed", 0)))
    snapshots, rejections = load_records(existing, Snapshot) if existing.exists() else [], []
    previous_counts = {s.id: sum(c.scene_id == s.id for c in snapshots) for s in scenes}
    progress = Progress("Collect real cuts", [(s.id, i) for s in scenes for i in range(wanted)],
                        {(s.id, i): "finished" for s in scenes for i in range(previous_counts[s.id])})
    for scene in scenes:
        instances = read_json(root / scene.instances_path)["instances"]
        if previous_counts[scene.id] >= wanted:
            # Consume the same scene-level draw as a fresh collection.
            rng.permutation(instances)
            continue
        scout = open_environment(scene, config, repo, repo / "cache/generation" / root.name / "scenes" / scene.id / "scout")
        scene_count = previous_counts[scene.id]
        try:
            candidates = _candidate_views(scout, scene, instances, rng)
            for instance, approaches, pairs in candidates:
                for start, endpoint in pairs:
                    if scene_count >= wanted:
                        break
                    identifier = stable_id(scene.id, instance["id"], start, endpoint)
                    if identifier in {snapshot.id for snapshot in snapshots}:
                        continue
                    destination = root / "snapshots" / identifier
                    work = repo / "cache/generation" / root.name / identifier
                    environment = session = None
                    try:
                        if work.exists():
                            shutil.rmtree(work)
                        pose = facing_pose(start, instance["center"])
                        environment = open_environment(scene, config, repo, work / "environment", pose)
                        # The target reference is photographed from a genuinely different stance.
                        environment.sim.restore_pose(facing_pose(endpoint, instance["center"]))
                        reference = environment.observe()
                        target_image = work / "reference.jpg"
                        target_semantic = np.load(reference.semantic, allow_pickle=False)
                        with Image.open(reference.rgb) as image:
                            image.convert("RGB").save(target_image, quality=90)
                        environment.sim.restore_pose(pose)
                        session = open_session(repo, config, work / "session", environment,
                                               root / scene.map_path, start_core=False)
                        session.begin_script(f"Find the {instance['category']} in the reference image and go next to it.", images=[target_image])
                        target_frame = session._target_frames[0]
                        np.save(session.private / "semantics" / f"{target_frame}.npy", target_semantic, allow_pickle=False)
                        initial_frame = session.current["frame"]
                        mask = np.load(session.private / "semantics" / f"{initial_frame}.npy", allow_pickle=False)
                        visible = visible_instances(mask, instances)
                        if instance["id"] not in {row["id"] for row in visible} or len(visible) < 2:
                            raise ValueError("Initial capture needs the target and a distinct visible distractor")
                        bindings, operations = {}, []
                        transform = session.transform
                        for other in visible:
                            if other["id"] == instance["id"]:
                                continue
                            identifier_memory = f"O{len(operations) + 1:02d}"
                            uv = [(other["center"][axis] - transform["origin"][i]) / transform["size_m"] for i, axis in enumerate((0, 2))]
                            if not all(0 <= value <= 1 for value in uv):
                                continue
                            operations.append({"op": "create", "id": identifier_memory, "type": "object", "fields": {
                                "name": other["category"], "category": other["category"], "point": uv,
                                "evidence": [initial_frame]}})
                            bindings[identifier_memory] = other["id"]
                        if operations:
                            reply = session.script_tool("memory_edit", {"operations": operations[:20]})
                            if reply["result"]["isError"]:
                                raise ValueError("Script memory could not be written")
                        session.script_tool("nav_observe", {"mode": "current"})
                        state = session.snapshot(destination)
                        current_frame = session.current["frame"]
                        before = environment.steps
                        _execute_reference(session, endpoint)
                        actions = environment.motions[before:]
                        reference_length = environment.path_length
                        if (not actions or len(actions) > 12 or reference_length > 2
                                or environment.distance([endpoint]) > .25):
                            raise ValueError("Actual short tool execution did not reach the verified approach")
                        replay = _verify_replay(scene, config, repo, work / "replay", pose, actions, endpoint)
                        if not replay["passed"]:
                            raise ValueError("Independent replay failed")
                        frame_visibility = {}
                        for path in (destination / "private/semantics").glob("*.npy"):
                            frame_visibility[path.stem] = [row["id"] for row in visible_instances(np.load(path, allow_pickle=False), instances)]
                        frame_visibility[target_frame] = [instance["id"]]
                        attrs = {"target_instance": instance["id"], "visible_instances": [row["id"] for row in visible], "current_frame": current_frame,
                                 "target_frame": target_frame, "frame_visibility": frame_visibility,
                                 "target_semantic_file": f"snapshots/{identifier}/private/semantics/{target_frame}.npy",
                                 "target_image": f"snapshots/{identifier}/workspace/{session.observations.get(target_frame)['path']}",
                                 "semantic_file": f"snapshots/{identifier}/private/semantics/{current_frame}.npy",
                                 "reference_actions": [row["action"] for row in actions], "reference_motions": actions,
                                 "reference_endpoint": replay["endpoint"],
                                 "reference_path_length": reference_length, "reference_steps": len(actions),
                                 "approach_points": approaches, "bindings": bindings, "map_transform": transform,
                                 "start_pose": state["pose"], "replay_passed": True, "reference_success": True,
                                 "initial_footprint_distance": footprint_distance(start, instance)}
                        snapshot = Snapshot(identifier, scene.id, f"snapshots/{identifier}",
                                            f"script:{scene.id}:{instance['id']}", attributes=attrs)
                        write_json(destination / "private/reference.json", {**replay, "actions": actions, "approach_points": approaches})
                        snapshots.append(snapshot)
                        save_records(existing, snapshots)
                        progress.set((scene.id, scene_count), "finished")
                        scene_count += 1
                    except Exception as error:
                        rejections.append({"scene": scene.id, "instance": instance["id"], "error": f"{type(error).__name__}: {error}"})
                        write_json(root / "reports/collection_rejections.json", rejections)
                        if destination.exists():
                            shutil.rmtree(destination)
                    finally:
                        if session is not None:
                            session.close()
                        if environment is not None:
                            environment.close()
                if scene_count >= wanted:
                    break
        finally:
            scout.close()
        if scene_count < wanted:
            raise RuntimeError(f"Collected {scene_count}/{wanted} usable cuts in {scene.id}; see collection_rejections.json")
    update_manifest(root, snapshot_count=len(snapshots), script_prefix_trainable=False)
    return refresh_reference_views(root, config, repo)
