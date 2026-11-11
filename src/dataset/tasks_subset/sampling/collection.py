"""Collect navigation snapshots and private observations from real execution."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import yaml

from src.common.io import read_json
from src.common.coordinates import world_to_uv
from src.common.scenes import canonical_scene_id
from src.dataset.episodes import EpisodeSpec, create_source
from src.dataset.maps import map_directory
from src.environment.habitat import Environment
from src.harness.runtime.executor import NavigationExecutor
from src.harness.runtime.workspace import read_snapshot

from ..bundle import current_observation_frames, reject, snapshot_digest, write_json
from .geometry import crossed_segment, floor_patch_fraction, project, read_depth, unproject, visible_point
from .scene import load_scene, room_at


def _hssd_episode(config, scene_id):
    dataset = Path(config["assets"]) / "hssd-hab"
    identifier = Path(scene_id).name.removesuffix(".scene_instance.json")
    split_file = dataset / "scene_splits.yaml"
    splits = yaml.safe_load(split_file.read_text())
    if identifier not in splits["train"] or identifier.split("_")[0] in {x.split("_")[0] for x in splits["val"]}:
        raise ValueError(f"HSSD scene is not an eligible train building: {identifier}")
    return EpisodeSpec(f"hssd:sample:{identifier}", identifier, "hssd", scene_id,
                       [0., 0., 0.], [0., 0., 0., 1.], [], str(split_file),
                       provenance="program_assisted")


def _environment_config(config, episode, gpu):
    result = {"scenes": config["assets"], "gpu": gpu, "seed": config.get("seed", 0), "depth": True}
    if episode.scene_id.startswith("hssd-hab/"):
        result["scene_dataset_config"] = str(Path(config["assets"]) / "hssd-hab/hssd-hab.scene_dataset_config.json")
    return result


def _place_hssd_start(env, episode, scene, rng):
    pathfinder = env.sim.sim.pathfinder
    pathfinder.seed(env.config["seed"])
    for _ in range(256):
        point = [float(x) for x in pathfinder.get_random_navigable_point()]
        if not all(map(math.isfinite, point)) or not any(abs(point[1]-l["floor_height"]) < .35 for l in scene["levels"]):
            continue
        if pathfinder.island_area(pathfinder.get_island(point)) < 10:
            continue
        angle = rng.uniform(0, math.tau)
        rotation = [0., math.sin(angle/2), 0., math.cos(angle/2)]
        episode.start_position, episode.start_rotation = point, rotation
        env.restore_pose({"position": point, "rotation_xyzw": rotation,
                          "camera_position": [point[0], point[1]+env.config["sensor_height"], point[2]],
                          "camera_rotation_xyzw": rotation})
        return
    raise ValueError("No HSSD navigable start on an annotated floor")


def _is_official_instruction(episode):
    return episode.provenance == "official" and episode.benchmark in {"r2r", "rxr"}


def _start_episodes(config, scene_id):
    hm3d = scene_id.startswith("hm3d/train/")
    name = config.get("collection", {}).get("episode_source", "objnav" if hm3d else "r2r")
    if name not in ({"objnav", "goat"} if hm3d else {"r2r", "rxr"}):
        raise ValueError(f"Incompatible episode source {name}: {scene_id}")
    source = create_source({"name": name, "root": config["datasets"][name], "split": "train"}, Path.cwd())
    if hm3d:
        identifier = Path(scene_id).name.removesuffix(".basis.glb")
        class SceneShard(type(source)):
            def _shards(self, requested):
                path = self.root / self.split / "content" / f"{identifier}.json.gz"
                if not path.is_file():
                    raise FileNotFoundError(path)
                return [path]
        source = SceneShard(source.config)
        goal_train = Path(config["datasets"]["goat"]) / "train/content" / f"{identifier}.json.gz"
        held_out = [Path(config["datasets"]["objnav"]) / "val/content" / goal_train.name,
                    Path(config["datasets"]["goat"]) / "val_unseen/content" / goal_train.name]
        if not goal_train.is_file() or any(path.is_file() for path in held_out):
            raise ValueError(f"HM3D scene not in the eligible train intersection: {scene_id}")
    else:
        if not scene_id.startswith("mp3d/"):
            raise ValueError(f"No verified training split configured for {scene_id}")
        for split in ("val_unseen", "test_challenge" if name == "rxr" else "test"):
            suffix = "_guide" if name == "rxr" else ""
            path = source.root / split / f"{split}{suffix}.json.gz"
            if any(canonical_scene_id(e["scene_id"]) == scene_id for e in read_json(path)["episodes"]):
                raise ValueError(f"MP3D scene overlaps {split}: {scene_id}")
    episodes = [e for e in source.load() if e.scene_id == scene_id]
    if not episodes:
        raise ValueError(f"No official training episodes for {scene_id}")
    return episodes


class _Capture(NavigationExecutor):
    def __init__(self, directory, environment, output, bundle, stream, initial_state=None, prefix=None):
        self.output, self.bundle, self.stream = output, bundle, stream
        self.frame_keys = _frame_index(bundle, prefix["frame_keys"] if prefix else [])
        super().__init__(directory, environment, toolsets=("navigation", "node_memory"),
                         initial_state=initial_state)

    def capture(self, observation, body_pose=None, **metadata):
        row = super().capture(observation, body_pose, **metadata)
        key = f"{self.stream}/{row['frame']}"
        self.frame_keys[row["frame"]] = key
        pose = observation.pose
        self.bundle["frames"][key] = {
            "key": key, "frame": row["frame"], "rgb": str(observation.rgb.relative_to(self.output)),
            "depth": str(observation.depth.relative_to(self.output)), "step": row["step"],
            "group": metadata.get("group"), "yaw": metadata.get("yaw"),
            "pose": pose, "public_pose": row["pose"], "camera": observation.camera,
            "world_from_optical": pose["world_from_optical"], "stream": self.stream,
            "room": room_at(self.bundle["scene"], pose["position"])}
        if self.stream.startswith("walk"):
            _sightings(self, [key])
        return row


def _call(executor, name, args):
    reply = executor.execute(f"{executor.stream}:{len(executor._calls)}", name, args)["result"]
    value = json.loads(reply["content"][0]["text"])
    if reply["isError"]:
        raise RuntimeError(f"{name}: {value}")
    return value


def _move(executor, actions, calls=None):
    args = {"mode": "steps", "actions": actions}
    value = _call(executor, "move", args)
    if calls is not None:
        calls.append({"name": "move", "arguments": args})
    return value


def _yaw_delta(pose, point):
    x, _, z = pose["position"]
    vector = np.asarray(pose["world_from_optical"])[:3, 2]
    desired = math.atan2(point[0]-x, -(point[2]-z))
    heading = math.atan2(vector[0], -vector[2])
    return math.degrees((desired-heading+math.pi) % math.tau - math.pi)


def _follow(executor, point, calls=None, budget=42, radius=0.36, on_step=None):
    env = executor.environment
    start = env.steps
    while env.steps-start < budget:
        p = env.position()
        if env.distance([point]) < radius:
            return True
        path = env.shortest_path([point])
        if not path:
            return False
        waypoint = next((q for q in path[1:] if math.dist(p, q) > 0.17), point)
        delta = _yaw_delta(env.pose(), waypoint)
        if abs(delta) > 16:
            _move(executor, ["turn_right" if delta > 0 else "turn_left"], calls)
        elif _move(executor, ["move_forward"], calls)["stop"] == "blocked":
            if env.steps-start+2 > budget:
                return False
            _move(executor, ["turn_right" if delta >= 0 else "turn_left", "move_forward"], calls)
        if on_step:
            on_step()
    return env.distance([point]) < radius


def _sightings(executor, keys):
    scene = executor.bundle["scene"]
    records = executor.bundle["sightings"]
    for key in keys:
        frame = executor.bundle["frames"][key]
        camera, matrix = frame["camera"], frame["world_from_optical"]
        depth = read_depth(executor.output / frame["depth"], camera)

        def add(kind, identifier, pixel, world, method, verified, quality):
            records.append({"kind": kind, "id": identifier, "frame": key, "pixel": pixel,
                            "world": world, "method": method, "verified": verified, "quality": quality})

        point_ids = {r["id"]: r["world"] for r in records if r["kind"] == "point" and
                     executor.bundle["frames"][r["frame"]]["stream"] == executor.stream}
        for identifier, world in point_ids.items():
            pixel = visible_point(world, depth, camera, matrix, tolerance=.045)
            if pixel:
                add("point", identifier, pixel, world, "optical_depth_reprojection", True, "runtime_depth")
        for i, (fx, fy) in enumerate(((.32, .60), (.50, .64), (.69, .60), (.49, .40),
                                     (.50, .85), (.40, .91), (.60, .91))):
            pixel = [int(camera["width"]*fx), int(camera["height"]*fy)]
            x, y = pixel
            world = unproject(pixel, depth[y, x], camera, matrix)
            if (world is not None and visible_point(world, depth, camera, matrix)
                    and all(math.dist(world, p) > .14 for p in point_ids.values())):
                add("point", f"{key}/P{i}", pixel, world, "optical_depth_surface", True, "runtime_depth")
        for obj in scene["objects"]:
            center, size = np.asarray(obj["center_xyz"]), np.asarray(obj["size_xyz"])
            projected = project(center, camera, matrix)
            if projected is None or projected[2] > 6:
                continue
            x, y, _ = projected
            hits = []
            for row in range(max(1, int(y)-15), min(camera["height"]-1, int(y)+16), 5):
                for col in range(max(1, int(x)-15), min(camera["width"]-1, int(x)+16), 5):
                    world = unproject((col, row), depth[row, col], camera, matrix)
                    if world is not None and np.all(np.abs(world-center) <= size/2 + .04):
                        hits.append(([col, row], world))
            if len(hits) >= 4:
                pixel, world = hits[len(hits)//2]
                add("object", obj["id"], pixel, world, "foreground_in_schematic_bounds",
                    False, "requires_runtime_instance_review")
        for opening in scene["openings"]:
            if opening["status"] != "confirmed_open":
                continue
            (a, b), (c, d) = opening["segment_xz"]
            y = next(level["floor_height"] for level in scene["levels"] if level["id"] == opening["level"])
            world = [(a+c)/2, y+.75, (b+d)/2]
            projection = project(world, camera, matrix)
            if projection is None:
                continue
            col, row, z = projection
            col, row = int(col+.5), int(row+.5)
            sample = depth[max(0, row-4):row+5, max(0, col-4):col+5]
            if sample.size == 81 and .4 < z < 5 and np.count_nonzero(sample > z+.18) >= 65:
                add("opening", opening["id"], [col, row], world,
                    "depth_clear_at_annotated_portal", True, "runtime_depth_and_schematic_portal")


def _panorama(executor, record_node):
    result = _call(executor, "observe", {"mode": "panorama"})
    keys = [f"{executor.stream}/{row['frame']}" for row in result["views"]]
    if record_node:
        _call(executor, "memory_edit", {"operations": [
            {"op": "create", "description": "Six RGB views recorded at this stop."}]})
    return keys


def _traversals(scene, motions, start):
    samples = [(0, start), *((m["step"], p["position"]) for m in motions for p in m["samples"])]
    rooms = [room_at(scene, p) for _, p in samples]
    events = []
    for i in range(1, len(samples)):
        a, b = samples[i-1][1], samples[i][1]
        for opening in scene["openings"]:
            sides = opening["connects"] or []
            floor = next(l["floor_height"] for l in scene["levels"] if l["id"] == opening["level"])
            if abs(a[1]-floor) > .45 or abs(b[1]-floor) > .45:
                continue
            if len(sides) != 2 or not crossed_segment(a, b, opening["segment_xz"]):
                continue
            before = next((r for r in reversed(rooms[:i]) if r), None)
            after = next((r for r in rooms[i:] if r), None)
            if before == after or {before, after} != set(sides):
                continue
            event = {"kind": "doorway_crossing", "opening": opening["id"], "from_room": before, "to_room": after,
                     "step": samples[i][0], "level": opening["level"]}
            if event not in events[-1:]:
                events.append(event)
    return events


def _frame_index(bundle, keys):
    index = {bundle["frames"][key]["frame"]: key for key in keys}
    if len(index) != len(keys):
        raise ValueError("Cut contains duplicate public frame IDs")
    return index


def _cut(executor, identifier):
    bundle, root = executor.bundle, executor.output
    directory = root / "states" / identifier / "snapshot"
    executor.snapshot(directory)
    env = executor.environment
    frames = [executor.frame_keys[fid] for fid in executor.observations.items]
    node_frames = {node["id"]: [key for key in frames
                   if bundle["frames"][key]["group"] == node["panorama"]]
                   for node in executor.memory.nodes}
    def level_at(p):
        floors = [l["id"] for l in bundle["scene"]["levels"] if abs(l["floor_height"]-p[1]) <= .45]
        return floors[0] if len(floors) == 1 else None

    history = [{"pos": world_to_uv(env.trajectory[0], executor.transform),
                "level": level_at(env.trajectory[0]), "step": 0}]
    for motion in env.motions:
        history.extend({"pos": world_to_uv(s["position"], executor.transform),
                        "level": level_at(s["position"]), "step": motion["step"]}
                       for s in motion["samples"])
    state = {"id": identifier, "snapshot": str(directory.relative_to(root)),
             "current_frame": executor.frame_keys[executor.current["frame"]], "frame_keys": frames,
             "node_frames": node_frames, "room": room_at(bundle["scene"], env.position()),
             "step": env.steps, "visited_rooms": list(dict.fromkeys(
                 room for p in env.trajectory if (room := room_at(bundle["scene"], p)))),
             "traversals": _traversals(bundle["scene"], env.motions, env.trajectory[0]), "transform": deepcopy(executor.transform),
             "public_trajectory": history, "trajectory_level": level_at(env.position())}
    for i, traversal in enumerate(state["traversals"]):
        events = state["traversals"]
        lower = events[i-1]["step"] if i else 0
        upper = events[i+1]["step"] if i+1 < len(events) else env.steps+1
        for step, position in [(0, env.trajectory[0]), *(
                (m["step"], p["position"]) for m in env.motions for p in m["samples"])]:
            room = room_at(bundle["scene"], position)
            if room is None:
                continue
            if step < traversal["step"] and room != traversal["from_room"]:
                lower = max(lower, step+1)
            if step > traversal["step"] and room != traversal["to_room"]:
                upper = min(upper, step)
        def evidence(room, start, end):
            return [key for key in frames if start <= bundle["frames"][key]["step"] < end
                    and bundle["frames"][key]["room"] == room
                    and level_at(bundle["frames"][key]["pose"]["position"]) == traversal["level"]]
        before = evidence(traversal["from_room"], lower, traversal["step"])
        after = evidence(traversal["to_room"], traversal["step"], upper)
        traversal["evidence_status"] = "available" if before and after else "missing"
        if before:
            traversal["before_frame"] = before[-1]
        if after:
            traversal["after_frame"] = after[0]
        for room_field, node_field, reverse in (("from_room", "from_node", True),
                                               ("to_room", "to_node", False)):
            valid = [(bundle["frames"][keys[0]]["step"], node)
                     for node, keys in node_frames.items() if keys and
                     keys[0] in evidence(traversal[room_field],
                                         lower if reverse else traversal["step"],
                                         traversal["step"] if reverse else upper)]
            if valid:
                traversal[node_field] = sorted(valid, reverse=reverse)[0][1]
    region_frames = [key for key in frames if bundle["frames"][key]["room"]]
    transitions = []
    for a, b in zip(region_frames, region_frames[1:]):
        first, last = bundle["frames"][a], bundle["frames"][b]
        level = level_at(first["pose"]["position"])
        if first["room"] != last["room"] and level and level == level_at(last["pose"]["position"]):
            if not any(t["from_room"] == first["room"] and t["to_room"] == last["room"] and
                       first["step"] <= t["step"] <= last["step"] for t in state["traversals"]):
                transitions.append({"kind": "zone_transition", "from_room": first["room"],
                                    "to_room": last["room"], "level": level,
                                    "before_frame": a, "after_frame": b, "step": last["step"]})
    state["zone_transitions"] = transitions
    bundle["states"].append(state)
    return state


def _branch(bundle, output, env_config, episode, state, kind, operation, cut=None, **fields):
    identifier = f"{state['id']}-{kind}-{len(bundle['trials'])}"
    snapshot = output / state["snapshot"]
    start_digest = snapshot_digest(snapshot)
    saved = read_snapshot(snapshot)
    calls = []
    records = []
    for phase in ("branch", "replay"):
        stream = f"{phase}-{identifier}"
        directory = output / "private" / stream
        env = Environment(episode, env_config, directory / "raw")
        executor = None
        try:
            executor = _Capture(directory, env, output, bundle, stream, saved, prefix=state)
            start = env.steps
            receipts = []
            if phase == "branch":
                outcome = operation(executor, calls)
                if cut and outcome.get("stop") == "blocked":
                    _cut(executor, cut)
            else:
                for call in calls:
                    call_reply = _call(executor, call["name"], call["arguments"])
                    receipts.append(call_reply)
            goals = fields.get("goal_positions") or outcome.get("positions")
            measurement = None
            if goals:
                distance = env.distance(goals)
                initial_distance = env.sim.distance_from(saved["pose"]["position"], goals)
                measurement = {"positions": goals, "distance": distance if math.isfinite(distance) else None,
                               "initial_distance": initial_distance if math.isfinite(initial_distance) else None}
            records.append({"motions": env.motions[start:], "positions": env.trajectory[start:],
                            "goal_measurement": measurement,
                            "end_frame": executor.frame_keys[executor.current["frame"]], "pose": env.pose(),
                            "nodes": deepcopy(executor.memory.nodes), "receipts": receipts,
                            "panoramas": list({f["group"] for f in bundle["frames"].values()
                                               if f["stream"] == stream and f["group"]})})
        finally:
            if executor is not None:
                executor.close()
            env.close()
    original, replayed = records
    agreement = (original["motions"] == replayed["motions"] and
                 original["positions"] == replayed["positions"] and original["pose"] == replayed["pose"])
    trial = {"id": identifier, "kind": kind, "start_state": state["id"], "start_snapshot_sha256": start_digest, "calls": calls,
             "motions": original["motions"], "positions": original["positions"],
             "end_frame": original["end_frame"], "reference": outcome,
             "goal_measurement": original["goal_measurement"],
             "replay": {**replayed, "passed": agreement, "calls": calls}, **fields}
    bundle["trials"].append(trial)
    return trial


def _near_opening(executor, opening):
    env, scene = executor.environment, executor.bundle["scene"]
    room = room_at(scene, env.position())
    sides = opening["connects"] or []
    if len(sides) != 2 or room not in sides:
        return None
    other = next((side for side in sides if side != room), None)
    (a, b), (c, d) = opening["segment_xz"]
    length = math.hypot(c-a, d-b)
    if length < .45:
        return None
    for sign in (1, -1):
        point = [(a+c)/2 - sign*(d-b)/length*.8, env.position()[1],
                 (b+d)/2 + sign*(c-a)/length*.8]
        snapped = env.snap_point(point)
        if snapped and math.dist(snapped, point) < .25 and room_at(scene, snapped) == other:
            return snapped
    return None


def _select_start(candidates, scene, rng):
    graph = {}
    for opening in scene["openings"]:
        if opening["status"] == "confirmed_open" and len(opening["connects"] or []) == 2:
            a, b = opening["connects"]
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)
    areas = {}
    for room in scene["rooms"]:
        points = room["polygon_xz"]
        areas[room["id"]] = abs(sum(a*d - b*c for (a, b), (c, d)
                                    in zip(points, points[1:]+points[:1])))/2
    def key(episode):
        room = room_at(scene, episode.start_position)
        neighbors = graph.get(room, set())
        second = {r for n in neighbors for r in graph.get(n, set())} - neighbors - {room}
        return (bool(second), areas.get(room, 0) >= 6, min(len(neighbors), 4),
                min(areas.get(room, 0), 24), rng.random())
    return max(candidates, key=key)


def _straight(executor, calls, point):
    env = executor.environment
    start = env.position()
    delta = _yaw_delta(env.pose(), point)
    n = round(abs(delta)/30)
    if n:
        _move(executor, ["turn_right" if delta > 0 else "turn_left"]*n, calls)
    for _ in range(min(30, round(math.dist(start, point)/.25))):
        if math.dist(env.position(), point) <= .40:
            break
        if _move(executor, ["move_forward"], calls)["stop"] == "blocked":
            break
    return {"reached": math.dist(env.position(), point) <= .40}


def _recording(executor, calls):
    result = _call(executor, "observe", {"mode": "panorama"})
    calls.append({"name": "observe", "arguments": {"mode": "panorama"}})
    room_id = room_at(executor.bundle["scene"], executor.environment.position())
    room = next((r["name"] for r in executor.bundle["scene"]["rooms"] if r["id"] == room_id), None)
    args = {"operations": [{"op": "create", "description": "Six RGB views recorded at this stop.",
                            **({"room": room} if room else {})}]}
    _call(executor, "memory_edit", args)
    calls.append({"name": "memory_edit", "arguments": args})
    return {"reached": True, "panorama": result["group"]}


def collect(config: dict, output: Path, gpu: int) -> dict:
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    scene_id = canonical_scene_id(config["scene"])
    directory = map_directory(config["maps"], scene_id)
    scene = load_scene(directory)
    source = scene.pop("source")
    rng = random.Random(config.get("seed", 0))
    if scene_id.startswith("hssd-hab/"):
        episode = _hssd_episode(config, scene_id)
        source["limitations"].append("HSSD geometry-only sampling; room and object tasks require annotations.")
    else:
        candidates = [ep for ep in _start_episodes(config, scene_id) if room_at(scene, ep.start_position)]
        route = config.get("collection", {}).get("room_route")
        if route:
            candidates = [ep for ep in candidates if room_at(scene, ep.start_position) == route[0]]
        if not candidates:
            raise ValueError("No training start unambiguously inside an annotated room")
        short = [ep for ep in candidates if ep.benchmark == "r2r" and ep.reference_path and
                 len(ep.goals[0]["instruction"].split()) <= 75 and
                 3.1 <= sum(math.dist(a, b) for a, b in zip(ep.reference_path, ep.reference_path[1:])) <= 7.5]
        episode = _select_start(short or candidates, scene, rng)
    source["split_evidence"] = {"split": "train", "benchmark": episode.benchmark,
                                "episode_shard": episode.source_file,
                                "sha256": hashlib.sha256(Path(episode.source_file).read_bytes()).hexdigest()}
    bundle = {"scene_id": scene_id, "split": "train", "seed": config.get("seed", 0),
              "episode": episode.to_dict(), "source": source, "scene": scene, "frames": {},
              "states": [], "sightings": [], "trials": [], "targets": [], "official_tasks": [],
              "rejections": []}
    env_config = _environment_config({**config, "seed": bundle["seed"]}, episode, gpu)
    bundle["environment_spec"] = {k: v for k, v in env_config.items() if k not in {"gpu", "depth"}}
    env = Environment(episode, env_config, output / "private" / "walk-0" / "raw")
    executor = None
    passages = []
    settings = config.get("collection", {})
    try:
        if episode.provenance == "program_assisted":
            _place_hssd_start(env, episode, scene, rng)
            bundle["episode"] = episode.to_dict()
        executor = _Capture(output / "private" / "walk-0", env, output, bundle, "walk-0")
        node_limit = settings.get("nodes", 3)
        start_state = _cut(executor, "start")
        register_official_tasks(bundle, output, episode, start_state, env)
        if settings.get("room_route"):
            from .routes import walk_room_route
            walk_room_route(executor, settings["room_route"], settings)
            node_limit = 0
        for i in range(node_limit):
            frames = _panorama(executor, True)
            state = _cut(executor, f"stop-{i}")
            options = []
            visible = {s["id"] for s in bundle["sightings"] if s["frame"] in frames and s["kind"] == "opening"}
            for opening in scene["openings"]:
                if opening["status"] == "confirmed_open":
                    point = _near_opening(executor, opening)
                    if point is not None:
                        options.append((opening, point))
            options.sort(key=lambda op: (room_at(scene, op[1]) in state["visited_rooms"],
                                         op[0]["id"] in {t["opening"] for t in state["traversals"]},
                                         op[0]["id"] not in visible,
                                         math.dist(env.position(), op[1])))
            if i+1 == node_limit:
                break
            if options:
                opening, point = options[0]
                visible_frame = next((s["frame"] for s in bundle["sightings"]
                    if s["id"] == opening["id"] and s["kind"] == "opening"
                    and s["frame"] in current_observation_frames(bundle, state)), None)
                if visible_frame:
                    passages.append((state, opening["id"], point, visible_frame))
            else:
                # Try a local accessible floor station; retain blocked attempts in the prefix.
                point = next((p for x, z in ((1.3, 0), (0, 1.3), (-1.3, 0), (0, -1.3))
                             if (p := env.snap_point([env.position()[0]+x, env.position()[1],
                                                      env.position()[2]+z])) is not None
                             and .8 < math.dist(env.position(), p) < 2.5), None)
            if point is None:
                break
            cut_taken = False
            start = env.position()
            def off_node():
                nonlocal cut_taken
                current = room_at(scene, env.position())
                if not cut_taken and current and current != state["room"] and math.dist(start, env.position()) > .65:
                    _cut(executor, f"cross-{i}")
                    cut_taken = True
            if not _follow(executor, point, budget=settings.get("route_actions", 42), on_step=off_node):
                bundle["rejections"].append(reject(output, "collection", state["id"],
                                                    "route_proposal_did_not_execute_to_endpoint", stage="capture"))
    finally:
        if executor is not None:
            executor.close()
        env.close()

    bundle.setdefault("passage_proposals", []).extend([{"state_id": state["id"], "opening": opening, "point": point, "visible_frame": visible}
                                   for state, opening, point, visible in passages])
    write_json(output / "private" / "bundle.json", bundle)
    return bundle


def register_official_tasks(bundle, output, episode, state, environment):
    from hashlib import sha256
    from src.bench.metrics import score_goal

    if episode.provenance != "official":
        return
    for index, goal in enumerate(episode.goals):
        shortest = environment.distance(goal["positions"])
        threshold = score_goal(episode.benchmark, submitted=False, distance=shortest,
                               shortest=shortest, path_length=0, steps=0)["official_success_distance"]
        if not threshold <= shortest <= 8 or len(goal["instruction"].split()) > 75:
            continue
        source_index = goal.get("source_index", index)
        identifier = f"{episode.id}:goal:{source_index}"
        row = {"id": identifier, "episode": episode.to_dict(), "state_id": state["id"],
               "goal_index": index, "source_goal_index": source_index, "shortest": shortest,
               "threshold": threshold}
        if goal.get("image_goal"):
            frame = "official-target/" + sha256(identifier.encode()).hexdigest()[:20]
            relative = Path("private") / (frame + ".png")
            environment.target_image(goal["image_goal"], output / relative)
            bundle["frames"][frame] = {"key": frame, "stream": "official-target", "rgb": str(relative)}
            bundle["targets"].append({"id": identifier, "reference_frames": [frame]})
            row["reference_frame"] = frame
        bundle["official_tasks"].append(row)


def collect_trials(config: dict, output: Path, gpu: int, kinds=(), bundle_path=None) -> dict:
    """Resume from frozen observations and independently replay selected branches."""
    output = Path(output).resolve()
    bundle_path = Path(bundle_path) if bundle_path else output / "private" / "bundle.json"
    bundle = read_json(bundle_path)
    episode = EpisodeSpec(**bundle["episode"])
    env_config = _environment_config({**config, "seed": bundle["seed"]}, episode, gpu)
    settings = config.get("collection", {})
    base_states = [state for state in bundle["states"] if state["id"] != "blocked"]
    by_id = {state["id"]: state for state in base_states}
    passages = [(by_id[p["state_id"]], p["opening"], p["point"], p["visible_frame"])
                for p in bundle.get("passage_proposals", [])]

    def branch(state, kind, operation, **details):
        if kinds and kind not in kinds:
            return None
        if any(t["kind"] == kind and t["start_state"] == state["id"] and all(t.get(field) == details.get(field) for field in ("official_task", "target", "opening", "pair_id", "variant")) and not t.get("invalidated")
               and t.get("start_snapshot_sha256") == snapshot_digest(output / state["snapshot"])
               for t in bundle["trials"]):
            return None
        return _branch(bundle, output, env_config, episode, state, kind, operation, **details)

    def follow(point, radius=.36):
        return lambda ex, calls: {"reached": _follow(ex, point, calls,
                                                    budget=settings.get("route_actions", 42), radius=radius)}

    def forward(ex, calls):
        for _ in range(3):
            result = _move(ex, ["move_forward"]*6, calls)
            if result["stop"] == "blocked":
                break
        return {"stop": result["stop"]}

    states = base_states
    for task in bundle["official_tasks"]:
        goal = task["episode"]["goals"][task["goal_index"]]
        state = by_id[task["state_id"]]
        threshold = task["threshold"]

        def official(ex, calls):
            env = ex.environment
            positions = goal["positions"]
            nearest = min(positions, key=lambda p: env.distance([p]))
            _follow(ex, nearest, calls, budget=55, radius=threshold)
            arrived = env.distance(positions) < threshold
            if arrived:
                _call(ex, "submit", {})
                calls.append({"name": "submit", "arguments": {}})
            return {"reached": arrived}

        trial = branch(state, "nav_18", official, official_task=task["id"], goal_positions=goal["positions"])
        if trial and _is_official_instruction(episode):
            trial["kinds"] = ["nav_16", "nav_18"]
    branch(states[0], "nav_04", _recording)
    branch(states[-1], "forward", forward, cut="blocked")
    blocked = next((state for state in bundle["states"] if state["id"] == "blocked"), None)
    if blocked:
        def recover(ex, calls):
            env = ex.environment
            start = np.asarray(env.position())
            alternatives = [p for p in reversed(env.trajectory[:-1])
                            if .7 < math.dist(start, p) <= 2 and abs(p[1] - start[1]) < .25]
            point = next((p for p in alternatives if .7 < env.distance([p]) <= 3), None)
            if point is None:
                return {"reached": False, "reason": "no_known_safe_waypoint"}
            return {"reached": _follow(ex, point, calls, budget=32, radius=.27),
                    "positions": [point]}
        trial = branch(blocked, "nav_03", recover, variant="safe_waypoint")
        if trial and trial["reference"].get("positions"):
            trial["goal_positions"] = trial["reference"]["positions"]
    for state, opening, point, visible in passages[:1]:
        def passage(ex, calls):
            from .routes import passage_points
            portal = next(o for o in bundle["scene"]["openings"] if o["id"] == opening)
            destination = next(r for r in portal["connects"] if r != state["room"])
            points = passage_points(ex, portal, destination)
            if points is None or not _follow(ex, points[0], calls, radius=.15):
                return {"reached":False}
            return _straight(ex, calls, points[2])
        branch(state, "nav_02", passage, opening=opening, visible_frame=visible)
    for state in states[1:]:
        key = state["current_frame"]
        for sighting in bundle["sightings"]:
            if sighting["kind"] != "point" or sighting["frame"] != key:
                continue
            world = sighting["world"]
            start = bundle["frames"][key]["pose"]["position"]
            if abs(world[1]-start[1]) > .13 or not .7 < math.dist(start, world) < 3:
                continue
            frame = bundle["frames"][key]
            if (not frame.get("depth") or floor_patch_fraction(
                    read_depth(output / frame["depth"], frame["camera"]), frame["camera"],
                    frame["world_from_optical"], sighting["pixel"],
                    state["transform"]["floor_height"]) < .9):
                continue
            identifier = sighting["id"]
            trial = branch(state, "nav_01", follow(world, .37),
                           target=identifier, goal_positions=[world])
            if trial:
                trial["kinds"] = ["nav_01", "nav_14"]
                if not any(t["id"] == identifier for t in bundle["targets"]):
                    bundle["targets"].append({"id": identifier, "object_id": None, "reference_frames": [key],
                                              "description": "Marked floor point", "positions": [world],
                                              "success_distance": .40, "source_benchmark": None, "source_id": None})
            break
    if len(states) > 2:
        old = bundle["frames"][states[1]["current_frame"]]["pose"]["position"]
        if math.dist(old, bundle["frames"][states[-1]["current_frame"]]["pose"]["position"]) > 1:
            branch(states[-1], "direct",
                    lambda ex, calls: _straight(ex, calls, old), goal_positions=[old],
                    target_node=next(iter(states[1]["node_frames"])))
    write_json(bundle_path, bundle)
    return bundle
