"""Collect navigation snapshots and private observations from real execution."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np

from src.common.io import read_json
from src.common.coordinates import world_to_uv
from src.common.scenes import canonical_scene_id
from src.dataset.episodes import EpisodeSpec, create_source
from src.dataset.maps import map_directory
from src.environment.habitat import Environment
from src.harness.runtime.executor import NavigationExecutor
from src.harness.runtime.workspace import read_snapshot

from ..bundle import reject, write_json
from .geometry import crossed_segment, project, read_depth, unproject, visible_point
from .scene import load_scene, room_at


def _start_episodes(config, scene_id):
    name = "objnav" if scene_id.startswith("hm3d/train/") else "r2r"
    source = create_source({"name": name, "root": config["datasets"][name], "split": "train"}, Path.cwd())
    if name == "objnav":
        identifier = Path(scene_id).name.removesuffix(".basis.glb")
        class SceneShard(type(source)):
            def _shards(self, requested):
                path = self.root / self.split / "content" / f"{identifier}.json.gz"
                if not path.is_file():
                    raise FileNotFoundError(path)
                return [path]
        source = SceneShard(source.config)
        goal_train = Path(config["datasets"]["goat"]) / "train/content" / f"{identifier}.json.gz"
        held_out = [source.root / "val/content" / goal_train.name,
                    Path(config["datasets"]["goat"]) / "val_unseen/content" / goal_train.name]
        if not goal_train.is_file() or any(path.is_file() for path in held_out):
            raise ValueError(f"HM3D scene not in the eligible train intersection: {scene_id}")
    else:
        if not scene_id.startswith("mp3d/"):
            raise ValueError(f"No verified training split configured for {scene_id}")
        for split in ("val_unseen", "test"):
            path = source.root / split / f"{split}.json.gz"
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
        if math.dist(p, point) <= radius:
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
    return math.dist(env.position(), point) <= radius


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


def _traversals(scene, motions):
    events = []
    for i in range(len(motions)-1):
        first = motions[i]["samples"][-1]["position"] if motions[i]["samples"] else None
        if first is None:
            continue
        for j in range(i+1, min(i+6, len(motions))):
            positions = [first, *(p["position"] for m in motions[i+1:j+1] for p in m["samples"])]
            before, after = room_at(scene, first), room_at(scene, positions[-1])
            if not before or not after or before == after:
                continue
            for opening in scene["openings"]:
                if set(opening["connects"] or []) == {before, after} and any(
                    crossed_segment(a, b, opening["segment_xz"]) for a, b in zip(positions, positions[1:])
                ):
                    event = {"opening": opening["id"], "from_room": before, "to_room": after,
                             "step": motions[j]["step"]}
                    if not events or any(event[k] != events[-1][k] for k in ("opening", "from_room", "to_room")):
                        events.append(event)
            break
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
             "traversals": _traversals(bundle["scene"], env.motions), "transform": deepcopy(executor.transform),
             "public_trajectory": history, "trajectory_level": level_at(env.position())}
    for traversal in state["traversals"]:
        before = [key for key in frames if bundle["frames"][key]["step"] < traversal["step"]]
        after = [key for key in frames if bundle["frames"][key]["step"] >= traversal["step"]]
        if before and after:
            traversal.update(before_frame=before[-1], after_frame=after[0])
        for room_field, node_field, reverse in (("from_room", "from_node", True),
                                               ("to_room", "to_node", False)):
            valid = [(bundle["frames"][keys[0]]["step"], node)
                     for node, keys in node_frames.items() if keys and
                     bundle["frames"][keys[0]]["room"] == traversal[room_field] and
                     (bundle["frames"][keys[0]]["step"] < traversal["step"]) == reverse]
            if valid:
                traversal[node_field] = sorted(valid, reverse=reverse)[0][1]
    bundle["states"].append(state)
    return state


def _branch(bundle, output, env_config, episode, state, kind, operation, cut=None, **fields):
    identifier = f"{state['id']}-{kind}-{len(bundle['trials'])}"
    saved = read_snapshot(output / state["snapshot"])
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
            records.append({"motions": env.motions[start:], "positions": env.trajectory[start:],
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
    trial = {"id": identifier, "kind": kind, "start_state": state["id"], "calls": calls,
             "motions": original["motions"], "positions": original["positions"],
             "end_frame": original["end_frame"], "reference": outcome,
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
    episodes = _start_episodes(config, scene_id)
    rng = random.Random(config.get("seed", 0))
    candidates = [ep for ep in episodes if room_at(scene, ep.start_position)]
    if not candidates:
        raise ValueError("No training start unambiguously inside an annotated room")
    if scene_id.startswith("mp3d/"):
        short = [ep for ep in candidates if ep.reference_path and
                 len(ep.goals[0]["instruction"].split()) <= 75 and
                 3.1 <= sum(math.dist(a, b) for a, b in zip(ep.reference_path, ep.reference_path[1:])) <= 7.5]
        episode = _select_start(short or candidates, scene, rng)
    else:
        episode = _select_start(candidates, scene, rng)
    source["split_evidence"] = {"split": "train", "benchmark": episode.benchmark,
                                "episode_shard": episode.source_file,
                                "sha256": hashlib.sha256(Path(episode.source_file).read_bytes()).hexdigest()}
    bundle = {"scene_id": scene_id, "split": "train", "seed": config.get("seed", 0),
              "episode": episode.to_dict(), "source": source, "scene": scene, "frames": {},
              "states": [], "sightings": [], "trials": [], "targets": [], "official_tasks": [],
              "rejections": []}
    env_config = {"scenes": config["assets"], "gpu": gpu, "seed": bundle["seed"], "depth": True}
    env = Environment(episode, env_config, output / "private" / "walk-0" / "raw")
    executor = None
    passages = []
    settings = config.get("collection", {})
    try:
        executor = _Capture(output / "private" / "walk-0", env, output, bundle, "walk-0")
        node_limit = settings.get("nodes", 3)
        start_state = _cut(executor, "start")
        if episode.benchmark == "r2r":
            shortest = env.distance(episode.goals[0]["positions"])
            if 0 < shortest <= 8 and len(episode.goals[0]["instruction"].split()) <= 75:
                bundle["official_tasks"].append({"id": episode.id, "episode": episode.to_dict(),
                                                 "state_id": start_state["id"], "shortest": shortest})
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
                if any(s["id"] == opening["id"] and s["frame"] == state["current_frame"]
                       for s in bundle["sightings"]):
                    passages.append((state, opening["id"], point))
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

    bundle["passage_proposals"] = [{"state_id": state["id"], "opening": opening, "point": point}
                                   for state, opening, point in passages]
    write_json(output / "private" / "bundle.json", bundle)
    return bundle


def collect_trials(config: dict, output: Path, gpu: int, kinds=()) -> dict:
    """Resume from frozen observations and independently replay selected branches."""
    output = Path(output).resolve()
    bundle = read_json(output / "private" / "bundle.json")
    episode = EpisodeSpec(**bundle["episode"])
    env_config = {"scenes": config["assets"], "gpu": gpu, "seed": bundle["seed"], "depth": True}
    settings = config.get("collection", {})
    base_states = [state for state in bundle["states"] if state["id"] != "blocked"]
    by_id = {state["id"]: state for state in base_states}
    passages = [(by_id[p["state_id"]], p["opening"], p["point"])
                for p in bundle.get("passage_proposals", [])]

    def branch(state, kind, operation, **details):
        if kinds and kind not in kinds:
            return None
        if any(t["kind"] == kind and t["start_state"] == state["id"] for t in bundle["trials"]):
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
        goal = episode.goals[0]["positions"][0]
        def official(ex, calls):
            # Follow the original route where available; benchmark endpoint is separate.
            for waypoint in [*episode.reference_path[1:], goal]:
                if not _follow(ex, waypoint, calls, budget=26, radius=.5):
                    break
            arrived = ex.environment.distance([goal]) < 3
            if arrived:
                _call(ex, "submit", {})
                calls.append({"name": "submit", "arguments": {}})
            return {"reached": arrived}
        trial = branch(states[0], "nav_18", official,
                        official_task=task["id"])
        if trial:
            trial["kinds"] = ["nav_16", "nav_18"]
    branch(states[0], "nav_04", _recording)
    branch(states[-1], "forward", forward, cut="blocked")
    blocked = next((state for state in bundle["states"] if state["id"] == "blocked"), None)
    if blocked:
        def recover(ex, calls):
            env = ex.environment
            start = np.asarray(env.position())
            matrix = np.asarray(env.pose()["world_from_optical"])
            forward, lateral = matrix[:3, 2], matrix[:3, 0]
            alternatives = []
            for advance in (.65, 1., 1.5):
                for offset in (-1., 1., -.6, .6, -1.5, 1.5):
                    proposal = start + advance*forward + offset*lateral
                    point = env.snap_point(proposal)
                    if point is None or math.dist(point, proposal) > .22:
                        continue
                    path = env.shortest_path([point])
                    length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
                    if .7 <= length <= 4:
                        alternatives.append((length, point))
            if not alternatives:
                return {"reached": False, "reason": "no_local_bypass"}
            _, point = min(alternatives)
            return {"reached": _follow(ex, point, calls, budget=32, radius=.30),
                    "positions": [point]}
        trial = branch(blocked, "nav_03", recover)
        if trial and trial["reference"].get("positions"):
            trial["goal_positions"] = trial["reference"]["positions"]
    for state, opening, point in passages[:1]:
        branch(state, "nav_02", follow(point), opening=opening)
    for state in states[1:]:
        key = state["current_frame"]
        for sighting in bundle["sightings"]:
            if sighting["kind"] != "point" or sighting["frame"] != key:
                continue
            world = sighting["world"]
            start = bundle["frames"][key]["pose"]["position"]
            if abs(world[1]-start[1]) > .13 or not .7 < math.dist(start, world) < 3:
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
    write_json(output / "private" / "bundle.json", bundle)
    return bundle
