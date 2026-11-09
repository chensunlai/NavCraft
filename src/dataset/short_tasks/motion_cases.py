"""Freeze short real-motion cases; private scripted replays are not model demos."""

from contextlib import contextmanager
import json
import math
from pathlib import Path
import shutil
import uuid

import numpy as np
from scipy.ndimage import distance_transform_edt

from src.common.coordinates import body_forward, heading_from_forward, relative_xz, world_to_uv
from src.common.io import write_json
from src.dataset.episodes import EpisodeSpec
from src.dataset.maps import prepare_map
from src.environment import Environment
from src.harness.executor import NavigationExecutor
from src.harness.workspace import read_snapshot


REPO = Path(__file__).resolve().parents[3]
LIMITS = {"max_requests": 20, "max_actions": 32, "max_path_length": 5,
          "max_observations": 4, "seconds": 600}


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _episode(source, seed):
    state = json.loads((source / seed["directory"] / "state.json").read_text())
    identifier = seed["scene_id"]
    asset = f"hm3d/train/{identifier}/{identifier.split('-')[-1]}.basis.glb"
    return EpisodeSpec("short:" + seed["id"], seed["id"], "short", asset,
                       state["pose"]["position"], state["pose"]["rotation_xyzw"], [],
                       str(source / "snapshots.jsonl"), provenance="program_assisted")


@contextmanager
def _executor(source, seed, directory, gpu, snapshot=None):
    episode = _episode(source, seed)
    directory = Path(directory)
    environment = Environment(episode, {"gpu": gpu, "depth": True,
                              "scenes": str(REPO / "data/nas/scene_datasets")}, directory / "private/environment")
    executor = None
    try:
        state = read_snapshot(snapshot) if snapshot else None
        if state is None:
            environment.restore_pose(json.loads((source / seed["directory"] / "state.json").read_text())["pose"])
        scene_map = (Path(state["workspace"]) / "scene/map.json" if state else
                     prepare_map(REPO / "data/scene_maps/hm3d/train" / seed["scene_id"],
                                 episode.start_position, directory / "private"))
        executor = NavigationExecutor(directory, environment, scene_map, toolsets=("navigation", "entity_memory", "geometry"), initial_state=state)
        yield executor, episode
    finally:
        try:
            if executor is not None:
                executor.close()
        finally:
            environment.close()


def _call(executor, name, arguments, calls):
    receipt = executor.execute("script-" + uuid.uuid4().hex, name, arguments)
    value = json.loads(receipt["result"]["content"][0]["text"])
    calls.append({"name": name, "arguments": arguments, "result": value,
                  "images": receipt["result"]["images"]})
    if receipt["result"]["isError"]:
        raise ValueError(value.get("error", "Tool execution failed"))
    return value


def _length(points):
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def _follow(executor, positions, calls, *, max_actions=28, max_path=4.5, tolerance=.25):
    """Use private path candidates, but execute every turn/translation as real tools."""
    env = executor.environment
    start_steps, start_path = env.steps, env.path_length
    while env.distance(positions) >= tolerance:
        remaining = max_actions - (env.steps - start_steps)
        if remaining < 2 or env.path_length - start_path > max_path:
            raise ValueError("Reference exceeded its fixed local budget")
        path = env.shortest_path(positions)
        if not path:
            raise ValueError("No private path candidate")
        waypoint = next((p for p in path[1:] if math.dist(env.position(), p) > .16), path[-1])
        _call(executor, "move", {"mode": "face", "target": {"uv": world_to_uv(waypoint, executor.transform)},
                                "max_actions": min(12, remaining - 1)}, calls)
        value = _call(executor, "move", {"mode": "steps", "actions": ["move_forward"]}, calls)
        if value["stop"] == "blocked":
            raise ValueError("Reference hit a blocked micro-step")
    return {"steps": env.steps - start_steps, "path_length": env.path_length - start_path,
            "distance": env.distance(positions), "endpoint": env.position()}


def _instance(source, seed):
    rows = json.loads((source / "private/scenes" / seed["scene_id"] / "instances.json").read_text())["instances"]
    return next(row for row in rows if row["id"] == seed["attributes"]["target_instance"])


def _target_pixel(source, seed, instance):
    mask = np.load(source / seed["attributes"]["semantic_file"]) == instance["semantic_id"]
    if mask.sum() < 100:
        raise ValueError("Target lacks clear source pixels")
    y, x = np.unravel_index(np.argmax(distance_transform_edt(mask)), mask.shape)
    return [int(x), int(y)]


def _outside_fov(executor, instance):
    """Conservative horizontal bounding-sphere test, not an occlusion claim."""
    env = executor.environment
    pos = env.position()
    forward = body_forward(env.pose()["rotation_xyzw"])
    heading = heading_from_forward(forward)
    dx, dz = instance["center"][0] - pos[0], instance["center"][2] - pos[2]
    relative = relative_xz(dx, dz, heading)
    radius = math.hypot(instance["size"][0], instance["size"][2]) / 2
    distance = relative["distance_m"]
    if distance <= radius:
        return False
    angular_radius = math.degrees(math.asin(radius / distance))
    return abs(relative["bearing_degrees"]) - angular_radius > env.config["hfov"] / 2 + 5


def _accepted_positions(executor, seed):
    env = executor.environment
    positions = []
    for point in seed["attributes"]["approach_points"]:
        snapped = env.snap_point(point)
        if (snapped is not None and math.dist(point, snapped) < .1
                and abs(snapped[1] - env.position()[1]) <= .2 and env.shortest_path([snapped])):
            positions.append(snapped)
    if not positions:
        raise ValueError("No current-navigation-mesh approach region")
    return positions


def _check_replay(source, seed, trial, snapshot, calls, positions, gpu):
    """Replay exactly the accepted tool calls in a separate restored environment."""
    with _executor(source, seed, trial / "replay", gpu, snapshot=snapshot) as (executor, _):
        env = executor.environment
        first_frame = executor.current["frame"]
        start_steps, start_path = env.steps, env.path_length
        replay = []
        for call in calls:
            _call(executor, call["name"], call["arguments"], replay)
        distance = env.distance(positions)
        if distance >= .35 or env.path_length - start_path > 5 or env.steps - start_steps > 32:
            raise ValueError("Independent real-motion replay failed")
        result = {"passed": True, "source": "program_assisted_not_model_demonstration",
                  "initial_frame": first_frame, "calls": replay, "endpoint": env.position(),
                  "distance": distance, "steps": env.steps - start_steps,
                  "path_length": env.path_length - start_path,
                  "motions": env.motions[-(env.steps - start_steps):],
                  "motion_trajectory": env.motion_trajectory}
        write_json(trial / "replay.json", result)
        return result


def _save_case(output, identifier, kind, seed, trial, episode, instruction, label, reference, images=()):
    case = output / "cases" / identifier
    case.mkdir(parents=True, exist_ok=False)
    shutil.copytree(trial / "snapshot", case / "snapshot")
    write_json(case / "private/episode.json", episode.to_dict())
    write_json(case / "private/label.json", label)
    # Verification frames are inspectable but never added to the model snapshot.
    for phase, directory in (("reference", "build"), ("replay", "replay")):
        shutil.copytree(trial / directory / "workspace/observations",
                        case / "private" / phase / "observations")
    write_json(case / "private/reference.json", {**reference, "image_root": "reference"})
    replay = json.loads((trial / "replay.json").read_text())
    write_json(case / "private/replay.json", {**replay, "image_root": "replay"})
    task = {"id": identifier, "kind": kind, "instruction": instruction, "snapshot": "snapshot",
            "label": "private/label.json", "episode": "private/episode.json", "limits": LIMITS,
            "provenance": {"scene_id": seed["scene_id"], "group_id": seed["group_id"],
                           "source_snapshot": seed["id"], "source": "program_assisted_not_model_demonstration"}}
    if images:
        task["images"] = list(images)
    write_json(case / "construction.json", {"status": "accepted", "reference_replayed": True,
               "initial_condition": label.get("initial_condition"), "snapshot_before_reference": True})
    write_json(case / "task.json", task)
    return {**task, "path": f"cases/{identifier}/task.json"}


def _revisit(source, output, seed, identifier, trial, gpu, candidate):
    instance = _instance(source, seed)
    with _executor(source, seed, trial / "build", gpu) as (executor, episode):
        env, prefix = executor.environment, []
        original = env.position()
        pixel = _target_pixel(source, seed, instance)
        projected = executor.spatial.target({"image": executor.current["frame"], "pixel": pixel})
        uv = projected["uv"]
        world = [uv[0] * executor.transform["size_m"] + executor.transform["origin"][0],
                 uv[1] * executor.transform["size_m"] + executor.transform["origin"][1]]
        if any(abs(world[i] - instance["center"][(0, 2)[i]]) > instance["size"][(0, 2)[i]] / 2 + .15 for i in (0, 1)):
            raise ValueError("Old semantic pixel disagrees with newly rendered geometry")
        _call(executor, "memory_edit", {"operations": [{"op": "create", "id": "O01", "type": "object",
              "fields": {"name": instance["category"], "category": instance["category"],
                         "description": "Previously observed; use the evidence image to identify this instance.",
                         "location": {"image": executor.current["frame"], "pixel": pixel}}}]}, prefix)
        positions = _accepted_positions(executor, seed)
        angle = candidate * math.tau / 12
        requested = [original[0] + 2.4 * math.cos(angle), original[1], original[2] + 2.4 * math.sin(angle)]
        destination = env.snap_point(requested)
        route = env.shortest_path([destination]) if destination is not None else []
        if (not route or not 2 <= _length(route) <= 4 or math.dist(destination, requested) > .35
                or abs(destination[1] - original[1]) > .2):
            raise ValueError("No short outbound route for this direction")
        _follow(executor, [destination], prefix, max_actions=28, max_path=4.2)
        if env.distance(positions) < 1.0:
            raise ValueError("Outbound endpoint is already near the accepted target region")
        pos = env.position()
        dx, dz = pos[0] - instance["center"][0], pos[2] - instance["center"][2]
        d = math.hypot(dx, dz)
        away = [pos[0] + dx / d * .5, pos[1], pos[2] + dz / d * .5]
        _call(executor, "move", {"mode": "face", "target": {"uv": world_to_uv(away, executor.transform)}, "max_actions": 12}, prefix)
        if not _outside_fov(executor, instance):
            raise ValueError("Target remains inside the initial field of view")
        executor.snapshot(trial / "snapshot")
        start = env.state()
        calls = []
        result = _follow(executor, positions, calls)
        label = {"positions": positions, "success_distance": .35, "minimum_motion_m": .1,
                 "target_instance": instance["id"], "reference_replayed": True,
                 "reference_path_length": result["path_length"], "reference_steps": result["steps"],
                 "initial_condition": "target_outside_current_horizontal_fov; occlusion_not_required",
                 "initial_target_visible": False, "target_record": "O01", "target_geometry": instance}
        reference = {"source": "program_assisted_not_model_demonstration", "prefix": prefix, "calls": calls,
                     "initial_state": start, "result": result,
                     "motions": env.motions[start["steps"]:], "motion_trajectory": env.motion_trajectory[len(start["motion_trajectory"])-1:]}
    _check_replay(source, seed, trial, trial / "snapshot", calls, positions, gpu)
    return _save_case(output, identifier, "revisit", seed, trial, episode,
                      "Return to the object recorded as O01, using its memory and evidence to identify it. Go next to it and submit.",
                      label, reference)


def _recover(source, output, seed, identifier, trial, gpu, candidate):
    with _executor(source, seed, trial / "build", gpu) as (executor, episode):
        env, prefix = executor.environment, []
        original = env.position()
        angle = candidate * math.tau / 12
        target = [original[0] + math.sin(angle) * .5, original[1], original[2] - math.cos(angle) * .5]
        _call(executor, "move", {"mode": "face", "target": {"uv": world_to_uv(target, executor.transform)}, "max_actions": 12}, prefix)
        blocked = False
        for _ in range(6):
            value = _call(executor, "move", {"mode": "steps", "actions": ["move_forward"]}, prefix)
            if value["stop"] == "blocked":
                blocked = True
                break
        if not blocked:
            raise ValueError("No real blocked action near this seed")
        start = env.position()
        heading = heading_from_forward(body_forward(env.pose()["rotation_xyzw"]))
        options = []
        for offset in (-90, 90, -135, 135):
            direction = math.radians(heading + offset)
            requested = [start[0] + 1.0 * math.sin(direction), start[1], start[2] - 1.0 * math.cos(direction)]
            point = env.snap_point(requested)
            path = env.shortest_path([point]) if point is not None else []
            if (path and math.dist(point, requested) < .3 and .7 <= _length(path) <= 3
                    and abs(point[1] - start[1]) <= .2):
                options.append((_length(path), point))
        if not options:
            raise ValueError("Blocked seed has no short side/retreat recovery route")
        goal = min(options, key=lambda pair: pair[0])[1]
        mapped = executor.renderer.render(executor.current["pose"], [], view={"around_self": 3},
                                         located_points=[{"label": "P1", "uv": world_to_uv(goal, executor.transform)}])
        # The last blocked tool result is copied into a public note exactly, not fabricated.
        write_json(executor.workspace / "scratch/last_motion.json", prefix[-1]["result"])
        executor.snapshot(trial / "snapshot")
        initial = env.state()
        calls = []
        result = _follow(executor, [goal], calls, max_actions=24, max_path=3.5)
        label = {"positions": [goal], "success_distance": .35, "minimum_motion_m": .1,
                 "initial_blocked": True, "reference_replayed": True,
                 "reference_path_length": result["path_length"], "reference_steps": result["steps"],
                 "initial_condition": "real_forward_blocked_then_side_or_retreat_recovery; no_door_crossing_claim",
                 "blocked_motion": initial["motions"][-1]}
        reference = {"source": "program_assisted_not_model_demonstration", "prefix": prefix, "calls": calls,
                     "initial_state": initial, "result": result,
                     "motions": env.motions[initial["steps"]:], "motion_trajectory": env.motion_trajectory[len(initial["motion_trajectory"])-1:]}
    _check_replay(source, seed, trial, trial / "snapshot", calls, [goal], gpu)
    return _save_case(output, identifier, "recover", seed, trial, episode,
                      "The last forward move was blocked. Its actual feedback is in scratch/last_motion.json. "
                      f"Open map image {mapped['view_id']} with view_images. Move away from the obstacle "
                      "and reach the marked P1 region, then submit.", label, reference)


def build_motion_cases(source: Path, output: Path, per_kind=4, gpu=0, on_case=None):
    """Generate small, bounded local cases; unverified candidates remain rejected."""
    source, output = Path(source).resolve(), Path(output).resolve()
    seeds = _rows(source / "snapshots.jsonl")
    # Cover different instances/scenes before trying another view of one instance.
    by_group = {}
    for seed in seeds:
        by_group.setdefault(seed["group_id"], []).append(seed)
    groups = list(by_group.values())
    directions = [0, 3, 6, 9, 2, 5, 8, 11, 1, 7]
    cache = REPO / "cache/short-tasks" / output.name
    tasks = []
    for kind, build in (("revisit", _revisit), ("recover", _recover)):
        used = set()
        accepted = 0
        for attempt in range(60):
            if accepted >= per_kind:
                break
            cycle, group_index = divmod(attempt, len(groups))
            group = groups[group_index]
            seed = group[cycle % len(group)]
            if seed["group_id"] in used:
                continue
            direction = directions[cycle % len(directions)]
            trial = cache / kind / f"candidate-{attempt:03d}"
            trial.mkdir(parents=True, exist_ok=False)
            try:
                task = build(source, output, seed, f"{kind}-{accepted + 1:02d}", trial, gpu, direction)
                tasks.append(task)
                accepted += 1
                used.add(seed["group_id"])
                if on_case:
                    on_case(task)
            except Exception as error:
                output.mkdir(parents=True, exist_ok=True)
                with (output / "rejections.jsonl").open("a") as stream:
                    stream.write(json.dumps({"kind": kind, "candidate": attempt, "snapshot": seed["id"],
                                             "error": f"{type(error).__name__}: {error}"}) + "\n")
    return tasks
