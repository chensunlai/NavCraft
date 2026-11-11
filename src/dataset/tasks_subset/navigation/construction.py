"""Freeze navigation tasks supported by collected cuts and independently replayed trials."""
import json
import hashlib
import math
import os
import shutil
from pathlib import Path

from PIL import Image, ImageDraw
from src.common.coordinates import world_to_uv

from ..bundle import current_observation_frames, freeze_case, reject, require_trial_snapshot, write_json
from ..catalog import NAVIGATION_KINDS
from ..review.memory import public_room
from .scoring import arrived_at_goal, doorway_crossed, in_approach_region
from ..sampling.scene import room_at
from ..sampling.geometry import floor_patch_fraction, read_depth


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _samples(run):
    return [s["position"] for motion in run.get("motions", []) for s in motion["samples"]]


def _verified_trial(trial, root, state):
    require_trial_snapshot(root, state, trial)
    calls = trial.get("calls", [])
    replay = trial.get("replay", {})
    _require(calls and replay.get("passed") is True and replay.get("calls"),
             "no independently replayed tool calls")
    def signature(rows):
        return [(c["name"], c["arguments"]) for c in rows]
    _require(signature(calls) == signature(replay["calls"]), "replay changed tool calls")
    _require(all(c["name"] != "move" or c["arguments"].get("mode") == "steps" for c in calls),
             "reference uses non-atomic navigation")
    points, repeated = _samples(trial), _samples(replay)
    _require(len(points) == len(repeated) and all(math.dist(p, q) < .03 for p, q in zip(points, repeated)),
             "physical replay does not match trial")
    return repeated


def _reference_budget(trial, points, position, limits):
    from src.harness.runtime.request import PromptLimits

    limits = PromptLimits(**limits)
    replay = trial["replay"]
    chain = [position, *points]
    usage = {"actions": len(replay["motions"]),
             "path_length": sum(math.dist(a, b) for a, b in zip(chain, chain[1:])),
             "observations": sum(c["name"] == "observe" for c in replay["calls"]),
             "script_tool_calls": len(replay["calls"])}
    _require(usage["actions"] < limits.max_actions, "reference exhausts the action budget")
    _require(limits.max_path_length is None or usage["path_length"] < limits.max_path_length,
             "reference exhausts the path budget")
    _require(limits.max_observations is None or usage["observations"] <= limits.max_observations,
             "reference exceeds the observation budget")
    return usage


def _require_goal_measurements(trial, label, endpoint):
    for phase in (trial, trial["replay"]):
        measurement = phase.get("goal_measurement") or {}
        _require(measurement.get("positions") == label["positions"],
                 "missing geodesic measurement bound to the labelled goals")
        _require(arrived_at_goal(label, endpoint, measurement.get("distance")),
                 "reference endpoint misses the labelled geodesic arrival criterion")


def _reviewed_absent(bundle, instance, keys):
    rows = {(r["object"], r["frame"]): r for r in
            bundle.get("navigation_evidence", {}).get("visibility", [])}
    return bool(keys) and all(rows.get((instance, key), {}).get("visible") is False for key in keys)


def _known_once(bundle, state, instance, present):
    nodes = state["node_frames"]
    seen_nodes = [node for node, keys in nodes.items() if set(keys) & present]
    return len(seen_nodes) == 1 and all(
        len(keys) == 6 and _reviewed_absent(bundle, instance, keys)
        for node, keys in nodes.items() if node != seen_nodes[0])


def _instruction(kind, trial, state, frame, bundle, points, root):
    label = {"kind": kind}
    images = []
    targets = {t["id"]: t for t in bundle.get("targets", [])}
    target = targets.get(trial.get("target"), {})
    objects = {o["id"]: o for o in bundle["scene"]["objects"]}
    obj = objects.get(target.get("object_id"), {})
    scene = bundle["scene"]
    prefix = set(state["frame_keys"])
    object_sightings = [s for s in bundle["sightings"]
                        if s["kind"] == "object" and s["id"] == obj.get("id") and s.get("semantic_verified")]
    current = state["current_frame"]
    seen = {s["frame"] for s in object_sightings}
    positions = target.get("positions") or trial.get("goal_positions", [])
    threshold = target.get("success_distance", .35)
    description = target.get("description") or obj.get("name")
    if positions:
        label.update(positions=positions, success_distance=threshold)
    if target.get("approach_region"):
        label.update(approach_region=target["approach_region"], scene=scene)

    def mark(sighting_id, sighting_kind, key=current, caption=None):
        match = next((s for s in bundle["sightings"] if s["frame"] == key
                      and s["id"] == sighting_id and s["kind"] == sighting_kind
                      and s.get("semantic_verified" if sighting_kind == "object" else "verified")), None)
        _require(match and match.get("pixel"), "no verified current-frame marker")
        view = bundle["frames"][key]
        with Image.open(root / view["rgb"]) as original:
            image = original.convert("RGB")
        x, y = match["pixel"]
        _require(0 <= x < image.width and 0 <= y < image.height, "marker outside source image")
        draw = ImageDraw.Draw(image)
        radius = max(5, image.width // 100)
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline="red", width=3)
        destination = root / "private" / "overlays" / (trial["id"] + "-" + kind + f"-{len(images):02}.png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination)
        explicit_caption = caption is not None
        caption = caption or "Current view with the requested location circled"
        if key != current and not explicit_caption:
            yaw = view["yaw"]
            caption = f"Current panorama view, {yaw if yaw <= 180 else yaw - 360:+g}° relative to body forward; positive is right."
        images.append({"path": destination.resolve(), "caption": caption,
                       "source": {"purpose": "prefix", "frames": [key]}})
        return match

    if kind == "nav_01" or (kind == "nav_14" and not obj):
        sighting = mark(trial["target"], "point")
        _require(frame.get("depth") and floor_patch_fraction(
            read_depth(root / frame["depth"], frame["camera"]), frame["camera"],
            frame["world_from_optical"], sighting["pixel"], state["transform"]["floor_height"]) >= .9,
            "marked point lacks a sufficiently clear floor patch")
        _require(positions and math.dist(positions[0], sighting["world"]) <= .4,
                 "reference destination does not match marked point")
        text = "Walk to the floor point circled in the marked current view."
    elif kind == "nav_02":
        opening = next((o for o in scene["openings"] if o["id"] == trial.get("opening")), None)
        _require(opening and state["room"] in opening["connects"], "opening lacks a known source room")
        mark(opening["id"], "opening")
        destination = next((r for r in opening["connects"] if r != state["room"]), None)
        _require(destination and doorway_crossed([frame["pose"]["position"], *points], scene, opening,
                                                 state["room"], destination),
                 "replay did not cross marked opening into the other room")
        label.update(scene=scene, opening=opening, from_room=state["room"], to_room=destination)
        text = "Go through the marked opening and stop on the other side."
    elif kind == "nav_03":
        snapshot = json.loads((root / state["snapshot"] / "state.json").read_text())
        motions = snapshot["environment"]["motions"]
        _require(motions and motions[-1]["blocked"] and motions[-1]["step"] == state["step"],
                 "initial cut does not end in real blocked motion")
        _require(trial.get("variant") == "safe_waypoint" and len(positions) == 1
                 and positions[0] in snapshot["environment"]["trajectory"][:-1]
                 and math.dist(frame["pose"]["position"], positions[0]) > .7,
                 "no previously visited recovery waypoint")
        reply = None
        for request, response in snapshot["tool_receipts"].values():
            if json.loads(request)[0] != "move":
                continue
            value = json.loads(response["result"]["content"][0]["text"])
            if value.get("step") == state["step"] and value.get("stop") == "blocked":
                reply = {key: value[key] for key in ("stop", "step", "done", "events")}
        _require(reply, "missing actual blocked move receipt at the cut")
        uv = world_to_uv(positions[0], state["transform"])
        label.update(variant="safe_waypoint", success_distance=.30, minimum_motion_m=.35)
        text = (f"Your last move returned {json.dumps(reply)}. Change course and walk to the previously "
                f"visited safe point P1 at map UV [{uv[0]:.4f}, {uv[1]:.4f}], then stop.")
    elif kind == "nav_04":
        room = next((r for r in scene["rooms"] if r["id"] == state["room"]), None)
        _require(room and room.get("name"), "starting room is not uniquely annotated")
        snapshot = root / state["snapshot"]
        memory = json.loads((snapshot / "workspace/memory/nodes.json").read_text())
        _require(not any(set(frames) & {current} for frames in state["node_frames"].values()),
                 "starting observation is already in a memory node")
        # Use the stored panorama anchor position, rather than the room name, to check novelty.
        current_position = frame["pose"]["position"]
        _require(all(math.dist(current_position, bundle["frames"][keys[0]]["pose"]["position"]) > .6
                     for keys in state["node_frames"].values() if keys),
                 "starting place already has a nearby node")
        replay = trial["replay"]
        nodes = replay.get("nodes", [])
        old_ids = {n["id"] for n in memory["nodes"]}
        _require(any(n["id"] not in old_ids and n["room"]
                     and n["panorama"] in replay.get("panoramas", []) for n in nodes),
                 "reference did not record a new panorama and room membership")
        alias = public_room(state, state["room"])
        label.update(room_names=[alias] if alias else [], origin_uv=frame["public_pose"]["pos"],
                     radius_uv=.35 / state["transform"]["size_m"])
        text = "Look all around, take a panorama, and create a memory node with a description and its room."
        if alias:
            text += f" Use the room label {alias!r} for this stop."
    elif kind in {"nav_05", "nav_07", "nav_09", "nav_10", "nav_12", "nav_14"}:
        _require(obj and description and positions and object_sightings, "missing grounded target instance")
        current_views = current_observation_frames(bundle, state)
        if kind in {"nav_05", "nav_14"}:
            visible_frame = trial.get("visible_frame", current)
            _require(visible_frame in seen and visible_frame in current_views and obj["room"] == state["room"],
                     "target not visibly in the starting room")
            mark(obj["id"], "object", visible_frame)
        if kind == "nav_05":
            variant = trial.get("variant") or "marked_target"
            submitted = any(c["name"] == "submit" for c in trial["replay"]["calls"])
            if variant == "marked_target":
                _require(submitted, "positive reference did not submit")
                text = "Walk up to the object circled in the supplied current observation and submit when you reach it."
                label["variant"] = variant
            else:
                pair = next((p for p in _reference_pairs(bundle, state, obj["id"], visible_frame)
                             if p["review_candidate"] == trial.get("comparison_id")), None)
                _require(pair is not None and variant in {"reference_match", "wrong_instance"},
                         "missing approved reference comparison")
                matches = variant == "reference_match"
                _require(submitted == matches, "replay submission does not match the comparison")
                reference = pair["matching_frame"] if matches else pair["reference_frames"][0]
                instance = obj["id"] if matches else pair["requested_object"]
                mark(instance, "object", reference, "Reference image with the comparison object circled.")
                text = REFERENCE_MATCH_INSTRUCTION
                label.update(variant="reference_match", matches=matches)
        else:
            text = f"Find the {description} and walk up to it."
        if kind == "nav_07":
            _require(_known_once(bundle, state, obj["id"], seen)
                     and _reviewed_absent(bundle, obj["id"], current_views)
                     and state["room"] != obj["room"],
                     "target not confined to one historical node in a different room")
        if kind == "nav_09":
            _require(any(set(keys) & seen & prefix for keys in state["node_frames"].values()),
                     "target lacks a verified sighting in the indexed memory prefix")
            known = set(state["visited_rooms"]) - {None}
            source, destination = state["room"], obj["room"]
            _require(source in known and _intervening_room(state, source, destination),
                     "target is not separated by an intervening room in the known route")
            label.update(scene=scene, known_rooms=sorted(known), room_aliases={
                identifier: public_room(state, identifier) for identifier in known
                if public_room(state, identifier)})
        if kind == "nav_10":
            _require(_reviewed_absent(bundle, obj["id"], state["frame_keys"])
                     and not (seen & prefix) and obj["room"] not in state["visited_rooms"],
                     "target or its room is already known in the starting prefix")
        if kind == "nav_12":
            repair = trial.get("repair")
            _require(repair and state.get("altered_memory"), "no isolated corrupted snapshot and repair evidence")
            _require(repair.get("review_candidate") and repair.get("evidence_frame") in prefix
                     and repair["evidence_frame"] in state["node_frames"].get(repair["node"], []),
                     "no approved repair evidence in the public node prefix")
            snapshot = root / state["snapshot"]
            memory = json.loads((snapshot / "workspace/memory/nodes.json").read_text())
            before = next((n for n in memory["nodes"] if n["id"] == repair["node"]), {})
            after = next((n for n in trial["replay"].get("nodes", []) if n["id"] == repair["node"]), {})
            _require(repair["wrong"] != repair["correct"] and before.get(repair["field"]) == repair["wrong"]
                     and after.get(repair["field"]) == repair["correct"], "replay did not repair the corrupted fact")
            label["repair"] = repair
            text += " Correct any memory entry that you can verify is wrong along the way."
    else:
        official = next((o for o in bundle.get("official_tasks", []) if o.get("id", o["episode"]["id"])
                         == trial.get("official_task") and o["state_id"] == state["id"]), None)
        _require(official and bundle["split"] == "train", "no matching official train episode")
        episode = official["episode"]
        goals = episode.get("goals", [])
        _require(0 <= official["goal_index"] < len(goals), "missing official goal")
        goal = goals[official["goal_index"]]
        label["source_goal_index"] = official["source_goal_index"]
        if official.get("reference_frame"):
            key = official["reference_frame"]
            images.append({"path": bundle["frames"][key]["rgb"], "caption": "Official RGB target reference",
                           "source": {"purpose": "target_reference", "frames": [key]}})
        aligned = official.get("alignment")
        if kind == "nav_16" and aligned and aligned.get("source") and aligned.get("positions"):
            text = aligned["instruction"]
            label.update(variant="aligned_segment", positions=aligned["positions"],
                         success_distance=aligned.get("success_distance", .35))
        else:
            text = goal["instruction"]
            shortest = official.get("shortest")
            _require(type(shortest) in (int, float) and 0 < shortest <= 8 and len(text.split()) <= 75,
                     "official instruction or route is not verified short")
            label.update(variant="short_complete_official", positions=goal["positions"],
                         benchmark=episode["benchmark"], shortest=shortest,
                         success_distance=official["threshold"])
            _require(any(c["name"] == "submit" for c in trial["replay"]["calls"]),
                     "official reference did not submit")
        label["_episode"] = episode

    if kind not in {"nav_02", "nav_04"}:
        positions = label.get("positions", [])
        _require(points and positions, "replay lacks motion or labelled goals")
        _require_goal_measurements(trial, label, points[-1])
        if kind not in {"nav_16", "nav_18"}:
            _require(min(math.dist(frame["pose"]["position"], p) for p in positions)
                     >= label["success_distance"], "starting pose is already in the target area")
    return text, label, images


def build_navigation(bundle, root, config):
    root = Path(root)
    states = {s["id"]: s for s in bundle["states"]}
    tasks = []
    comparisons = {}
    for kind in NAVIGATION_KINDS:
        candidates = [t for t in bundle["trials"] if t["kind"] == kind or kind in t.get("kinds", [])]
        if not candidates:
            reject(root, kind, bundle["scene_id"], "no eligible trial was collected")
        for trial in candidates:
            try:
                state = states[trial["start_state"]]
                frame = bundle["frames"][state["current_frame"]]
                points = _verified_trial(trial, root, state)
                text, label, images = _instruction(kind, trial, state, frame, bundle, points, root)
                identifier = f'{kind}-{trial["id"]}'
                episode_path = root / "private" / "episodes" / (identifier + ".json")
                write_json(episode_path, label.pop("_episode", bundle["episode"]))
                env_path = root / "private" / "environment" / (identifier + ".json")
                write_json(env_path, bundle["environment_spec"])
                case = root / "cases" / identifier
                task = {"id": identifier, "kind": kind, "mode": "navigation", "instruction": text,
                        "snapshot": os.path.relpath(root / state["snapshot"], case),
                        "episode": os.path.relpath(episode_path, case),
                        "environment": os.path.relpath(env_path, case),
                        "limits": config.get("navigation_limits", {"max_requests": 20, "max_actions": 90,
                                                                    "max_observations": 20, "seconds": 240}),
                        "provenance": {"scene_id": bundle["scene_id"], "state_id": state["id"],
                                       "source_kind": "replayed_trial"}}
                usage = _reference_budget(trial, points, frame["pose"]["position"], task["limits"])
                entry = (root, task, label, {"trial": trial, "reference_usage": usage,
                    "physical_replay": "passed", "semantic_review": "needs_review",
                    "image_sources": [image["source"] for image in images]}, images)
                if label.get("variant") == "reference_match":
                    key = (state["id"], trial["visible_frame"], trial["comparison_id"])
                    comparisons.setdefault(key, {})[label["matches"]] = entry
                else:
                    tasks.append(freeze_case(*entry))
            except (KeyError, ValueError, FileNotFoundError) as error:
                reject(root, kind, trial["id"], str(error))
    for key, variants in comparisons.items():
        if set(variants) != {True, False}:
            reject(root, "nav_05", "/".join(key), "missing independently replayed comparison counterpart")
            continue
        tasks.extend(freeze_case(*variants[matches]) for matches in (True, False))
    return tasks


_OBJECT_KINDS = ("nav_05", "nav_07", "nav_09", "nav_10", "nav_12")


REFERENCE_MATCH_INSTRUCTION = (
    "Walk up to the circled object in the current observation and inspect it. Compare it with the circled "
    "object in the reference image. If they are the same physical instance, submit when you reach it. "
    "If they differ, do not submit; answer “mismatch”."
)


def _reference_pairs(bundle, state, instance, visible):
    from ..judgment.targets import different_view

    sightings = {(s["id"], s["frame"]): s for s in bundle["sightings"]
                 if s["kind"] == "object" and s.get("semantic_verified") and s.get("pixel")}
    if not sightings.get((instance, visible), {}).get("identity_verified"):
        return
    frames = bundle["frames"]
    prefix = set(state["frame_keys"])
    for pair in bundle.get("navigation_evidence", {}).get("distinct_instances", []):
        refs = pair.get("reference_frames", [])
        other = pair.get("requested_object")
        if (not pair.get("review_candidate") or pair.get("same_category") is not True
                or pair.get("observed_object") != instance or other == instance
                or pair.get("current_frame") != visible or len(refs) != 1
                or refs[0] not in prefix or (other, refs[0]) not in sightings):
            continue
        for key in state["frame_keys"]:
            if (sightings.get((instance, key), {}).get("identity_verified")
                    and different_view(frames[visible], frames[key])):
                yield {**pair, "matching_frame": key}
                break


def _object_candidates(bundle, kind):
    """Keep frame-level presence and verified absence distinct from missing annotations."""
    reviews = bundle.get("navigation_evidence", {})
    reviewed = {(r["object"], r["frame"]): r for r in reviews.get("visibility", [])}
    reviewed.update({(s["id"], s["frame"]): {"object": s["id"], "frame": s["frame"], "visible": True}
                     for s in bundle["sightings"] if s["kind"] == "object" and s.get("semantic_verified")})
    for obj in bundle["scene"]["objects"]:
        present = {frame for (identifier, frame), row in reviewed.items()
                   if identifier == obj["id"] and row["visible"]}
        if not present:
            continue
        for state in bundle["states"]:
            if state.get("altered_memory") or not state["room"]:
                continue
            prefix = set(state["frame_keys"])
            current_views = current_observation_frames(bundle, state)
            visible = next((key for key in current_views if key in present), None)
            absent = bool(current_views) and all(
                reviewed.get((obj["id"], key), {}).get("visible") is False for key in current_views)
            same = state["room"] == obj["room"]
            correction = None
            if kind == "nav_05" and not (visible and same):
                continue
            if kind == "nav_07" and (not absent or same or not _known_once(bundle, state, obj["id"], present)):
                continue
            if kind == "nav_09":
                if (same or not any(set(keys) & present & prefix for keys in state["node_frames"].values())
                        or not _intervening_room(state, state["room"], obj["room"])):
                    continue
            if kind == "nav_10" and (not _reviewed_absent(bundle, obj["id"], state["frame_keys"])
                                    or present & prefix or obj["room"] in state["visited_rooms"]):
                continue
            if kind == "nav_12":
                correction = next((r for r in reviews.get("corrections", [])
                                   if r.get("review_candidate") and r.get("state") == state["id"]
                                   and r.get("target") == obj["id"]
                                   and r.get("field") in {"room", "description"}
                                   and r.get("evidence_frame") in prefix
                                   and r.get("node") in state["node_frames"]
                                   and r.get("wrong") and r.get("correct")
                                   and r["wrong"] != r["correct"]), None)
                if not correction:
                    continue
            candidate = {"state": state["id"], "object": obj["id"], "reference_frames": sorted(present),
                         "repair": correction, "visible_frame": visible}
            yield candidate
            if kind == "nav_05":
                for pair in _reference_pairs(bundle, state, obj["id"], visible):
                    for variant in ("reference_match", "wrong_instance"):
                        yield {**candidate, "variant": variant, "comparison_id": pair["review_candidate"]}
                    break


def _intervening_room(state, source, destination):
    known = set(state["visited_rooms"]) - {None}
    links = [{e["from_room"], e["to_room"]} for e in state["traversals"]
             if {e["from_room"], e["to_room"]} <= known]
    if source == destination or any({source, destination} <= edge for edge in links):
        return False
    reached = {source}
    for _ in links:
        reached |= set().union(*(edge for edge in links if edge & reached))
    return destination in reached


def _altered_cut(bundle, root, state, correction):
    memory_path = "workspace/memory/nodes.json"
    source = root / state["snapshot"]
    memory = json.loads((source / memory_path).read_text())
    node = next((n for n in memory["nodes"] if n["id"] == correction["node"]), {})
    field = correction["field"]
    _require(node.get(field) == correction["correct"], "review does not match source node")
    node[field] = correction["wrong"]
    identifier = state["id"] + "-altered-" + correction["node"]
    relative = Path("states") / identifier / "snapshot"
    destination = root / relative
    if not destination.exists():
        shutil.copytree(source, destination)
        write_json(destination / memory_path, memory)
    else:
        _require(json.loads((destination / memory_path).read_text()) == memory,
                 "altered cut already exists with a different correction")
    altered = {**state, "id": identifier, "snapshot": str(relative), "altered_memory": True}
    if not any(s["id"] == identifier for s in bundle["states"]):
        bundle["states"].append(altered)
    return altered


def approach_positions(environment, scene, obj):
    """Sample the reachable part of the footprint's same-floor, same-room vicinity."""
    level = next((row for row in scene["levels"] if row["id"] == obj["level"]), None)
    _require(level and obj.get("rectangles_xz"), "object lacks a floor or footprint")
    region = {"room": obj["room"], "floor_height": level["floor_height"], "floor_tolerance": .25,
              "room_margin": .04, "rectangles": obj["rectangles_xz"], "max_footprint_distance": .85,
              "spacing": .25, "max_snap": .15}
    points = {}
    for rectangle in region["rectangles"]:
        x, z = rectangle["center_xz"]
        w, d = rectangle["size"]
        radius = math.hypot(w, d) / 2 + region["max_footprint_distance"]
        n = math.ceil(radius / region["spacing"])
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                query = [x + i * region["spacing"], region["floor_height"], z + j * region["spacing"]]
                if not in_approach_region(scene, query, region):
                    continue
                point = environment.snap_point(query)
                if (point is None or math.dist(query, point) > region["max_snap"]
                        or not in_approach_region(scene, point, region)):
                    continue
                key = tuple(round(value, 3) for value in point)
                if key not in points and math.isfinite(distance := environment.distance([point])):
                    points[key] = (distance, point)
    return [point for _, point in sorted(points.values())], region


def prepare_navigation(bundle, root, config, gpu, kinds=()):
    """Offline proposal and physical replay for object-dependent navigation families.

    The reviewed bundle supplies visibility, distinguishable instances and
    verifiable corrections. Missing reviews remain candidate rejections.
    """
    from src.dataset.episodes import EpisodeSpec
    from ..sampling.collection import _branch, _call, _follow

    root = Path(root)
    if type(gpu) is not int or gpu not in (0, 1):
        raise ValueError("Subset navigation requires physical GPU 0 or 1")
    objects = {obj["id"]: obj for obj in bundle["scene"]["objects"]}
    states = {state["id"]: state for state in bundle["states"]}
    selected = [kind for kind in _OBJECT_KINDS if not kinds or kind in kinds]
    episode = EpisodeSpec(**bundle["episode"])
    limit = config.get("navigation_candidates_per_kind", 1)
    env_config = {**bundle["environment_spec"], "gpu": gpu, "depth": True}
    for kind in selected:
        candidates = list(_object_candidates(bundle, kind))
        for candidate in candidates:
            digest = hashlib.sha256(json.dumps(candidate, sort_keys=True).encode()).hexdigest()[:12]
            candidate["id"] = f"{kind}-{candidate['state']}-{digest}"
        write_json(root / "private/navigation_candidates" / f"{kind}.json", candidates)
        if not candidates:
            reject(root, kind, bundle["scene_id"], "no candidate with the required independent evidence",
                   stage="proposal")
            continue
        chosen = candidates[:limit]
        if kind == "nav_05":
            pairs = {c["comparison_id"] for c in candidates if c.get("variant") == "reference_match"}
            pairs = sorted(pairs)[:limit]
            chosen = ([c for c in candidates if not c.get("variant")][:limit]
                      + [c for c in candidates if c.get("comparison_id") in pairs])
        for candidate in chosen:
            identifier = candidate["id"]
            if any(t.get("candidate") == identifier for t in bundle["trials"]):
                continue
            state = states[candidate["state"]]
            obj = objects[candidate["object"]]
            repair = candidate["repair"]
            try:
                if repair:
                    state = _altered_cut(bundle, root, state, repair)

                def operation(executor, calls):
                    environment = executor.environment
                    def call(name, args):
                        result = _call(executor, name, args)
                        calls.append({"name": name, "arguments": args})
                        return result
                    threshold = .32
                    if repair:
                        source = bundle["frames"][repair["evidence_frame"]]
                        if math.dist(environment.position(), source["pose"]["position"]) > .35:
                            _require(_follow(executor, source["pose"]["position"], calls, radius=.3),
                                     "could not reach correction evidence")
                        call("observe", {"mode": "panorama"})
                        call("memory_edit", {"operations": [{"op": "update", "id": repair["node"],
                                                             repair["field"]: repair["correct"]}]})
                    positions, region = approach_positions(environment, bundle["scene"], obj)
                    _require(positions, "no reachable same-room approach region")
                    interior = {**region, "max_footprint_distance": region["max_footprint_distance"] - .25,
                                "room_margin": region["room_margin"] + .20}
                    goal = next((point for point in positions if in_approach_region(bundle["scene"], point, interior)),
                                positions[0])
                    _require(environment.distance(positions) >= .4, "already inside the approach area")
                    reached = _follow(executor, goal, calls,
                                      budget=config.get("collection", {}).get("route_actions", 42), radius=.25)
                    reached = reached and in_approach_region(bundle["scene"], environment.position(), region)
                    if reached and kind == "nav_05" and candidate.get("variant") != "wrong_instance":
                        call("submit", {})
                    return {"positions": positions, "reference_goal": goal, "approach_region": region,
                            "success_distance": threshold, "reached": reached}

                trial = _branch(bundle, root, env_config, episode, state, kind, operation,
                                candidate=identifier, target=identifier, repair=repair,
                                variant=candidate.get("variant"), comparison_id=candidate.get("comparison_id"),
                                reference_frames=candidate["reference_frames"], visible_frame=candidate["visible_frame"])
                outcome = trial["reference"]
                if kind == "nav_09":
                    route_rooms = {room_at(bundle["scene"], point) for point in _samples(trial["replay"])}
                    _require(len(route_rooms & (set(state["visited_rooms"]) - {None})) >= 3,
                             "replayed route did not traverse an intervening known room")
                bundle["targets"].append({"id": identifier, "object_id": obj["id"],
                    "description": obj["name"], "positions": outcome["positions"],
                    "success_distance": outcome["success_distance"], "approach_region": outcome["approach_region"]})
            except ValueError as error:
                reject(root, kind, identifier, str(error), stage="reference")
    return bundle


def navigation_review_proposals(bundle, root):
    """Propose targeted RGB reviews; the central reviewer binds and accepts them."""
    root = Path(root)
    objects = {obj["id"]: obj for obj in bundle["scene"]["objects"]}
    observed = {}
    absence_proposals = set()
    for sight in bundle["sightings"]:
        if sight["kind"] == "object" and sight.get("pixel"):
            observed.setdefault(sight["id"], {})[sight["frame"]] = sight

    def proposal(kind, name, frames, prompt, check, key, value, annotations, markers=None):
        return {"kind": kind, "name": name, "frames": frames, "prompt": prompt,
                "check_keys": [check] if isinstance(check, str) else check,
                "annotations": annotations, "markers": markers,
                "fact": {"section": "navigation_evidence", "key": key, "value": value}}

    for state in bundle["states"]:
        current = state["current_frame"]
        current_views = current_observation_frames(bundle, state)
        prefix = set(state["frame_keys"])
        if state.get("altered_memory") or not state.get("room"):
            continue
        for identifier, sightings in observed.items():
            obj = objects.get(identifier)
            if not obj:
                continue
            name = obj["name"]
            remembered = sorted(set(sightings) & prefix)
            if sightings:
                reference = sorted(sightings)[0]
                for view in state["frame_keys"]:
                    binding = (identifier, view, reference)
                    if view in sightings or binding in absence_proposals:
                        continue
                    absence_proposals.add(binding)
                    yield proposal(
                        "navigation_absence", name, [view, reference],
                        "Image 2 marks the proposed target. Is that physical instance absent from image 1? "
                        "If the reference is unclear or the target could be partly visible, return uncertain.",
                        "target_absent", "visibility", {"object": identifier, "frame": view, "visible": False},
                        [obj], [None, sightings[reference]["pixel"]])
            visible = next((view for view in current_views if view in sightings), None)
            if visible and state["room"] == obj["room"]:
                for other_id, other_sightings in observed.items():
                    other = objects.get(other_id)
                    refs = sorted((set(other_sightings) & prefix) - {visible})
                    if (not other or other_id == identifier or not refs or
                            (obj.get("category") and other.get("category") and
                             obj["category"] != other["category"])):
                        continue
                    reference = refs[0]
                    yield proposal(
                        "navigation_instance_pair", name, [visible, reference],
                        "Are both marked objects of the same category, and are they clearly different physical instances? "
                        "Require visible distinguishing details; a changed viewpoint alone is insufficient.",
                        ["same_category", "visibly_distinct"], "distinct_instances",
                        {"observed_object": identifier, "requested_object": other_id,
                         "current_frame": visible, "reference_frames": [reference], "same_category": True}, [obj, other],
                        [sightings[visible]["pixel"], other_sightings[reference]["pixel"]])
        memory_file = root / state["snapshot"] / "workspace/memory/nodes.json"
        if not memory_file.is_file():
            continue
        nodes = json.loads(memory_file.read_text())["nodes"]
        for node in nodes:
            room = node.get("room")
            frames = [k for k in state["node_frames"].get(node["id"], []) if k in prefix]
            if not room or len(frames) != 6:
                continue
            known_rooms = {public_room(state, row["id"]): row for row in bundle["scene"]["rooms"]
                           if public_room(state, row["id"])}
            alternate = next((n["room"] for n in nodes if n.get("room") and n["room"] != room
                              and n["room"] in known_rooms), None)
            if room not in known_rooms or not alternate:
                continue
            comparison = _correction_comparison(bundle, state, nodes, node, alternate)
            if comparison is None:
                continue
            comparison_frames, comparison_prompt = comparison
            room_row = known_rooms[room]
            for identifier, sightings in observed.items():
                target = objects.get(identifier)
                if not target or target["room"] != room_row["id"]:
                    continue
                evidence = next((key for key in frames if key in sightings), None)
                if not evidence:
                    continue
                yield proposal(
                    "navigation_correction", node["id"], comparison_frames,
                    comparison_prompt,
                    "room_correction", "corrections",
                    {"state": state["id"], "target": identifier, "node": node["id"], "field": "room",
                     "wrong": alternate, "correct": room, "evidence_frame": evidence,
                     "comparison_frames": comparison_frames},
                    [room_row, known_rooms[alternate], target, node])


def _correction_comparison(bundle, state, nodes, node, alternate):
    """Bind aliases to other observed nodes, not to private room names."""
    prefix = set(state["frame_keys"])
    frames = bundle["frames"]

    def panorama(row):
        keys = state["node_frames"].get(row["id"], [])
        if (len(keys) != 6 or not set(keys) <= prefix
                or {frames[k].get("yaw") for k in keys} != set(range(0, 360, 60))
                or len({frames[k].get("group") for k in keys}) != 1):
            return None
        return sorted(keys, key=lambda k: frames[k]["yaw"])

    own = panorama(node)
    same = next((n for n in nodes if n["id"] != node["id"] and n.get("room") == node["room"]
                 and panorama(n)), None)
    other = next((n for n in nodes if n.get("room") == alternate and panorama(n)), None)
    if own is None or same is None or other is None:
        return None
    keys, descriptions = [], []
    for row, caption in ((node, "Unlabelled panorama under review"),
                         (same, f"Reference for public alias {node['room']!r}"),
                         (other, f"Reference for public alias {alternate!r}")):
        local = panorama(row)
        start = len(keys) + 1
        keys.extend(local)
        pose = frames[local[0]]["public_pose"]
        descriptions.append(f"Images {start}–{len(keys)}: {caption}; node {row['id']}; "
                            f"capture pose {json.dumps(pose)}; yaw 0,60,120,180,240,300.")
    prompt = " ".join(descriptions) + (
        f" Does the unlabelled panorama belong with {node['room']!r}, rather than {alternate!r}? "
        "Require visual continuity or shared landmarks with the independent reference panoramas. "
        "Positions alone and arbitrary label names are not enough. Return uncertain if ambiguous.")
    return keys, prompt
