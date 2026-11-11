"""Score private navigation events independently of tool success messages."""
import math
import json

from src.bench.metrics import score_goal
from ..sampling.geometry import crossed_segment
from ..sampling.scene import room_at


def _samples(motions):
    return [s["position"] for m in motions for s in m["samples"]]


def doorway_crossed(points, scene, opening, source, destination):
    """Check that real motion through the finite opening leads into its other room."""
    level = next((l for l in scene["levels"] if l["id"] == opening["level"]), None)
    if not level:
        return False
    previous_room = None
    crossed = False
    for index, point in enumerate(points):
        if index and all(abs(p[1] - level["floor_height"]) <= .45
                                   for p in (points[index - 1], point)) and crossed_segment(
                                       points[index - 1], point, opening["segment_xz"]):
            crossed = previous_room == source
        room = room_at(scene, point)
        if room == destination and crossed:
            return True
        if room is not None and room != previous_room:
            previous_room = room
            crossed = False
    return False


def footprint_distance(point, rectangles):
    distances = []
    for rectangle in rectangles:
        x, z = rectangle["center_xz"]
        w, d = rectangle["size"]
        a = math.radians(rectangle["angle_deg"])
        dx, dz = point[0] - x, point[2] - z
        u = math.cos(a) * dx + math.sin(a) * dz
        v = -math.sin(a) * dx + math.cos(a) * dz
        distances.append(math.hypot(max(0., abs(u) - w / 2), max(0., abs(v) - d / 2)))
    return min(distances, default=math.inf)


def in_approach_region(scene, position, region):
    return (abs(position[1] - region["floor_height"]) <= region["floor_tolerance"]
            and room_at(scene, position, margin=region["room_margin"]) == region["room"]
            and footprint_distance(position, region["rectangles"]) <= region["max_footprint_distance"])


def arrived_at_goal(label, position, distance):
    return (distance is not None and distance < label.get("success_distance", .35)
            and (not label.get("approach_region")
                 or in_approach_region(label["scene"], position, label["approach_region"])))


def _result_value(result, key, default):
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def score_navigation(environment, executor, label, initial, result=None):
    kind = label["kind"]
    steps = environment.steps - initial["steps"]
    length = environment.path_length - initial["path_length"]
    motions = [m for m in environment.motions if m["step"] > initial["steps"]]
    points = [initial["position"], *_samples(motions)]
    submitted = bool(_result_value(result, "submitted", executor.request.submitted))
    answer = _result_value(result, "answer", "") or ""
    details = {"actions": steps, "path_length": length, "submitted": submitted,
               "semantic": "not_required"}
    distance = environment.distance(label["positions"]) if label.get("positions") else math.inf
    arrived = arrived_at_goal(label, environment.position(), distance)
    if math.isfinite(distance):
        details["distance"] = distance
    event = arrived
    if kind == "nav_02":
        event = doorway_crossed(points, label["scene"], label["opening"],
                                label["from_room"], label["to_room"]) and (
                                    room_at(label["scene"], environment.position()) == label["to_room"])
    elif kind == "nav_03":
        blocked = next((m for m in reversed(environment.motions)
                        if m["step"] == initial["steps"]), None)
        event = (bool(blocked and blocked["blocked"]) and arrived
                 and length >= label.get("minimum_motion_m", .35))
    elif kind == "nav_04":
        existing = set(initial["nodes"])
        nodes = [n for n in executor.memory.nodes if n["id"] not in existing
                 and n["room"] and (not label["room_names"] or n["room"] in label["room_names"])
                 and n["description"].strip()]
        valid = []
        for node in nodes:
            panorama = executor.observations.panorama(node["panorama"])
            if (node["panorama"] not in initial.get("panoramas", ()) and len(panorama["frames"]) == 6
                    and math.dist(panorama["pose"]["pos"], label["origin_uv"]) <= label["radius_uv"]):
                valid.append(node["id"])
        event = bool(valid)
        details.update(new_nodes=valid, semantic="needs_review")
    elif kind == "nav_05":
        if label.get("variant") == "reference_match" and label.get("matches") is False:
            event = (not submitted and length >= .1 and arrived
                     and answer.strip().casefold().rstrip(".。") in {"mismatch", "不匹配"})
        else:
            event = submitted and arrived
    elif kind == "nav_09":
        rooms = {room_at(label["scene"], point) for point in points} - {None}
        details["rooms"] = sorted(rooms)
        event = arrived
    elif kind == "nav_12":
        repair = label["repair"]
        original = initial["memory"].get("nodes", [])
        before = next((n for n in original if n["id"] == repair["node"]), {})
        after = next((n for n in executor.memory.nodes if n["id"] == repair["node"]), {})
        event = (arrived and before.get(repair["field"]) == repair["wrong"]
                 and after.get(repair["field"]) == repair["correct"])
    elif kind in {"nav_16", "nav_18"} and label.get("benchmark"):
        if math.isfinite(distance):
            metrics = score_goal(label["benchmark"], submitted=submitted, distance=distance,
                                 shortest=label["shortest"], path_length=length, steps=steps)
            details["official_metrics"] = metrics
            event = bool(metrics["official_success"])
        else:
            event = False
    if kind in {"nav_07", "nav_14"}:
        calls = set(executor._calls) - set(initial.get("receipt_ids", ()))
        names = [json.loads(executor._calls[c][0])[0] for c in calls]
        details["memory_queries"] = names.count("memory_read")
    details["event"] = bool(event)
    pending = event and details["semantic"] == "needs_review"
    return {"success": None if pending else bool(event),
            "status": "needs_review" if pending else "correct" if event else "incorrect",
            "details": details}
