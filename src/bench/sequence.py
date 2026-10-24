"""Run a frozen continuous task chain in one environment and Pi session."""

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import traceback

from src.common.io import redact, write_json
from src.dataset.episodes import EpisodeSpec
from src.environment import Environment
from src.harness.session import PromptLimits, open_session

from .metrics import spl
from .runner import INFRASTRUCTURE_REASONS, validate_single_floor
from .samples import read_config


def inside(point, polygon):
    x, y = point
    hit = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if (a[1] > y) != (b[1] > y) and x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]:
            hit = not hit
    return hit


def _room_side(point, normal, polygon):
    # Archived room polygons use X,-Z; portal segments and motion use X,Z.
    for distance in (.0001, .001, .01, .05, .1):
        sides = [inside([point[0] + sign * distance * normal[0],
                         -(point[1] + sign * distance * normal[1])], polygon)
                 for sign in (-1, 1)]
        if all(sides):
            return 0
        if any(sides):
            return 1 if sides[1] else -1
    return 0


def crosses(before, after, event, rooms):
    """Require the designated opening and direction, never proximity to its centre."""
    a, b = event["segment_xz"]
    p, q = [before[0], before[2]], [after[0], after[2]]
    vx, vz = b[0] - a[0], b[1] - a[1]
    length = math.hypot(vx, vz)
    if length < 1e-8:
        return False
    ps = vx * (p[1] - a[1]) - vz * (p[0] - a[0])
    qs = vx * (q[1] - a[1]) - vz * (q[0] - a[0])
    if abs(ps - qs) < 1e-8 or ps * qs > 1e-8:
        return False
    t = ps / (ps - qs)
    if not 0 <= t <= 1:
        return False
    intersection = [p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]
    u = ((intersection[0] - a[0]) * vx + (intersection[1] - a[1]) * vz) / length**2
    if not -1e-6 <= u <= 1 + 1e-6:
        return False
    u = min(1 - 1e-4, max(1e-4, u))
    local = [a[0] + u * vx, a[1] + u * vz]
    normal = [-vz / length, vx / length]
    destination = _room_side(local, normal, rooms[event["to_room"]]["polygon"])
    if not destination:
        return False
    if event.get("from_room") and _room_side(local, normal, rooms[event["from_room"]]["polygon"]) != -destination:
        return False
    return (qs - ps) * destination > 1e-8


def _segment_distance(point, a, b):
    delta = [y - x for x, y in zip(a, b)]
    squared = sum(x * x for x in delta)
    t = min(1., max(0., sum((p - x) * d for p, x, d in zip(point, a, delta)) / squared)) if squared else 0.
    return math.dist(point, [x + t * d for x, d in zip(a, delta)])


def score_route(specification, trajectory, rooms):
    cursor, rows = 1, []
    for event in specification.get("route_events", []):
        found = None
        for index in range(cursor, len(trajectory)):
            before, after = trajectory[index - 1:index + 1]
            matched = (crosses(before, after, event, rooms) if event["kind"] == "portal"
                       else any(_segment_distance(point, before, after) < event["radius"]
                                for point in event["positions"]))
            if matched:
                found = index
                break
        rows.append({"id": event["id"], "kind": event["kind"], "passed": found is not None,
                     "trajectory_index": found})
        if found is None:
            rows.extend({"id": item["id"], "kind": item["kind"], "passed": False,
                         "trajectory_index": None} for item in specification["route_events"][len(rows):])
            break
        cursor = found + 1
    return {"passed": all(row["passed"] for row in rows), "events": rows}


def _memory_report(session, before, call_start, trace_start):
    after = {entity["id"]: entity for entity in session.memory.entities}
    evidence = [(entity["id"], frame) for entity in after.values() for frame in entity.get("evidence", [])]
    invalid = [(identifier, frame) for identifier, frame in evidence
               if frame not in session.observations.items or not
               (session.workspace / session.observations.items[frame]["path"]).is_file()]
    calls = session.trace.calls[call_start:]
    operations = Counter({name: 0 for name in ("create", "update", "delete", "merge")})
    trace_path = session.private / "trace.jsonl"
    if trace_path.exists():
        with trace_path.open("rb") as stream:
            stream.seek(trace_start)
            for line in stream:
                event = json.loads(line)
                data = event.get("data", {})
                if (event["type"] == "public_tool_reply" and data.get("name") == "memory_edit"
                        and not data["result"].get("isError")):
                    operations.update(item["op"] for item in data["args"]["operations"])
    return {"entities": len(after), "types": dict(Counter(item["type"] for item in after.values())),
            "operations": dict(operations), "operation_basis": "Accepted memory_edit batches only.",
            "entity_delta": {"created": sorted(after.keys() - before.keys()),
                             "deleted": sorted(before.keys() - after.keys()),
                             "revised": sorted(identifier for identifier in after.keys() & before.keys()
                                               if after[identifier] != before[identifier])},
            "evidence_references": len(evidence), "valid_evidence_references": len(evidence) - len(invalid),
            "invalid_evidence_references": invalid,
            "evidence_check": "Known frame and existing public RGB only; not semantic correctness.",
            "tools": dict(Counter(item["name"] for item in calls)),
            "tool_errors": sum(bool(item.get("isError")) for item in calls)}


def _validate(scenario):
    rounds = scenario["rounds"]
    if not rounds or len({row["id"] for row in rounds}) != len(rounds):
        raise ValueError("A sequence requires nonempty rounds with unique IDs")
    for row in rounds:
        if row["kind"] not in {"explore", "navigation"} or not row["instruction"].strip():
            raise ValueError("Each round needs an explore/navigation kind and instruction")
        PromptLimits(row["max_requests"], row["max_actions"], row["seconds"])
        if row["kind"] == "navigation" and not row.get("positions"):
            raise ValueError("Navigation rounds need frozen accepted positions")
        if any(event["kind"] not in {"portal", "waypoint"} for event in row.get("route_events", [])):
            raise ValueError("Unknown route event kind")
    if rounds[0].get("teleport"):
        raise ValueError("The first round uses the scenario start pose")


def _relocation_preflight(environment, rounds):
    rows, floor = [], environment.position()[1]
    for specification in rounds:
        if not specification.get("teleport"):
            continue
        pose = specification["teleport"]
        position, rotation = pose["position"], pose["rotation_xyzw"]
        finite = (len(position) == 3 and len(rotation) == 4 and
                  all(type(value) in (int, float) and math.isfinite(value) for value in [*position, *rotation]))
        valid = finite and math.isclose(sum(value * value for value in rotation), 1, abs_tol=1e-3)
        snapped = environment.snap_point(position) if valid else None
        drift = math.dist(position, snapped) if snapped is not None else None
        local = [point for point in specification.get("positions", []) if abs(point[1] - floor) <= .5]
        path = environment.shortest_path(local, snapped) if snapped is not None and local else []
        span = max(point[1] for point in path) - min(point[1] for point in path) if path else None
        rows.append({"id": specification["id"], "position": position, "snapped": snapped,
                     "snap_drift_m": drift, "max_snap_drift_m": .1,
                     "path_height_span_m": span, "frozen_pose_changed": False,
                     "passed": bool(valid and abs(position[1] - floor) <= .5 and drift is not None
                                    and drift <= .1 and path and span <= .5)})
    return rows


def run_sequence(repo, model_config, scenario_path, scene_map, directory, environment_config):
    repo, scenario_path, directory = Path(repo).resolve(), Path(scenario_path).resolve(), Path(directory).resolve()
    scenario = json.loads(scenario_path.read_text())
    _validate(scenario)
    if type(environment_config.get("gpu")) is not int or environment_config["gpu"] not in (0, 1):
        raise ValueError("Sequence simulation requires an explicit gpu of 0 or 1")
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "private/scenario.json", scenario)
    planned = sum(row["kind"] == "navigation" for row in scenario["rounds"])
    result = {"episode_id": "sequence:" + Path(scenario["scene_id"]).name.split(".")[0],
              "benchmark": "sequence", "protocol": scenario.get("protocol"),
              "model": model_config["model"], "reasoning_effort": model_config["reasoning_effort"],
              "tool_tier": model_config.get("tool_tier", "basic"), "gpu": environment_config["gpu"],
              "scenario_sha256": hashlib.sha256(scenario_path.read_bytes()).hexdigest(),
              "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "planned_rounds": len(scenario["rounds"]), "planned_navigation": planned, "rounds": []}
    environment = session = None

    def save():
        navigation = [row for row in result["rounds"] if row["kind"] == "navigation" and row["status"] == "completed"]
        successes = sum(row["success"] for row in navigation)
        result["completed_rounds"] = len(result["rounds"])
        result["metrics"] = {"navigation_completed": len(navigation), "navigation_planned": planned,
                             "navigation_successes": successes,
                             "observed_success_rate": successes / len(navigation) if navigation else None,
                             "success_rate": successes / planned if result["status"] == "completed" and planned else None}
        if environment is not None:
            result["environment"] = {"steps": environment.steps, "path_length": environment.path_length,
                                     "trajectory": environment.trajectory,
                                     "trajectory_breaks": environment.trajectory_breaks}
        write_json(directory / "result.json", result)

    save()
    try:
        goals = [{"modality": "instruction", "instruction": row["instruction"], "positions": row["positions"]}
                 for row in scenario["rounds"] if row["kind"] == "navigation"]
        episode = EpisodeSpec(result["episode_id"], result["episode_id"], "sequence", scenario["scene_id"],
                              scenario["start_position"], scenario["start_rotation"], goals, str(scenario_path),
                              provenance="custom_continuous_campaign")
        environment = Environment(episode, {**environment_config, "tool_tier": result["tool_tier"]},
                                  directory / "private/environment")
        preflight = validate_single_floor(environment, episode)
        preflight["relocations"] = _relocation_preflight(environment, scenario["rounds"])
        preflight["passed"] &= all(row["passed"] for row in preflight["relocations"])
        write_json(directory / "private/preflight.json", preflight)
        if not preflight["passed"]:
            raise ValueError("Single-floor/relocation preflight failed; inspect private/preflight.json")
        session = open_session(repo, model_config, directory, environment, Path(scene_map))
        rooms = {room["id"]: room for room in scenario.get("private_rooms", [])}
        for index, specification in enumerate(scenario["rounds"], 1):
            result["active_round"] = {"round": index, "id": specification["id"], "kind": specification["kind"]}
            save()
            if specification.get("teleport"):
                environment.relocate(specification["teleport"])
                session.refresh_observation()
            start = environment.position()
            start_steps, start_length = environment.steps, environment.path_length
            motion_start = len(environment.motion_trajectory)
            before = {entity["id"]: deepcopy(entity) for entity in session.memory.entities}
            call_start = len(session.trace.calls)
            trace_path = session.private / "trace.jsonl"
            trace_start = trace_path.stat().st_size if trace_path.exists() else 0
            shortest = environment.distance(specification["positions"]) if specification["kind"] == "navigation" else None
            instruction = specification["instruction"]
            if specification.get("teleport"):
                instruction = "你已被重新放置，当前图片来自传送后的位置。\n" + instruction
            prompt = session.prompt(instruction, limits=PromptLimits(specification["max_requests"],
                                     specification["max_actions"], specification["seconds"]))
            trajectory = [start, *environment.motion_trajectory[motion_start:]]
            failed = prompt.reason in INFRASTRUCTURE_REASONS
            row = {"round": index, "id": specification["id"], "kind": specification["kind"],
                   "status": "infrastructure_error" if failed else "completed",
                   "submitted": prompt.submitted, "reason": prompt.reason, "answer": prompt.answer,
                   "requests": prompt.requests, "actions": environment.steps - start_steps,
                   "path_length": environment.path_length - start_length, "elapsed_seconds": prompt.elapsed_seconds,
                   "record": prompt.record, "start_position": start, "end_position": environment.position(),
                   "teleported": bool(specification.get("teleport")),
                   "memory": _memory_report(session, before, call_start, trace_start)}
            if prompt.error:
                row["error"] = prompt.error
            if row["kind"] == "navigation":
                distance = environment.distance(specification["positions"])
                route = score_route(specification, trajectory, rooms)
                arrived = distance < specification["success_distance"]
                row.update(navigation_error=distance, success_distance=specification["success_distance"],
                           arrived=arrived, route=route, shortest_path=shortest,
                           success=bool(prompt.submitted and arrived and route["passed"]))
                row["spl"] = spl(row["success"], shortest, row["path_length"])
            write_json(directory / f"private/rounds/{index:02d}-{row['id']}.json", {**row, "trajectory": trajectory})
            result["rounds"].append(row)
            result["active_round"] = None
            save()
            if failed:
                result.update(status="infrastructure_error", finish_reason=prompt.reason)
                break
        else:
            result.update(status="completed", finish_reason="sequence_complete")
    except Exception as error:
        result.update(status="infrastructure_error", finish_reason="worker_error",
                      error=redact(str(error)), error_type=type(error).__name__)
        (directory / "private/error.txt").write_text(traceback.format_exc())
    finally:
        for resource in (session, environment):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    result.update(status="infrastructure_error", cleanup_error=redact(str(error)))
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--scenes", type=Path, default=Path("data/nas/scene_datasets"))
    args = parser.parse_args()
    result = run_sequence(Path(__file__).resolve().parents[2], read_config(args.model_config), args.scenario,
                          args.map, args.output, {"gpu": args.gpu, "scenes": str(args.scenes.resolve())})
    print(json.dumps({"directory": str(args.output), "status": result["status"], "completed_rounds": result["completed_rounds"]}))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
