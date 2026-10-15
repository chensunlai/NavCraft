"""Re-score archived campaign trajectories against a versioned private goal set.

This reads stored positions and a matching NavMesh. It does not call a model,
move an agent, or overwrite an original run's metrics.
"""

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from src.bench.sequence import inside, score_route
from src.environment.tasks.metrics import spl


def load_prepared_dataset(dataset):
    """Read an already prepared pair; never repair or regenerate its sources."""
    dataset = Path(dataset)
    documents = []
    for name in ("scenario.json", "catalog.json"):
        path = dataset / name
        if not path.is_file():
            raise FileNotFoundError(f"Prepared dataset is missing {path}; run dataset preparation first")
        documents.append(json.loads(path.read_text()))
    scenario, catalog = documents
    if (scenario.get("protocol") != "navcraft-map-free-12-round-goal-catalog-v2"
            or scenario.get("goal_catalog_version") != 2 or catalog.get("schema_version") != 2):
        raise ValueError("Unsupported prepared goal-catalog-v2 dataset")
    if Path(scenario["scene_id"]).name != catalog["scene_id"] + ".basis.glb":
        raise ValueError("Prepared scenario and catalog scene disagree")
    for key, value in catalog["environment"].items():
        if scenario["environment"].get(key) != value:
            raise ValueError(f"Prepared scenario and catalog environment disagree: {key}")
    for identifier, target in catalog["targets"].items():
        positions = target.get("positions")
        if (target.get("id") != identifier or not positions or any(
                not isinstance(p, list) or len(p) != 3
                or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in p)
                for p in positions)):
            raise ValueError(f"Invalid prepared target identity or positions: {identifier}")
    for spec in scenario["rounds"]:
        if spec["kind"] != "navigation":
            continue
        private = spec["private_verification"]
        target = catalog["targets"].get(private["target_object_id"])
        if (target is None or spec["positions"] != target["positions"]
                or private.get("target_object_ids") != target["object_ids"]
                or private.get("target") != {k: v for k, v in target.items() if k != "positions"}):
            raise ValueError(f"Prepared scenario and catalog target disagree: {spec['id']}")
        for event in spec.get("route_events", []):
            if event["kind"] == "waypoint" and event.get("target_object_id"):
                waypoint = catalog["targets"].get(event["target_object_id"])
                if (waypoint is None or event["positions"] != waypoint["positions"]
                        or event.get("viewpoint_source") != waypoint["source_type"]):
                    raise ValueError(f"Prepared scenario and catalog waypoint disagree: {event['id']}")
    return scenario, catalog


def _validate_rescore_scenario(original, scenario):
    """Only goal sets and their provenance may change in this protocol correction."""
    if original["protocol"] not in {
            "navcraft-map-free-12-round-exploratory-campaign", scenario["protocol"]}:
        raise ValueError("Archived protocol is not compatible with goal-catalog-v2")
    ignored = {"schema_version", "goal_catalog_version", "protocol", "limitations", "rounds"}
    if ({k: v for k, v in original.items() if k not in ignored}
            != {k: v for k, v in scenario.items() if k not in ignored}):
        raise ValueError("Prepared scenario changed archived environment or sequence settings")
    if len(original["rounds"]) != len(scenario["rounds"]):
        raise ValueError("Prepared scenario changed archived round count")
    for raw_spec, spec in zip(original["rounds"], scenario["rounds"]):
        comparable = []
        for item in (raw_spec, spec):
            item = deepcopy(item)
            if item["kind"] == "navigation":
                for key in ("positions", "snap_viewpoints"):
                    item.pop(key, None)
                for key in ("target", "target_object_ids"):
                    item["private_verification"].pop(key, None)
                for event in item.get("route_events", []):
                    if event["kind"] == "waypoint":
                        for key in ("positions", "target_object_id", "viewpoint_source"):
                            event.pop(key, None)
            comparable.append(item)
        if comparable[0] != comparable[1]:
            raise ValueError(f"Prepared scenario changed archived task definition: {raw_spec['id']}")


def audit(run, output, navmesh, dataset):

    run, output, navmesh, dataset = map(Path, (run, output, navmesh, dataset))
    output = output.resolve()
    if output == run.resolve() or run.resolve() in output.parents:
        raise ValueError("Offline evaluation must be outside the archived run")
    if output == dataset.resolve():
        raise ValueError("Offline evaluation must not overwrite the prepared dataset")
    scenario, catalog = load_prepared_dataset(dataset)
    result_path = run / "result.json"
    scenario_path = run / "private/scenario.json"
    old = json.loads(result_path.read_text())
    original = json.loads(scenario_path.read_text())
    if len(old["rounds"]) != len(original["rounds"]):
        raise ValueError("Archived sequence is incomplete; refusing a silently partial re-score")
    _validate_rescore_scenario(original, scenario)
    if hashlib.sha256(navmesh.read_bytes()).hexdigest() != catalog["generation_provenance"]["navmesh_sha256"]:
        raise ValueError("NavMesh differs from the one used to verify custom viewpoints")
    import habitat_sim
    import numpy as np

    pf = habitat_sim.PathFinder()
    if not pf.load_nav_mesh(str(navmesh)):
        raise ValueError(f"Cannot read NavMesh: {navmesh}")
    settings = pf.nav_mesh_settings
    expected = original["environment"]
    for field, config in (("agent_height", "agent_height"), ("agent_radius", "agent_radius"),
                          ("cell_height", "navmesh_cell_height")):
        if not math.isclose(float(getattr(settings, field)), expected[config], abs_tol=1e-6):
            raise ValueError(f"Archived environment and NavMesh disagree: {config}")

    def distances(starts, goals):
        query = habitat_sim.MultiGoalShortestPath()
        query.requested_ends = np.asarray(goals, dtype=np.float32)
        values = []
        for point in starts:
            query.requested_start = np.asarray(point, dtype=np.float32)
            pf.find_path(query)
            values.append(float(query.geodesic_distance))
        return values

    rooms = {r["id"]: r for r in scenario["private_rooms"]}
    records = []
    for spec, archived, raw_spec in zip(scenario["rounds"], old["rounds"], original["rounds"]):
        if spec["id"] != archived["id"]:
            raise ValueError("Archived round order disagrees with private scenario")
        raw = archived["result"]
        record = dict(round=archived["round"], id=spec["id"], instruction=spec["instruction"],
                      kind=spec["kind"], old_distance=None, new_distance=None,
                      old_passed=None, new_passed=None, route=None)
        if spec["kind"] == "explore":
            record["exploration"] = raw.get("exploration")
            records.append(record)
            continue
        trajectory = raw["trajectory"]
        old_distance = distances([trajectory[-1]], raw_spec["positions"])[0]
        if abs(old_distance - raw["metrics"]["navigation_error"]) > .025:
            raise ValueError(f"Cannot reproduce archived distance: {spec['id']}")
        target = catalog["targets"][spec["private_verification"]["target_object_id"]]
        point_checks = distances(spec["positions"], [original["start_position"]])
        if not all(math.isfinite(v) for v in point_checks):
            raise ValueError(f"Target set unreachable: {spec['id']}")
        ds = distances(trajectory, spec["positions"])
        route = score_route(spec, trajectory, rooms)
        stopped = bool(raw["subtasks"] and raw["subtasks"][-1]["reason"] == "stop")
        arrival = bool(stopped and ds[-1] < spec["success_distance"])
        record.update(target_object_id=target["id"], target_name=target["label"],
                      target_room=target["room_id"], source_type=target["source_type"],
                      old_viewpoint_count=len(raw_spec["positions"]),
                      new_viewpoint_count=len(spec["positions"]),
                      reachable_viewpoint_count=len(point_checks),
                      old_distance=old_distance, new_distance=ds[-1],
                      old_passed=raw["independent"]["passed"],
                      new_passed=bool(arrival and route["passed"]),
                      stopped=stopped, arrival=arrival, route=route,
                      old_route=raw["independent"]["route"],
                      endpoint=trajectory[-1], start=trajectory[0],
                      endpoint_in_target_room=inside([trajectory[-1][0], -trajectory[-1][2]],
                                                    rooms[target["room_id"]]["polygon"]),
                      success_distance=spec["success_distance"],
                      new_shortest=ds[0], new_oracle_arrival=min(ds) < spec["success_distance"],
                      new_spl=spl(arrival, ds[0], raw["metrics"]["path_length"]))
        records.append(record)
    nav_records = [r for r in records if r["kind"] == "navigation"]
    report = {"schema_version": 2, "run": str(run), "original_protocol": original["protocol"],
              "new_protocol": scenario["protocol"], "rounds": records,
              "navigation_rounds": len(nav_records),
              "old_passed": sum(r["old_passed"] for r in nav_records),
              "new_passed": sum(r["new_passed"] for r in nav_records),
              "source_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (result_path, scenario_path, navmesh,
                                          dataset / "scenario.json", dataset / "catalog.json")},
              "limitations": [
                  "Post-hoc protocol correction on a single dependent chain, not an unbiased benchmark or new model run.",
                  "Original STOP events, trajectories, thresholds and budgets are unchanged.",
                  "Room membership is a diagnostic, not an added success condition.",
                  "Visibility of generated seeds does not imply final STOP visibility within the 1 m geodesic tolerance.",
                  "Per-round SPL measures arrival and ignores required route detours, matching the existing metric semantics.",
              ]}
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("catalog.json", catalog), ("scenario-v2.json", scenario),
                        ("reevaluation.json", report)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=repo / "runs/harness-pi/20261014T125449Z-astra-sequence")
    parser.add_argument("--output", type=Path, default=repo / "outputs/goal-completion-20261015")
    parser.add_argument("--navmesh", type=Path, default=repo / "outputs/goal-coverage-20261015/objnav.navmesh")
    parser.add_argument("--dataset", type=Path, default=repo / "data/datasets/navcraft/qa_campaign_v2")
    args = parser.parse_args()
    report = audit(args.run, args.output, args.navmesh, args.dataset)
    print(json.dumps({k: report[k] for k in ("navigation_rounds", "old_passed", "new_passed")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
