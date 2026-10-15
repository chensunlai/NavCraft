"""Re-score archived campaign trajectories against a versioned private goal set.

This reads stored positions and a matching NavMesh. It does not call a model,
move an agent, or overwrite an original run's metrics.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from src.bench.goal_catalog import complete_scenario, load_goal_catalog
from src.bench.sequence import inside, score_route
from src.environment.tasks.metrics import spl


def audit(run, output, navmesh, repo, generated_path=None):
    import habitat_sim
    import numpy as np

    run, output, navmesh, repo = map(Path, (run, output, navmesh, repo))
    output = output.resolve()
    if output == run.resolve() or run.resolve() in output.parents:
        raise ValueError("Offline evaluation must be outside the archived run")
    result_path = run / "result.json"
    scenario_path = run / "private/scenario.json"
    old = json.loads(result_path.read_text())
    original = json.loads(scenario_path.read_text())
    if len(old["rounds"]) != len(original["rounds"]):
        raise ValueError("Archived sequence is incomplete; refusing a silently partial re-score")
    catalog = load_goal_catalog(repo, generated_path)
    if hashlib.sha256(navmesh.read_bytes()).hexdigest() != catalog["generation_provenance"]["navmesh_sha256"]:
        raise ValueError("NavMesh differs from the one used to verify custom viewpoints")
    scenario = complete_scenario(original, catalog)
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
                                for p in (result_path, scenario_path, navmesh)},
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
    parser.add_argument("--generated", type=Path)
    args = parser.parse_args()
    report = audit(args.run, args.output, args.navmesh, repo, args.generated)
    print(json.dumps({k: report[k] for k in ("navigation_rounds", "old_passed", "new_passed")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
