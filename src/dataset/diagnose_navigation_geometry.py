"""Privileged offline diagnosis of ObjectNav target/navmesh compatibility.

No model is called and no task, scoring threshold, scene asset or online input
is modified. Teleports and exact-heading actions are explicitly oracle probes.
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import yaml

from src.common.io import read_json, write_json
from src.dataset.episodes.base import EpisodeSpec


def settings_dict(settings):
    return {
        name: getattr(settings, name)
        for name in dir(settings)
        if not name.startswith("_") and not callable(getattr(settings, name))
    }


def distance(pathfinder, start, ends):
    import habitat_sim

    query = habitat_sim.MultiGoalShortestPath()
    query.requested_start = np.asarray(start, dtype=np.float32)
    query.requested_ends = np.asarray(ends, dtype=np.float32)
    found = pathfinder.find_path(query)
    return float(query.geodesic_distance) if found else None


def target_report(pathfinder, positions):
    rows = []
    for index, target in enumerate(positions):
        snapped = np.asarray(
            pathfinder.snap_point(np.asarray(target, dtype=np.float32))
        )
        if not np.isfinite(snapped).all():
            continue
        vertical = abs(float(snapped[1]) - target[1])
        budget = math.sqrt(max(0, 0.1**2 - vertical**2))
        rows.append(
            {
                "index": index,
                "target": target,
                "snapped": snapped.astype(float).tolist(),
                "navigable": bool(
                    pathfinder.is_navigable(np.asarray(target, dtype=np.float32))
                ),
                "snap_displacement_m": math.dist(target, snapped),
                "vertical_displacement_m": vertical,
                "target_to_itself_geodesic_m": distance(pathfinder, target, [target]),
                "snapped_to_original_geodesic_m": distance(
                    pathfinder, snapped, [target]
                ),
                "snapped_to_itself_geodesic_m": distance(
                    pathfinder, snapped, [snapped]
                ),
                "horizontal_success_budget_m_if_euclidean": budget,
            }
        )
    best = min(rows, key=lambda row: row["snapped_to_original_geodesic_m"])
    return {
        "viewpoint_count": len(positions),
        "finite_snaps": len(rows),
        "navigable_viewpoints": sum(row["navigable"] for row in rows),
        "minimum_snap_displacement_m": min(row["snap_displacement_m"] for row in rows),
        "maximum_snap_displacement_m": max(row["snap_displacement_m"] for row in rows),
        "minimum_snapped_to_target_m": best["snapped_to_original_geodesic_m"],
        "best": best,
        "samples": rows[:: max(1, len(rows) // 8)][:8],
    }


def native_action_probe(simulator, targets, best):
    import habitat_sim
    import quaternion

    target = np.asarray(best["snapped"], dtype=np.float32)
    original = simulator.agent.get_state()
    results = []
    try:
        for angle in np.linspace(0, 2 * math.pi, 8, endpoint=False):
            forward = np.array([-math.sin(angle), 0, -math.cos(angle)])
            requested = target - 0.75 * forward
            start = np.asarray(simulator.sim.pathfinder.snap_point(requested))
            if not np.isfinite(start).all() or math.dist(start, requested) > 0.01:
                continue
            state = habitat_sim.AgentState()
            state.position = start
            state.rotation = quaternion.from_rotation_vector([0, angle, 0])
            simulator.agent.set_state(state)
            records = [
                {
                    "position": simulator.position(),
                    "distance_to_original_goals_m": simulator.distance(targets),
                }
            ]
            for _ in range(3):
                collided = simulator.step("move_forward")
                records.append(
                    {
                        "position": simulator.position(),
                        "collision": collided,
                        "distance_to_original_goals_m": simulator.distance(targets),
                    }
                )
            results.append(
                {
                    "label": "privileged_teleport_0.75m_away_then_three_0.25m_forward_actions",
                    "angle_radians": float(angle),
                    "steps": records,
                }
            )
            if len(results) == 2:
                break
    finally:
        simulator.agent.set_state(
            original, reset_sensors=False, infer_sensor_states=False
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("cache/visual-harness-v1/objnav_dev.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cache/visual-harness-v1/geometry-diagnosis"),
    )
    parser.add_argument("--gpu", type=int, default=3)
    args = parser.parse_args()
    os.environ.setdefault(
        "__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    )
    import habitat_sim
    from src.environment.simulator import HabitatSimulator

    episodes = [EpisodeSpec(**raw) for raw in read_json(args.manifest)["episodes"]]
    config = yaml.safe_load(
        Path("config/environment/objectnav_visual.yaml").read_text()
    )["environment"]
    # Reproduce the pre-fix bake even after the live visual configuration changes.
    config.update(
        gpu=args.gpu, depth=False, width=64, height=48, navmesh_cell_height=0.2
    )
    simulator = HabitatSimulator(episodes[0], config)
    try:
        original_pathfinder = habitat_sim.PathFinder()
        asset = Path(config["scenes"]) / episodes[0].scene_id
        original_asset = asset.with_suffix(".navmesh")
        assert original_pathfinder.load_nav_mesh(str(original_asset))
        original_settings = original_pathfinder.nav_mesh_settings
        current_settings = simulator.sim.pathfinder.nav_mesh_settings
        report = {
            "protocol": "privileged_geometry_oracle_not_navigation_metrics",
            "threshold_m": 0.1,
            "original_asset": str(original_asset),
            "original_settings": settings_dict(original_settings),
            "current_settings": settings_dict(current_settings),
            "episodes": [],
        }
        for episode in episodes:
            targets = episode.goals[0]["positions"]
            current = target_report(simulator.sim.pathfinder, targets)
            current["oracle_atomic_probe"] = native_action_probe(
                simulator, targets, current["best"]
            )
            report["episodes"].append(
                {
                    "episode_id": episode.id,
                    "current": current,
                    "original_asset": target_report(original_pathfinder, targets),
                }
            )

        # Keep requested agent dimensions but retain the archived voxel settings.
        preserved = original_pathfinder.nav_mesh_settings
        preserved.agent_height = config["agent_height"]
        preserved.agent_radius = config["agent_radius"]
        preserved.agent_max_climb = 0.2
        assert simulator.sim.recompute_navmesh(simulator.sim.pathfinder, preserved)
        report["preserved_voxel_settings_requested_agent_dimensions"] = settings_dict(
            simulator.sim.pathfinder.nav_mesh_settings
        )
        for episode, row in zip(episodes, report["episodes"]):
            targets = episode.goals[0]["positions"]
            variant = target_report(simulator.sim.pathfinder, targets)
            variant["oracle_atomic_probe"] = native_action_probe(
                simulator, targets, variant["best"]
            )
            row["preserved_voxel_variant"] = variant

        for cell_height in (0.1, 0.05):
            settings = habitat_sim.NavMeshSettings()
            settings.set_defaults()
            settings.agent_height = config["agent_height"]
            settings.agent_radius = config["agent_radius"]
            settings.agent_max_climb = 0.2
            settings.cell_height = cell_height
            assert simulator.sim.recompute_navmesh(simulator.sim.pathfinder, settings)
            for episode, row in zip(episodes, report["episodes"]):
                targets = episode.goals[0]["positions"]
                variant = target_report(simulator.sim.pathfinder, targets)
                variant["oracle_atomic_probe"] = native_action_probe(
                    simulator, targets, variant["best"]
                )
                # Deliberately inspect points near an endpoint, without changing targets.
                anchor = np.asarray(variant["best"]["snapped"])
                variant["nearby_offsets"] = []
                for dx in (0, 0.0001, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075):
                    point = anchor + [dx, 0, 0]
                    variant["nearby_offsets"].append(
                        {
                            "horizontal_offset_m": dx,
                            "original_target_distance_m": distance(
                                simulator.sim.pathfinder,
                                point,
                                [variant["best"]["target"]],
                            ),
                            "all_goals_distance_m": distance(
                                simulator.sim.pathfinder, point, targets
                            ),
                        }
                    )
                row[f"cell_height_{cell_height:g}_variant"] = variant

        write_json(args.output / "report.json", report)
        for row in report["episodes"]:
            print(
                row["episode_id"],
                "minimum snapped-to-target",
                {
                    name: row[name]["minimum_snapped_to_target_m"]
                    for name in (
                        "current",
                        "original_asset",
                        "preserved_voxel_variant",
                        "cell_height_0.1_variant",
                        "cell_height_0.05_variant",
                    )
                },
                flush=True,
            )
        print(
            "settings differences",
            {
                key: [report["original_settings"][key], value]
                for key, value in report["current_settings"].items()
                if report["original_settings"][key] != value
            },
            flush=True,
        )
    finally:
        simulator.close()


if __name__ == "__main__":
    main()
