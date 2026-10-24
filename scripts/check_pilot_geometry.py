"""Offline physical validation of the frozen pilot; no model or prompt calls."""

import argparse
import html
import json
import math
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np

from src.bench.runner import validate_single_floor
from src.bench.samples import load_episodes, plan, read_config
from src.common.io import write_json
from src.environment import Environment


def wall_intersections(trajectory, walls):
    """Mark geometric disagreements for inspection, not as an evaluation score."""
    hits = []
    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]
    for first, second in zip(trajectory, trajectory[1:]):
        p, r = np.array(first)[[0, 2]], np.array(second)[[0, 2]] - np.array(first)[[0, 2]]
        if np.linalg.norm(r) < .001:
            continue
        for wall in walls:
            q, end = np.asarray(wall[0]), np.asarray(wall[1])
            s = end - q
            denominator = cross(r, s)
            if abs(denominator) < 1e-9:
                continue
            t, u = cross(q - p, s) / denominator, cross(q - p, r) / denominator
            # Ignore endpoint touches and tiny mesh fragments.
            if .01 < t < .99 and .01 < u < .99 and np.linalg.norm(s) > .02:
                point = (p + t * r).tolist()
                if all(math.dist(point, old) > .05 for old in hits):
                    hits.append(point)
    return hits


def overlay(environment, episode, map_data, target, destination):
    vertices = np.asarray(environment.sim.sim.pathfinder.build_navmesh_vertices()).reshape((-1, 3, 3))
    floor = episode.start_position[1]
    triangles = vertices[np.abs(vertices[:, :, 1].mean(axis=1) - floor) < .5][:, :, [0, 2]]
    walls = map_data["walls"]
    fig, axis = plt.subplots(figsize=(11, 11))
    axis.add_collection(PolyCollection(triangles, facecolors="#cedfe8", edgecolors="none", alpha=.8))
    axis.add_collection(LineCollection(walls, colors="#222222", linewidths=.8, label="Semantic wall slice"))
    positions = np.asarray(episode.goals[0]["positions"])
    local = positions[np.abs(positions[:, 1] - floor) <= .5]
    if len(local):
        axis.scatter(local[:, 0], local[:, 2], s=7, c="#3b94bd", alpha=.35, label="Official same-level goal positions")
    reference = np.asarray(episode.reference_path)
    if len(reference):
        axis.plot(reference[:, 0], reference[:, 2], "--", color="#984ea3", lw=2,
                  label="Official sparse instruction path (not replayed)")
    actual = np.asarray(environment.motion_trajectory)
    axis.plot(actual[:, 0], actual[:, 2], color="#e68125", lw=2, label="Actual adaptive-motion path")
    axis.scatter([actual[0, 0]], [actual[0, 2]], c="#2c9b42", s=75, marker="o", label="Start")
    axis.scatter([actual[-1, 0]], [actual[-1, 2]], c="#e13c4e", s=80, marker="x", label="Final")
    if target is not None:
        axis.scatter([target[0]], [target[2]], color="#edb000", edgecolor="black", s=160, marker="*", label="Reference destination")
    intersections = wall_intersections(environment.motion_trajectory, walls)
    if intersections:
        values = np.asarray(intersections)
        axis.scatter(values[:, 0], values[:, 1], c="magenta", s=60, marker="x", label="Wall/path intersections: inspect")
    origin, size = map_data["origin"], map_data["size_m"]
    axis.set(xlim=(origin[0], origin[0] + size), ylim=(origin[1] + size, origin[1]),
             xlabel="Habitat world X / m", ylabel="Habitat world Z / m",
             title=episode.id + "\nCurrent Lite NavMesh + semantic walls + program reference")
    axis.set_aspect("equal")
    axis.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return intersections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "config/pilot.yaml")
    parser.add_argument("--output", type=Path, default=REPO / "cache/pilot-geometry")
    parser.add_argument("--gpu", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()
    import habitat_sim

    config = read_config(args.config)
    planned = plan(REPO, config)
    episodes = load_episodes(REPO, planned["dispatch_order"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    reports = []
    for job, episode in zip(planned["jobs"], episodes):
        started = time.monotonic()
        folder = output / f"{job['index']:02d}-{episode.benchmark}"
        folder.mkdir()
        record = {"episode_id": episode.id, "scene": episode.scene_id, "gpu": args.gpu,
                  "source": "program-assisted physical reference, not a model demonstration",
                  "goal_radius_m": .2, "maximum_actions": 500, "status": "initializing"}
        env = None
        target = None
        try:
            env = Environment(episode, {"gpu": args.gpu, "scenes": str((REPO / config['environment']['scenes']).resolve())}, folder / "environment")
            initial = env.observe()
            record["initial_rgb"] = str(initial.rgb.relative_to(output))
            record["preflight"] = validate_single_floor(env, episode)
            same_level = [p for p in episode.goals[0]["positions"] if abs(p[1] - episode.start_position[1]) <= .5]
            route = env.sim.shortest_path(env.position(), same_level)
            target = route[-1] if route else None
            record["reference_destination"] = target
            record["initial_geodesic_m"] = env.distance(episode.goals[0]["positions"])
            record["status"] = "preflight_failed"
            if record["preflight"]["passed"]:
                follower = habitat_sim.GreedyGeodesicFollower(
                    env.sim.sim.pathfinder, env.sim.agent, goal_radius=.2,
                    stop_key="stop", forward_key="move_forward", left_key="turn_left", right_key="turn_right")
                record["status"] = "action_budget"
                for _ in range(500):
                    action = follower.next_action_along(np.asarray(target, dtype=np.float32))
                    if action == "stop":
                        record["status"] = "reached"
                        break
                    env.step(action)
                    if env.steps % 100 == 0:
                        print(json.dumps({"episode": episode.id, "steps": env.steps,
                                          "distance": env.distance(episode.goals[0]['positions'])}), flush=True)
            record["final_geodesic_m"] = env.distance(episode.goals[0]["positions"])
            record["reached_selected_point"] = env.distance([target]) < .2 if target is not None else False
            record["final_rgb"] = str(env.current.rgb.relative_to(output))
        except Exception as error:
            record.update(status="error", error=f"{type(error).__name__}: {error}")
        finally:
            if env is not None:
                record.update(actions=env.steps, path_length_m=env.path_length,
                              collision_actions=sum(m["collision"] for m in env.motions),
                              blocked_actions=sum(m["blocked"] for m in env.motions))
                write_json(folder / "trajectory.json", {"endpoints": env.trajectory, "microsteps": env.motion_trajectory})
                env.sim.sim.pathfinder.save_nav_mesh(str(folder / "current.navmesh"))
                try:
                    record["wall_path_intersections"] = overlay(env, episode, json.loads(Path(job["map"]).read_text()), target, folder / "geometry.png")
                    record["overlay"] = str((folder / "geometry.png").relative_to(output))
                finally:
                    env.close()
            record["seconds"] = time.monotonic() - started
            write_json(folder / "report.json", record)
            reports.append(record)
            write_json(output / "summary.json", reports)
            print(json.dumps({k: record[k] for k in ("episode_id", "status", "seconds")}), flush=True)
    cards = []
    for report in reports:
        pictures = "".join(f'<figure><img src="{html.escape(report[key])}"><figcaption>{key}</figcaption></figure>'
                           for key in ("initial_rgb", "overlay", "final_rgb") if key in report)
        cards.append(f'<section><h2>{html.escape(report["episode_id"])}</h2><p>{html.escape(report["status"])}</p>{pictures}'
                     f'<details><summary>Private checks</summary><pre>{html.escape(json.dumps(report, indent=2))}</pre></details></section>')
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Pilot physical validation</title>'
        '<style>body{font:16px system-ui;max-width:1250px;margin:auto}section{padding:20px;border-bottom:1px solid #ddd}'
        'figure{display:inline-block;width:30%;vertical-align:top;margin:1%}img{width:100%}pre{white-space:pre-wrap}</style>'
        '<h1>Current Lite embodiment: program reference checks</h1><p>No model was called. Sparse official R2R routes are shown for comparison; '
        'the greedy follower validates physical reachability, not instruction compliance.</p>' + "".join(cards), encoding="utf-8")


if __name__ == "__main__":
    main()
