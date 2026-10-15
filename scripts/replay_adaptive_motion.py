"""Replay fixed Qa collision cases without model requests or changing old runs."""

import argparse
from copy import deepcopy
from dataclasses import asdict
import html
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

import habitat_sim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image
import quaternion
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from src.dataset.atomic.geometry import crossed_opening, horizontal
from src.dataset.episodes.base import EpisodeSpec
from src.environment.base import MotionResult, TRANSLATION_ACTIONS
from src.environment.simulator import HabitatSimulator


class ArchivedMovementReference(HabitatSimulator):
    """Historical comparison fixture only; production movement has one behavior."""

    def step(self, action):
        if action not in TRANSLATION_ACTIONS:
            return super().step(action)
        before = self._motion_sample()
        collision = bool(self.sim.step(action).get("collided", False))
        return MotionResult(collision, collision,
                            samples=[before, self._motion_sample(collision=collision)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/adaptive-motion-20261015")
    parser.add_argument("--gpu", type=int, default=3)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run = ROOT / "runs/harness-pi/20261014T125449Z-astra-sequence"
    scenario = json.loads((run / "private/scenario.json").read_text())
    frames = {r["id"]: r for r in map(json.loads, (run / "private/environment/frames.jsonl").open())}
    floor_y = scenario["start_position"][1]
    pf = habitat_sim.PathFinder()
    navmesh = ROOT / "outputs/goal-coverage-20261015/objnav.navmesh"
    assert pf.load_nav_mesh(str(navmesh))
    vertices = np.asarray(pf.build_navmesh_vertices())
    triangles = vertices[np.asarray(pf.build_navmesh_vertex_indices()).reshape(-1, 3)]
    triangles = triangles[np.max(np.abs(triangles[:, :, 1] - floor_y), axis=1) < .25]
    navigable = unary_union([Polygon(t[:, [0, 2]] * [1, -1]) for t in triangles]).buffer(.0001)
    pf.seed(20261015)
    for _ in range(1000):
        open_point = np.asarray(pf.get_random_navigable_point())
        if abs(float(open_point[1]) - floor_y) < .05 and pf.distance_to_closest_obstacle(open_point) > .6:
            break
    else:
        raise RuntimeError("Cannot find a same-floor open-space test point")

    cases = [
        {"id": "step32-yaw210", "frame": "frame_000673", "yaw": 210, "count": 4,
         "door": [[3.8, 2.55], [5.12, 2.55]]},
        {"id": "step32-yaw180", "frame": "frame_000673", "yaw": 180, "count": 6,
         "door": [[3.8, 2.55], [5.12, 2.55]]},
        {"id": "step127-D03", "frame": "frame_000774", "yaw": 0, "count": 6,
         "door": [[1.4, 2.55], [2.28, 2.55]]},
    ]
    modes = [("native", ArchivedMovementReference), ("sliding", ArchivedMovementReference),
             ("adaptive", HabitatSimulator)]
    rows, open_checks = [], []
    for name, factory in modes:
        config = deepcopy(scenario["environment"])
        config.update(gpu=args.gpu, depth=False)
        episode = EpisodeSpec("adaptive-replay", "adaptive-replay", "audit", scenario["scene_id"],
                              open_point.tolist(), [0, 0, 0, 1], [], "offline real action replay")
        simulator = factory(episode, config)
        try:
            assert simulator.sim.pathfinder.load_nav_mesh(str(navmesh))
            if name == "native":
                simulator.agent.controls.move_filter_fn = simulator.sim.pathfinder.try_step_no_sliding
            elif name == "sliding":
                simulator.agent.controls.move_filter_fn = simulator.sim.pathfinder.try_step
            for case in cases:
                directory = output / case["id"] / name
                directory.mkdir(parents=True, exist_ok=True)
                state = habitat_sim.AgentState()
                state.position = np.asarray(frames[case["frame"]]["pose"]["position"], dtype=np.float32)
                state.rotation = quaternion.from_rotation_vector([0, math.radians(case["yaw"]), 0])
                simulator.agent.set_state(state, reset_sensors=True)
                simulator.capture(directory, 0)
                records, positions = [], [simulator.position()]
                for index in range(case["count"]):
                    started = time.perf_counter()
                    motion = simulator.step("move_forward")
                    motion_seconds = time.perf_counter() - started
                    observation = simulator.capture(directory, index + 1)
                    for a, b in zip(motion.positions, motion.positions[1:]):
                        assert np.linalg.norm(np.asarray(simulator.sim.pathfinder.snap_point(b)) - b) < 1e-4
                    turn_deltas = np.diff([s["auto_yaw_delta_degrees"] for s in motion.samples])
                    assert np.max(np.abs(turn_deltas), initial=0) <= 3.00001
                    assert abs(motion.auto_yaw_delta_degrees) <= 15.00001
                    segments = [{"start": a, "end": b,
                                 "straight_filter_error_m": float(np.linalg.norm(
                                     np.asarray(simulator.sim.pathfinder.try_step_no_sliding(a, b)) - b)),
                                 "outside_navmesh_m": LineString([horizontal(a), horizontal(b)]).difference(navigable).length}
                                for a, b in zip(motion.positions, motion.positions[1:])]
                    if name == "adaptive":
                        assert all(s["outside_navmesh_m"] < 1e-6 for s in segments), segments
                    records.append({"step": index + 1, "action": "move_forward", **asdict(motion),
                                    "motion_seconds": motion_seconds,
                                    "image": observation.rgb.relative_to(output).as_posix(),
                                    "segments": segments})
                    positions.extend(motion.positions[1:])
                    if motion.blocked:
                        break
                crossed = any(crossed_opening(horizontal(a), horizontal(b), case["door"])
                              for a, b in zip(positions, positions[1:]))
                row = {"case": case["id"], "mode": name, "requested_actions": case["count"],
                       "executed_actions": len(records), "crossed_door": crossed,
                       "path_length_m": sum(math.dist(a, b) for a, b in zip(positions, positions[1:])),
                       "trajectory": positions, "actions": records, "config": asdict(simulator.config),
                       "initial_image": (directory / "observations/rgb_000000.png").relative_to(output).as_posix()}
                rows.append(row)
                print(json.dumps({k: row[k] for k in ["case", "mode", "executed_actions", "crossed_door", "path_length_m"]}), flush=True)

            for amount in ([.25, .1] if name == "adaptive" else [.25]):
                if amount != config["forward_step"]:
                    from dataclasses import replace
                    simulator.config = replace(simulator.config, forward_step=amount)
                for yaw in [0, 90]:
                    for action in TRANSLATION_ACTIONS:
                        state = habitat_sim.AgentState()
                        state.position = open_point.copy()
                        state.rotation = quaternion.from_rotation_vector([0, math.radians(yaw), 0])
                        simulator.agent.set_state(state, reset_sensors=True)
                        simulator.step("look_up")
                        before = simulator.agent.get_state()
                        local_camera = before.rotation.inverse() * before.sensor_states["rgb"].rotation
                        motion = simulator.step(action)
                        after = simulator.agent.get_state()
                        after_local = after.rotation.inverse() * after.sensor_states["rgb"].rotation
                        rotation_error = float(np.max(np.abs(quaternion.as_rotation_matrix(local_camera) - quaternion.as_rotation_matrix(after_local))))
                        displacement = math.dist(motion.positions[0], motion.positions[-1])
                        assert rotation_error < 1e-5
                        assert math.isclose(displacement, amount, abs_tol=1e-5)
                        assert not motion.blocked and abs(motion.auto_yaw_delta_degrees) < 1e-5
                        open_checks.append({"mode": name, "action": action, "start_yaw": yaw,
                                            "requested_m": amount, "displacement_m": displacement,
                                            "camera_local_rotation_error": rotation_error})
        finally:
            simulator.close()

    font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, case in zip(axes, cases):
        if (ROOT / "outputs/navmesh-rgb-bev-20261015/assets/cut-1.65m-rgb.png").exists():
            ax.imshow(Image.open(ROOT / "outputs/navmesh-rgb-bev-20261015/assets/cut-1.65m-rgb.png"),
                      extent=[.6, 6, .3, 5.7], origin="upper")
        for mode, color in [("native", "#d44b45"), ("sliding", "#dfaa00"), ("adaptive", "#137dd5")]:
            row = next(r for r in rows if r["case"] == case["id"] and r["mode"] == mode)
            path = np.asarray(row["trajectory"])
            ax.plot(path[:, 0], -path[:, 2], color=color, marker=".", lw=2,
                    label=f"{mode}: {row['executed_actions']}动作, {row['path_length_m']:.2f}m, 穿门{row['crossed_door']}")
        door = np.asarray(case["door"])
        ax.plot(*door.T, color="#9736a1", lw=3)
        all_points = np.concatenate([np.asarray(r["trajectory"])[:, [0, 2]] * [1, -1]
                                     for r in rows if r["case"] == case["id"]])
        center = (all_points.max(axis=0) + all_points.min(axis=0)) / 2
        span = max(1.2, float(np.ptp(all_points, axis=0).max()) + .4)
        ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
        ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
        ax.set_aspect("equal"); ax.set_title(case["id"]); ax.legend(fontsize=8)
        ax.set_xlabel("X / m"); ax.set_ylabel("−Z / m")
    fig.suptitle("真实原子动作重放：红色原动作／黄色仅滑移／蓝色滑移＋渐进转向\n一次批次，受阻则中断；无模型请求，不作为导航成功率")
    fig.tight_layout(rect=[0, 0, 1, .87])
    fig.savefig(output / "comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    data = {"model_requests": 0, "source_run": str(run), "navmesh": str(navmesh),
            "open_point_xyz": open_point.tolist(), "open_space_checks": open_checks, "cases": rows}
    (output / "result.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    cards = []
    for row in rows:
        pictures = [row["initial_image"]] + [a["image"] for a in row["actions"]]
        cards.append(f'<article><h2>{html.escape(row["case"])} / {row["mode"]}</h2>'
                     f'<p>{row["executed_actions"]}动作；实际路程{row["path_length_m"]:.3f}m；穿门{row["crossed_door"]}</p>'
                     '<div class="frames">' + "".join(f'<a href="{p}"><img src="{p}" title="动作后状态{i}"></a>' for i, p in enumerate(pictures)) + '</div></article>')
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>自适应运动重放</title><style>body{font-family:system-ui;max-width:1500px;margin:24px auto}article{padding:16px;border:1px solid #ccd;margin:15px 0}.frames{display:flex;gap:8px;flex-wrap:wrap}.frames a{width:240px}img{width:100%}</style><h1>无模型真实动作重放</h1><p>旧归档位置重新初始化后实际执行；不是历史模型轨迹，也不报告导航成功率。点图片查看原图。</p><a href="result.json">完整微步数据</a><img src="comparison.png">' + "".join(cards))
    markdown = ["# 自适应原子移动：真实局部重放", "",
                "固定旧归档位置，比较原动作、仅滑移、滑移与渐进转向；无模型请求，不作为导航成功率。", "",
                "![轨迹比较](comparison.png)", "",
                "| 案例 | 条件 | 执行动作 | 实际路程 | 穿门 |", "|---|---|---:|---:|---|"]
    markdown.extend(f"| {r['case']} | {r['mode']} | {r['executed_actions']} | {r['path_length_m']:.3f}m | {r['crossed_door']} |" for r in rows)
    markdown += ["", f"开阔区检查{len(open_checks)}项；自适应实际线段以0.1mm数值容差独立检查NavMesh覆盖。", "",
                 "[逐动作RGB图册](index.html) · [完整微步与配置](result.json)", "",
                 "复现：在仓库根目录执行 `python scripts/replay_adaptive_motion.py --gpu 3`。需要旧Astra运行归档、Qa原始场景和对应NavMesh。", ""]
    (output / "README.md").write_text("\n".join(markdown))
    assert next(r for r in rows if r["case"] == "step32-yaw210" and r["mode"] == "adaptive")["crossed_door"]
    assert next(r for r in rows if r["case"] == "step127-D03" and r["mode"] == "adaptive")["crossed_door"]
    print(json.dumps({"output": str(output), "open_space_checks": len(open_checks), "replays": len(rows), "model_requests": 0}), flush=True)


if __name__ == "__main__":
    main()
