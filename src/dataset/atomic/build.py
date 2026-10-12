"""Rule sampling and bounded teacher collection for the 28-example showcase.

Run with ``python -m src.dataset.atomic.build --help``. No benchmark defaults are
changed; a private stage configuration adds semantic verification to the same GLB.
"""

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from src.harness.map import MapCalibration
from src.harness.assets import compose_images
from .geometry import crossed_opening, distance, horizontal, map_point
from .runtime import AtomicWorld, RequestBudget, collect_attempt, parse_answer, text_message
from .schema import AtomicTask, COVERAGE, write_json
from .export import export_build


def font(size=22):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)


def draw_labels(source, destination, entries):
    with Image.open(source) as image:
        image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    for label, box in entries:
        x0, y0, x1, y1 = box
        draw.rectangle(box, outline="#ffcc00", width=4)
        draw.rectangle([x0, max(0, y0 - 27), x0 + 42, max(0, y0 - 27) + 27], fill="#ffcc00")
        draw.text((x0 + 4, max(0, y0 - 27)), label, fill="black", font=font())
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return destination


def semantic_boxes(mask, mapping, minimum=300):
    result = []
    for label, instance in mapping.items():
        rows, cols = np.where(mask == instance)
        if len(rows) >= minimum:
            result.append((label, [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]))
    return result


class Sampler:
    def __init__(self, archive, world, root):
        self.archive, self.world, self.root = archive, world, root
        self.calibration = MapCalibration(archive)
        self.doors = [d for d in json.loads((archive / "raw/doors.json").read_text())["doors"]
                      if d["status"] == "rgb_and_local_navmesh_verified"]
        self.objects = json.loads((archive / "raw/objects.json").read_text())["objects"]
        self.by_instance = {i: obj for obj in self.objects for i in obj.get("reference_instance_ids", [])}
        self.regions = json.loads((archive / "raw/regions.json").read_text())["regions"]

    def region(self, point):
        # Exact boundary points are not used as task starts or endpoints.
        for region in self.regions:
            polygon = region["polygon"]
            inside = False
            x, y = point
            for a, b in zip(polygon, polygon[1:] + polygon[:1]):
                if (a[1] > y) != (b[1] > y) and x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]:
                    inside = not inside
            if inside:
                return region["id"]
        return None

    def seed(self, serial):
        """Try a finite deterministic sequence; prove crossing by actual actions."""
        for offset in range(len(self.doors)):
            door = self.doors[(serial + offset) % len(self.doors)]
            a, b = np.array(door["segment"], dtype=float)
            tangent = (b - a) / np.linalg.norm(b - a)
            normal = np.array([-tangent[1], tangent[0]])
            for sign in (1, -1):
                direction = normal * sign
                yaw = math.atan2(-direction[0], direction[1])
                center = (a + b) / 2 + tangent * ((serial % 5) - 2) * 0.025
                raw = center - direction * (0.55 + (serial % 4) * 0.05)
                position = self.world.sim.sim.pathfinder.snap_point(np.array([raw[0], 0.0, -raw[1]], dtype=np.float32))
                position = [float(v) for v in position]
                if not all(math.isfinite(v) for v in position) or distance(horizontal(position), raw) > 0.16 or abs(position[1]) > 0.3:
                    continue
                pose = {"position": position, "rotation_xyzw": [0, math.sin(yaw / 2), 0, math.cos(yaw / 2)]}
                self.world.reset(pose)
                history = [{"pose": self.world.sim.pose(), "action": None}]
                actions = []
                for _ in range(5):
                    collision = self.world.sim.step("move_forward")
                    actions.append("move_forward")
                    history.append({"pose": self.world.sim.pose(), "action": "move_forward", "collision": collision})
                    if collision:
                        break
                points = [horizontal(row["pose"]["position"]) for row in history]
                passed = any(crossed_opening(p, q, door["segment"]) for p, q in zip(points, points[1:]))
                if len(actions) == 5 and not any(row.get("collision") for row in history) and passed:
                    self.world.reset(pose)
                    return {"start_pose": pose, "door": door, "destination": points[-1],
                            "start_room": self.region(points[0]), "destination_room": self.region(points[-1]),
                            "reference_actions": actions, "preflight": history}
        raise RuntimeError(f"No verified short seed for {serial}")

    def map(self, path, *, arrows=(), points=(), objects=(), doors=False, route=None):
        with Image.open(self.archive / "floor_plan.png") as image:
            image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        def pixel(point):
            u, v = map_point(self.calibration, point)
            return (u * image.width, v * image.height)
        for label, point in points:
            x, y = pixel(point)
            draw.ellipse((x - 15, y - 15, x + 15, y + 15), fill="#fbd245", outline="black", width=2)
            draw.text((x + 18, y - 20), label, fill="#9a1008", font=font(30))
        for label, obj in objects:
            x, y = pixel(obj["primitives"][0]["center"])
            draw.ellipse((x - 23, y - 23, x + 23, y + 23), outline="#dc201a", width=6)
            draw.text((x + 25, y - 28), label, fill="#ab120c", font=font(32))
        for label, pose in arrows:
            _, origin, target = self.calibration.initial_arrow(pose)
            start = np.array([origin[0] * image.width, origin[1] * image.height])
            end = np.array([target[0] * image.width, target[1] * image.height])
            unit = (end - start) / np.linalg.norm(end - start)
            end = start + unit * 65
            side = np.array([-unit[1], unit[0]])
            draw.line([tuple(start), tuple(end)], fill="#1476cf", width=8)
            draw.polygon([tuple(end), tuple(end - unit * 21 + side * 11), tuple(end - unit * 21 - side * 11)], fill="#1476cf")
            draw.ellipse((*tuple(start - 7), *tuple(start + 7)), fill="#1476cf")
            draw.text(tuple(start + 13), label, fill="#084c96", font=font(28))
        if doors:
            for door in self.doors:
                pa, pb = [pixel(p) for p in door["segment"]]
                draw.line([pa, pb], fill="#007f66", width=7)
                draw.text(((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2 - 28), door["id"], fill="#00715b", font=font(23))
        if route:
            draw.line([pixel(p) for p in route], fill="#b032bc", width=6)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return path

    def observation(self, directory, mode):
        return self.world.capture(directory, mode)

    def candidate_map(self, path, candidates):
        """Separate panels prevent different headings at one point hiding labels."""
        entries = []
        for label, pose in candidates:
            panel = self.map(path.with_name(f"{path.stem}_{label}.png"), arrows=[(label, pose)])
            entries.append((f"Candidate {label}", panel))
        return compose_images(entries, path, columns=3)

    def object_view(self, start_pose, directory, serial):
        """Find a new actor-height RGB containing multiple verified mapped objects."""
        best = None
        for delta in (0, 90, 180, 270):
            pose = deepcopy(start_pose)
            q = pose["rotation_xyzw"]
            yaw = 2 * math.atan2(q[1], q[3]) + math.radians(delta)
            pose["rotation_xyzw"] = [0, math.sin(yaw / 2), 0, math.cos(yaw / 2)]
            self.world.reset(pose)
            mask = self.world.semantic()
            values, counts = np.unique(mask, return_counts=True)
            candidates = [(int(n), int(v)) for v, n in zip(values, counts)
                          if int(v) in self.by_instance and n >= 1200]
            candidates.sort(reverse=True)
            score = sum(min(n, 15000) for n, _ in candidates[:3])
            if len(candidates) >= 2 and (best is None or score > best[0]):
                best = score, pose, candidates, mask
        if best is None:
            raise RuntimeError("No sufficiently visible mapped objects at seed")
        _, pose, candidates, mask = best
        self.world.reset(pose)
        rgb = self.world.capture(directory)
        target_index = serial % min(2, len(candidates))
        instance = candidates[target_index][1]
        return pose, rgb, mask, instance, candidates

    def task(self, cluster, form, mode, serial):
        identifier = f"{cluster}-{form}-{'S' if mode == 'single' else 'P'}"
        directory = self.root / "samples" / identifier
        directory.mkdir(parents=True, exist_ok=True)
        seed = self.seed(serial)
        pose = deepcopy(seed["start_pose"])
        initial, resources = [], {}
        anchor = "initial_observation"
        labels = {**seed, "anchor": anchor, "door_segment": seed["door"]["segment"],
                  "radius": 0.30, "seed_serial": serial,
                  "label_method": "private simulator pose + calibrated schematic geometry + actual local replay"}
        pose_condition = "One initial map arrow; no live pose updates"
        def keep(path, ref=None, show=True):
            rel = str(path.relative_to(directory))
            resources[ref or path.name] = rel
            resources[rel] = rel
            if show:
                initial.append(rel)
            return rel
        if cluster in {"A05", "A06"}:
            pose, rgb, mask, instance, candidates = self.object_view(pose, directory, serial)
            labels["start_pose"] = pose
            target = self.by_instance[instance]
            np.save(directory / "private_mask.npy", mask)
            private_dir = directory / "private"
            private_dir.mkdir(exist_ok=True)
            (directory / "private_mask.npy").rename(private_dir / "initial_semantic.npy")
            center = target["primitives"][0]["center"]
            alternatives = sorted((obj for obj in self.objects if obj["id"] != target["id"]
                                   and distance(center, obj["primitives"][0]["center"]) > .75),
                                  key=lambda obj: (obj["room"] != target["room"], obj["category"] != target["category"]))
            objects = [target] + alternatives[:2]
            random.Random(serial).shuffle(objects)
            labels.update(target_object=target["id"], reference_instance=instance,
                          target_pixels=int((mask == instance).sum()), target_box=semantic_boxes(mask, {"T": instance})[0][1],
                          label_method="same-camera semantic instance mask + archived reference_instance_ids + map calibration")
            if cluster == "A05":
                pose_condition = "No true-pose hint; match the marked observation instance to map candidates"
                marked = draw_labels(rgb, directory / "target_observation.png", [("T", labels["target_box"])])
                keep(marked, "target_observation.png")
                if mode == "panorama":
                    keep(self.observation(directory, "panorama"), "initial_panorama.jpg")
                task_map = self.map(directory / "task_map.png", objects=[(chr(65 + i), obj) for i, obj in enumerate(objects)])
                labels["choice"] = chr(65 + objects.index(target))
                question = "初始 RGB 中黄色框 T 的物体，对应俯视图红圈 A/B/C 中的哪一个？按实例而不是仅按类别匹配。回答 choice=A/B/C。"
                keep(task_map, "task_map.png", show=form == "simple_qa")
                if form == "retrieval_qa":
                    question += " 俯视图在固定资源库中，请实际读取图像查证后回答。"
            else:
                mapping = {chr(65 + i): candidate[1] for i, candidate in enumerate(candidates[:3])}
                labels["choice"] = next(k for k, v in mapping.items() if v == instance)
                labels["observation_labels"] = mapping
                task_map = self.map(directory / "task_map.png", objects=[("T", target)], arrows=[("start", pose)])
                keep(task_map, "task_map.png")
                keep(self.observation(directory, mode), "initial_observation")
                question = "俯视图红圈 T 的物体对应 RGB 中哪个黄色候选框（A/B/C）？回答候选框字母，不能只回答物体类别。"
                if form == "simple_qa":
                    keep(draw_labels(rgb, directory / "candidates.png", semantic_boxes(mask, mapping)), "candidates.png")
                else:
                    labels["annotate_observation"] = True
                    question += " 请调用 nav_observe 获取当前候选框观测后回答；工具在真实新 RGB 上标出候选区域，不返回实例身份。"
                    if mode == "panorama":
                        question += " 先向左转一次，再获取新观测，判断转向后候选框与地图的对应。"
                        labels["require_motion"] = True
                anchor = "map_target_T_identity"
                labels["anchor"] = anchor
        else:
            # A04/A07 demand orientation correction, not a scripted forward-only walk.
            turn_count = 0
            if cluster in {"A04", "A07"}:
                turn_count = 1 if mode == "single" else -2
                q = pose["rotation_xyzw"]
                yaw = 2 * math.atan2(q[1], q[3]) + math.radians(30 * turn_count)
                pose["rotation_xyzw"] = [0, math.sin(yaw / 2), 0, math.cos(yaw / 2)]
                labels["start_pose"] = pose
                labels["reference_actions"] = ["turn_right" if turn_count > 0 else "turn_left"] * abs(turn_count) + seed["reference_actions"]
            self.world.reset(pose)
            observation = self.observation(directory, mode)
            keep(observation, "initial_observation")
            if cluster == "A01":
                pose_condition = "No current or initial true-pose arrow; A/B/C are hypotheses"
                alternative = deepcopy(pose)
                alternate_yaw = 2 * math.atan2(pose["rotation_xyzw"][1], pose["rotation_xyzw"][3]) + math.pi
                alternative["rotation_xyzw"] = [0, math.sin(alternate_yaw / 2), 0, math.cos(alternate_yaw / 2)]
                other_place = deepcopy(pose)
                other_place["position"] = seed["preflight"][-1]["pose"]["position"]
                candidates = [("correct", pose), ("opposite", alternative), ("other_side", other_place)]
                candidates = candidates[serial % 3:] + candidates[:serial % 3]
                labels["choice"] = chr(65 + [name for name, _ in candidates].index("correct"))
                labels["pose_candidates"] = {chr(65 + i): p for i, (_, p) in enumerate(candidates)}
                keep(self.candidate_map(directory / "task_map.png", [(chr(65 + i), p) for i, (_, p) in enumerate(candidates)]), "task_map.png")
                question = "地图上 A/B/C 是三个候选位姿（位置和朝向），哪一个与初始观测一致？回答 choice=A/B/C，并用场景结构解释。"
                if form == "interactive_qa":
                    question += " 必须主动取得一次新观测查证；问题始终问初始位姿，不是移动后的位姿。"
                    if mode == "panorama":
                        question += " 初始全景之外，请至少执行一次短移动，从门口视差查证。"
                        labels["require_translation"] = True
            elif cluster == "A02":
                pose_condition = "Only the ORIGINAL start arrow is supplied; final A/B/C are hypotheses"
                keep(self.map(directory / "start_map.png", arrows=[("start", pose)]), "start_map.png")
                motion = ["move_forward", "move_forward", "turn_right"]
                history = []
                for action in motion:
                    collision = self.world.sim.step(action)
                    history.append({"action": action, "collision": collision, "pose": self.world.sim.pose()})
                if any(h["collision"] for h in history):
                    raise RuntimeError("Pose-update sample collided")
                end_pose = self.world.sim.pose()
                keep(self.observation(directory, mode), "after_motion_observation")
                wrong = deepcopy(end_pose)
                wrong["rotation_xyzw"] = pose["rotation_xyzw"]
                candidates = [(True, end_pose), (False, wrong), (False, pose)]
                random.Random(serial).shuffle(candidates)
                labels["choice"] = chr(65 + next(i for i, (yes, _) in enumerate(candidates) if yes))
                labels["motion_history"] = history
                labels["pose_candidates"] = {chr(65 + i): p for i, (_, p) in enumerate(candidates)}
                keep(self.candidate_map(directory / "candidates_map.png", [(chr(65 + i), p) for i, (_, p) in enumerate(candidates)]), "candidates_map.png")
                question = "从 start 位姿实际执行 move_forward ×2（共约0.5米），再 turn_right ×1（30°）。结合前后观测，末状态是候选 A/B/C 的哪一个？只给出了原始起点，不提供末状态真值标记。"
                anchor = "after_recorded_actions"
                labels["anchor"] = anchor
            elif cluster in {"A03", "A08"}:
                map_path = self.map(directory / "task_map.png", arrows=[("start", pose)], points=[("T", seed["destination"])], doors=True)
                keep(map_path, "task_map.png", show=form != "retrieval_qa")
                if cluster == "A03":
                    question = f"从初始位置进入相邻的 {seed['destination_room']}，到地图 T 所在的门内侧。应使用图上哪个绿色标记开口？回答开口编号。只考虑这一次局部通行。"
                else:
                    destination_pose = deepcopy(pose)
                    destination_pose["position"] = seed["preflight"][-1]["pose"]["position"]
                    landmark = {"R01": [5.53, -.22], "R08": [-5.5, 5.0]}.get(seed["destination_room"])
                    if landmark:
                        delta = np.asarray(landmark) - np.asarray(horizontal(destination_pose["position"]))
                        yaw = math.atan2(-delta[0], delta[1])
                        destination_pose["rotation_xyzw"] = [0, math.sin(yaw / 2), 0, math.cos(yaw / 2)]
                    self.world.reset(destination_pose)
                    goal = self.observation(directory, "single")
                    keep(goal, "goal_view")
                    self.world.reset(pose)
                    question = "goal_view 是相邻目标区域的参考观察，拍摄朝向可能不同于当前朝向。先将目标观察与俯视图对应，再确定进入该区域要经过哪个绿色标记开口。回答开口编号。"
                    # T is not supplied for A08: actor must ground the observed target.
                    self.map(map_path, arrows=[("start", pose)], doors=True)
                    labels["goal_pose"] = destination_pose
                if form == "retrieval_qa":
                    question += " 必须读取固定资源中的 task_map.png 查证；不会产生新观察。"
                labels["choice"] = seed["door"]["id"]
                if form == "interactive_ia":
                    question = "把 goal_view 对应到地图上的相邻目标区域，实际穿过正确开口到门内侧（一次穿门，距离约1.25米）。参考图的拍摄朝向可不同，不要求最终朝向一致。可查询地图、执行原子动作；由工具报告局部事件完成。"
            elif cluster == "A04":
                keep(self.map(directory / "task_map.png", arrows=[("start", pose)], points=[("T", seed["destination"])], doors=True), "task_map.png")
                labels["choice"] = "turn_right" if turn_count > 0 else "turn_left"
                question = f"要沿 {seed['door']['id']} 通过门口接近局部目标 T，先选择哪种转向来对齐门口通路？回答 turn_left 或 turn_right。结合当前视角和初始地图朝向，避免直接朝旁边的墙行走。"
                if form == "interactive_ia":
                    question = f"对齐 {seed['door']['id']} 门口，实际穿过它，到 T 点附近。先处理当前朝向偏差，再进行一次短程通行。"
            else:
                keep(self.map(directory / "task_map.png", arrows=[("start", pose)],
                              points=[("T", seed["destination"])], doors=True,
                              route=[horizontal(pose["position"]), [(a + b) / 2 for a, b in zip(*seed["door"]["segment"])], seed["destination"]]), "task_map.png")
                question = "将地图上的紫色短路线落实为真实动作：先把当前朝向与路线对齐，再经路线经过的开口到 T 附近。只执行这一段，不继续探索。"
        # Frozen index is an ordinary public lookup document, never a label dump.
        index = directory / "resources.md"
        index.write_text("# 本样本的固定资源\n" + "\n".join(f"- {ref}: image" for ref in resources) + "\n门口标签表示已核验开口；地图不提供当前实时位姿。\n")
        resources["resources.md"] = "resources.md"
        task = AtomicTask(identifier, cluster, form, mode, question, initial, resources, anchor, pose_condition, labels)
        task.save(directory)
        write_json(directory / "private" / "generation.json", {"source": "rule_generated", "seed": seed,
                   "sensor": {"height": .88, "hfov": 79, "resolution": [480, 640]},
                   "panorama": "six right-offset views 0,60,120,180,240,300 degrees; originals preserved"})
        self.world.reset(labels["start_pose"])
        return task


def load_task(directory):
    public = json.loads((directory / "task.json").read_text())
    labels = json.loads((directory / "private/labels.json").read_text())
    return AtomicTask(**{k: v for k, v in public.items() if k not in {"tools", "capabilities"}}, private=labels)


def model_config(config):
    # Same credential resolution as the existing probes, without logging secrets.
    from src.bench.probe_models import load_model_config
    return {**load_model_config(config["model_config"]), **config["model_overrides"]}


def setup_world(config, root):
    env = yaml.safe_load(Path(config["environment_config"]).read_text())["environment"]
    env["gpu"] = config["gpu"]
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = env["egl_vendor_file"]
    scene = (Path(env["scenes"]) / config["scene_id"]).resolve()
    stage = root / "private" / "scene.stage_config.json"
    write_json(stage, {"render_asset": str(scene), "collision_asset": str(scene),
                      "semantic_asset": str(scene).replace(".basis.glb", ".semantic.glb"),
                      "semantic_descriptor_filename": str(scene).replace(".basis.glb", ".semantic.txt"),
                      "up": [0, 0, 1], "front": [0, 1, 0]})
    return AtomicWorld(str(stage), env)


def review(task, directory, budget):
    """Author-side QA is separate from every actor context and never labels success."""
    question = ("You are an image-quality editor. Do NOT solve or calculate this navigation task. "
                "Only check whether images are readable, candidate labels overlap, or question wording contradicts the images. "
                "Return a short JSON {supported:bool, issues:[...], wording:string}, at most 80 words. "
                "You may propose clearer wording without changing candidate IDs or the answer anchor. "
                "Tool-retrieval tasks intentionally withhold the map initially; it is supplied below for YOUR review only.\n"
                + task.question + "\nPrivate reference for verification only: "
                + json.dumps({k: task.private[k] for k in ["choice", "target_object", "door_segment"] if k in task.private}))
    paths = [directory / path for path in task.initial_images]
    for ref in ["task_map.png"]:
        if ref in task.resources and directory / task.resources[ref] not in paths:
            paths.append(directory / task.resources[ref])
    turn = budget.complete([text_message(question, images=paths)], [], f"{task.id}/author_review")
    result = parse_answer(turn.items)
    write_json(directory / "private" / "author_review.json", {"source": "deepseek_review", "result": result, "items": turn.items})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/dataset/atomic_demo.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=["probe", "sample", "collect", "export"], required=True)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--curator-reviewed", action="store_true",
                        help="Skip automated author review only when private/visual_review.json exists")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.config).read_text())
    if args.phase == "export":
        print(export_build(root), flush=True)
        return
    if args.phase == "probe":
        from src.bench.probe_models import probe_model, scene_images
        settings = model_config(config)
        budget = RequestBudget(settings, root, config["request_limit"])
        class ProbeBackend:
            def complete(self, request):
                return budget.complete(request.items, request.tools, "compatibility")
            def request_options(self):
                return budget.model.request_options()
            def close(self):
                budget.model.close()
        result = probe_model(settings, scene_images(config["scene_archive"]), root / "private/probe",
                             timeout_seconds=300, backend=ProbeBackend())
        write_json(root / "compatibility.json", result)
        print(result["status"], flush=True)
        return
    world = setup_world(config, root)
    try:
        if args.phase == "sample":
            sampler = Sampler(Path(config["scene_archive"]).resolve(), world, root)
            serial = 0
            for cluster, (_, _, forms) in COVERAGE.items():
                for form in forms:
                    for mode in ["single", "panorama"]:
                        serial += 1
                        identifier = f"{cluster}-{form}-{'S' if mode == 'single' else 'P'}"
                        if args.ids and identifier not in args.ids:
                            continue
                        if (root / "samples" / identifier / "task.json").exists():
                            print("exists", identifier, flush=True)
                            continue
                        task = sampler.task(cluster, form, mode, serial)
                        print("sampled", task.id, flush=True)
        else:
            budget = RequestBudget(model_config(config), root, config["request_limit"])
            try:
                for directory in sorted((root / "samples").iterdir()):
                    if args.ids and directory.name not in args.ids:
                        continue
                    if (directory / "selected.json").exists() or (directory / "attempts/teacher/record.json").exists():
                        continue
                    task = load_task(directory)
                    if args.curator_reviewed:
                        if not (directory / "private/visual_review.json").exists():
                            raise ValueError("Missing explicit curator review")
                        result = {"source": "agent_visual_review", "reference": "private/visual_review.json"}
                    else:
                        try:
                            result = review(task, directory, budget)
                        except Exception as exc:
                            result = {"error": str(exc)}
                            write_json(directory / "private/author_review_error.json", result)
                    print("reviewed", task.id, result, flush=True)
                    record = collect_attempt(task, directory, world, budget)
                    print("collected", task.id, record["validation"], record["error"], flush=True)
                    if record["validation"]["passed"]:
                        write_json(directory / "selected.json", {"record": "attempts/teacher/record.json",
                                   "reason": "Independent private scoring passed; visual review pending",
                                   "visual_review": "pending"})
            finally:
                budget.model.close()
    finally:
        world.close()


if __name__ == "__main__":
    main()
