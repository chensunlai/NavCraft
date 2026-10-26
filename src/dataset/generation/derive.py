"""Derive tasks from frozen real cuts, with public conditions and private labels."""

from collections import Counter
from dataclasses import asdict
import html
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

from src.common.io import read_json, write_json
from src.common.progress import Progress
from src.harness.geometry import Geometry
from src.harness.session import read_snapshot

from .records import (SceneRecord, Snapshot, TaskSample, load_records, save_records,
                      stable_id, update_manifest)
from .tasks import TASKS


def _context(root, snapshot, scene, task_id, index, directory):
    state = read_snapshot(root / snapshot.directory)
    workspace = Path(state["workspace"])
    observations = {row["frame"]: row for row in load_records(workspace / "observations/index.jsonl")}
    frame = state["current_frame"]
    row = observations[frame]
    semantic_file = root / snapshot.attributes["semantic_file"]
    semantic = np.load(semantic_file, allow_pickle=False)
    instances = read_json(root / scene.instances_path)["instances"]
    by_id = {item["id"]: item for item in instances}
    target = by_id[str(snapshot.attributes["target_instance"])]
    mask = semantic == target["semantic_id"]
    if mask.sum() < 500:
        raise ValueError("Target lacks sufficient actual semantic pixels")
    transform = read_json(root / scene.map_path)
    geometry = Geometry(workspace, state["private_dir"], transform)
    visible = [item for item in instances if np.count_nonzero(semantic == item["semantic_id"]) >= 500]
    memory, bindings, sources = [], {}, {}
    original = {item["id"]: item for item in read_json(workspace / "memory/entities.json")}
    previous_bindings = snapshot.attributes.get("bindings", {})
    by_instance = {instance: original[identifier] for identifier, instance in previous_bindings.items()
                   if identifier in original}
    used_file = Path(state["private_dir"]) / "memory_used_ids.json"
    used = set(read_json(used_file)) if used_file.exists() else set(original)
    next_number = max((int(identifier[1:]) for identifier in used if identifier.startswith("O")), default=0) + 1
    for instance in visible[:30]:
        region = semantic == instance["semantic_id"]
        y, x = np.unravel_index(np.argmax(distance_transform_edt(region)), region.shape)
        projected = geometry.resolve_frame(frame, [int(x), int(y)])
        if not projected.get("valid"):
            continue
        if instance["id"] in by_instance:
            entity = by_instance[instance["id"]].copy()
            entity["evidence"] = list(dict.fromkeys([*entity.get("evidence", []), frame]))
        else:
            identifier = f"O{next_number:02d}"
            next_number += 1
            entity = {"id": identifier, "type": "object", "name": instance["category"],
                      "category": instance["category"], "point": projected["uv"], "room": None,
                      "description": f"Observed {instance['category']}.", "evidence": [frame]}
        identifier = entity["id"]
        memory.append(entity)
        bindings[identifier] = instance["id"]
        sources[identifier] = {"frame": frame, "uv": entity["point"], "height": projected["height"]}
    target_id = next((key for key, value in bindings.items() if value == target["id"]), None)
    if target_id is None:
        raise ValueError("Target has no stable valid surface projection")
    public = directory / "public"
    public.mkdir(parents=True, exist_ok=True)
    with Image.open(workspace / row["path"]) as source:
        source = source.convert("RGB")
        y, x = np.where(mask)
        bounds = [int(x.min()), int(y.min()), int(x.max()), int(y.max())]
        ImageDraw.Draw(source).rectangle(bounds, outline="#ed3f4d", width=3)
        source.save(public / "q.jpg", quality=90)
    category_mask = np.zeros(semantic.shape, dtype=np.uint8)
    for item in instances:
        if item["category"] == target["category"]:
            category_mask[semantic == item["semantic_id"]] = 1
    private = directory / "private"
    private.mkdir(exist_ok=True)
    np.save(private / "category.npy", category_mask, allow_pickle=False)
    categories = sorted({item["category"] for item in instances})
    present = {item["category"] for item in instances if np.any(semantic == item["semantic_id"])}
    absent = next((category for category in categories if category not in present), None)
    comparison = []
    if task_id == "6c":
        if index % 2 == 0:
            duplicate_id = "O99"
            if duplicate_id in bindings:
                raise ValueError("Reserved comparison ID is already occupied")
            original = next(item for item in memory if item["id"] == target_id)
            memory.append({**original, "id": duplicate_id, "description": "Another view of the recorded object."})
            bindings[duplicate_id] = target["id"]
            sources[duplicate_id] = sources[target_id].copy()
            comparison = [target_id, duplicate_id]
        else:
            other = next((item for item in memory if item["id"] != target_id
                          and item["category"] == target["category"]), None)
            other = other or next((item for item in memory if item["id"] != target_id), None)
            if other is None:
                raise ValueError("Different-instance comparison needs a second observed object")
            comparison = [target_id, other["id"]]
    if task_id == "1b" or task_id == "6a" and index % 2:
        memory = [item for item in memory if item["id"] != target_id]
        bindings.pop(target_id)
        sources.pop(target_id)
    target_image = None
    if task_id == "5c":
        target_frame = snapshot.attributes.get("target_frame")
        reference_mask = snapshot.attributes.get("target_semantic_file")
        if not target_frame or not reference_mask:
            raise ValueError("Image goal needs a verified separate reference view")
        target_mask = np.load(root / reference_mask, allow_pickle=False) == target["semantic_id"]
        if target_mask.sum() < 500:
            raise ValueError("Reference view has insufficient target evidence")
        ys, xs = np.where(target_mask)
        reference = workspace / observations[target_frame]["path"]
        with Image.open(reference) as source:
            bounds = [max(0, int(xs.min()) - 24), max(0, int(ys.min()) - 24),
                      min(source.width, int(xs.max()) + 25), min(source.height, int(ys.max()) + 25)]
            source.crop(bounds).save(public / "reference.jpg", quality=90)
        target_image = (public / "reference.jpg").relative_to(root).as_posix()
    write_json(public / "memory.json", memory)
    return {"snapshot": snapshot, "target": target, "frame": frame, "memory": memory,
            "bindings": bindings, "location_sources": sources, "target_memory_id": target_id,
            "comparison_ids": comparison, "instances": instances, "transform": transform,
            "heading": row["pose"]["heading"], "absent_category": absent,
            "frame_visibility": snapshot.attributes["frame_visibility"],
            "semantic_file": semantic_file.relative_to(root).as_posix(),
            "category_mask": (private / "category.npy").relative_to(root).as_posix(),
            "boxed_image": (public / "q.jpg").relative_to(root).as_posix(),
            "target_image": target_image}


def derive(root, config, repo):
    root = Path(root).resolve()
    if (root / "samples.jsonl").exists():
        return load_records(root / "samples.jsonl", TaskSample)
    options = config.get("generation", {})
    tasks = [str(task) for task in options.get("tasks", TASKS)]
    if set(tasks) - set(TASKS):
        raise ValueError(f"Unsupported task templates: {sorted(set(tasks) - set(TASKS))}")
    scenes = {scene.id: scene for scene in load_records(root / "scenes.jsonl", SceneRecord)}
    snapshots = load_records(root / "snapshots.jsonl", Snapshot)
    rng = random.Random(options.get("seed", 0))
    # Interleave scenes so a short acceptance subset is not dominated by one scene.
    by_scene = {identifier: [cut for cut in snapshots if cut.scene_id == identifier] for identifier in scenes}
    for cuts in by_scene.values():
        rng.shuffle(cuts)
    ordered = [cuts[i] for i in range(max(map(len, by_scene.values()), default=0))
               for cuts in by_scene.values() if i < len(cuts)]
    wanted = int(options.get("per_task", 10))
    samples, rejected = [], []
    progress = Progress("Derive tasks", tasks)
    for task_id in tasks:
        count = 0
        for snapshot in ordered:
            if count >= wanted:
                break
            identifier = f"{task_id}-{stable_id(snapshot.id, task_id, count)}"
            directory = root / "samples" / identifier
            try:
                context = _context(root, snapshot, scenes[snapshot.scene_id], task_id, count, directory)
                generated = TASKS[task_id](task_id, context, count)
                label = generated.pop("label")
                label["location_sources"] = context["location_sources"]
                sample = TaskSample(identifier, task_id, generated["form"], generated["entry"],
                                    snapshot.id, generated["question"],
                                    {"images": generated["images"],
                                     "memory_file": (directory / "public/memory.json").relative_to(root).as_posix()},
                                    generated["limits"], (directory / "private/label.json").relative_to(root).as_posix(),
                                    snapshot.group_id, scenes[snapshot.scene_id].split,
                                    {"scene_id": snapshot.scene_id,
                                     "tool_tier": config.get("tool_tier", "super"), "source": snapshot.source})
                write_json(root / sample.label_file, label)
                write_json(directory / "task.json", asdict(sample))
                samples.append(sample)
                count += 1
            except (ValueError, KeyError, FileNotFoundError) as error:
                rejected.append({"task_id": task_id, "snapshot": snapshot.id, "reason": str(error)})
        progress.set(task_id, "finished" if count == wanted else "failed")
    write_json(root / "reports/derivation_rejections.json", rejected)
    counts = Counter(sample.task_id for sample in samples)
    if any(counts[task] != wanted for task in tasks):
        save_records(root / "reports/partial_samples.jsonl", samples)
        raise RuntimeError(f"Insufficient eligible cuts: {dict(counts)}; expected {wanted} per task")
    save_records(root / "samples.jsonl", samples)
    teacher_count = int(options.get("teacher_per_task", 2))
    selection = []
    for task in tasks:
        selection.extend([sample.id for sample in samples if sample.task_id == task][:teacher_count])
    write_json(root / "teacher_selection.json", selection)
    update_manifest(root, tasks=tasks, sample_counts=dict(counts), samples=len(samples),
                    teacher_selection=selection, tool_tier=config.get("tool_tier", "super"))
    render_gallery(root)
    return samples


def render_gallery(root):
    root = Path(root).resolve()
    snapshots = {cut.id: cut for cut in load_records(root / "snapshots.jsonl", Snapshot)}
    solutions = load_records(root / "solutions.jsonl") if (root / "solutions.jsonl").exists() else []
    cards = []
    for sample in load_records(root / "samples.jsonl", TaskSample):
        snapshot = snapshots[sample.snapshot]
        state = read_snapshot(root / snapshot.directory)
        index = {row["frame"]: row for row in load_records(Path(state["workspace"]) / "observations/index.jsonl")}
        current = (Path(state["workspace"]) / index[state["current_frame"]]["path"]).relative_to(root).as_posix()
        images = [current, *[(Path(state["workspace"]) / index[frame]["path"]).relative_to(root).as_posix()
                            for frame in state.get("target_frames", [])], *sample.inputs.get("images", [])]
        captions = [f"Initial RGB: {state['current_frame']}",
                    *[f"Active navigation target: {frame}" for frame in state.get("target_frames", [])],
                    *["Question / request attachment" for _ in sample.inputs.get("images", [])]]
        map_image = root / snapshot.directory / "review-map.png"
        if not map_image.exists():
            transform = read_json(Path(state["workspace"]) / "scene/map.json")
            image = Image.new("RGB", (720, 720), "#fcfaf5")
            draw = ImageDraw.Draw(image)
            origin, scale = transform["origin"], transform["size_m"]
            for segment in transform["walls"]:
                points = [tuple((point[i] - origin[i]) / scale * 680 + 20 for i in (0, 1)) for point in segment]
                draw.line(points, fill="#343b45", width=3)
            pose = index[state["current_frame"]]["pose"]
            x, y = [value * 680 + 20 for value in pose["pos"]]
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="#287cc9")
            angle = np.deg2rad(pose["heading"])
            draw.line((x, y, x + np.sin(angle) * 24, y - np.cos(angle) * 24), fill="#287cc9", width=4)
            image.save(map_image)
        images.append(map_image.relative_to(root).as_posix())
        captions.append("Geometry review; the Agent obtains its map through map_view")
        figures = "".join(f'<figure><a href="{html.escape(ref)}" target="_blank"><img src="{html.escape(ref)}" loading="lazy"></a>'
                          f'<figcaption>{html.escape(caption)}</figcaption></figure>' for ref, caption in zip(images, captions))
        attempts = [row for row in solutions if row["sample_id"] == sample.id and row.get("model_started")]
        teacher = ("Verified teacher demonstration" if any(row["score"]["success"] for row in attempts)
                   else "No verified demonstration" if attempts else "Not selected for teacher testing")
        label = read_json(root / sample.label_file)
        for key in ("instances", "location_sources", "frame_visibility"):
            label.pop(key, None)
        memory = read_json(root / sample.inputs["memory_file"])
        cards.append(f'<article data-task="{sample.task_id}" data-form="{sample.form}"><h2>{html.escape(sample.id)}</h2>'
                     f'<p>{sample.form} · {html.escape(sample.attributes["scene_id"])} · {sample.entry}</p>'
                     f'<p>{teacher} · {sample.split}</p>'
                     f'<details><summary>Active navigation request</summary><p>{html.escape(state.get("current_request", ""))}</p></details>'
                     f'<p>{html.escape(sample.question or "Continue the original navigation request without a new user message.")}</p>'
                     f'<div class="images">{figures}</div>'
                     f'<details><summary>Public initial memory</summary><pre>{html.escape(json.dumps(memory,ensure_ascii=False,indent=2))}</pre></details>'
                     f'<details><summary>Private verification label</summary><pre>{html.escape(json.dumps(label,ensure_ascii=False,indent=2))}</pre></details>'
                     f'<details><summary>Actual teacher outcomes</summary><pre>{html.escape(json.dumps(attempts,ensure_ascii=False,indent=2))}</pre></details></article>')
    options = "".join(f'<option>{task}</option>' for task in TASKS)
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Spatial task collection</title>'
            '<style>body{font:15px system-ui;background:#f7f7f8;color:#222;margin:0 auto;max-width:1150px;padding:28px}'
            'article{background:white;border:1px solid #ddd;border-radius:12px;padding:20px;margin:18px 0}h2{font-size:17px}'
            '.images{display:flex;gap:10px;flex-wrap:wrap}.images figure{margin:0}.images figcaption{font-size:12px;color:#666}'
            '.images img{max-width:500px;max-height:360px;border-radius:8px}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}select{padding:8px}</style>'
            f'<h1>Real spatial tasks</h1><p>Public inputs and private verification are shown separately. Teacher results are in solutions.jsonl.</p>'
            f'<select id="task"><option value="">All tasks</option>{options}</select><main>{"".join(cards)}</main>'
            '<script>document.querySelector("#task").onchange=e=>document.querySelectorAll("article").forEach(a=>a.hidden=!!e.target.value&&a.dataset.task!==e.target.value)</script></html>')
    (root / "index.html").write_text(page, encoding="utf-8")
