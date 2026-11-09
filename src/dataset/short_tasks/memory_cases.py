"""Freeze eight short memory cases from real observations, without model calls."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, distance_transform_edt

from src.common.coordinates import map_pose
from src.common.io import write_json
from src.dataset.episodes import EpisodeSpec
from src.harness.geometry import Geometry
from src.harness.memory import MemoryStore
from src.harness.observations import Observations


# Each row names a real cut, not a generated answer or a preferred tool sequence.
GROUND = [
    ("18d371f804cf4d64", "shelf_229"),
    ("04e818f30b07a7ae", "cabinet_112"),
    ("8a897e388e2b2abc", "tv_414"),
    ("3d5b99da778461bd", "chair_427"),
]
RECOGNIZE = [
    ("f12dc3d64f763b21", "04e818f30b07a7ae", "cabinet_112", "cabinet_112"),
    ("55eb200363930f66", "8a897e388e2b2abc", "tv_414", "tv_414"),
    ("0b637d13ace4aebd", "18d371f804cf4d64", "shelf_229", "shelf_223"),
    ("ad1de4f371950e47", "3d5b99da778461bd", "chair_425", "chair_426"),
]
SEED_DESCRIPTIONS = {
    "shelf_229": "The lower of the two wall shelves above the kitchen counter; the other shelf is above-left in the evidence view.",
    "chair_425": "Carved wooden dining chair at the foreground-left edge of the table in the evidence view, with its back facing the camera.",
}


def _read(path):
    return json.loads(Path(path).read_text())


def _sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def instance_mask(labels, instance, minimum=1000):
    """Reject small/absent targets before creating a public question image."""
    mask = labels == instance["semantic_id"]
    if mask.ndim != 2 or int(mask.sum()) < minimum:
        raise ValueError(f"Insufficient visible evidence for {instance['id']}")
    return mask


def interior_pixel(mask):
    """Use a pixel safely inside the instance only for private seed construction."""
    if not np.any(mask):
        raise ValueError("No visible instance pixels")
    y, x = np.unravel_index(np.argmax(distance_transform_edt(mask)), mask.shape)
    return [int(x), int(y)]


def _import_observation(source, snapshot, observations, private, *, current):
    """Import an observed frame with its original camera/depth; never target art."""
    directory = source / snapshot["directory"]
    state = _read(directory / "state.json")
    old_frame = snapshot["attributes"]["current_frame"]
    index = {row["frame"]: row for row in map(json.loads,
             (directory / "workspace/observations/index.jsonl").read_text().splitlines())}
    original = index[old_frame]
    if original["kind"] != "observation":
        raise ValueError("Seed evidence must be an actual scene observation")
    row = observations.add(directory / "workspace" / original["path"], kind="observation",
                           step=0 if current else None,
                           pose=map_pose(state["pose"], observations.transform))
    cameras = {r["frame"]: r for r in map(json.loads,
               (directory / "private/depth/index.jsonl").read_text().splitlines())}
    camera = deepcopy(cameras[old_frame])
    camera["frame"], camera["depth"] = row["frame"], row["frame"] + ".png"
    depth = private / "depth"
    depth.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(directory / "private/depth" / cameras[old_frame]["depth"], depth / camera["depth"])
    with (depth / "index.jsonl").open("a") as stream:
        stream.write(json.dumps(camera) + "\n")
    return row, state


def _episode(source, scene, state, identifier, repo):
    scene_name = scene["id"].split("-", 1)[1]
    canonical = f"hm3d/train/{scene['id']}/{scene_name}.basis.glb"
    asset = repo / "data/nas/scene_datasets" / canonical
    original = _read(source / "private/scenes" / scene["id"] / "source.json")["stage"]["render_asset"]
    digest = _sha256(asset)
    if digest != _sha256(original):
        raise ValueError("Frozen observations do not use the current canonical render asset")
    episode = EpisodeSpec(identifier, identifier, "short_task", canonical,
                          state["pose"]["position"], state["pose"]["rotation_xyzw"], [],
                          str(source / "snapshots.jsonl"), provenance="frozen_real_observation")
    return episode.to_dict(), digest


def _build_case(source, output, snapshots, scenes, identifier, current_id, target_id,
                previous_id=None, previous_instance=None):
    kind = "recognize" if previous_id else "ground"
    directory = output / "cases" / identifier
    directory.mkdir(parents=True, exist_ok=False)
    snapshot_dir = directory / "snapshot"
    workspace, private = snapshot_dir / "workspace", snapshot_dir / "private"
    for name in ("scene", "memory", "scratch"):
        (workspace / name).mkdir(parents=True, exist_ok=True)
    current = snapshots[current_id]
    scene = scenes[current["scene_id"]]
    source_map = source / current["directory"] / "workspace/scene/map.json"
    shutil.copyfile(source_map, workspace / "scene/map.json")
    transform = _read(source_map)
    observations = Observations(workspace, transform)
    instances = _read(source / scene["instances_path"])["instances"]
    indexed = {row["id"]: row for row in instances}
    masks, imported, evidence_visibility = {}, {}, {}
    if previous_id:
        previous = snapshots[previous_id]
        if previous["scene_id"] != current["scene_id"]:
            raise ValueError("Cross-view evidence must belong to the same scene")
        old, _ = _import_observation(source, previous, observations, private, current=False)
        imported[previous_id] = old
        masks[old["frame"]] = np.load(source / previous["attributes"]["semantic_file"])
        instance_mask(masks[old["frame"]], indexed[previous_instance])
    row, state = _import_observation(source, current, observations, private, current=True)
    imported[current_id] = row
    masks[row["frame"]] = np.load(source / current["attributes"]["semantic_file"])
    target_mask = instance_mask(masks[row["frame"]], indexed[target_id])
    for frame, mask in masks.items():
        with Image.open(workspace / observations.get(frame)["path"]) as image:
            if mask.shape != (image.height, image.width):
                raise ValueError("Semantic mask does not match the actual RGB dimensions")
        relative = f"private/semantics/{frame}.npy"
        (directory / relative).parent.mkdir(parents=True, exist_ok=True)
        np.save(directory / relative, mask)
        evidence_visibility[frame] = [item["id"] for item in instances
                                      if int(np.count_nonzero(mask == item["semantic_id"])) >= 100]
    write_json(workspace / "memory/entities.json", [])
    memory = MemoryStore(workspace, private)
    baseline, seeded = {}, {}
    if previous_id:
        old = imported[previous_id]
        if old["sha256"] == row["sha256"] or old["pose"] == row["pose"]:
            raise ValueError("Cross-view task needs different real images and camera poses")
        old_mask = instance_mask(masks[old["frame"]], indexed[previous_instance])
        pixel = interior_pixel(old_mask)
        projected = Geometry(workspace, private, transform).resolve_frame(old["frame"], pixel)
        if not projected["valid"]:
            raise ValueError("Seed memory cannot project its visible evidence")
        category = indexed[previous_instance]["category"]
        edit = memory.edit([{"op": "create", "id": "O01", "type": "object", "fields": {
            "name": category.replace("_", " "), "category": category,
            "point": projected["uv"], "description": SEED_DESCRIPTIONS.get(previous_instance,
                "Previously observed; compare the evidence image before adding another record."),
            "evidence": [old["frame"]]}}])
        if "error" in edit:
            raise ValueError(edit["error"])
        baseline = {"O01": previous_instance}
        seeded = {"O01": {"frame": old["frame"], "uv": projected["uv"], "height": projected["height"]}}
        write_json(private / "location_sources.json", seeded)
    write_json(private / "memory_used_ids.json", list(baseline))
    with Image.open(workspace / row["path"]) as original:
        question = original.convert("RGB")
    ys, xs = np.where(target_mask)
    box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    # Nearby chairs overlap in an axis-aligned box; a silhouette identifies the
    # intended visible instance without highlighting the table between its legs.
    outline = kind == "recognize" and indexed[target_id]["category"] == "chair"
    if outline:
        pixels = np.array(question)
        pixels[target_mask & ~binary_erosion(target_mask, iterations=2)] = [227, 39, 39]
        question = Image.fromarray(pixels)
    else:
        ImageDraw.Draw(question).rectangle(box, outline="#e32727", width=3)
    marking = "red outline" if outline else "red box"
    (directory / "public").mkdir()
    question.save(directory / "public/question.jpg", quality=90)
    instruction = (
        f"Record the object marked by the {marking} in spatial memory, including its location, a brief identifying description, "
        f"and supporting evidence. The marking is shown on a copy of current observation {row['frame']}; "
        "use the original observation for image selection and evidence. Inspect the saved result and correct it if needed, then submit."
        if kind == "ground" else
        f"Check the object marked by the {marking} against the existing spatial-memory records. Reuse its record if already known; "
        f"otherwise record the new object. Add useful evidence without duplicating or merging distinct objects, then submit. "
        f"The marking is shown on a copy of current observation {row['frame']}; use the original observation as evidence."
    )
    environment = deepcopy(state["environment"])
    environment.update(steps=0, path_length=0.0, frame=len(observations.items),
                       trajectory=[state["pose"]["position"]], motion_trajectory=[state["pose"]["position"]],
                       trajectory_breaks=[], motions=[])
    write_json(snapshot_dir / "state.json", {
        "pose": state["pose"], "current_frame": row["frame"], "workspace": "workspace",
        "private_dir": "private", "history": [], "current_request": "", "target_frames": [],
        "submitted_requests": [], "observations_count": 0, "tool_receipts": {}, "environment": environment,
    })
    repo = Path(__file__).resolve().parents[3]
    episode, asset_hash = _episode(source, scene, state, identifier, repo)
    write_json(directory / "private/episode.json", episode)
    visible = set().union(*map(set, evidence_visibility.values()))
    expected = "O01" if previous_instance == target_id else None
    label = {"target": indexed[target_id], "instances": [e for e in instances if e["id"] in visible],
             "expected_existing_id": expected, "baseline_bindings": baseline,
             "evidence_masks": {frame: f"private/semantics/{frame}.npy" for frame in masks},
             "current_frame": row["frame"], "position_tolerance_m": .3, "require_evidence": True,
             "transform": transform, "target_box": box, "frame_visibility": evidence_visibility}
    write_json(directory / "private/label.json", label)
    task = {"id": identifier, "kind": kind, "instruction": instruction,
            "images": ["public/question.jpg"], "snapshot": "snapshot", "label": "private/label.json",
            "episode": "private/episode.json", "limits": {"max_requests": 10, "max_actions": 12,
                "max_path_length": 3, "max_observations": 2, "seconds": 600},
            "provenance": {"source_snapshot": current_id, "previous_snapshot": previous_id,
                "group_id": current["group_id"], "scene_id": scene["id"],
                "memory_source": "program_seeded_from_prior_observed_evidence" if baseline else "empty",
                "seed_description_source": "agent_visual_review" if previous_instance in SEED_DESCRIPTIONS else "generic",
                "source_collection": str(source), "map_sha256": _sha256(source_map)}}
    write_json(directory / "task.json", task)
    write_json(directory / "construction.json", {
        "status": "constructed", "source": "frozen_real_observations_no_model_demonstration",
        "checks": {"images_and_masks_aligned": True, "target_pixels": int(target_mask.sum()),
                   "cross_view": bool(previous_id), "empty_history": True, "empty_scratch": True,
                   "no_target_reference_leak": True, "asset_hash_matches": True},
        "asset_sha256": asset_hash, "seeded_projection_sources": seeded,
        "visual_review": "pending", "current_simulator_restore": "pending",
        "note": "Prior evidence is an independent real view; no traversed connection is claimed."})
    return {**task, "path": f"cases/{identifier}/task.json"}


def build_memory_cases(source: Path, output: Path, per_kind=4, on_case=None):
    """Build deterministic pilot cases; case directories must be new."""
    if not 1 <= per_kind <= 4:
        raise ValueError("This pilot has 1-4 frozen choices per kind")
    source, output = Path(source).resolve(), Path(output).resolve()
    snapshots = {row["id"]: row for row in map(json.loads, (source / "snapshots.jsonl").read_text().splitlines())}
    scenes = {row["id"]: row for row in map(json.loads, (source / "scenes.jsonl").read_text().splitlines())}
    result = []
    for index, (current, target) in enumerate(GROUND[:per_kind], 1):
        task = _build_case(source, output, snapshots, scenes, f"ground-{index:02d}", current, target)
        result.append(task)
        if on_case:
            on_case(task)
    for index, (old, current, previous, target) in enumerate(RECOGNIZE[:per_kind], 1):
        task = _build_case(source, output, snapshots, scenes, f"recognize-{index:02d}",
                           current, target, old, previous)
        result.append(task)
        if on_case:
            on_case(task)
    return result
