"""Prepare private instance labels from actual semantic Habitat scenes."""

import csv
import json
import math
from pathlib import Path
import random
import shutil
import struct

import numpy as np

from src.common.io import read_json, write_json
from src.common.progress import Progress
from src.dataset.episodes import EpisodeSpec
from src.dataset.maps import load_wall_map
from src.environment import Environment

from .records import SceneRecord, load_records, save_records, update_manifest


CATEGORIES = {"chair", "sofa", "bed", "table", "cabinet", "dresser", "shelf", "bookshelf",
              "plant", "toilet", "sink", "bathtub", "tv", "refrigerator", "oven", "desk"}
ALIASES = {"couch": "sofa", "potted plant": "plant", "television": "tv", "dining table": "table",
           "chest of drawers": "dresser", "bookcase": "bookshelf", "fridge": "refrigerator"}


def generation_options(config):
    return config.get("generation", {})


def environment_config(config, repo):
    values = config.get("environment", {})
    gpu = values.get("gpu", values.get("gpus", [0])[0])
    if type(gpu) is not int or gpu not in (0, 1):
        raise ValueError("Generation simulation is restricted to physical GPU 0 or 1")
    return {**values, "gpu": gpu, "tool_tier": config.get("tool_tier", "super"),
            "private_depth": True, "private_semantic": True,
            "scenes": str((Path(repo) / values.get("scenes", "data/nas/scene_datasets")).resolve())}


def episode_for(scene, pose=None):
    pose = pose or scene.start_pose
    return EpisodeSpec(id=f"generated:{scene.id}", source_id=scene.id, benchmark="generated",
                       scene_id=scene.scene_id, start_position=pose["position"],
                       start_rotation=pose["rotation_xyzw"], goals=[], source_file="generated",
                       provenance="script")


def open_environment(scene, config, repo, directory, pose=None):
    values = environment_config(config, repo)
    if scene.scene_dataset_config:
        values["scene_dataset_config"] = scene.scene_dataset_config
    return Environment(episode_for(scene, pose), values, Path(directory))


def semantic_instances(semantic_scene, floor_height, mesh_bounds):
    """Bind semantic pixel IDs to native object identities, never list indices."""
    rows = []
    for obj in semantic_scene.objects:
        if obj is None or obj.category is None:
            continue
        category = obj.category.name().strip().lower()
        category = ALIASES.get(category, category)
        if category not in CATEGORIES:
            continue
        semantic_id = getattr(obj, "semantic_id", None)
        if callable(semantic_id):
            semantic_id = semantic_id()
        if semantic_id is None:
            semantic_id = int(str(obj.id).rsplit("_", 1)[-1])
        if int(semantic_id) not in mesh_bounds:
            continue
        center = np.asarray(mesh_bounds[int(semantic_id)]["center"])
        size = np.asarray(mesh_bounds[int(semantic_id)]["size"])
        if center.shape != (3,) or size.shape != (3,) or not np.isfinite([center, size]).all():
            continue
        if not floor_height - .2 <= center[1] - size[1] / 2 <= floor_height + 1.8:
            continue
        rows.append({"id": str(obj.id), "semantic_id": int(semantic_id), "category": category,
                     "center": center.tolist(), "size": size.tolist()})
    identifiers = [row["semantic_id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Semantic pixel IDs are not unique")
    return rows


def semantic_mesh_bounds(mesh, descriptor):
    """Decode HM3D instance-coloured vertices; native AABBs can include origin."""
    colors = {}
    for row in csv.reader(Path(descriptor).read_text().splitlines()[1:]):
        if len(row) >= 3:
            category = ALIASES.get(row[2].strip().lower(), row[2].strip().lower())
            if category in CATEGORIES:
                colors[int(row[1], 16)] = int(row[0])
    if not colors:
        return {}
    with Path(mesh).open("rb") as stream:
        header = stream.read(12)
        if struct.unpack("<III", header)[:2] != (0x46546C67, 2):
            raise ValueError("Expected a binary glTF 2.0 semantic mesh")
        length, kind = struct.unpack("<II", stream.read(8))
        if kind != 0x4E4F534A:
            raise ValueError("Missing glTF JSON chunk")
        document = json.loads(stream.read(length))
        length, kind = struct.unpack("<II", stream.read(8))
        if kind != 0x004E4942:
            raise ValueError("Missing glTF binary chunk")
        binary = stream.read(length)
    if any(set(node) & {"matrix", "translation", "rotation", "scale"} for node in document.get("nodes", [])):
        raise ValueError("Transformed semantic nodes require an explicit coordinate adapter")
    types = {5121: np.dtype("u1"), 5123: np.dtype("<u2"), 5125: np.dtype("<u4"), 5126: np.dtype("<f4")}

    def accessor(index):
        row = document["accessors"][index]
        if row.get("sparse"):
            raise ValueError("Sparse semantic attributes are unsupported")
        view = document["bufferViews"][row["bufferView"]]
        dtype = types[row["componentType"]]
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[row["type"]]
        offset = view.get("byteOffset", 0) + row.get("byteOffset", 0)
        stride = view.get("byteStride", dtype.itemsize * width)
        return np.ndarray((row["count"], width), dtype=dtype, buffer=binary,
                          offset=offset, strides=(stride, dtype.itemsize))

    palette = np.asarray(sorted(colors), dtype=np.int64)
    lower = np.full((len(palette), 3), np.inf)
    upper = np.full((len(palette), 3), -np.inf)
    for mesh_row in document["meshes"]:
        for primitive in mesh_row["primitives"]:
            attributes = primitive["attributes"]
            positions, rgba = accessor(attributes["POSITION"]), accessor(attributes["COLOR_0"])
            rgb = rgba[:, :3].astype(float)
            if np.issubdtype(rgba.dtype, np.integer):
                rgb /= np.iinfo(rgba.dtype).max
            # glTF vertex colours are linear; HM3D descriptor hex values are sRGB.
            rgb = np.where(rgb <= .0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - .055)
            rgb = np.rint(rgb * 255).astype(np.int64)
            packed = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
            groups = np.searchsorted(palette, packed)
            valid = (groups < len(palette)) & (palette[np.minimum(groups, len(palette) - 1)] == packed)
            world = positions[valid][:, [0, 2, 1]].astype(float)
            world[:, 2] *= -1
            np.minimum.at(lower, groups[valid], world)
            np.maximum.at(upper, groups[valid], world)
    return {colors[int(color)]: {"center": ((lo + hi) / 2).tolist(), "size": (hi - lo).tolist()}
            for color, lo, hi in zip(palette, lower, upper) if np.isfinite([lo, hi]).all()}


def verify_bounds(observation, instances):
    """Independently compare decoded bounds against actual semantic/depth pixels."""
    from src.harness.geometry import Geometry
    from PIL import Image

    semantic = np.load(observation.semantic, allow_pickle=False)
    with Image.open(observation.depth) as image:
        depth = np.asarray(image, dtype=float) * observation.camera["depth_scale"]
    camera = {**observation.camera, "world_from_optical": observation.pose["world_from_optical"]}
    checks = []
    for row in instances:
        y, x = np.nonzero((semantic == row["semantic_id"]) & (depth > .01) & (depth < 10))
        if len(x) < 500:
            continue
        indices = np.linspace(0, len(x) - 1, min(2000, len(x)), dtype=int)
        points = Geometry._project(camera, np.column_stack([x[indices], y[indices]]), depth[y[indices], x[indices]])
        low = np.asarray(row["center"]) - np.asarray(row["size"]) / 2 - .05
        high = np.asarray(row["center"]) + np.asarray(row["size"]) / 2 + .05
        fraction = float(np.mean(((points >= low) & (points <= high)).all(axis=1)))
        checks.append({"instance": row["id"], "points": len(points), "inside_mesh_aabb_fraction": fraction})
    if not checks or any(row["inside_mesh_aabb_fraction"] < .95 for row in checks):
        raise ValueError(f"Semantic geometry/depth coordinate check failed: {checks}")
    return checks


def floor_points(pathfinder, height):
    vertices = np.asarray(pathfinder.build_navmesh_vertices(), dtype=float)
    indices = np.asarray(pathfinder.build_navmesh_vertex_indices()).reshape(-1, 3)
    triangles = vertices[indices]
    mask = (np.ptp(triangles[:, :, 1], axis=1) <= .08) & (np.abs(triangles[:, :, 1].mean(axis=1) - height) <= .35)
    triangles = triangles[mask]
    if not len(triangles):
        raise ValueError("No navigable triangles on the chosen floor")
    centers = triangles.mean(axis=1)
    # Deterministic spatial thinning keeps candidate inspection inexpensive.
    _, unique = np.unique(np.round(centers[:, [0, 2]] / .5).astype(int), axis=0, return_index=True)
    return centers[np.sort(unique)]


def _stage_assets(repo, directory, private, scenes_root):
    sid = directory.name.split("-", 1)[-1]
    original = scenes_root / "hm3d/train" / directory.name
    cached = repo / "cache/scene-selection/nonhssd-20261026/assets" / sid
    local = repo / "cache/generation/assets" / sid
    local.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, extension in (("render_asset", ".basis.glb"), ("semantic_asset", ".semantic.glb"),
                            ("semantic_descriptor_filename", ".semantic.txt"), ("nav_asset", ".basis.navmesh")):
        name_asset = sid + extension
        candidate = local / name_asset
        if candidate.is_file():
            files[name] = str(candidate.resolve())
            continue
        candidate = cached / name_asset
        if not candidate.is_file():
            candidate = original / name_asset
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        # Native GLB readers perform random reads; stage remote resources once.
        target = local / name_asset
        temporary = target.with_suffix(target.suffix + ".tmp")
        shutil.copyfile(candidate, temporary)
        temporary.replace(target)
        files[name] = str(target.resolve())
    private.mkdir(parents=True, exist_ok=True)
    stage = private / "scene.stage_config.json"
    write_json(stage, {**files, "up": [0, 0, 1], "front": [0, 1, 0], "semantic_up": [0, 0, 1],
                       "semantic_front": [0, 1, 0], "requires_lighting": False})
    return stage


def assign_splits(root, records, options):
    """Reserve complete scene groups before any teacher call."""
    count = int(options.get("dev_scene_count", 0))
    if not 0 <= count < len(records):
        raise ValueError("dev_scene_count must leave at least one training scene")
    for index, record in enumerate(records):
        record.split = "dev" if count and index >= len(records) - count else "train"
    save_records(Path(root) / "scenes.jsonl", records)
    update_manifest(root, source_scene_split="train", scene_splits={row.id: row.split for row in records})
    return records


def prepare(root, config, repo):
    """Freeze a seeded set only after real RGB/semantic rendering succeeds."""
    root, repo = Path(root).resolve(), Path(repo).resolve()
    output = root / "scenes.jsonl"
    options = generation_options(config)
    seed, count = int(options.get("seed", 0)), int(options.get("scene_count", 3))
    previous = load_records(output, SceneRecord) if output.exists() else []
    frozen_ids = [row.id for row in previous]
    records = [row for row in previous if (root / "private/scenes" / row.id / "source.json").exists()
               and read_json(root / "private/scenes" / row.id / "source.json").get("bounds_method") == "semantic_glb_vertices"]
    if len(records) == count:
        return assign_splits(root, records, options)
    maps = repo / options.get("maps_root", "data/scene_map_datasets") / "hm3d/train"
    candidates = sorted(p.parent for p in maps.glob("*/scene.json"))
    random.Random(seed).shuffle(candidates)
    if len(frozen_ids) == count:
        by_id = {directory.name: directory for directory in candidates}
        candidates = [by_id[identifier] for identifier in frozen_ids]
    rejected = []
    progress = Progress("Prepare semantic scenes", range(count), {i: "finished" for i in range(len(records))})
    for directory in candidates:
        if len(records) == count:
            break
        identifier = directory.name
        if identifier in {row.id for row in records}:
            continue
        private = root / "private/scenes" / identifier
        environment = None
        try:
            scene = read_json(directory / "scene.json")
            levels = scene.get("levels") or read_json(directory / "levels.json")
            scenes_root = Path(environment_config(config, repo)["scenes"])
            stage = _stage_assets(repo, directory, private, scenes_root)
            stage_data = read_json(stage)
            bounds = semantic_mesh_bounds(stage_data["semantic_asset"], stage_data["semantic_descriptor_filename"])
            def objects_on_level(level):
                height = float(level.get("floor_height", level.get("height_m")))
                return sum(height - .2 <= value["center"][1] - value["size"][1] / 2 <= height + 1.8
                           for value in bounds.values())
            level = max(levels, key=objects_on_level)
            height = float(level.get("floor_height", level.get("height_m")))
            initial = {"position": [0, height, 0], "rotation_xyzw": [0, 0, 0, 1]}
            record = SceneRecord(identifier, str(stage), "hm3d", "train", height,
                                 f"public/scenes/{identifier}/map.json", f"private/scenes/{identifier}/instances.json", initial)
            probe = repo / "cache/generation" / root.name / "scenes" / identifier / "probe"
            environment = open_environment(record, config, repo, probe)
            points = floor_points(environment.sim.sim.pathfinder, height)
            instances = semantic_instances(environment.sim.sim.semantic_scene, height, bounds)
            if len(instances) < 5:
                raise ValueError(f"Only {len(instances)} eligible semantic objects")
            by_semantic = {row["semantic_id"]: row for row in instances}
            best = None
            rng = np.random.default_rng(seed)
            for index in rng.permutation(len(points))[:30]:
                position = points[index].tolist()
                for yaw in (0, 90, 180, 270):
                    angle = -math.radians(yaw) / 2
                    rotation = [0, math.sin(angle), 0, math.cos(angle)]
                    pose = {"position": position, "rotation_xyzw": rotation,
                            "camera_position": [position[0], position[1] + .88, position[2]],
                            "camera_rotation_xyzw": rotation}
                    environment.sim.restore_pose(pose)
                    observation = environment.observe()
                    semantic = np.load(observation.semantic, allow_pickle=False)
                    ids, pixels = np.unique(semantic, return_counts=True)
                    visible = [by_semantic[int(i)] for i, n in zip(ids, pixels) if i in by_semantic and n >= 500]
                    if visible:
                        best = (environment.pose(), visible)
                        break
                if best:
                    break
            if best is None:
                raise ValueError("No verified RGB/semantic view of eligible objects")
            record.start_pose = best[0]
            coordinate_checks = verify_bounds(observation, instances)
            record.capabilities = ["rgb", "depth", "instances", "walls", "local_motion"]
            public_map, provenance = load_wall_map(directory, record.start_pose["position"])
            write_json(root / record.map_path, public_map)
            write_json(root / record.instances_path, {"scene": identifier, "floor_height": height, "instances": instances})
            write_json(private / "source.json", {"published_map": str(directory), "map_provenance": provenance,
                                                  "stage": stage_data, "semantic_probe": "passed",
                                                  "bounds_method": "semantic_glb_vertices", "coordinate_checks": coordinate_checks})
            records.append(record)
            progress.set(len(records) - 1, "finished")
            save_records(output, records)
        except Exception as error:
            rejected.append({"scene": identifier, "error": f"{type(error).__name__}: {error}"})
            write_json(root / "reports/scene_rejections.json", rejected)
        finally:
            if environment is not None:
                environment.close()
    if len(records) != count:
        raise RuntimeError(f"Prepared {len(records)}/{count} usable train scenes; see reports/scene_rejections.json")
    update_manifest(root, seed=seed, scene_ids=[row.id for row in records],
                    scene_split="train", engineering_acceptance=True)
    write_json(root / "reports/scene_rejections.json", rejected)
    return assign_splits(root, records, options)
