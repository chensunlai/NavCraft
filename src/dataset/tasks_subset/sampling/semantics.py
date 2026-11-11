"""Private HM3D instance evidence rendered at immutable observation poses."""
from collections import Counter
from copy import deepcopy
import csv
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image

from ..bundle import write_json
from .scene import load_scene


def _digest(path):
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def descriptor_ids(path):
    with path.open() as stream:
        return {int(row[0]): row[1] for row in csv.reader(stream)
                if len(row) >= 2 and row[0].isdigit()}


def visibility(pixels, *, valid, associated, complete, threshold):
    if not valid or not associated:
        return "unknown"
    if pixels >= threshold:
        return "visible"
    return "absent" if pixels == 0 and complete else "unknown"


def _object_pixels(mask, objects, descriptor, owners):
    present = set(np.unique(mask))
    rows = {}
    for identifier, mapping in objects.items():
        ambiguous = any(owners[i] > 1 for i in mapping)
        valid = bool(mapping) and not ambiguous and all(i in descriptor for i in mapping)
        matched = present.intersection(mapping) if valid else set()
        ys, xs = np.nonzero(np.isin(mask, list(matched))) if matched else ([], [])
        rows[identifier] = {
            "mapping_valid": valid, "mapping_ambiguous": ambiguous, "mapped_ids": mapping, "pixels": len(xs),
            "bbox": [int(min(xs)), int(min(ys)), int(max(xs))+1, int(max(ys))+1] if len(xs) else None,
        }
    return rows


def link_native_identity(bundle, root):
    """Associate reviewed foreground pixels across views using unique native IDs."""
    root = Path(root)
    groups = {}
    for sight in bundle["sightings"]:
        if sight.pop("native_identity", None):
            sight["identity_verified"] = False
    capability = bundle.get("instance_capability", {})
    if not capability.get("available"):
        return bundle
    source = bundle["source"]
    try:
        if any(_digest(Path(capability["stage"][field])) != expected
               for field, expected in capability["asset_sha256"].items()):
            return bundle
        if _digest(Path(source["map_directory"])/"objects.json") != capability["object_map_sha256"]:
            return bundle
        native_map = {obj["id"]: obj["semantic_ids"] for obj in load_scene(Path(source["map_directory"]))["objects"]}
    except (OSError, ValueError, KeyError):
        return bundle
    owners = Counter(i for mapped in native_map.values() for i in set(mapped))
    validated = {}
    for sight in bundle["sightings"]:
        key = sight["frame"]
        row = bundle.get("instance_evidence", {}).get(key, {})
        frame = bundle["frames"][key]
        obj = row.get("objects", {}).get(sight["id"], {})
        binding = sight.get("review_source", {})
        mapped = native_map.get(sight["id"], [])
        if (not row.get("valid") or not obj.get("mapping_valid") or obj.get("mapping_ambiguous")
                or obj.get("mapped_ids") != mapped or not mapped or any(owners[i] != 1 for i in mapped)
                or not sight.get("semantic_verified")
                or sight.get("review_candidate") not in obj.get("association_reviews", [])
                or binding.get("frame") != key or binding.get("sha256") != row.get("source_rgb_sha256")
                or binding.get("pixel") != sight.get("pixel") or binding.get("world") != sight.get("world")
                or row.get("pose") != frame.get("pose") or row.get("camera") != frame.get("camera")):
            continue
        if key not in validated:
            try:
                valid = (_digest(root/frame["rgb"]) == row["source_rgb_sha256"] and
                         _digest(root/row["mask"]) == row["mask_sha256"])
                validated[key] = np.load(root/row["mask"], allow_pickle=False, mmap_mode="r") if valid else None
            except (OSError, ValueError, KeyError):
                validated[key] = None
        mask = validated[key]
        if mask is None:
            continue
        x,y = sight["pixel"]
        if not 0 <= y < mask.shape[0] or not 0 <= x < mask.shape[1] or int(mask[y,x]) not in mapped:
            continue
        groups.setdefault((sight["id"],int(mask[y,x])), []).append(sight)
    for (identifier, native), sightings in groups.items():
        for sight in sightings:
            pose = bundle["frames"][sight["frame"]]["pose"]["camera_position"]
            others = [s for s in sightings if s["frame"] != sight["frame"] and
                      np.linalg.norm(np.asarray(bundle["frames"][s["frame"]]["pose"]["camera_position"])-pose) >= .2]
            if others:
                sight.update(identity_verified=True, native_identity={
                    "descriptor_id": native, "object": identifier,
                    "review_candidates": [sight["review_candidate"], *[s["review_candidate"] for s in others]],
                    "method": "reviewed_unique_native_instance_pixels"})
    return bundle


def collect_instance_evidence(bundle, root, config, gpu, frame_keys=None):
    root = Path(root).resolve()
    result = deepcopy(bundle)
    evidence = result.setdefault("instance_evidence", {})
    asset = (Path(config["assets"]) / bundle["scene_id"]).resolve()
    if not bundle["scene_id"].startswith("hm3d/"):
        result["instance_capability"] = {"available": False, "reason": "unsupported_dataset"}
        return result
    prefix = str(asset).removesuffix(".basis.glb")
    assets = {"render_asset": asset, "semantic_asset": Path(prefix + ".semantic.glb"),
              "semantic_descriptor_filename": Path(prefix + ".semantic.txt"),
              "nav_asset": Path(prefix + ".basis.navmesh")}
    if not all(p.is_file() for p in assets.values()):
        result["instance_capability"] = {"available": False, "reason": "missing_assets"}
        return result
    if type(gpu) is not int or gpu not in (0, 1):
        raise ValueError("Semantic render requires physical GPU 0 or 1")
    prefix_keys = {key for state in bundle["states"] for key in state["frame_keys"]}
    keys = list(dict.fromkeys(frame_keys)) if frame_keys is not None else sorted(prefix_keys)
    if not keys or not set(keys) <= prefix_keys:
        raise ValueError("Semantic frames must belong to a saved observation prefix")
    keys = [k for k in keys if not evidence.get(k, {}).get("valid")]
    threshold = config.get("semantics", {}).get("min_pixels", 100)
    if type(threshold) is not int or threshold <= 0:
        raise ValueError("semantics.min_pixels must be a positive integer")
    directory = root / "private/semantics"
    directory.mkdir(parents=True, exist_ok=True)
    stage = {k: str(v) for k, v in assets.items()} | {"up": [0, 0, 1], "front": [0, 1, 0]}
    stage_path = directory / "scene.stage_config.json"
    write_json(stage_path, stage)
    ids = descriptor_ids(assets["semantic_descriptor_filename"])
    source = bundle["source"]
    map_root = Path(source["map_directory"])
    if _digest(map_root / "objects.json") != source["sha256"]["objects.json"]:
        raise ValueError("Semantic associations require unchanged source objects")
    source_objects = {o["id"]: o for o in load_scene(map_root)["objects"]}
    for obj in bundle["scene"]["objects"]:
        if any(obj[field] != source_objects[obj["id"]][field]
               for field in ("name", "center_xyz", "size_xyz", "rectangles_xz")):
            raise ValueError("Bundle object geometry differs from the source map")
    objects = {o["id"]: source_objects[o["id"]]["semantic_ids"] for o in bundle["scene"]["objects"]}
    owners = Counter(i for obj in source_objects.values() for i in set(obj["semantic_ids"]))
    result["instance_capability"] = {"available": True, "backend": "habitat_native_hm3d",
        "asset_sha256": {k: _digest(v) for k, v in assets.items()}, "stage": stage,
        "descriptor_ids": {str(k): v for k, v in ids.items()}, "object_map_sha256": source["sha256"]["objects.json"],
        "rgb_tolerance": {"fraction_pixels_above_2": .001, "mean_channel_error": .02}, "min_pixels": threshold}
    if any(row["valid"] for row in evidence.values()):
        if bundle.get("instance_capability") != result["instance_capability"]:
            raise ValueError("Cached instance evidence configuration or assets changed")
        for key, row in evidence.items():
            if not row["valid"]:
                continue
            frame = bundle["frames"][key]
            if (key not in prefix_keys or row["camera"] != frame["camera"] or row["pose"] != frame["pose"]
                    or row["source_rgb_sha256"] != _digest(root / frame["rgb"])
                    or row["mask_sha256"] != _digest(root / row["mask"])):
                raise ValueError(f"Cached instance evidence changed: {key}")
            mask = np.load(root / row["mask"], allow_pickle=False, mmap_mode="r")
            row["objects"] = _object_pixels(mask, objects, ids, owners)
    if keys:
        import habitat_sim as h
        from habitat_sim.utils.common import quat_from_coeffs
        native = h.SimulatorConfiguration()
        native.scene_id, native.gpu_device_id = str(stage_path), gpu
        agent = h.agent.AgentConfiguration()
        camera = bundle["frames"][keys[0]]["camera"]
        shape = [camera["height"], camera["width"]]
        for name, kind in (("rgb", h.SensorType.COLOR), ("semantic", h.SensorType.SEMANTIC)):
            sensor = h.CameraSensorSpec()
            sensor.uuid, sensor.sensor_type = name, kind
            sensor.resolution, sensor.hfov = shape, camera["hfov_degrees"]
            agent.sensor_specifications.append(sensor)
        with h.Simulator(h.Configuration(native, [agent])) as sim:
            actor = sim.get_agent(0)
            for key in keys:
                frame = bundle["frames"][key]
                pose = frame["pose"]
                rgb_path = root / frame["rgb"]
                row = evidence[key] = {"source_rgb_sha256": _digest(rgb_path), "camera": frame["camera"],
                                       "pose": pose, "valid": False, "objects": {}}
                if frame["camera"] != camera:
                    row["reason"] = "camera_intrinsics_changed"
                    continue
                state = actor.get_state()
                state.position = np.asarray(pose["position"])
                state.rotation = quat_from_coeffs(pose["rotation_xyzw"])
                for sensor in state.sensor_states.values():
                    sensor.position = np.asarray(pose["camera_position"])
                    sensor.rotation = quat_from_coeffs(pose["camera_rotation_xyzw"])
                actor.set_state(state, reset_sensors=False, infer_sensor_states=False)
                capture = sim.get_sensor_observations()
                with Image.open(rgb_path) as image:
                    delta = np.abs(np.asarray(image.convert("RGB"), dtype=np.int16) - capture["rgb"][..., :3])
                    row["rgb_max_error"] = int(delta.max())
                    row["rgb_mean_error"] = float(delta.mean())
                    row["rgb_fraction_pixels_above_2"] = float((delta.max(axis=2) > 2).mean())
                    aligned = row["rgb_fraction_pixels_above_2"] <= .001 and row["rgb_mean_error"] <= .02
                mask = capture["semantic"]
                observed = set(np.unique(mask).tolist())
                if not aligned or len(observed & ids.keys()) < 2:
                    row["reason"] = "rgb_mismatch" if not aligned else "unusable_sensor"
                    continue
                path = directory / (sha256(key.encode()).hexdigest()[:20] + ".npy")
                np.save(path, mask, allow_pickle=False)
                row.update(valid=True, mask=str(path.relative_to(root)), mask_sha256=_digest(path))
                row["objects"] = _object_pixels(mask, objects, ids, owners)
    associated = {}
    for sight in bundle["sightings"]:
        key = sight["frame"]
        row = evidence.get(key, {})
        binding = sight.get("review_source", {})
        if (not row.get("valid") or sight.get("semantic_verified") is not True
                or not sight.get("review_candidate") or binding.get("sha256") != row["source_rgb_sha256"]
                or not row["objects"].get(sight["id"], {}).get("mapping_valid") or binding.get("frame") != key
                or binding.get("pixel") != sight.get("pixel") or binding.get("world") != sight.get("world")):
            continue
        x, y = sight["pixel"]
        mask = np.load(root / row["mask"], allow_pickle=False, mmap_mode="r")
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and int(mask[y, x]) in objects[sight["id"]]:
            associated.setdefault(sight["id"], set()).add(sight["review_candidate"])
    bindings_path = root / "private/instance_bindings.json"
    from src.common.io import read_json
    bindings = read_json(bindings_path) if bindings_path.exists() else {}
    for identifier, binding in bindings.items():
        frames = binding.get("frames", [])
        if (identifier not in objects or binding.get("ids") != objects[identifier]
                or binding.get("objects_sha256") != source["sha256"]["objects.json"]
                or binding.get("descriptor_sha256") != result["instance_capability"]["asset_sha256"]["semantic_descriptor_filename"]
                or not binding.get("scope") or not binding.get("reviewer") or not binding.get("method") or not frames
                or not all(k in evidence and evidence[k]["valid"] and evidence[k]["source_rgb_sha256"] == digest
                           for k, digest in frames.items())):
            raise ValueError(f"Unbound instance completeness declaration: {identifier}")
    for row in evidence.values():
        for identifier, record in row["objects"].items():
            reviews = sorted(associated.get(identifier, ()))
            complete = identifier in bindings
            record.update(association_verified=bool(reviews), association_reviews=reviews,
                          mapping_complete=complete, completeness_binding=bindings.get(identifier),
                          visibility=visibility(record["pixels"], valid=record["mapping_valid"],
                                                associated=bool(reviews), complete=complete, threshold=threshold))
    return link_native_identity(result, root)


def derive_instance_sightings(bundle, root):
    """Reproject a reviewed static surface anchor, with native and depth checks."""
    from .geometry import read_depth, visible_point
    root = Path(root)
    clean = deepcopy(bundle)
    clean["sightings"] = [s for s in clean["sightings"] if s.get("method") != "native_instance_reprojection"]
    if not clean.get("instance_capability", {}).get("available"):
        return []
    checked = link_native_identity(clean, root)
    skipped = Counter()
    bundle["instance_reprojection_report"] = {"skipped": skipped}
    seeds = [s for s in checked["sightings"] if s.get("native_identity") and s.get("semantic_verified")]
    if not seeds:
        skipped["no_valid_reviewed_native_seed"] += 1
    rows = []
    masks, depths = {}, {}
    prefix = {k for state in checked["states"] for k in state["frame_keys"]}
    for seed in seeds:
        native = seed["native_identity"]["descriptor_id"]
        for key, evidence in checked["instance_evidence"].items():
            frame = checked["frames"][key]
            if key not in prefix or key == seed["frame"]:
                continue
            if not evidence.get("valid") or not frame.get("depth"):
                skipped["missing_valid_cache_or_depth"] += 1
                continue
            if (evidence.get("camera") != frame.get("camera") or evidence.get("pose") != frame.get("pose")):
                skipped["camera_or_pose_changed"] += 1
                continue
            try:
                valid = (evidence.get("source_rgb_sha256") == _digest(root/frame["rgb"]) and
                         evidence.get("mask_sha256") == _digest(root/evidence["mask"]))
            except (OSError, KeyError):
                valid = False
            if not valid:
                skipped["rgb_or_mask_changed"] += 1
                continue
            record = evidence["objects"].get(seed["id"], {})
            if not record.get("mapping_valid") or native not in record.get("mapped_ids", []):
                skipped["invalid_mapping"] += 1
                continue
            if key not in masks:
                masks[key] = np.load(root/evidence["mask"],allow_pickle=False,mmap_mode="r")
                depths[key] = read_depth(root/frame["depth"],frame["camera"])
            if np.count_nonzero(masks[key] == native) < checked["instance_capability"]["min_pixels"]:
                skipped["insufficient_native_pixels"] += 1
                continue
            pixel = visible_point(seed["world"],depths[key],frame["camera"],frame["world_from_optical"],tolerance=.045)
            if pixel is None:
                continue
            x,y = pixel
            if int(masks[key][y,x]) != native:
                continue
            rows.append({"kind":"object","id":seed["id"],"frame":key,"pixel":pixel,
                "world":seed["world"],"method":"native_instance_reprojection",
                "verified":True,"geometry_verified":True,"identity_verified":True,
                "derived_from_review":seed["review_candidate"],"source_frame":seed["frame"],
                "semantic_verified":False,
                "native_identity":{"descriptor_id":native,"object":seed["id"],
                    "method":"depth_consistent_reprojection_from_reviewed_native_surface"},
                "evidence_binding":{"rgb_sha256":evidence["source_rgb_sha256"],
                    "mask_sha256":evidence["mask_sha256"],"depth_sha256":_digest(root/frame["depth"]),
                    "camera":frame["camera"],"pose":frame["pose"]}})
    return rows
