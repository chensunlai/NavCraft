"""Build private Qa target approach samples from semantic mesh and real renders.

This is a finite, declared-grid custom protocol, not an official benchmark goal set.
Run with the Habitat-enabled Python environment; never expose this file to an agent.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import struct

import numpy as np


TARGETS = {
    "O099": ("浴缸", "R03", [99]),
    "O175": ("餐桌", "R05", [175]),
    "O204": ("备餐区水槽", "R06", [204]),
    "O243": ("独立卫生间洗手盆", "R07", [243]),
    "O108": ("主卫双台盆洗手台", "R03", [108, 109, 110]),
    "O188": ("客厅方形木茶几", "R05", [188]),
}


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def semantic_geometry(mesh, annotations, identifiers):
    """Decode HM3D's linear vertex colors; compute bounds without origin padding."""
    raw = Path(mesh).read_bytes()
    magic, version, length = struct.unpack_from("<III", raw)
    if (magic, version, length) != (0x46546C67, 2, len(raw)):
        raise ValueError("Expected a complete GLB 2 file")
    size, kind = struct.unpack_from("<II", raw, 12)
    if kind != 0x4E4F534A:
        raise ValueError("GLB JSON chunk missing")
    doc = json.loads(raw[20:20 + size])
    binary_size, binary_kind = struct.unpack_from("<II", raw, 20 + size)
    if binary_kind != 0x004E4942:
        raise ValueError("GLB binary chunk missing")
    binary = raw[28 + size:28 + size + binary_size]
    for node in doc["nodes"]:
        if any(key in node for key in ("matrix", "translation", "rotation", "scale")):
            raise ValueError("Transformed nodes require an explicit geometry adapter")

    def accessor(index):
        a = doc["accessors"][index]
        if "sparse" in a:
            raise ValueError("Sparse accessors are not supported")
        view = doc["bufferViews"][a["bufferView"]]
        types = {5123: "<u2", 5125: "<u4", 5126: "<f4", 5121: "u1"}
        dtype = np.dtype(types[a["componentType"]])
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
        offset = view.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = view.get("byteStride", width * dtype.itemsize)
        values = np.ndarray((a["count"], width), dtype, binary, offset,
                            strides=(stride, dtype.itemsize)).copy()
        if a.get("normalized"):
            values = values.astype(np.float64) / np.iinfo(dtype).max
        return values

    labels = {}
    with Path(annotations).open() as stream:
        for row in csv.reader(stream):
            if row and row[0].isdigit():
                labels[int(row[0])] = {"color": int(row[1], 16), "label": row[2],
                                      "semantic_region_id": int(row[3])}
    selected = {identifier: [] for identifier in identifiers}
    for mesh_record in doc["meshes"]:
        for primitive in mesh_record["primitives"]:
            attributes = primitive["attributes"]
            positions = accessor(attributes["POSITION"])
            linear = accessor(attributes["COLOR_0"])[:, :3]
            srgb = np.where(linear <= .0031308, 12.92 * linear,
                            1.055 * linear ** (1 / 2.4) - .055)
            colors = np.clip(np.rint(srgb * 255), 0, 255).astype(np.uint32)
            codes = (colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]
            for identifier in identifiers:
                points = positions[codes == labels[identifier]["color"]]
                if len(points):
                    # HM3D asset is Z-up. Habitat stage uses X, Z, -Y.
                    world = points[:, [0, 2, 1]].copy()
                    world[:, 2] *= -1
                    selected[identifier].append(world)
    result = {}
    for identifier, parts in selected.items():
        if not parts:
            raise ValueError(f"No decoded semantic vertices for {identifier}")
        points = np.concatenate(parts)
        low, high = points.min(axis=0), points.max(axis=0)
        result[identifier] = {
            "semantic_instance_id": identifier, **labels[identifier],
            "aabb_min_xyz": low.tolist(), "aabb_max_xyz": high.tolist(),
            "center_xyz": ((low + high) / 2).tolist(),
            "aim_xyz": np.median(points, axis=0).tolist(),
            "decoded_vertex_count": len(points),
        }
    return result


def horizontal_distance(position, bounds):
    point = np.asarray(position)[[0, 2]]
    low = np.asarray(bounds["aabb_min_xyz"])[[0, 2]]
    high = np.asarray(bounds["aabb_max_xyz"])[[0, 2]]
    return float(np.linalg.norm(np.maximum(np.maximum(low - point, point - high), 0)))


def validate_grid(pathfinder, points):
    """Fail before rendering if the cached grid and configured NavMesh disagree."""
    expected = {"agent_height": .88, "agent_radius": .18, "cell_height": .05,
                "agent_max_climb": .2}
    actual = {key: float(getattr(pathfinder.nav_mesh_settings, key)) for key in expected}
    for key, value in expected.items():
        if not math.isclose(actual[key], value, abs_tol=1e-6):
            raise ValueError(f"NavMesh {key}={actual[key]} disagrees with declared {value}")
    if len(points) == 0 or np.asarray(points).shape[1:] != (3,):
        raise ValueError("Expected a nonempty XYZ candidate grid")
    maximum_error = 0.
    for index, point in enumerate(points):
        snapped = np.asarray(pathfinder.snap_point(point))
        error = float(np.linalg.norm(snapped - point))
        if not math.isfinite(error) or error > 1e-4:
            raise ValueError(f"Grid point {index} is not on this NavMesh: snap error {error}")
        maximum_error = max(maximum_error, error)
    return {"navmesh_settings": actual, "grid_points_checked": len(points),
            "maximum_snap_error_m": maximum_error, "snap_tolerance_m": 1e-4}


def generate(args):
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                          "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
    import habitat_sim
    import magnum as mn
    import quaternion
    from habitat_sim.utils.common import quat_to_coeffs
    from PIL import Image, ImageDraw
    from shapely.geometry import Point, Polygon

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence = output.parent / "evidence"
    evidence.mkdir(exist_ok=True)
    scene = Path(args.scene)
    mesh = scene.with_name("QaLdnwvtxbs.semantic.glb")
    annotations = scene.with_name("QaLdnwvtxbs.semantic.txt")
    identifiers = sorted({i for _, _, ids in TARGETS.values() for i in ids})
    geometry = semantic_geometry(mesh, annotations, identifiers)
    native = habitat_sim.SimulatorConfiguration()
    native.scene_id = str(scene.resolve())
    native.scene_dataset_config_file = str(scene.parent.parent / "hm3d_annotated_val_basis.scene_dataset_config.json")
    native.gpu_device_id = args.gpu
    agent = habitat_sim.agent.AgentConfiguration()
    agent.height, agent.radius = .88, .18
    agent.sensor_specifications = []
    for name, kind in [("rgb", habitat_sim.SensorType.COLOR),
                       ("semantic", habitat_sim.SensorType.SEMANTIC)]:
        sensor = habitat_sim.CameraSensorSpec()
        sensor.uuid, sensor.sensor_type = name, kind
        sensor.resolution, sensor.hfov = [480, 640], 79
        sensor.position = mn.Vector3(0, .88, 0)
        agent.sensor_specifications.append(sensor)
    grid = np.load(args.grid)
    points = grid["points"]
    rooms = {r["id"]: Polygon(r["polygon"]) for r in json.loads(Path(args.rooms).read_text())["rooms"]}
    simulator = habitat_sim.Simulator(habitat_sim.Configuration(native, [agent]))

    def observe(position, aim):
        eye = np.asarray(position) + [0, .88, 0]
        delta = np.asarray(aim) - eye
        yaw = math.atan2(-delta[0], -delta[2])
        pitch = math.atan2(delta[1], np.hypot(delta[0], delta[2]))
        body = quaternion.from_rotation_vector([0, yaw, 0])
        camera = body * quaternion.from_rotation_vector([pitch, 0, 0])
        state = habitat_sim.AgentState()
        state.position, state.rotation = np.asarray(position), body
        simulator.get_agent(0).set_state(state)
        state = simulator.get_agent(0).get_state()
        for sensor in state.sensor_states.values():
            sensor.position, sensor.rotation = eye, camera
        simulator.get_agent(0).set_state(state, reset_sensors=False, infer_sensor_states=False)
        return simulator.get_sensor_observations(), {
            "position": np.asarray(position).tolist(), "rotation_xyzw": quat_to_coeffs(body).tolist(),
            "camera_position_xyz": eye.tolist(), "camera_rotation_xyzw": quat_to_coeffs(camera).tolist(),
            "camera_pitch_degrees": math.degrees(pitch), "aim_xyz": np.asarray(aim).tolist(),
        }

    results = {}
    try:
        if not simulator.pathfinder.load_nav_mesh(str(args.navmesh)):
            raise RuntimeError("Cannot load audited navigation mesh")
        grid_validation = validate_grid(simulator.pathfinder, points)
        for target_id, (label, room_id, ids) in TARGETS.items():
            bounds = [geometry[i] for i in ids]
            low = np.min([b["aabb_min_xyz"] for b in bounds], axis=0)
            high = np.max([b["aabb_max_xyz"] for b in bounds], axis=0)
            # Compound vanity aims between its basins; other targets aim at actual mesh median.
            aim_ids = [109, 110] if target_id == "O108" else ids
            aim = np.mean([geometry[i]["aim_xyz"] for i in aim_ids], axis=0)
            candidates, rejected = [], {"other_room_or_floor": 0, "outside_1m_geometry": 0,
                                        "unreachable": 0, "insufficient_visibility": 0}
            for index, point in enumerate(points):
                if abs(float(point[1]) - .035739123821258545) > .25 or not rooms[room_id].covers(Point(float(point[0]), -float(point[2]))):
                    rejected["other_room_or_floor"] += 1
                    continue
                distance = min(horizontal_distance(point, b) for b in bounds)
                if distance > 1:
                    rejected["outside_1m_geometry"] += 1
                    continue
                query = habitat_sim.ShortestPath()
                query.requested_start = np.asarray([-.7, .035739123821258545, -.5], dtype=np.float32)
                query.requested_end = point
                simulator.pathfinder.find_path(query)
                if not math.isfinite(query.geodesic_distance):
                    rejected["unreachable"] += 1
                    continue
                candidates.append((index, point, distance, float(query.geodesic_distance)))
            accepted, checks = [], []
            for index, point, distance, path_distance in candidates:
                observations, record = observe(point, aim)
                counts = {str(i): int(np.sum(observations["semantic"] == i)) for i in ids}
                pixels = sum(counts.values())
                good = pixels >= math.ceil(.001 * 640 * 480)
                record.update(grid_index=index, target_pixels=pixels, pixels_by_instance=counts,
                              pixel_fraction=pixels / (640 * 480), geometry_distance_m=distance,
                              geodesic_from_fixed_start_m=path_distance, accepted=good)
                checks.append(record)
                if good:
                    accepted.append(record)
                else:
                    rejected["insufficient_visibility"] += 1
            refs = []
            # Preserve a best-visible view in each target-relative quadrant, including failures.
            for good in (True, False):
                matching = [record for record in checks if record["accepted"] == good]
                bins = {}
                for record in matching:
                    delta = np.asarray(record["position"])[[0, 2]] - aim[[0, 2]]
                    angle = math.atan2(delta[1], delta[0])
                    quadrant = int(((angle + math.pi) % (2 * math.pi)) / (math.pi / 2))
                    if quadrant not in bins or record["target_pixels"] > bins[quadrant]["target_pixels"]:
                        bins[quadrant] = record
                for quadrant, record in bins.items():
                    observations, _ = observe(record["position"], aim)
                    rgb = observations["rgb"][..., :3]
                    mask = np.isin(observations["semantic"], ids)
                    overlay = rgb.copy()
                    overlay[mask] = (.45 * overlay[mask] + .55 * np.array([255, 0, 255])).astype(np.uint8)
                    canvas = Image.new("RGB", (1280, 510), "white")
                    canvas.paste(Image.fromarray(rgb), (0, 30))
                    canvas.paste(Image.fromarray(overlay), (640, 30))
                    ImageDraw.Draw(canvas).text((8, 8), f"{target_id} grid={record['grid_index']} accepted={good} pixels={record['target_pixels']} (>=308) | raw RGB / semantic mask", fill="black")
                    name = f"{target_id}-{'accepted' if good else 'rejected'}-q{quadrant}.png"
                    canvas.save(evidence / name)
                    refs.append({"path": "evidence/" + name, "grid_index": record["grid_index"],
                                 "accepted": good, "target_pixels": record["target_pixels"]})
            results[target_id] = {
                "label": label, "room_id": room_id, "semantic_instance_ids": ids,
                "positions": [r["position"] for r in accepted], "viewpoints": accepted,
                "candidate_checks": checks, "candidate_count": len(candidates), "accepted_count": len(accepted),
                "grid_point_count": len(points), "rejection_counts": rejected, "evidence_refs": refs,
                "source": "custom_semantic_mesh_grid_visibility_v1",
                "geometry": {"aabb_min_xyz": low.tolist(), "aabb_max_xyz": high.tolist(),
                             "center_xyz": ((low + high) / 2).tolist(), "aim_xyz": aim.tolist(),
                             "instances": bounds, "source": "actual HM3D GLB semantic vertices; not schematic map boxes"},
                "limitations": ["Finite 0.1 m grid, single declared gaze per point; not exhaustive continuous viewpoints.",
                                "Semantic surface AABB approximates physical footprint; same-room filter uses reviewed schematic room polygons.",
                                "Only visibility at 0.88 m sensor height is certified; original evidence used a taller camera."]
            }
            print(json.dumps({"target": target_id, "candidates": len(candidates), "accepted": len(accepted),
                              "max_pixels": max((r["target_pixels"] for r in checks), default=0)}, ensure_ascii=False), flush=True)
    finally:
        simulator.close()
    result = {
        "schema_version": 1, "scene_id": "QaLdnwvtxbs", "targets": results,
        "method": {"name": "custom_semantic_mesh_grid_visibility_v1", "grid_spacing_m": .1,
                   "max_horizontal_aabb_distance_m": 1, "minimum_visible_fraction": .001,
                   "minimum_visible_pixels": 308, "same_room_required": True,
                   "gaze": "body yaw faces target mesh median; camera pitch looks at target median",
                   "fixed_reachability_start": [-.7, .035739123821258545, -.5],
                   "no_old_endpoint_selection": True},
        "environment": {"width": 640, "height": 480, "hfov": 79, "sensor_height": .88,
                        "agent_height": .88, "agent_radius": .18, "navmesh_cell_height": .05},
        "input_validation": grid_validation,
        "provenance": {"mesh": str(mesh), "mesh_sha256": checksum(mesh),
                       "annotations_sha256": checksum(annotations), "navmesh": str(args.navmesh),
                       "navmesh_sha256": checksum(args.navmesh), "grid": str(args.grid),
                       "grid_sha256": checksum(args.grid), "rooms": str(args.rooms),
                       "rooms_sha256": checksum(args.rooms)},
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="/home2/chen/NAS/Data/scene_datasets/hm3d/val/00829-QaLdnwvtxbs/QaLdnwvtxbs.basis.glb")
    parser.add_argument("--navmesh", default="outputs/goal-coverage-20261015/objnav.navmesh")
    parser.add_argument("--grid", default="outputs/goal-coverage-20261015/objnav-grid.npz")
    parser.add_argument("--rooms", default="data/scene_maps/hm3d/val/00829-QaLdnwvtxbs/raw/rooms.json")
    parser.add_argument("--output", default="data/datasets/navcraft/qa_campaign_goals_v2.json")
    parser.add_argument("--gpu", type=int, default=2)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
