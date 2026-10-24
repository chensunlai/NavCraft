"""Offline architectural wall slices from original HM3D and MP3D semantics."""

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import struct

import numpy as np
from PIL import Image, ImageDraw


def _world(points):
    """Raw semantic assets use Z-up; Habitat uses X, Z, -Y."""
    result = np.asarray(points, dtype=np.float64)[..., [0, 2, 1]].copy()
    result[..., 2] *= -1
    return result


def hm3d_walls(scene):
    scene = Path(scene)
    stem = scene.name.removesuffix(".basis.glb")
    mesh = scene.with_name(stem + ".semantic.glb")
    annotations = scene.with_name(stem + ".semantic.txt")
    wall_colors = set()
    with annotations.open() as stream:
        for row in csv.reader(stream):
            if row and row[0].isdigit() and row[2].strip().lower() == "wall":
                wall_colors.add(int(row[1], 16))
    raw = mesh.read_bytes()
    if struct.unpack_from("<III", raw) != (0x46546C67, 2, len(raw)):
        raise ValueError("Expected GLB 2")
    length, kind = struct.unpack_from("<II", raw, 12)
    if kind != 0x4E4F534A:
        raise ValueError("Missing GLB JSON chunk")
    doc = json.loads(raw[20:20 + length])
    binary_length, binary_kind = struct.unpack_from("<II", raw, 20 + length)
    if binary_kind != 0x004E4942:
        raise ValueError("Missing GLB binary chunk")
    binary = memoryview(raw)[28 + length:28 + length + binary_length]
    if any(any(key in node for key in ("matrix", "translation", "rotation", "scale")) for node in doc["nodes"]):
        raise ValueError("Transformed semantic mesh nodes are unsupported")

    def accessor(index):
        item = doc["accessors"][index]
        if "sparse" in item:
            raise ValueError("Sparse semantic accessors are unsupported")
        view = doc["bufferViews"][item["bufferView"]]
        dtype = np.dtype({5121: "u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}[item["componentType"]])
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[item["type"]]
        values = np.ndarray((item["count"], width), dtype=dtype, buffer=binary,
                            offset=view.get("byteOffset", 0) + item.get("byteOffset", 0),
                            strides=(view.get("byteStride", width * dtype.itemsize), dtype.itemsize))
        return values.astype(np.float64) / np.iinfo(dtype).max if item.get("normalized") else values

    for record in doc["meshes"]:
        for primitive in record["primitives"]:
            if primitive.get("mode", 4) != 4:
                raise ValueError("Semantic primitive is not triangles")
            attributes = primitive["attributes"]
            vertices = accessor(attributes["POSITION"])
            linear = accessor(attributes["COLOR_0"])[:, :3]
            srgb = np.where(linear <= .0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - .055)
            colors = np.clip(np.rint(srgb * 255), 0, 255).astype(np.uint32)
            codes = (colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]
            faces = accessor(primitive["indices"]).reshape(-1, 3)
            selected = np.isin(codes, list(wall_colors))[faces].all(axis=1)
            if selected.any():
                yield _world(vertices[faces[selected]])


def mp3d_walls(scene):
    from plyfile import PlyData

    scene = Path(scene)
    categories, objects = {}, {}
    with scene.with_suffix(".house").open() as stream:
        for line in stream:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "C":
                categories[int(parts[1])] = parts[5]
            elif parts[0] == "O":
                objects[int(parts[1])] = int(parts[3])
    ids = [identifier for identifier, category in objects.items() if categories.get(category) == "wall"]
    data = PlyData.read(scene.with_name(scene.stem + "_semantic.ply"), known_list_len={"face": {"vertex_indices": 3}})
    vertices, faces = data["vertex"].data, data["face"].data
    for start in range(0, len(faces), 200_000):
        chunk = faces[start:start + 200_000]
        indices = chunk["vertex_indices"][np.isin(chunk["object_id"], ids)]
        if len(indices):
            points = np.stack([vertices[key][indices] for key in ("x", "y", "z")], axis=-1)
            yield _world(points)


def slice_triangles(triangles, height):
    """Intersect triangles with a horizontal plane, never project overhead lintels."""
    triangles = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    if not np.isfinite(triangles).all() or not math.isfinite(height):
        raise ValueError("Geometry and slice height must be finite")
    keep = (triangles[:, :, 1].min(1) <= height) & (triangles[:, :, 1].max(1) >= height)
    triangles = triangles[keep]
    if not len(triangles):
        return np.empty((0, 2, 2))
    crossings = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        low, high = triangles[:, a], triangles[:, b]
        dy = high[:, 1] - low[:, 1]
        valid = (np.abs(dy) > 1e-10) & ((low[:, 1] <= height) & (high[:, 1] >= height) |
                                             (high[:, 1] <= height) & (low[:, 1] >= height))
        fraction = np.divide(height - low[:, 1], dy, out=np.zeros_like(dy), where=valid)
        positions = low[:, [0, 2]] + fraction[:, None] * (high[:, [0, 2]] - low[:, [0, 2]])
        positions[~valid] = np.nan
        crossings.append(positions)
    candidates = np.stack(crossings, axis=1)
    segments = []
    for points in candidates:
        points = np.unique(np.round(points[np.isfinite(points).all(axis=1)], 5), axis=0)
        if len(points) >= 2:
            segments.append([points[0], points[-1]])
    return np.asarray(segments).reshape(-1, 2, 2)


def canvas(segments):
    points = np.asarray(segments).reshape(-1, 2)
    low, high = points.min(0) - 2, points.max(0) + 2
    side = max(10., math.ceil(float(max(high - low)) / 10) * 10.)
    origin = (low + high - side) / 2
    return origin.tolist(), side


def _navmesh_overlay(scene, floor_height, segments, origin, side, output):
    """Private comparison only; the asset NavMesh may use different robot sizes."""
    navmesh = scene.with_suffix(".navmesh")
    if not navmesh.exists():
        return {"navmesh_overlay": None}
    import habitat_sim

    pathfinder = habitat_sim.PathFinder()
    if not pathfinder.load_nav_mesh(str(navmesh)):
        raise ValueError(f"Cannot load audit NavMesh: {navmesh}")
    vertices = np.asarray(pathfinder.build_navmesh_vertices())
    indices = np.asarray(pathfinder.build_navmesh_vertex_indices()).reshape(-1, 3)
    triangles = vertices[indices]
    triangles = triangles[np.max(np.abs(triangles[:, :, 1] - floor_height), axis=1) <= .3]
    image = Image.new("RGB", (768, 768), "white")
    draw = ImageDraw.Draw(image)
    for triangle in triangles:
        pixels = [tuple(20 + (point[[0, 2]] - origin) / side * 728) for point in triangle]
        draw.polygon(pixels, fill="#cbdfee")
    for segment in segments:
        pixels = [tuple(20 + (np.asarray(point) - origin) / side * 728) for point in segment]
        draw.line(pixels, fill="#232629", width=2)
    image.save(output / "navmesh-overlay.png")
    settings = pathfinder.nav_mesh_settings
    return {"navmesh_overlay": "navmesh-overlay.png", "navmesh_source": str(navmesh),
            "navmesh_floor_triangles": len(triangles),
            "navmesh_agent_radius": float(settings.agent_radius) if settings else None,
            "navmesh_note": "Original asset NavMesh, for geometric comparison only; not current robot movement validation"}


def build(scene_path, floor_height, output_dir):
    scene = Path(scene_path).resolve()
    floor_height = float(floor_height)
    if not math.isfinite(floor_height):
        raise ValueError("floor_height must be finite")
    reader = hm3d_walls if scene.name.endswith(".basis.glb") else mp3d_walls
    pieces, triangle_count = [], 0
    for triangles in reader(scene):
        triangle_count += len(triangles)
        pieces.append(slice_triangles(triangles, floor_height + 1.0))
    if not pieces or not any(len(piece) for piece in pieces):
        raise ValueError("No semantic walls intersect the floor+1m slice; explicit fallback/review required")
    segments = np.unique(np.concatenate(pieces), axis=0)
    origin, side = canvas(segments)
    data = {"origin": origin, "size_m": side, "walls": segments.tolist(),
            "floor_height": floor_height, "method": "semantic_wall_slice", "slice_height": floor_height + 1.0}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "map.json").write_text(json.dumps(data, separators=(",", ":")) + "\n")
    image = Image.new("RGB", (768, 768), "white")
    draw = ImageDraw.Draw(image)
    for segment in segments:
        pixels = [tuple(20 + (np.asarray(point) - origin) / side * 728) for point in segment]
        draw.line(pixels, fill="#222222", width=2)
    image.save(output / "audit.png")
    report = {"scene": str(scene), "floor_height": floor_height, "slice_height": floor_height + 1.,
              "method": data["method"], "wall_triangles": triangle_count, "segments": len(segments),
              "geometry_sha256": hashlib.sha256(segments.tobytes()).hexdigest(),
              "validation": "CPU extraction only; doorway/route and RGB alignment need simulator review"}
    report.update(_navmesh_overlay(scene, floor_height, segments, origin, side, output))
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "audit.html").write_text('<!doctype html><meta charset="utf-8"><title>Wall slice audit</title>'
        '<h1>Semantic wall slice</h1><p>Original semantic assets; no room or object annotations are supplied to the agent.</p>'
        '<img src="audit.png" width="768">'
        + ('<h2>Original NavMesh comparison (private)</h2><img src="navmesh-overlay.png" width="768">' if report["navmesh_overlay"] else '')
        + '<pre>' + html.escape(json.dumps(report, indent=2)) + '</pre>')
    return {"map_path": str(output / "map.json"), **report}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--height", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.scene, args.height, args.output), indent=2))
