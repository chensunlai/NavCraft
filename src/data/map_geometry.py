"""Unlabelled scene geometry and optional architectural semantic priors.

HM3D and MP3D assets are Z-up.  Mesh-node transforms are applied before the
single conversion to Habitat's Y-up frame: ``(x, y, z) -> (x, z, -y)``.
Semantic triangles are a separate candidate source, never geometry truth.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import struct
import time

import numpy as np


@dataclass(frozen=True)
class SceneGeometry:
    scene: Path
    triangles: np.ndarray
    categories: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def to_habitat(points):
    """Convert raw asset Z-up coordinates to world Y-up, preserving handedness."""
    points = np.asarray(points)
    result = points[..., [0, 2, 1]].astype(np.float32, copy=True)
    result[..., 2] *= -1
    return result


def _node_transform(node):
    if "matrix" in node:
        matrix = np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T
    else:
        x, y, z, w = node.get("rotation", [0, 0, 0, 1])
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1e-12:
            raise ValueError("GLB node has a zero quaternion")
        x, y, z, w = (value / norm for value in (x, y, z, w))
        rotation = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        matrix = np.eye(4)
        matrix[:3, :3] = rotation @ np.diag(node.get("scale", [1, 1, 1]))
        matrix[:3, 3] = node.get("translation", [0, 0, 0])
    if not np.isfinite(matrix).all():
        raise ValueError("GLB node transform is not finite")
    return matrix


class _Glb:
    """Read mesh accessors without loading textures or adding a glTF dependency."""

    def __init__(self, path):
        self.path = Path(path)
        self.raw = self.path.read_bytes()
        if len(self.raw) < 20 or struct.unpack_from("<III", self.raw) != (0x46546C67, 2, len(self.raw)):
            raise ValueError(f"Expected a valid GLB 2 file: {self.path}")
        self.doc, self.binary = None, None
        offset = 12
        while offset + 8 <= len(self.raw):
            length, kind = struct.unpack_from("<II", self.raw, offset)
            offset += 8
            if offset + length > len(self.raw):
                raise ValueError("GLB chunk extends beyond the file")
            if kind == 0x4E4F534A:
                self.doc = json.loads(self.raw[offset:offset + length])
            elif kind == 0x004E4942:
                self.binary = memoryview(self.raw)[offset:offset + length]
            offset += length
        if self.doc is None or self.binary is None:
            raise ValueError("GLB needs JSON and binary chunks")

    def accessor(self, index):
        item = self.doc["accessors"][index]
        if "sparse" in item:
            raise ValueError("Sparse GLB accessors are unsupported")
        view = self.doc["bufferViews"][item["bufferView"]]
        if view.get("buffer", 0) != 0:
            raise ValueError("External GLB buffers are unsupported")
        types = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
        widths = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
        dtype, width = np.dtype(types[item["componentType"]]), widths[item["type"]]
        offset = view.get("byteOffset", 0) + item.get("byteOffset", 0)
        stride = view.get("byteStride", width * dtype.itemsize)
        count = item["count"]
        if count < 0 or stride < width * dtype.itemsize or offset < 0:
            raise ValueError("Invalid GLB accessor shape or offset")
        end = offset + max(0, count - 1) * stride + (width * dtype.itemsize if count else 0)
        if end > len(self.binary):
            raise ValueError("GLB accessor extends beyond its binary buffer")
        values = np.ndarray((count, width), dtype=dtype, buffer=self.binary, offset=offset,
                            strides=(stride, dtype.itemsize))
        if item.get("normalized") and dtype.kind in "iu":
            limit = np.iinfo(dtype).max
            values = np.maximum(values.astype(np.float32) / limit, -1)
        return values

    def primitives(self):
        nodes = self.doc.get("nodes", [])
        scenes = self.doc.get("scenes", [])
        if scenes:
            roots = scenes[self.doc.get("scene", 0)].get("nodes", [])
        else:
            children = {child for node in nodes for child in node.get("children", [])}
            roots = [index for index in range(len(nodes)) if index not in children]

        def visit(index, parent, ancestors):
            if index in ancestors:
                raise ValueError("GLB node hierarchy contains a cycle")
            node = nodes[index]
            transform = parent @ _node_transform(node)
            if "mesh" in node:
                for primitive in self.doc["meshes"][node["mesh"]]["primitives"]:
                    if primitive.get("mode", 4) != 4:
                        raise ValueError("Only triangular GLB primitives are supported")
                    vertices = self.accessor(primitive["attributes"]["POSITION"])
                    if vertices.shape[1] != 3 or not np.isfinite(vertices).all():
                        raise ValueError("GLB positions must be finite 3D points")
                    transformed = vertices @ transform[:3, :3].T + transform[:3, 3]
                    if "indices" in primitive:
                        faces = self.accessor(primitive["indices"]).reshape(-1)
                    else:
                        faces = np.arange(len(vertices), dtype=np.int64)
                    if len(faces) % 3 or (len(faces) and faces.max() >= len(vertices)):
                        raise ValueError("Invalid triangle indices")
                    yield to_habitat(transformed), faces.reshape(-1, 3), primitive
            for child in node.get("children", []):
                yield from visit(child, transform, ancestors | {index})

        for root in roots:
            yield from visit(root, np.eye(4), set())


def _category(value):
    value = value.lower().replace("_", " ").replace("#", " ").strip()
    for category in ("wall", "door", "window"):
        if value.startswith(category):
            return category
    return None


def _hm3d_priors(scene):
    stem = scene.name.removesuffix(".basis.glb")
    mesh = scene.with_name(stem + ".semantic.glb")
    annotations = scene.with_name(stem + ".semantic.txt")
    if not mesh.is_file() or not annotations.is_file():
        return {}, []
    colors = {}
    with annotations.open(encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if len(row) >= 3 and row[0].isdigit() and (category := _category(row[2])):
                colors[int(row[1], 16)] = category
    pieces = {category: [] for category in ("wall", "door", "window")}
    glb = _Glb(mesh)
    for vertices, faces, primitive in glb.primitives():
        color_index = primitive["attributes"].get("COLOR_0")
        if color_index is None:
            raise ValueError("HM3D semantic mesh has no vertex colors")
        linear = glb.accessor(color_index)[:, :3].astype(np.float32)
        srgb = np.where(linear <= .0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - .055)
        rgb = np.clip(np.rint(srgb * 255), 0, 255).astype(np.uint32)
        codes = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
        for category in pieces:
            selected_colors = [code for code, name in colors.items() if name == category]
            selected = np.isin(codes, selected_colors)[faces].all(axis=1)
            if selected.any():
                pieces[category].append(vertices[faces[selected]])
    return {name: np.concatenate(parts) for name, parts in pieces.items() if parts}, [str(mesh), str(annotations)]


def _mp3d_priors(scene):
    house = scene.with_suffix(".house")
    mesh = scene.with_name(scene.stem + "_semantic.ply")
    if not house.is_file() or not mesh.is_file():
        return {}, []
    from plyfile import PlyData

    categories, objects = {}, {}
    with house.open(encoding="utf-8") as stream:
        for line in stream:
            parts = line.split()
            if parts and parts[0] == "C":
                categories[int(parts[1])] = _category(parts[5])
            elif parts and parts[0] == "O":
                objects[int(parts[1])] = int(parts[3])
    ids = {name: [identifier for identifier, category in objects.items() if categories.get(category) == name]
           for name in ("wall", "door", "window")}
    ply = PlyData.read(mesh, known_list_len={"face": {"vertex_indices": 3}})
    raw_vertices = ply["vertex"].data
    vertices = np.column_stack([raw_vertices[key] for key in ("x", "y", "z")])
    vertices = to_habitat(vertices)
    pieces = {name: [] for name in ids}
    faces = ply["face"].data
    for start in range(0, len(faces), 200_000):
        chunk = faces[start:start + 200_000]
        for name, identifiers in ids.items():
            selected = chunk["vertex_indices"][np.isin(chunk["object_id"], identifiers)]
            if len(selected):
                if selected.dtype.kind == "O":
                    selected = np.stack(selected).astype(np.int64)
                if selected.shape[1:] != (3,):
                    raise ValueError("MP3D semantic faces must be triangles")
                pieces[name].append(vertices[selected])
    return {name: np.concatenate(parts) for name, parts in pieces.items() if parts}, [str(mesh), str(house)]


def load_scene_geometry(scene, *, include_semantics=True):
    """Load the raw render mesh, with optional independent semantic candidates.

    ``triangles`` always comes from the supplied .basis.glb/.glb, including
    unlabeled walls and closed doors.  ``categories`` may be empty when semantic
    assets are absent; its labels must subsequently be checked geometrically.
    """
    started = time.monotonic()
    scene = Path(scene).resolve(strict=True)
    pieces = [vertices[faces] for vertices, faces, _ in _Glb(scene).primitives() if len(faces)]
    if not pieces:
        raise ValueError(f"Scene has no triangles: {scene}")
    triangles = np.concatenate(pieces)
    categories, sources = {}, []
    if include_semantics:
        categories, sources = (_hm3d_priors(scene) if scene.name.endswith(".basis.glb") else _mp3d_priors(scene))
    return SceneGeometry(scene, triangles, categories, {
        "geometry_source": str(scene), "geometry_triangles": len(triangles),
        "semantic_sources": sources, "semantic_triangles": {name: len(value) for name, value in categories.items()},
        "coordinate_transform": "raw Z-up to Habitat Y-up: (x,y,z)->(x,z,-y); mesh nodes applied first",
        "seconds": time.monotonic() - started,
    })


def _triangles(value):
    value = np.asarray(value, dtype=np.float64).reshape(-1, 3, 3)
    if not np.isfinite(value).all():
        raise ValueError("Triangles must be finite")
    return value


def vertical_triangles(triangles, max_normal_y=.35):
    """Select near-vertical surfaces, without semantic category filtering."""
    triangles = _triangles(triangles)
    if not 0 <= max_normal_y < 1:
        raise ValueError("max_normal_y must be in [0,1)")
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    keep = (lengths > 1e-10) & (np.abs(normals[:, 1]) <= max_normal_y * lengths)
    return triangles[keep]


def slice_triangles(triangles, height):
    """Exact horizontal intersections as world-XZ segments, excluding coplanar faces."""
    triangles = _triangles(triangles)
    if not math.isfinite(height):
        raise ValueError("Slice height must be finite")
    keep = (triangles[:, :, 1].min(1) <= height) & (triangles[:, :, 1].max(1) >= height)
    triangles = triangles[keep]
    if not len(triangles):
        return np.empty((0, 2, 2), dtype=np.float64)
    points = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        low, high = triangles[:, a], triangles[:, b]
        dy = high[:, 1] - low[:, 1]
        valid = (np.abs(dy) > 1e-10) & ((low[:, 1] - height) * (high[:, 1] - height) <= 0)
        fraction = np.divide(height - low[:, 1], dy, out=np.zeros_like(dy), where=valid)
        position = low[:, [0, 2]] + fraction[:, None] * (high[:, [0, 2]] - low[:, [0, 2]])
        position[~valid] = np.nan
        points.append(position)
    points = np.stack(points, axis=1)
    # A plane through a triangle vertex can produce a duplicate intersection.
    # Choose the farthest finite pair, preserving complete coplanar edge cuts.
    candidates, distances = [], []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        pair = points[:, [a, b]]
        distance = np.sum((pair[:, 1] - pair[:, 0]) ** 2, axis=1)
        distance[~np.isfinite(pair).all(axis=(1, 2))] = -1
        candidates.append(pair)
        distances.append(distance)
    distances = np.stack(distances, axis=1)
    best = distances.argmax(axis=1)
    segments = np.stack(candidates, axis=1)[np.arange(len(triangles)), best]
    segments = segments[distances.max(axis=1) > 1e-16]
    return segments


def multiheight_slices(triangles, floor_height, relative_heights=(.9, 1.3, 1.7, 2.0), *, vertical_only=True):
    """Independent raw-mesh cuts; rasterize each cut before counting height votes."""
    if not math.isfinite(floor_height) or not relative_heights:
        raise ValueError("Need a finite floor and at least one slice height")
    triangles = vertical_triangles(triangles) if vertical_only else _triangles(triangles)
    return {float(offset): slice_triangles(triangles, floor_height + float(offset)) for offset in relative_heights}


def height_projection(triangles, floor_height, lower=.9, upper=2.0, *, vertical_only=True):
    """Clip surfaces to a floor-relative height slab, returning XZ triangles.

    This is a geometry-only projection baseline.  It can include tall furniture;
    it must not be described as an architectural ground-truth wall layer.
    """
    if not all(math.isfinite(value) for value in (floor_height, lower, upper)) or lower >= upper:
        raise ValueError("Projection needs finite heights with lower < upper")
    triangles = vertical_triangles(triangles) if vertical_only else _triangles(triangles)
    low, high = floor_height + lower, floor_height + upper
    minima, maxima = triangles[:, :, 1].min(1), triangles[:, :, 1].max(1)
    selected = triangles[(minima <= high) & (maxima >= low)]
    inside = ((selected[:, :, 1] >= low) & (selected[:, :, 1] <= high)).all(1)
    result = [selected[inside][:, :, [0, 2]]]
    for triangle in selected[~inside]:
        polygon = list(triangle)
        for boundary, sign in ((low, 1), (high, -1)):
            clipped = []
            for first, second in zip(polygon, polygon[1:] + polygon[:1]):
                first_inside, second_inside = sign * (first[1] - boundary) >= 0, sign * (second[1] - boundary) >= 0
                if first_inside:
                    clipped.append(first)
                if first_inside != second_inside:
                    fraction = (boundary - first[1]) / (second[1] - first[1])
                    clipped.append(first + fraction * (second - first))
            polygon = clipped
            if not polygon:
                break
        for index in range(1, len(polygon) - 1):
            result.append(np.asarray([[polygon[0][[0, 2]], polygon[index][[0, 2]], polygon[index + 1][[0, 2]]]]))
    return np.concatenate(result).reshape(-1, 3, 2)
