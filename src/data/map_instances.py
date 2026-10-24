"""Instance-level semantic surface priors, separate from unlabelled render mesh.

These triangles originate in semantic.glb/_semantic.ply and their annotations.
They are not independent raw-geometry evidence and must be checked against the
render mesh, observations and the current robot's actual traversability.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import time

import numpy as np

from .map_geometry import _Glb, to_habitat


CATEGORIES = frozenset({"wall", "door", "window", "floor", "ceiling", "furniture", "unknown"})
ARCHITECTURE = {
    "wall": "wall", "walls": "wall", "door": "door", "doors": "door",
    "door frame": "door", "doorframe": "door", "doorway": "door", "archway": "door",
    "window": "window", "windows": "window", "window frame": "window", "windowframe": "window",
    "floor": "floor", "floors": "floor", "ceiling": "ceiling", "ceilings": "ceiling",
}
FURNITURE = frozenset({
    "bed", "bunk bed", "sofa", "couch", "loveseat", "chair", "armchair", "stool", "bench",
    "table", "dining table", "coffee table", "side table", "desk", "nightstand", "dresser",
    "cabinet", "cabinets", "wall cabinet", "wallcabinet", "bathroom cabinet", "kitchen cabinet",
    "wardrobe", "closet", "cupboard", "shelf", "shelves", "bookshelf", "bookcase", "rack",
    "counter", "countertop", "kitchen counter", "kitchen island", "island", "tv stand",
    "fridge", "refrigerator", "freezer", "oven", "stove", "microwave", "dishwasher",
    "washing machine", "washer", "dryer", "washingmachine", "sink", "toilet", "bathtub",
    "shower", "shower stall", "shower curtain", "curtain", "curtains", "blinds", "mirror",
    "tv", "television", "monitor", "plant", "potted plant", "lamp", "floor lamp", "floorlamp",
    "table lamp", "ceiling lamp", "ceiling light", "chandelier", "rug", "mat", "floor mat",
})


@dataclass(frozen=True)
class SemanticInstance:
    id: int
    label: str
    category: str
    region_id: int | None
    triangles: np.ndarray
    # [active-scene primitive ordinal, original primitive face index] for GLB;
    # [0, original global PLY face row] for MP3D. These index semantic geometry.
    triangle_ids: np.ndarray
    source: dict


def _normalise(value):
    return re.sub(r"\s+", " ", re.sub(r"[#_/\\]+", " ", value.casefold())).strip()


def _category(label, canonical=None):
    """Exact names prevent wall cabinets/floor lamps becoming building surfaces."""
    name = _normalise(label)
    if name in FURNITURE:
        return "furniture"
    if name in ARCHITECTURE:
        return ARCHITECTURE[name]
    canonical = _normalise(canonical or "")
    if canonical in FURNITURE:
        return "furniture"
    return ARCHITECTURE.get(canonical, "unknown")


def _region(value):
    try:
        identifier = int(value)
    except (TypeError, ValueError):
        return None
    return identifier if identifier >= 0 else None


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hm3d(scene, wanted):
    stem = scene.name.removesuffix(".basis.glb")
    mesh = scene.with_name(stem + ".semantic.glb")
    annotation = scene.with_name(stem + ".semantic.txt")
    definitions, by_color = {}, {}
    with annotation.open(encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if len(row) < 3 or not row[0].isdigit():
                continue
            identifier, color = int(row[0]), int(row[1], 16)
            if identifier in definitions or color in by_color:
                raise ValueError("HM3D instance IDs and colors must be unique")
            definitions[identifier] = {"label": row[2], "category": _category(row[2]),
                                       "region_id": _region(row[3] if len(row) > 3 else None),
                                       "source": {"annotation_instance_id": identifier, "color_hex": f"{color:06X}"}}
            by_color[color] = identifier
    glb = _Glb(mesh)
    geometry, identifiers = {}, {}
    mixed, unassigned, total = 0, 0, 0
    for primitive_id, (vertices, faces, primitive) in enumerate(glb.primitives()):
        color_accessor = primitive["attributes"].get("COLOR_0")
        if color_accessor is None:
            raise ValueError("HM3D semantic mesh is missing COLOR_0")
        linear = glb.accessor(color_accessor)[:, :3].astype(np.float32)
        srgb = np.where(linear <= .0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - .055)
        rgb = np.clip(np.rint(srgb * 255), 0, 255).astype(np.uint32)
        codes = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
        face_colors = codes[faces]
        uniform = (face_colors == face_colors[:, :1]).all(1)
        total += len(faces)
        mixed += int((~uniform).sum())
        unassigned += int((uniform & ~np.isin(face_colors[:, 0], list(by_color))).sum())
        for color in np.unique(face_colors[uniform, 0]):
            identifier = by_color.get(int(color))
            if identifier is None or definitions[identifier]["category"] not in wanted:
                continue
            selected = np.flatnonzero(uniform & (face_colors[:, 0] == color))
            geometry.setdefault(identifier, []).append(vertices[faces[selected]])
            identifiers.setdefault(identifier, []).append(np.column_stack([np.full(len(selected), primitive_id), selected]))
    instances = [SemanticInstance(identifier, definition["label"], definition["category"], definition["region_id"],
                                  np.concatenate(geometry[identifier]), np.concatenate(identifiers[identifier]),
                                  {**definition["source"], "mesh": str(mesh), "annotation": str(annotation),
                                   "geometry_id_space": "semantic_glb_active_scene_primitive_ordinal_and_face_index"})
                 for identifier, definition in sorted(definitions.items()) if identifier in geometry]
    return instances, {"sources": [{"path": str(mesh), "sha256": hashlib.sha256(glb.raw).hexdigest()},
                                   {"path": str(annotation), "sha256": _sha256(annotation)}],
                       "annotated_instance_count": len(definitions), "semantic_mesh_triangle_count": total,
                       "mixed_color_triangle_count": mixed, "unassigned_triangle_count": unassigned}


def _mp3d(scene, wanted):
    from plyfile import PlyData

    mesh, annotation = scene.with_name(scene.stem + "_semantic.ply"), scene.with_suffix(".house")
    categories, definitions = {}, {}
    with annotation.open(encoding="utf-8") as stream:
        for line in stream:
            row = line.split()
            if row and row[0] == "C":
                categories[int(row[1])] = {"raw_label": row[3], "canonical_label": row[5],
                                           "canonical_id": int(row[4])}
            elif row and row[0] == "O":
                identifier = int(row[1])
                if identifier in definitions:
                    raise ValueError("MP3D object IDs must be unique")
                definitions[identifier] = {"region_id": _region(row[2]), "category_index": int(row[3])}
    for identifier, definition in definitions.items():
        category = categories.get(definition["category_index"], {})
        label = category.get("raw_label", "unknown")
        definition.update(label=label, category=_category(label, category.get("canonical_label")),
                          source={"object_id": identifier, "category_index": definition["category_index"], **category})
    ply = PlyData.read(mesh, known_list_len={"face": {"vertex_indices": 3}})
    vertex_data = ply["vertex"].data
    vertices = to_habitat(np.column_stack([vertex_data[key] for key in ("x", "y", "z")]))
    if not np.isfinite(vertices).all():
        raise ValueError("MP3D semantic vertices must be finite")
    faces = ply["face"].data
    geometry, identifiers, unassigned = {}, {}, 0
    for start in range(0, len(faces), 200_000):
        chunk = faces[start:start + 200_000]
        object_ids = chunk["object_id"]
        unassigned += int((~np.isin(object_ids, list(definitions))).sum())
        for identifier in np.unique(object_ids):
            identifier = int(identifier)
            if identifier not in definitions or definitions[identifier]["category"] not in wanted:
                continue
            selected = np.flatnonzero(object_ids == identifier)
            indices = chunk["vertex_indices"][selected]
            if indices.dtype.kind == "O":
                indices = np.stack(indices).astype(np.int64)
            if indices.shape[1:] != (3,):
                raise ValueError("MP3D semantic faces must be triangles")
            geometry.setdefault(identifier, []).append(vertices[indices])
            identifiers.setdefault(identifier, []).append(np.column_stack([np.zeros(len(selected), dtype=np.int64), start + selected]))
    instances = [SemanticInstance(identifier, definition["label"], definition["category"], definition["region_id"],
                                  np.concatenate(geometry[identifier]), np.concatenate(identifiers[identifier]),
                                  {**definition["source"], "mesh": str(mesh), "annotation": str(annotation),
                                   "geometry_id_space": "semantic_ply_global_face_row"})
                 for identifier, definition in sorted(definitions.items()) if identifier in geometry]
    return instances, {"sources": [{"path": str(mesh), "sha256": _sha256(mesh)},
                                   {"path": str(annotation), "sha256": _sha256(annotation)}],
                       "annotated_instance_count": len(definitions), "semantic_mesh_triangle_count": len(faces),
                       "unassigned_triangle_count": unassigned}


def load_semantic_instances(scene, *, wanted_categories=None):
    """Return independent instances and provenance, with no geometric acceptance.

    IDs and triangle IDs refer to the *semantic* source, not to the independent
    raw .basis.glb/.glb triangles. ``wanted_categories`` selects normalized
    categories; original labels, region IDs and source IDs remain intact.
    """
    started = time.monotonic()
    scene = Path(scene).resolve(strict=True)
    wanted = CATEGORIES if wanted_categories is None else frozenset(wanted_categories)
    if wanted - CATEGORIES:
        raise ValueError(f"Unknown wanted_categories: {sorted(wanted - CATEGORIES)}")
    instances, metadata = _hm3d(scene, wanted) if scene.name.endswith(".basis.glb") else _mp3d(scene, wanted)
    metadata.update(scene=str(scene), role="semantic_surface_priors_not_independent_raw_geometry",
                    independent_raw_render_mesh=str(scene), geometry_acceptance="not_performed",
                    coordinate_frame="Habitat XYZ;Y-up;(rawx,rawy,rawz)->(rawx,rawz,-rawy);GLBnodes_appliedfirst",
                    wanted_categories=sorted(wanted), returned_instance_count=len(instances),
                    returned_triangle_count=sum(len(instance.triangles) for instance in instances),
                    category_instance_counts={name: sum(instance.category == name for instance in instances) for name in sorted(wanted)},
                    seconds=time.monotonic() - started)
    return instances, metadata
