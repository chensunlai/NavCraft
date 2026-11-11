"""Private scene geometry adapted from the reviewed map files."""
import hashlib
import json
import math
from pathlib import Path

from src.dataset.maps import _number, _scene_coordinates, _source_point


def load_scene(directory: Path) -> dict:
    directory = Path(directory)
    hashes = {}

    def read(name, default=None):
        path = directory / (name + ".json")
        if default is not None and not path.exists():
            return default
        raw = path.read_bytes()
        hashes[path.name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    source = read("scene")
    _scene_coordinates(source)
    manifest = read("manifest", {})
    expected = directory.name.split("-", 1)[-1] if source.get("dataset") == "hm3d" else directory.name
    if source["scene_id"] != expected or manifest.get("scene_id", expected) != expected:
        raise ValueError(f"Map scene identity does not match {directory}")
    if source.get("status") in {"partial", "blocked", "building", "pending"} or manifest.get("status") == "partial":
        raise ValueError(f"Scene map is incomplete: {directory}")
    raw_levels = source.get("levels") or read("levels")
    if isinstance(raw_levels, dict):
        raw_levels = raw_levels["levels"]
    levels = [{"id": row["id"], "floor_height": _number(row.get("floor_height", row.get("height_m")))}
              for row in raw_levels if all(row.get(key) is not False for key in
              ("is_building_floor", "has_walkable_floor", "walkable_floor",
               "occupied_floor_confirmed", "occupied_floor"))]
    level_ids = {row["id"] for row in levels}
    if not levels or len(level_ids) != len(levels):
        raise ValueError("Scene requires unique occupied floors")

    def rows(name):
        value = read(name, {})
        if value and value["scene_id"] != source["scene_id"]:
            raise ValueError(f"Mismatching scene in {name}")
        selected = [row for row in value.get(name, []) if row["level"] in level_ids]
        if len({row["id"] for row in selected}) != len(selected):
            raise ValueError(f"Duplicate IDs in {name}")
        return selected

    rooms = [{"id": r["id"], "level": r["level"], "name": r["name"],
              "polygon_xz": [_source_point(p) for p in r["polygon"]],
              "description": r.get("description", ""), "evidence": r.get("evidence_frames", [])}
             for r in rows("rooms") if len(r.get("polygon", [])) >= 3]
    openings = [{"id": r["id"], "level": r["level"], "kind": r["kind"],
                 "segment_xz": [_source_point(p) for p in r["segment"]],
                 "connects": r.get("connects", []), "status": r.get("status"),
                 "evidence": r.get("evidence_frames", [])}
                for r in rows("doors") if len(r.get("segment", [])) == 2]
    objects = []
    for row in rows("objects"):
        height = row.get("height_range_m")
        if not height or len(height) != 2:
            continue
        height = [_number(h) for h in height]
        floor = next(l["floor_height"] for l in levels if l["id"] == row["level"])
        if height[0] >= height[1] or height[0] < floor-1.0 or height[1] > floor+5.0:
            # Runtime projection requires a plausible absolute Habitat Y range.
            # Incompatible records need a source correction rather than an implicit offset.
            continue
        corners = []
        rectangles = []
        for primitive in row.get("primitives", []):
            if primitive["type"] != "rectangle":
                continue
            x, z = _source_point(primitive["center"])
            w, d = map(_number, primitive["size"])
            if min(w, d) <= 0:
                continue
            angle = math.radians(_number(primitive.get("angle_deg", 0)))
            c, s = math.cos(angle), math.sin(angle)
            rectangles.append({"center_xz": [x, z], "size": [w, d], "angle_deg": -math.degrees(angle)})
            corners.extend([x + c * u - s * v, z - s * u - c * v]
                           for u in (-w / 2, w / 2) for v in (-d / 2, d / 2))
        if not corners:
            continue
        xmin, zmin = map(min, zip(*corners))
        xmax, zmax = map(max, zip(*corners))
        objects.append({"id": row["id"], "level": row["level"], "room": row.get("room"),
                        "name": row["label"], "category": row.get("category"),
                        "description": row.get("description", ""),
                        "center_xyz": [(xmin+xmax)/2, sum(height)/2, (zmin+zmax)/2],
                        "size_xyz": [xmax-xmin, height[1]-height[0], zmax-zmin],
                        "rectangles_xz": rectangles, "height_range_m": height,
                        "semantic_ids": row.get("semantic_mesh_ids", row.get("semantic_reference_ids", row.get("semantic_instances", []))),
                        "evidence": row.get("evidence_frames", []),
                        "annotation_quality": "schematic_source_unverified_in_runtime"})
    return {"levels": levels, "rooms": rooms, "openings": openings, "objects": objects,
            "source": {"map_directory": str(directory), "sha256": hashes,
                       "map_status": manifest.get("status"),
                       "limitations": ["Schematic polygons and rectangles; rooms and objects may be incomplete.",
                                       "Map review does not verify identity or visibility in runtime observations."]}}


def room_at(scene, xyz, margin=0.12):
    """Return a room only if uniquely inside a floor's polygon away from its edge."""
    x, y, z = xyz
    levels = [r["id"] for r in scene["levels"] if abs(r["floor_height"] - y) <= 0.45]
    if len(levels) != 1:
        return None
    hits = []
    for room in scene["rooms"]:
        if room["level"] != levels[0]:
            continue
        polygon = room["polygon_xz"]
        inside, near = False, False
        for (a, b), (c, d) in zip(polygon, polygon[1:] + polygon[:1]):
            dx, dz = c - a, d - b
            length2 = dx * dx + dz * dz
            if not length2:
                continue
            t = max(0.0, min(1.0, ((x-a)*dx + (z-b)*dz) / length2))
            near |= math.hypot(x-a-t*dx, z-b-t*dz) <= margin
            if (b > z) != (d > z) and x < a + (z-b)*dx/dz:
                inside = not inside
        if near:
            return None
        if inside:
            hits.append(room["id"])
    return hits[0] if len(hits) == 1 else None
