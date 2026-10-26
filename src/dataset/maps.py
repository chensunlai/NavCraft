"""Read original scene files and select the public geometry for one run.

Scene files use metric (Habitat X,-Z). The adapter creates Habitat (X,Z)
geometry in memory; only run directories receive a public map.json snapshot.
"""

import hashlib
import json
import math
from pathlib import Path
import re

from src.common.coordinates import validate_map, world_to_uv
from src.common.io import write_json
from src.common.scenes import canonical_scene_id

COORDINATES = "habitat_xz_metres"
LAYERS = ("walls", "wall_real")


def map_directory(root, scene_id):
    """Find original scene files without a second directory tree or asset index."""
    scene = Path(canonical_scene_id(scene_id))
    root = Path(root)
    if scene.parts[0] in {"hm3d", "mp3d"}:
        return root / scene.parent
    if scene.parts[0] == "hssd-hab" and scene.name.endswith(".scene_instance.json"):
        identifier = scene.name.removesuffix(".scene_instance.json")
        matches = sorted(path for path in (root / "hssd-hab").glob(f"*/{identifier}")
                         if (path / "scene.json").is_file())
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one scene directory for {scene_id}, found {len(matches)}")
        return matches[0]
    raise ValueError(f"Unsupported scene asset layout: {scene_id}")


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Expected finite numeric map coordinate, got {value!r}")
    return float(value)


def _source_point(point):
    """The sole source-plan [X,-Z] -> runtime [X,Z] boundary, both in metres."""
    if not isinstance(point, list) or len(point) != 2:
        raise ValueError("Source map points require two coordinates")
    return [_number(point[0]), -_number(point[1])]


def _scene_coordinates(scene):
    declaration = next((scene[key] for key in ("coordinate_system", "coordinates",
                       "coordinate_convention", "map_coordinates") if key in scene), None)
    if isinstance(declaration, dict):
        declaration = next((declaration[key] for key in ("horizontal", "map", "map_xy", "map_axes", "plan")
                            if key in declaration), declaration)
    compact = json.dumps(declaration).lower().replace("habitat", "").replace('"', '')
    # Require the declared map pair; a source-mesh transform mentioning -Z is
    # insufficient. Unknown conventions require an explicit offline correction.
    if not re.search(r'(?<![a-z])x\s*,\s*-\s*z(?![a-z])', compact):
        raise ValueError("Scene must explicitly declare map coordinates (Habitat X,-Z)")
    if scene.get("units", scene.get("unit")) not in {"m", "metre", "metres", "meter", "meters"}:
        raise ValueError("Scene map coordinates must use metres")


def _read_geometry(source):
    """Read scene/levels and wall layers together; no persistent converted copy."""
    hashes = {}

    def read(name):
        path = source / name
        raw = path.read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    scene = read("scene.json")
    if scene.get("status") in {"partial", "blocked", "building", "pending"}:
        raise ValueError("Cannot load an unfinished scene")
    _scene_coordinates(scene)
    levels = scene.get("levels") or read("levels.json")
    if not isinstance(levels, list) or not levels:
        raise ValueError("Scene requires floor definitions")
    ids = [row["id"] for row in levels]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate floor ID")
    layers = {}
    for name in LAYERS:
        path = source / f"{name}.json"
        if name == "wall_real" and not path.is_file():
            continue
        data = read(path.name)
        if data.get("scene_id") != scene["scene_id"]:
            raise ValueError("Scene and wall IDs disagree")
        layers[name] = data["segments"]
        unknown = {row.get("level") for row in data["segments"]} - set(ids)
        if unknown:
            raise ValueError(f"Walls refer to undeclared floors: {unknown}")
    floors, excluded = [], []
    for level in levels:
        height = _number(level.get("floor_height", level.get("height_m")))
        if any(level.get(key) is False for key in ("is_building_floor", "has_walkable_floor",
                    "walkable_floor", "occupied_floor_confirmed", "occupied_floor")):
            excluded.append({"id": level["id"], "reason": "source_marks_non_floor"})
            continue
        geometry = {}
        for name, segments in layers.items():
            walls = []
            for segment in segments:
                if segment["level"] != level["id"]:
                    continue
                converted = [_source_point(segment[key]) for key in ("start", "end")]
                if converted[0] == converted[1]:
                    raise ValueError("Zero-length wall segment")
                walls.append(converted)
            if walls:
                geometry[name] = walls
        if "walls" not in geometry:
            excluded.append({"id": level["id"], "reason": "no_wall_geometry"})
            continue
        # Use one calibration for both layers; switching to evidence-only walls
        # must not silently move all remembered UV points.
        points = [p for walls in geometry.values() for segment in walls for p in segment]
        bounds = level.get("bounds")
        if bounds is not None:
            if len(bounds) == 4:
                bounds = [bounds[:2], bounds[2:]]
            if len(bounds) != 2 or any(len(axis) != 2 for axis in bounds):
                raise ValueError("Invalid scene floor bounds")
            x, y = [[_number(v) for v in axis] for axis in bounds]
            if x[0] >= x[1] or y[0] >= y[1]:
                raise ValueError("Invalid scene floor bounds ordering")
            points.extend([_source_point([x[0], y[1]]), _source_point([x[1], y[0]])])
        low = [min(p[i] for p in points) - 2 for i in (0, 1)]
        high = [max(p[i] for p in points) + 2 for i in (0, 1)]
        size = math.ceil(max(high[i] - low[i] for i in (0, 1)) / 10) * 10
        origin = [(low[i] + high[i] - size) / 2 for i in (0, 1)]
        for walls in geometry.values():
            validate_map({"origin": origin, "size_m": size, "floor_height": height, "walls": walls})
        floors.append({"id": level["id"], "floor_height": height,
                       "origin": origin, "size_m": size, "layers": geometry})
    if not floors:
        raise ValueError("No floors with wall geometry")
    value = {"scene_id": scene["scene_id"], "coordinate_system": COORDINATES, "levels": floors,
             "source": {"directory": str(source.resolve()), "coordinates": "habitat_x_minus_z_metres",
                        "excluded_levels": excluded,
                        "sha256": hashes}}
    if any(hashlib.sha256((source / name).read_bytes()).hexdigest() != digest
           for name, digest in hashes.items()):
        raise ValueError("Scene changed while loading; retry after its build finishes")
    return value


def load_wall_map(directory, start_position, layer="walls"):
    """Load one floor/layer from original files, without modifying them."""
    if layer not in LAYERS:
        raise ValueError("Map layer must be walls or wall_real")
    directory = Path(directory)
    document = _read_geometry(directory)
    if (not isinstance(start_position, (list, tuple)) or len(start_position) != 3):
        raise ValueError("Start position must be Habitat [X,Y,Z] metres")
    start_position = [_number(v) for v in start_position]
    levels = document["levels"]
    matches = [row for row in levels if abs(_number(row["floor_height"]) - start_position[1]) <= .5]
    if len(matches) != 1:
        raise ValueError(f"Start height {start_position[1]} must match exactly one floor within 0.5m: {directory}")
    level = matches[0]
    if layer not in level["layers"]:
        raise ValueError(f"Layer {layer} is unavailable on floor {level['id']}")
    public = validate_map({key: level[key] for key in ("origin", "size_m", "floor_height")} |
                          {"walls": level["layers"][layer]})
    if any(not 0 <= v <= 1 for v in world_to_uv(start_position, public)):
        raise ValueError(f"Start position is outside the scene canvas: {directory}")
    provenance = {"directory": str(directory.resolve()), "scene_id": document["scene_id"],
                  "layer": layer, "level": level["id"], "floor_height": level["floor_height"],
                  "coordinate_system": COORDINATES,
                  "map_sha256": hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest(),
                  "source": document["source"]}
    return public, provenance


def prepare_map(source, start_position, private_directory, layer="walls"):
    """Save a public geometry snapshot and private source hashes under one run."""
    source = Path(source)
    if source.is_file():
        validate_map(json.loads(source.read_text()))
        return source
    public, provenance = load_wall_map(source, start_position, layer)
    private_directory = Path(private_directory)
    destination = private_directory / "map.json"
    write_json(destination, public)
    write_json(private_directory / "map_source.json", provenance)
    return destination
