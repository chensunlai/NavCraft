"""Read published scene walls and adapt one floor to the Harness map format.

This is format/coordinate conversion only: no reconstruction, geometry repair,
NavMesh lookup, target information, or source-file mutation.
"""

import hashlib
import json
import math
from pathlib import Path

from src.common.io import write_json


def map_directory(root, scene_id):
    scene = Path(scene_id.removeprefix("data/scene_datasets/"))
    if scene.is_absolute() or ".." in scene.parts:
        raise ValueError(f"Scene ID must be dataset-relative: {scene_id}")
    if scene.parts[0] not in {"hm3d", "mp3d"}:
        raise ValueError(f"No benchmark scene-map mapping defined for {scene_id}")
    return Path(root) / scene.parent


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Expected finite numeric map coordinate, got {value!r}")
    return float(value)


def load_wall_map(directory, start_position, layer="walls"):
    """Return a public geometry-only map and private source provenance."""
    if layer not in {"walls", "wall_real"}:
        raise ValueError("Map layer must be walls or wall_real")
    directory = Path(directory)
    sources = {}

    def read(name):
        path = directory / name
        raw = path.read_bytes()
        sources[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    scene, wall_data = read("scene.json"), read(f"{layer}.json")
    if scene.get("scene_id") != wall_data.get("scene_id"):
        raise ValueError(f"Scene and wall IDs disagree: {directory}")
    convention = json.dumps(scene.get("coordinate_system", scene.get("coordinates", scene.get("coordinate_convention", "")))).lower().replace("habitat", "").replace(" ", "")
    if "-z" not in convention:
        raise ValueError(f"Expected published (Habitat X, -Z) map coordinates: {directory}")
    levels = scene.get("levels") or read("levels.json")
    if not isinstance(levels, list) or not levels:
        raise ValueError(f"Missing floor definitions: {directory}")
    heights = [(row, _number(row.get("floor_height", row.get("height_m")))) for row in levels]
    matches = [(row, height) for row, height in heights if abs(height - _number(start_position[1])) <= .5]
    if len(matches) != 1:
        raise ValueError(f"Start height {start_position[1]} must match exactly one floor within 0.5m: {directory}")
    level, height = matches[0]
    if sum(row["id"] == level["id"] for row in levels) != 1:
        raise ValueError(f"Duplicate floor ID: {level['id']}")
    segments = wall_data["segments"]
    walls = []
    for segment in segments:
        if segment.get("level") != level["id"] or segment.get("kind") == "railing":
            continue
        points = [segment["start"], segment["end"]]
        if any(len(point) != 2 for point in points):
            raise ValueError("Wall segments must have two 2D endpoints")
        walls.append([[_number(point[0]), -_number(point[1])] for point in points])
    if not walls:
        raise ValueError(f"No wall segments on floor {level['id']}: {directory}")
    points = [p for segment in walls for p in segment]
    bounds = level.get("bounds")
    if bounds is not None:
        # Published bounds: [[xmin,xmax],[ymin,ymax]], or the same four scalars.
        if len(bounds) == 4:
            bounds = [bounds[:2], bounds[2:]]
        if len(bounds) != 2 or any(len(axis) != 2 for axis in bounds):
            raise ValueError("Invalid published floor bounds")
        x, y = [[_number(v) for v in axis] for axis in bounds]
        if x[0] >= x[1] or y[0] >= y[1]:
            raise ValueError("Invalid published floor bounds ordering")
        points.extend([[x[0], -y[1]], [x[1], -y[0]]])
    low = [min(p[i] for p in points) - 2 for i in (0, 1)]
    high = [max(p[i] for p in points) + 2 for i in (0, 1)]
    size = math.ceil(max(high[i] - low[i] for i in (0, 1)) / 10) * 10
    origin = [(low[i] + high[i] - size) / 2 for i in (0, 1)]
    if any(not origin[i] <= start_position[axis] <= origin[i] + size for i, axis in enumerate((0, 2))):
        raise ValueError(f"Start position is outside the published map: {directory}")
    public = {"origin": origin, "size_m": size, "floor_height": height, "walls": walls}
    provenance = {"directory": str(directory.resolve()), "scene_id": scene["scene_id"],
                  "layer": layer, "level": level["id"], "floor_height": height,
                  "conversion": "published (X,-Z) to Harness (X,Z)", "source_sha256": sources}
    return public, provenance


def prepare_map(source, start_position, private_directory, layer="walls"):
    """Materialize only the normalized public wall map inside this run.

An explicitly supplied normalized map.json remains a valid input for local
sequence fixtures; benchmark planning resolves published scene directories.
"""
    source = Path(source)
    if source.is_file():
        return source
    public, provenance = load_wall_map(source, start_position, layer)
    private_directory = Path(private_directory)
    destination = private_directory / "map.json"
    write_json(destination, public)
    write_json(private_directory / "map_source.json", provenance)
    return destination
