"""Publish construction archives once; load frozen map geometry at runtime.

Construction uses a metric plan (Habitat X, -Z). Published map.json uses Habitat
(X,Z) metres, one calibrated square per floor shared by walls/wall_real. The
runtime only selects a floor/layer and copies its public geometry.
"""

import argparse
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
    scene = Path(canonical_scene_id(scene_id))
    root = Path(root)
    if scene.parts[0] in {"hm3d", "mp3d"}:
        return root / scene.parent
    # Scene instances (e.g. HSSD) do not encode the train/val map directory.
    # Their asset-to-map association is published explicitly from the scene queue.
    index = json.loads((root / "index.json").read_text())
    relative = Path(index[scene.as_posix()])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid published map index path")
    return root / relative


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Expected finite numeric map coordinate, got {value!r}")
    return float(value)


def _archive_coordinates(scene):
    declaration = next((scene[key] for key in ("coordinate_system", "coordinates",
                       "coordinate_convention", "map_coordinates") if key in scene), None)
    if isinstance(declaration, dict):
        declaration = next((declaration[key] for key in ("horizontal", "map", "map_xy", "map_axes", "plan")
                            if key in declaration), declaration)
    compact = json.dumps(declaration).lower().replace("habitat", "").replace('"', '')
    # Require the declared map pair; a source-mesh transform mentioning -Z is
    # insufficient. Unknown conventions require an explicit offline correction.
    if not re.search(r'(?<![a-z])x\s*,\s*-\s*z(?![a-z])', compact):
        raise ValueError("Archive must explicitly declare map coordinates (Habitat X,-Z)")
    if scene.get("units", scene.get("unit")) not in {"m", "metre", "metres", "meter", "meters"}:
        raise ValueError("Archive map coordinates must use metres")


def publish_map(source, destination):
    """Convert a completed archive to one geometry-only map.json, atomically.

    Source can be an archive root with raw/, or the directory containing
    scene.json directly. Existing source files are never changed.
    """
    source, destination = Path(source), Path(destination)
    if (source / "raw/scene.json").is_file():
        source = source / "raw"
    hashes = {}

    def read(name):
        path = source / name
        raw = path.read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    scene = read("scene.json")
    if scene.get("status") in {"partial", "blocked", "building", "pending"}:
        raise ValueError("Cannot publish an unfinished scene archive")
    _archive_coordinates(scene)
    levels = scene.get("levels") or read("levels.json")
    if not isinstance(levels, list) or not levels:
        raise ValueError("Archive requires floor definitions")
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
                if segment["level"] != level["id"] or segment.get("kind") == "railing":
                    continue
                points = [segment["start"], segment["end"]]
                if any(not isinstance(p, list) or len(p) != 2 for p in points):
                    raise ValueError("Wall segments require two 2D endpoints")
                converted = [[_number(p[0]), -_number(p[1])] for p in points]
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
                raise ValueError("Invalid archive floor bounds")
            x, y = [[_number(v) for v in axis] for axis in bounds]
            if x[0] >= x[1] or y[0] >= y[1]:
                raise ValueError("Invalid archive floor bounds ordering")
            points.extend([[x[0], -y[1]], [x[1], -y[0]]])
        low = [min(p[i] for p in points) - 2 for i in (0, 1)]
        high = [max(p[i] for p in points) + 2 for i in (0, 1)]
        size = math.ceil(max(high[i] - low[i] for i in (0, 1)) / 10) * 10
        origin = [(low[i] + high[i] - size) / 2 for i in (0, 1)]
        for walls in geometry.values():
            validate_map({"origin": origin, "size_m": size, "floor_height": height, "walls": walls})
        floors.append({"id": level["id"], "floor_height": height,
                       "origin": origin, "size_m": size, "layers": geometry})
    if not floors:
        raise ValueError("No floors with publishable wall geometry")
    value = {"scene_id": scene["scene_id"], "coordinate_system": COORDINATES, "levels": floors,
             "source": {"directory": str(source.resolve()), "coordinates": "habitat_x_minus_z_metres",
                        "excluded_levels": excluded,
                        "sha256": hashes}}
    if any(hashlib.sha256((source / name).read_bytes()).hexdigest() != digest
           for name, digest in hashes.items()):
        raise ValueError("Archive changed while publishing; retry after its build finishes")
    write_json(destination / "map.json", value)
    return destination / "map.json"


def load_wall_map(directory, start_position, layer="walls"):
    """Select already-published geometry; never read construction archives."""
    if layer not in LAYERS:
        raise ValueError("Map layer must be walls or wall_real")
    directory = Path(directory)
    path = directory / "map.json"
    if not path.is_file():
        raise FileNotFoundError(f"Publish this map before running: {path} (python -m src.dataset.maps publish)")
    raw = path.read_bytes()
    document = json.loads(raw)
    if document.get("coordinate_system") != COORDINATES:
        raise ValueError("Published map must declare habitat_xz_metres coordinates")
    if (not isinstance(start_position, (list, tuple)) or len(start_position) != 3):
        raise ValueError("Start position must be Habitat [X,Y,Z] metres")
    start_position = [_number(v) for v in start_position]
    levels = document["levels"]
    if len({row["id"] for row in levels}) != len(levels):
        raise ValueError("Duplicate floor ID")
    matches = [row for row in levels if abs(_number(row["floor_height"]) - start_position[1]) <= .5]
    if len(matches) != 1:
        raise ValueError(f"Start height {start_position[1]} must match exactly one floor within 0.5m: {directory}")
    level = matches[0]
    if layer not in level["layers"]:
        raise ValueError(f"Published layer {layer} is unavailable on floor {level['id']}")
    public = validate_map({key: level[key] for key in ("origin", "size_m", "floor_height")} |
                          {"walls": level["layers"][layer]})
    if any(not 0 <= v <= 1 for v in world_to_uv(start_position, public)):
        raise ValueError(f"Start position is outside the published map: {directory}")
    provenance = {"directory": str(directory.resolve()), "scene_id": document["scene_id"],
                  "layer": layer, "level": level["id"], "floor_height": level["floor_height"],
                  "coordinate_system": COORDINATES, "map_sha256": hashlib.sha256(raw).hexdigest(),
                  "source": document["source"]}
    return public, provenance


def prepare_map(source, start_position, private_directory, layer="walls"):
    """Copy the selected public floor into a run; accept explicit local fixtures."""
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


def publish_tree(source_root, output_root, queues=()):
    """Publish only declared scene directories, without walking raw assets."""
    from tqdm import tqdm
    source_root, output_root = Path(source_root), Path(output_root)
    paths = set()
    for pattern in ("*/*/scene.json", "*/*/*/scene.json", "*/*/raw/scene.json", "*/*/*/raw/scene.json"):
        paths.update(source_root.glob(pattern))
    sources = sorted({p.parent.parent if p.parent.name == "raw" else p.parent for p in paths})
    results = []
    published = set()
    for source in tqdm(sources, desc="Publish maps", unit="scene"):
        relative = source.relative_to(source_root)
        try:
            path = publish_map(source, output_root / relative)
            results.append({"scene": relative.as_posix(), "status": "published", "path": str(path)})
            published.add(relative.as_posix())
        except (OSError, ValueError, KeyError, TypeError) as error:
            results.append({"scene": relative.as_posix(), "status": "error", "error": str(error)})
    index = {}
    for queue_path in queues:
        queue = json.loads(Path(queue_path).read_text())
        scene_root, archive_root = Path(queue["scene_root"]), Path(queue["output_root"])
        for scene in queue["scenes"]:
            relative = Path(scene["output_dir"]).relative_to(archive_root).as_posix()
            if relative not in published:
                continue
            asset = canonical_scene_id(Path(scene["scene_path"]).relative_to(scene_root).as_posix())
            if asset in index and index[asset] != relative:
                raise ValueError(f"Conflicting map directories for {asset}")
            index[asset] = relative
    write_json(output_root / "index.json", index)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["publish"])
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--queue", type=Path, action="append", default=[])
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    results = publish_tree(args.source_root, args.output_root, args.queue)
    write_json(args.report, results)
    print(json.dumps({"published": sum(r["status"] == "published" for r in results),
                      "errors": sum(r["status"] == "error" for r in results), "report": str(args.report)}))
    return int(any(r["status"] == "error" for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
