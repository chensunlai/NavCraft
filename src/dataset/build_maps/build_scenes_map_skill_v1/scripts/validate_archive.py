#!/usr/bin/env python3
"""Read-only archive/geometry checks. Does not certify visual truth or navigation."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from shapely.affinity import rotate, scale
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union


def validate(scene_dir, tolerance=0.005):
    root = scene_dir.resolve()
    raw = root / "raw"
    errors, warnings = [], []

    def fail(kind, detail):
        errors.append({"kind": kind, "detail": detail})

    def local(base, name):
        p = (base / name).resolve()
        if not p.is_relative_to(root):
            raise ValueError(f"archive reference escapes scene directory: {name}")
        return p

    def load(name):
        return json.loads((raw / name).read_text())

    def xy(value):
        if len(value) != 2 or not all(math.isfinite(float(v)) for v in value):
            raise ValueError(f"invalid XY: {value}")
        return value

    expected = {
        "floor_plan.png",
        "scene_description.md",
        "evidence",
        "evidence_rooms.jpg",
        "raw",
    }
    actual = {p.name for p in root.iterdir()}
    if actual != expected:
        fail(
            "top_level_layout",
            {"missing": sorted(expected - actual), "extra": sorted(actual - expected)},
        )
    for name in ["floor_plan.png", "scene_description.md", "evidence_rooms.jpg"]:
        if not (root / name).is_file():
            fail("required_file", name)
    if not (root / "evidence").is_dir():
        fail("required_directory", "evidence")
    scene = load("scene.json")
    levels = {l["id"] for l in scene["levels"]}
    regions = load("regions.json")["regions"]
    rooms = load("rooms.json")["rooms"]
    objects = load("objects.json")["objects"]
    walls = load("walls.json")["segments"]
    doors = load("doors.json")["doors"]
    evidence = load("evidence.json")["frames"]
    manifest = load("manifest.json")
    for filename in [
        "walls.json",
        "regions.json",
        "rooms.json",
        "objects.json",
        "doors.json",
        "evidence.json",
        "manifest.json",
    ]:
        if load(filename).get("scene_id") != scene["scene_id"]:
            fail("scene_id_mismatch", filename)
    for name in scene.get("files", {}).values():
        if not local(raw, name).is_file():
            fail("scene_file_reference", name)
    byframe = {f["id"]: f for f in evidence}
    byobject = {o["id"]: o for o in objects}
    if len(byframe) != len(evidence):
        fail("duplicate_frame_id", len(evidence) - len(byframe))
    if len(byobject) != len(objects):
        fail("duplicate_object_id", len(objects) - len(byobject))
    for frame in evidence:
        if not local(raw, frame["rgb"]).is_file():
            fail("missing_frame_image", frame["id"])
        if not frame.get("source"):
            warnings.append({"kind": "frame_source_missing", "id": frame["id"]})

    def check_refs(record, required=False):
        refs = record.get("evidence_frames", [])
        if required and not refs:
            fail("missing_evidence", record["id"])
        for fid in refs:
            if fid not in byframe:
                fail("unknown_evidence_frame", [record["id"], fid])

    polygons = {}
    for region in regions:
        key = (region["level"], region["id"])
        if key in polygons:
            fail("duplicate_region", key)
        if key[0] not in levels:
            fail("unknown_level", key)
        poly = Polygon([xy(p) for p in region["polygon"]])
        if not poly.is_valid or poly.area <= 0:
            fail("invalid_region_geometry", key)
        polygons[key] = poly
    if errors and any(e["kind"] == "invalid_region_geometry" for e in errors):
        return {"structural_pass": False, "errors": errors, "warnings": warnings}
    keys = list(polygons)
    for i, key in enumerate(keys):
        for other in keys[i + 1 :]:
            if (
                key[0] == other[0]
                and polygons[key].intersection(polygons[other]).area > tolerance**2
            ):
                fail("overlapping_regions", [key, other])
    room_keys = set()
    for room in rooms:
        key = (room["level"], room["id"])
        if key in room_keys:
            fail("duplicate_room_record", key)
        room_keys.add(key)
        if key not in polygons:
            fail("room_without_region", key)
        check_refs(room, required=True)
        if not room.get("description") or not room.get("function"):
            fail("missing_room_description", key)
        for oid in room.get("object_ids", []):
            if (
                oid not in byobject
                or (byobject[oid]["level"], byobject[oid]["room"]) != key
            ):
                fail("room_object_mismatch", [key, oid])
        for observed in room.get("additional_observed_objects", []):
            check_refs({"id": key, **observed})
    if room_keys != set(polygons):
        fail("region_room_set_mismatch", sorted(set(polygons) - room_keys))
    shapes = {}
    for obj in objects:
        check_refs(obj, required=True)
        pieces = []
        for primitive in obj["primitives"]:
            x, y = xy(primitive["center"])
            w, h = xy(primitive["size"])
            angle = float(primitive.get("angle_deg", 0))
            if min(w, h) <= 0 or not math.isfinite(angle):
                raise ValueError(f'invalid primitive size/angle for {obj["id"]}')
            if primitive["type"] == "rectangle":
                shape = box(x - w / 2, y - h / 2, x + w / 2, y + h / 2)
            elif primitive["type"] == "ellipse":
                shape = scale(Point(x, y).buffer(1), w / 2, h / 2)
                warnings.append({"kind": "non_box_shape_review", "id": obj["id"]})
            else:
                raise ValueError(f'unsupported primitive type: {primitive["type"]}')
            pieces.append(rotate(shape, angle, origin=(x, y)))
        shape = unary_union(pieces)
        if shape.is_empty:
            fail("empty_object", obj["id"])
        shapes[obj["id"]] = shape
        key = (obj["level"], obj["room"])
        if key not in polygons:
            fail("unknown_object_room", [obj["id"], key])
        elif shape.difference(polygons[key].buffer(tolerance)).area > tolerance**2:
            fail("object_outside_room", obj["id"])
        if obj.get("supported_by") and obj["supported_by"] not in byobject:
            fail("unknown_support_object", obj["id"])
    wall_lines = []
    for wall in walls:
        line = LineString([xy(wall["start"]), xy(wall["end"])])
        if line.length <= 0 or wall["level"] not in levels:
            fail("invalid_wall", wall["id"])
        wall_lines.append((wall, line))
    nesting = []
    for i, obj in enumerate(objects):
        shape = shapes[obj["id"]]
        for wall, line in wall_lines:
            if wall["level"] == obj["level"] and line.intersects(
                shape.buffer(-tolerance)
            ):
                fail(
                    "wall_object_intersection_requires_review", [wall["id"], obj["id"]]
                )
        for other in objects[i + 1 :]:
            if obj["level"] != other["level"]:
                continue
            overlap = shape.intersection(shapes[other["id"]]).area
            if overlap <= tolerance**2:
                continue
            child, parent = (
                (obj, other) if obj.get("supported_by") == other["id"] else (other, obj)
            )
            if (
                child.get("supported_by") == parent["id"]
                and shapes[child["id"]]
                .difference(shapes[parent["id"]].buffer(tolerance))
                .area
                <= tolerance**2
            ):
                nesting.append([child["id"], parent["id"]])
            else:
                fail(
                    "independent_object_overlap",
                    [obj["id"], other["id"], round(overlap, 5)],
                )
    open_statuses = {
        "open",
        "confirmed_open",
        "rgb_and_local_navmesh_verified",
        "reported_open_in_xhigh",
    }
    for door in doors:
        check_refs(door)
        if door.get("status") not in open_statuses:
            continue
        if not door.get("evidence_frames"):
            fail("confirmed_portal_without_evidence", door["id"])
        line = LineString([xy(p) for p in door["segment"]])
        interior = LineString(
            [
                line.interpolate(0.02, normalized=True),
                line.interpolate(0.98, normalized=True),
            ]
        )
        for wall, wall_line in wall_lines:
            if wall["level"] == door["level"] and interior.intersects(wall_line):
                fail("open_portal_blocked_by_wall", [door["id"], wall["id"]])
    for file in [root / "scene_description.md", raw / "README.md", raw / "index.html"]:
        if not file.exists():
            continue
        text = file.read_text()
        refs = (
            re.findall(r"\]\(([^)]+)\)", text)
            if file.suffix == ".md"
            else re.findall(r'(?:href|src)="([^\"]+)"', text)
        )
        for ref in refs:
            if "://" in ref or ref.startswith("#"):
                continue
            if not local(file.parent, ref.split("#")[0]).exists():
                fail("broken_document_link", [str(file.relative_to(root)), ref])
    if manifest.get("checksum_path_base") is None:
        warnings.append({"kind": "unspecified_checksum_path_base"})
    else:
        base = local(raw, manifest["checksum_path_base"])
        for name, digest in manifest.get("checksums", {}).items():
            file = local(base, name)
            if (
                not file.is_file()
                or hashlib.sha256(file.read_bytes()).hexdigest() != digest
            ):
                fail("checksum_mismatch", name)
    warnings.append(
        {
            "kind": "manual_review_required",
            "detail": "Rendered evidence, actual room/door identity, floor assignment, graph reachability and local path crossing must be reviewed separately.",
        }
    )
    return {
        "scene_id": scene["scene_id"],
        "structural_pass": not errors,
        "counts": {
            "regions": len(regions),
            "objects": len(objects),
            "frames": len(evidence),
        },
        "intended_fixture_nesting": nesting,
        "errors": errors,
        "warnings": warnings,
        "tolerance_m": tolerance,
        "scope": "File and geometric consistency only; not a reviewed-quality or navigation success certificate.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_directory", type=Path)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.005,
        help="Numerical geometric tolerance in metres, not a positioning accuracy claim.",
    )
    args = parser.parse_args()
    if not 0 < args.tolerance <= 0.02:
        parser.error(
            "tolerance must be > 0 and <= 0.02 m; do not hide layout errors with a large tolerance"
        )
    try:
        report = validate(args.scene_directory, args.tolerance)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report = {
            "structural_pass": False,
            "errors": [{"kind": "invalid_archive", "detail": str(exc)}],
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["structural_pass"] else 1)


if __name__ == "__main__":
    main()
