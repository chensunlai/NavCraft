"""Public spatial memory conversion, validation, and rendering.

CLI: ``memory.py prepare|check|render|view`` reads one JSON request on stdin.
Only prepare reads an immutable scene archive. Runtime operations read public
workspace files and never consult the simulator or calibration.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import uuid

from PIL import Image, ImageDraw, ImageFont


class MemoryError(ValueError):
    pass


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _path(workspace, ref, roots=None):
    if not isinstance(ref, str) or not ref or "\\" in ref:
        raise MemoryError("Expected a workspace-relative path")
    relative = Path(ref)
    if relative.is_absolute() or ".." in relative.parts:
        raise MemoryError(f"Path is outside the public workspace: {ref}")
    if roots and relative.parts[0] not in roots:
        raise MemoryError(f"Path is not an allowed public resource: {ref}")
    root = Path(workspace).resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise MemoryError(f"Path resolves outside the public workspace: {ref}")
    if roots and not any(result.is_relative_to(root / name) for name in roots):
        # Do not resolve the allowed roots: a symlink must not redefine which
        # workspace tree is writable (e.g. scratch -> scene).
        raise MemoryError(f"Path resolves outside allowed resource roots {sorted(roots)}: {ref}")
    return result


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", value):
        raise MemoryError(f"Invalid ID: {value!r}")
    return value


def _point(value):
    if (not isinstance(value, list) or len(value) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in value)):
        raise MemoryError(f"Expected normalized [x,y], got {value!r}")
    return value


def _box(value):
    if not isinstance(value, list) or len(value) != 4:
        raise MemoryError("Expected [left,top,right,bottom]")
    _point(value[:2])
    _point(value[2:])
    if value[0] >= value[2] or value[1] >= value[3]:
        raise MemoryError("Box/crop must have positive width and height")
    return value


def _geometry(value, kind):
    if value is None:
        return
    if not isinstance(value, dict) or len(value) != 1:
        raise MemoryError("Geometry must be null or one shape: box/polygon/segment/shapes")
    key, points = next(iter(value.items()))
    if key == "shapes" and kind == "object":
        if not isinstance(points, list) or not points:
            raise MemoryError("shapes must be a nonempty list")
        for shape in points:
            if "shapes" in shape:
                raise MemoryError("Nested shapes are unsupported")
            _geometry(shape, kind)
        return
    allowed = {"region": {"polygon"}, "portal": {"segment"}, "object": {"box", "polygon"}}
    if key not in allowed.get(kind, set()):
        raise MemoryError(f"Unsupported {kind} geometry: {key}")
    if key == "box":
        _box(points)
    else:
        minimum = 3 if key == "polygon" else 2
        if not isinstance(points, list) or len(points) < minimum:
            raise MemoryError(f"{key} needs at least {minimum} points")
        for point in points:
            _point(point)


def _maps(workspace):
    index = _json(_path(workspace, "scene/maps/index.json"))
    maps = {}
    for row in index["maps"]:
        name = _id(row["map_id"])
        if name in maps:
            raise MemoryError(f"Duplicate map ID: {name}")
        if any(type(row.get(k)) is not int or row[k] <= 0 for k in ("width", "height")):
            raise MemoryError(f"Invalid map dimensions: {name}")
        _path(workspace, row["base_ref"], {"scene"})
        _path(workspace, row["current_entities_ref"], {"memory"})
        maps[name] = row
    return maps


def _image_index(workspace):
    result = {}
    for name in ("scene/evidence/index.jsonl", "observations/index.jsonl", "scratch/images/index.jsonl"):
        path = _path(workspace, name, {name.split("/", 1)[0]})
        if not path.exists():
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = _id(row["id"])
            if key in result:
                raise MemoryError(f"Duplicate image ID {key} at {name}:{number}")
            _path(workspace, row["path"], {"scene", "observations", "scratch"})
            result[key] = row
    return result


def _font(size=20):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _record_image(workspace, picture, prefix, metadata):
    image_id = f"{prefix}_{uuid.uuid4().hex[:12]}"
    ref = f"scratch/images/{image_id}.png"
    path = _path(workspace, ref, {"scratch"})
    index_path = _path(workspace, "scratch/images/index.jsonl", {"scratch"})
    path.parent.mkdir(parents=True, exist_ok=True)
    picture.save(path)
    row = {"id": image_id, "path": ref, "source": "derived", "step": None,
           "view": None, "derived_from": [], **metadata,
           "width": picture.width, "height": picture.height, "sha256": _sha(path)}
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def prepare(scene_dir, workspace, mode="prebuilt", allowed_ids=None, initial_pose=None,
            goal_id=None, initial_frame=None):
    """Trusted initialization: expose only task-approved authored information."""
    scene_dir, workspace = Path(scene_dir).resolve(), Path(workspace).resolve()
    if workspace == scene_dir or workspace.is_relative_to(scene_dir):
        raise MemoryError("Workspace must be outside the scene archive")
    if (workspace / "scene/index.md").exists():
        raise MemoryError("Refusing to overwrite an initialized workspace")
    if mode not in {"prebuilt", "partial", "empty"}:
        raise MemoryError("mode must be prebuilt, partial or empty")
    if mode == "partial" and allowed_ids is None:
        raise MemoryError("partial mode needs explicit allowed_ids")
    # Calibration is used only by this trusted archive conversion entry point.
    repository = Path(__file__).resolve().parents[3]
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    from src.harness.map import MapCalibration

    calibration = MapCalibration(scene_dir)
    raw = scene_dir / "raw"
    rooms = _json(raw / "rooms.json")["rooms"]
    objects = _json(raw / "objects.json")["objects"]
    doors = [item for item in _json(raw / "doors.json")["doors"]
             if item.get("status") == "rgb_and_local_navmesh_verified"]
    walls = _json(raw / "walls.json")["segments"]
    evidence = _json(raw / "evidence.json")["frames"]
    authored = [(item, kind) for rows, kind in ((rooms, "region"), (doors, "portal"), (objects, "object")) for item in rows]
    available = {item["id"] for item, _ in authored}
    selected = set() if mode == "empty" else available if allowed_ids is None else set(allowed_ids)
    if selected - available:
        raise MemoryError(f"Unknown allowed entity IDs: {sorted(selected - available)}")
    for folder in ("scene/maps", "scene/initial/maps", "scene/initial/entities", "scene/evidence",
                   "memory/maps", "memory/entities", "observations", "scratch/images"):
        _path(workspace, folder).mkdir(parents=True, exist_ok=True)
    width, height = 1800, 1440
    vx, vy, vw, vh = calibration.viewbox
    maps, entities_by_map, cards = [], {}, {}
    evidence_ids = set()
    for panel in calibration.panels:
        level = _id(panel.level_id)
        map_id = f"map_{level}"

        def normalized(point):
            return [round((panel.x_scale * point[0] + panel.x_offset - vx) / vw, 8),
                    round((panel.y_scale * point[1] + panel.y_offset - vy) / vh, 8)]

        base = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(base)
        for wall in walls:
            if wall["level"] == level:
                pts = [normalized(wall[key]) for key in ("start", "end")]
                draw.line([(x * width, y * height) for x, y in pts], fill="#222d38", width=5)
        base_ref = f"scene/maps/{level}.base.png"
        base.save(_path(workspace, base_ref))
        row = {"map_id": map_id, "base_ref": base_ref,
               "current_entities_ref": f"memory/maps/{level}.entities.json",
               "width": width, "height": height, "coordinates": "normalized x-right/y-down; original map"}
        maps.append(row)
        entities = []
        for item, kind in authored:
            if item["level"] != level or item["id"] not in selected:
                continue
            name = _id(item["id"])
            if kind == "region":
                geometry = {"polygon": [normalized(p) for p in item["polygon"]]}
            elif kind == "portal":
                geometry = {"segment": [normalized(p) for p in item["segment"]]}
            else:
                shapes = []
                for primitive in item["primitives"]:
                    if primitive["type"] != "rectangle":
                        raise MemoryError("Archive contains an unsupported object primitive")
                    cx, cy = primitive["center"]
                    sx, sy = primitive["size"]
                    angle = math.radians(primitive.get("angle_deg", 0))
                    corners = [[cx + x * math.cos(angle) - y * math.sin(angle),
                                cy + x * math.sin(angle) + y * math.cos(angle)]
                               for x, y in [(-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2)]]
                    points = [normalized(point) for point in corners]
                    if angle == 0:
                        shapes.append({"box": [min(p[0] for p in points), min(p[1] for p in points),
                                                max(p[0] for p in points), max(p[1] for p in points)]})
                    else:
                        shapes.append({"polygon": points})
                geometry = shapes[0] if len(shapes) == 1 else {"shapes": shapes}
            _geometry(geometry, kind)
            entities.append({"id": name, "map_id": map_id, "type": kind, "geometry": geometry})
            refs = item.get("evidence_frames", [])
            evidence_ids.update(refs)
            title = item.get("name", item.get("label", "门洞" if kind == "portal" else name))
            description = item.get("description", title)
            description = re.sub(r"\b[RODPC]\d+\b", lambda m: m.group() if m.group() in selected else "未公开实体", description)
            fields = [("地图", f"{map_id} 中的 {name}"), ("类别", item.get("category", kind)),
                      ("描述", description)]
            if kind == "object":
                fields.append(("所属区域", item.get("room") if item.get("room") in selected else "未知"))
            if kind == "portal":
                fields.append(("连接", " 与 ".join(x if x in selected else "未知" for x in item.get("connects", []))))
            fields.extend([("证据", ", ".join(f"`{ref}`" for ref in refs) or "无"),
                           ("来源", f"scene/initial/entities/{name}.md；尚未在本次运行重新核实"),
                           ("未确认", "当前外观、阻挡和可通过状态需结合新观察；图形为近似位置")])
            cards[name] = f"# {name}：{title}\n\n" + "\n".join(f"- {key}：{value}" for key, value in fields) + "\n"
        entities_by_map[map_id] = entities
        _save_json(_path(workspace, row["current_entities_ref"]), entities)
        _save_json(_path(workspace, f"scene/initial/maps/{level}.entities.json"), entities)
    _save_json(_path(workspace, "scene/maps/index.json"), {"maps": maps})
    for name, text in cards.items():
        for folder in ("scene/initial/entities", "memory/entities"):
            _path(workspace, f"{folder}/{name}.md").write_text(text, encoding="utf-8")
    if mode == "prebuilt" and allowed_ids is None:
        evidence_ids = {item["id"] for item in evidence}
    rows = []
    for item in evidence:
        if item["id"] not in evidence_ids:
            continue
        source = (raw / item["rgb"]).resolve()
        if not source.is_relative_to(scene_dir / "evidence") or source.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise MemoryError(f"Invalid public RGB source for {item['id']}")
        name = _id(item["id"])
        ref = f"scene/evidence/{name}{source.suffix.lower()}"
        destination = _path(workspace, ref)
        shutil.copyfile(source, destination)
        with Image.open(destination) as image:
            size = image.size
        rows.append({"id": name, "path": ref, "source": "prior", "step": None, "view": None,
                     "derived_from": [], "width": size[0], "height": size[1], "sha256": _sha(destination)})
    missing = evidence_ids - {row["id"] for row in rows}
    if missing:
        raise MemoryError(f"Archive cards reference missing evidence: {sorted(missing)}")
    _path(workspace, "scene/evidence/index.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    state = {"goal_id": goal_id, "at_step": 0, "pose_estimate": None,
             "subgoal": None, "open_questions": [], "evidence_refs": []}
    if initial_pose is not None:
        level, position, heading = calibration.initial_arrow(initial_pose)
        dx, dy = heading[0] - position[0], heading[1] - position[1]
        scale = min(1.0, 0.04 / max(math.hypot(dx, dy), 1e-9))
        while not all(0 <= v <= 1 for v in (position[0] + dx * scale, position[1] + dy * scale)):
            scale *= 0.5
        state["pose_estimate"] = {"map_id": f"map_{level}", "at_frame": initial_frame,
                                  "position": list(position), "heading_point": [position[0] + dx * scale, position[1] + dy * scale],
                                  "status": "initial_hint"}
    _save_json(_path(workspace, "memory/state.json"), state)
    _path(workspace, "scene/index.md").write_text(
        "# 场景资料\n\n地图入口：`scene/maps/index.json`。\n"
        "当前图元：`memory/maps/`；当前实体卡片：`memory/entities/`；状态：`memory/state.json`。\n"
        "证据索引：`scene/evidence/index.jsonl`；运行观察：`observations/index.jsonl`。\n"
        "坐标以原地图左上角为[0,0]，右下角为[1,1]；比例尺未公开，图上空白不保证可通行。\n"
        "初始资料仅供追溯，绘图和查询以当前记忆为准；删除实体不会恢复初始版本。\n"
        "实体JSON允许region/polygon、portal/segment、object/box或polygon或shapes；geometry:null表示未定位。\n"
        "卡片使用地图、所属区域、连接、关系、证据、来源、未确认等标签；证据ID用反引号引用。\n",
        encoding="utf-8")
    return {"ok": True, "data": {"maps": maps, "entity_count": len(cards), "evidence_count": len(rows),
                                   "mode": mode, "entry_ref": "scene/index.md"}, "images": []}


def check(workspace):
    """Validate public syntax and explicit references, never spatial truth."""
    errors, entities = [], {}
    try:
        maps = _maps(workspace)
        images = _image_index(workspace)
        for map_id, row in maps.items():
            base = _path(workspace, row["base_ref"])
            with Image.open(base) as image:
                if image.size != (row["width"], row["height"]):
                    raise MemoryError(f"Base image size disagrees with {map_id}")
            records = _json(_path(workspace, row["current_entities_ref"]))
            if not isinstance(records, list):
                raise MemoryError(f"{row['current_entities_ref']}: expected an array")
            for entity in records:
                name = _id(entity["id"])
                if name in entities:
                    raise MemoryError(f"Duplicate current entity ID: {name}")
                if entity.get("map_id") != map_id or entity.get("type") not in {"region", "portal", "object"}:
                    raise MemoryError(f"{name}: invalid map_id/type")
                if "geometry" not in entity:
                    raise MemoryError(f"{name}: missing geometry")
                try:
                    _geometry(entity["geometry"], entity["type"])
                except MemoryError as exc:
                    raise MemoryError(f"{name}: {exc}") from exc
                entities[name] = entity
        card_paths = list(_path(workspace, "memory/entities").glob("*.md"))
        if {p.stem for p in card_paths} != set(entities):
            raise MemoryError("Current cards and entity IDs differ: " + str(sorted({p.stem for p in card_paths} ^ set(entities))))
        for name, entity in entities.items():
            text = _path(workspace, f"memory/entities/{name}.md").read_text(encoding="utf-8")
            for label, value in re.findall(r"^-\s*([^：:]+)[：:]\s*(.*)$", text, flags=re.M):
                if label == "地图" and entity["map_id"] not in value:
                    raise MemoryError(f"{name}: card map disagrees with geometry")
                if label in {"所属区域", "连接", "关系"}:
                    refs = set(re.findall(r"\b[RODPC]\d+\b", value))
                    refs.update(re.findall(r"`([A-Za-z][A-Za-z0-9_-]*)`", value))
                    for ref in sorted(refs):
                        if ref not in entities:
                            raise MemoryError(f"{name}: unknown entity reference {ref}")
                if label == "证据":
                    refs = set(re.findall(r"`([^`]+)`", value))
                    refs.update(re.findall(r"\b(?:render_|review_|frame_|rgb_|map_|crop_|grid_|f_|E_)[A-Za-z0-9_-]+", value))
                    for ref in sorted(refs):
                        if ref not in images:
                            raise MemoryError(f"{name}: unknown evidence {ref}")
        for name, row in images.items():
            path = _path(workspace, row["path"], {"scene", "observations", "scratch"})
            if not path.is_file():
                raise MemoryError(f"{name}: missing image {row['path']}")
        state = _json(_path(workspace, "memory/state.json"))
        pose = state.get("pose_estimate")
        if pose is not None:
            if pose.get("map_id") not in maps:
                raise MemoryError("state.pose_estimate: unknown map_id")
            if pose.get("position") is not None:
                _point(pose["position"])
                _point(pose["heading_point"])
                if pose["position"] == pose["heading_point"]:
                    raise MemoryError("Position and heading point must differ")
            frame = pose.get("at_frame")
            if frame is not None and frame not in images:
                raise MemoryError(f"state.pose_estimate: unknown frame {frame}")
        for ref in state.get("evidence_refs", []):
            if ref not in images:
                raise MemoryError(f"state: unknown evidence {ref}")
    except (MemoryError, OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(str(exc))
    return {"ok": not errors, "errors": errors, "entity_count": len(entities),
            "data": {"errors": errors, "entity_count": len(entities)}, "images": []}


def _draw_shape(draw, geometry, width, height, kind, color):
    if "shapes" in geometry:
        for shape in geometry["shapes"]:
            _draw_shape(draw, shape, width, height, kind, color)
        return
    key, data = next(iter(geometry.items()))
    if key == "box":
        points = [(data[0] * width, data[1] * height), (data[2] * width, data[3] * height)]
        draw.rectangle(points, fill=(*color, 42), outline=(*color, 230), width=3)
    else:
        points = [(x * width, y * height) for x, y in data]
        if key == "polygon":
            draw.polygon(points, fill=(*color, 20 if kind == "region" else 42))
            draw.line(points + [points[0]], fill=(*color, 210), width=2)
        else:
            draw.line(points, fill=(*color, 255), width=8)


def render(workspace, map_id, region=None, highlight_ids=None, grid=False, layers=None):
    validation = check(workspace)
    if not validation["ok"]:
        raise MemoryError("; ".join(validation["errors"]))
    maps = _maps(workspace)
    if map_id not in maps:
        raise MemoryError(f"Unknown map_id: {map_id}")
    row = maps[map_id]
    layers = ["walls", "region", "portal", "object", "pose"] if layers is None else layers
    if not isinstance(layers, list) or set(layers) - {"walls", "region", "portal", "object", "pose"}:
        raise MemoryError("Unknown map layer")
    region = [0, 0, 1, 1] if region is None else _box(region)
    entities_path = _path(workspace, row["current_entities_ref"])
    entities = _json(entities_path)
    highlight_ids = [] if highlight_ids is None else highlight_ids
    missing = set(highlight_ids) - {item["id"] for item in entities}
    if missing:
        raise MemoryError(f"Unknown highlight IDs for {map_id}: {sorted(missing)}")
    base = Image.open(_path(workspace, row["base_ref"])).convert("RGBA")
    if "walls" not in layers:
        base = Image.new("RGBA", base.size, "white")
    overlay = Image.new("RGBA", base.size)
    draw = ImageDraw.Draw(overlay)
    width, height = base.size
    unlocated = []
    labels = []
    colors = {"region": (55, 110, 170), "portal": (0, 150, 105), "object": (169, 102, 34)}
    for item in entities:
        geometry, kind = item["geometry"], item["type"]
        if geometry is None:
            unlocated.append(item["id"])
            continue
        if kind not in layers:
            continue
        color = (210, 35, 60) if item["id"] in highlight_ids else colors[kind]
        _draw_shape(draw, geometry, width, height, kind, color)
        shape = geometry.get("shapes", [geometry])[0]
        key, points = next(iter(shape.items()))
        if key == "box":
            center = [(points[0] + points[2]) / 2, (points[1] + points[3]) / 2]
        else:
            center = [sum(p[i] for p in points) / len(points) for i in (0, 1)]
        labels.append((item["id"], center, color, kind))
    for name, center, color, kind in sorted(labels, key=lambda label: label[3] == "region"):
        point = (center[0] * width, center[1] * height)
        font = _font(25 if kind == "region" else 20)
        bounds = draw.textbbox(point, name, font=font, anchor="mm")
        draw.rounded_rectangle((bounds[0] - 3, bounds[1] - 2, bounds[2] + 3, bounds[3] + 2),
                               radius=3, fill=(255, 255, 255, 225))
        draw.text(point, name, font=font, fill=(*color, 255), anchor="mm")
    state_path = _path(workspace, "memory/state.json")
    state = _json(state_path)
    pose = state.get("pose_estimate")
    shown_pose = pose if pose and pose.get("map_id") == map_id and pose.get("position") is not None else None
    if "pose" in layers and shown_pose:
        a = (pose["position"][0] * width, pose["position"][1] * height)
        b = (pose["heading_point"][0] * width, pose["heading_point"][1] * height)
        dx, dy = b[0] - a[0], b[1] - a[1]
        norm = math.hypot(dx, dy)
        ux, uy = dx / norm, dy / norm
        draw.line([a, b], fill="#e02953", width=7)
        draw.polygon([b, (b[0] - 18 * ux + 8 * uy, b[1] - 18 * uy - 8 * ux),
                      (b[0] - 18 * ux - 8 * uy, b[1] - 18 * uy + 8 * ux)], fill="#e02953")
        draw.ellipse((a[0] - 7, a[1] - 7, a[0] + 7, a[1] + 7), fill="#e02953")
        label = "Initial hint" if pose.get("status") == "initial_hint" else "Estimate"
        draw.text((a[0] + 12, a[1] + 8), f"{label}: {pose.get('at_frame') or 'initial'}", font=_font(18), fill="#a01842")
    if grid:
        for i in range(1, 10):
            draw.line([(width * i / 10, 0), (width * i / 10, height)], fill=(80, 80, 80, 65))
            draw.line([(0, height * i / 10), (width, height * i / 10)], fill=(80, 80, 80, 65))
            draw.text((width * i / 10 + 3, 5), f"{i / 10:.1f}", font=_font(17), fill="#666666")
            draw.text((5, height * i / 10 + 3), f"{i / 10:.1f}", font=_font(17), fill="#666666")
    picture = Image.alpha_composite(base, overlay).convert("RGB")
    bounds = [round(region[0] * width), round(region[1] * height), round(region[2] * width), round(region[3] * height)]
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        raise MemoryError("Map crop is smaller than one pixel")
    picture = picture.crop(bounds)
    actual_region = [bounds[0] / width, bounds[1] / height, bounds[2] / width, bounds[3] / height]
    versions = {row["base_ref"]: _sha(_path(workspace, row["base_ref"])),
                row["current_entities_ref"]: _sha(entities_path), "memory/state.json": _sha(state_path)}
    info = _record_image(workspace, picture, "map", {"map_id": map_id, "map_region": actual_region,
                         "versions": versions, "pose_at_frame": shown_pose.get("at_frame") if shown_pose else None,
                         "derived_from": [row["base_ref"]], "unlocated_ids": unlocated})
    return {"ok": True, "data": info, "images": [info["path"]]}


def view(workspace, refs, crops=None, layout="separate"):
    if not isinstance(refs, list) or not 1 <= len(refs) <= 10 or layout not in {"separate", "grid"}:
        raise MemoryError("Provide 1–10 refs and separate/grid layout")
    if crops is not None and (not isinstance(crops, list) or len(crops) != len(refs)):
        raise MemoryError("crops must contain one value per ref")
    index = _image_index(workspace)
    rows, pictures = [], []
    for ref, crop in zip(refs, crops if crops is not None else [None] * len(refs)):
        if ref in index:
            info = dict(index[ref])
        else:
            info = next((dict(row) for row in index.values() if row["path"] == ref), None)
            if info is None:
                path = _path(workspace, ref, {"scene", "observations", "scratch"})
                if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                    raise MemoryError(f"Not a public image: {ref}")
                info = {"id": ref, "path": ref, "source": "unindexed", "step": None, "view": None, "derived_from": []}
                maps = _maps(workspace) if _path(workspace, "scene/maps/index.json").exists() else {}
                for map_id, map_row in maps.items():
                    if ref == map_row["base_ref"]:
                        info.update(map_id=map_id, map_region=[0, 0, 1, 1], source="base_map")
        path = _path(workspace, info["path"], {"scene", "observations", "scratch"})
        with Image.open(path) as source:
            picture = source.convert("RGB")
        info["sha256"] = _sha(path)
        info.update(width=picture.width, height=picture.height)
        if crop is not None:
            crop = _box(crop)
            w, h = picture.size
            box = [round(crop[0] * w), round(crop[1] * h), round(crop[2] * w), round(crop[3] * h)]
            if box[0] >= box[2] or box[1] >= box[3]:
                raise MemoryError("Image crop is smaller than one pixel")
            actual = [box[0] / w, box[1] / h, box[2] / w, box[3] / h]
            metadata = {"derived_from": [info["id"]], "source_sha256": info["sha256"],
                        "source_crop": actual, "step": info.get("step"), "view": info.get("view")}
            if info.get("map_id"):
                a, b, c, d = info.get("map_region", [0, 0, 1, 1])
                metadata.update(map_id=info["map_id"], map_region=[a + actual[0] * (c - a), b + actual[1] * (d - b),
                                                                 a + actual[2] * (c - a), b + actual[3] * (d - b)])
            picture = picture.crop(box)
            info = _record_image(workspace, picture, "crop", metadata)
        rows.append(info)
        pictures.append(picture)
    if layout == "grid":
        cols = min(3, len(pictures))
        cell_w, cell_h = 640, 520
        grid_image = Image.new("RGB", (cols * cell_w, math.ceil(len(pictures) / cols) * cell_h), "#f6f7f8")
        draw = ImageDraw.Draw(grid_image)
        for i, (picture, info) in enumerate(zip(pictures, rows)):
            image = picture.copy()
            image.thumbnail((cell_w - 12, cell_h - 48))
            x, y = i % cols * cell_w, i // cols * cell_h
            grid_image.paste(image, (x + (cell_w - image.width) // 2, y + 40))
            view_info = info.get("view") or {}
            direction = []
            if view_info.get("right_degrees") is not None:
                direction.append(f"right {view_info['right_degrees']} deg")
            if view_info.get("pitch_degrees") is not None:
                direction.append(f"pitch {view_info['pitch_degrees']} deg")
            label = info["id"][:38]
            if direction:
                label += " | " + ", ".join(direction)
            draw.text((x + 6, y + 8), label, fill="black", font=_font(15))
        grid_info = _record_image(workspace, grid_image, "grid", {"derived_from": [row["id"] for row in rows],
                                  "sources": rows, "map_region": None})
        return {"ok": True, "data": {"images": rows, "grid": grid_info}, "images": [grid_info["path"]]}
    return {"ok": True, "data": {"images": rows}, "images": [row["path"] for row in rows]}


def dispatch(request):
    args = dict(request)
    operation = args.pop("operation", args.pop("op", None))
    if operation not in {"prepare", "check", "render", "view"}:
        raise MemoryError(f"Unknown operation: {operation}")
    return globals()[operation](**args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["prepare", "check", "render", "view"])
    options = parser.parse_args()
    try:
        request = json.load(sys.stdin)
        request["operation"] = options.operation
        result = dispatch(request)
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
