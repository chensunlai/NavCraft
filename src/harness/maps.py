"""Public wall/memory views with reversible image-to-map coordinates."""

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .memory import MemoryStore, location
from .map_labels import layout_labels


COLORS = {"room": "#2563eb", "door": "#c57813", "object": "#748096", "locate": "#9333ea"}


def _font(size, bold=False):
    candidates = (("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf") if bold else ())
    for candidate in (*candidates, "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                      "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


class MapRenderer:
    def __init__(self, workspace, memory=None):
        self.workspace = Path(workspace)
        self.memory = memory if memory is not None else MemoryStore(self.workspace)

    def render(self, pose, trajectory, view="explored", labels="id", highlight=None, *, overlay_points=None,
               trajectory_breaks=(), located_points=()):
        if labels not in {"id", "name", "both", "none"}:
            raise ValueError("labels must be id, name, both, or none")
        for point in located_points:
            if (not isinstance(point, dict) or not isinstance(point.get("label"), str) or not point["label"]
                    or not isinstance(point.get("uv"), (list, tuple)) or len(point["uv"]) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in point["uv"])):
                raise ValueError("located_points require a label and finite uv in [0,1]")
        data = json.loads((self.workspace / "scene/map.json").read_text())
        size = float(data["size_m"])
        if not math.isfinite(size) or size <= 0:
            raise ValueError("map size_m must be positive")
        entities = self.memory.entities
        indexed = {e["id"]: e for e in entities}
        warnings = [f"{e['id']} has no location" for e in entities if location(e) is None]
        ego = isinstance(view, dict) and set(view) == {"ego"}
        points, padding = [], 2 / size
        if ego:
            radius = view["ego"]
            if type(radius) not in (int, float) or not math.isfinite(radius) or radius <= 0:
                raise ValueError("ego must be a positive radius in metres")
            angle = math.radians(pose["heading"])
            cosine, sine = math.cos(angle), math.sin(angle)
            scale = 692 / (2 * radius / size)
            width = height = 768
            cx, cy = 384., 380.
            affine = [[cosine / scale, -sine / scale,
                       pose["pos"][0] - (cosine * cx - sine * cy) / scale],
                      [sine / scale, cosine / scale,
                       pose["pos"][1] - (sine * cx + cosine * cy) / scale]]

            def ego_pixel(point):
                du, dv = point[0] - pose["pos"][0], point[1] - pose["pos"][1]
                return (cx + scale * (cosine * du + sine * dv),
                        cy + scale * (-sine * du + cosine * dv))

            corners = [[a * x + b * y + c for a, b, c in affine]
                       for x, y in ((38, 34), (730, 34), (730, 726), (38, 726))]
            bounds = [min(p[0] for p in corners), min(p[1] for p in corners),
                      max(p[0] for p in corners), max(p[1] for p in corners)]
        elif view == "full":
            bounds = [0., 0., 1., 1.]
        else:
            if view == "explored":
                points = list(trajectory or []) + [pose["pos"]] + [p["uv"] for p in located_points]
            elif isinstance(view, dict) and set(view) == {"around"}:
                ids = view["around"]
                if not isinstance(ids, list) or not ids:
                    raise ValueError("around must be a nonempty ID list")
                for identifier in ids:
                    if identifier not in indexed:
                        warnings.append(f"{identifier} does not exist")
                    elif location(indexed[identifier]) is not None:
                        points.append(location(indexed[identifier]))
                if not points:
                    raise ValueError("around contains no located entities")
            elif isinstance(view, dict) and set(view) == {"around_self"}:
                radius = view["around_self"]
                if type(radius) not in (int, float) or not math.isfinite(radius) or radius <= 0:
                    raise ValueError("around_self must be a positive radius in metres")
                points, padding = [pose["pos"]], radius / size
            else:
                raise ValueError("invalid map view")
            bounds = [max(0., min(p[0] for p in points) - padding),
                      max(0., min(p[1] for p in points) - padding),
                      min(1., max(p[0] for p in points) + padding),
                      min(1., max(p[1] for p in points) + padding)]
        if not ego:
            if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
                raise ValueError("requested view is outside the map canvas")
            span_u, span_v = bounds[2] - bounds[0], bounds[3] - bounds[1]
            scale = 692 / max(span_u, span_v)
            width, height = round(span_u * scale) + 76, round(span_v * scale) + 76
            affine = [[1 / scale, 0., bounds[0] - 38 / scale],
                      [0., 1 / scale, bounds[1] - 34 / scale]]

            def north_pixel(point):
                return (38 + (point[0] - bounds[0]) * scale,
                        34 + (point[1] - bounds[1]) * scale)

        pixel = ego_pixel if ego else north_pixel
        plot_bounds = [38., 34., float(width - 38), float(height - 42)]
        image = Image.new("RGB", (width, height), "#ffffff")
        draw = ImageDraw.Draw(image)
        font, bold, small = _font(13), _font(14, bold=True), _font(11)

        def inside(point):
            x, y = pixel(point)
            return plot_bounds[0] <= x <= plot_bounds[2] and plot_bounds[1] <= y <= plot_bounds[3]

        if overlay_points is not None:
            for u, v, red, green, blue in overlay_points:
                px, py = pixel((u, v))
                if plot_bounds[0] <= px <= plot_bounds[2] and plot_bounds[1] <= py <= plot_bounds[3]:
                    draw.ellipse((px - 1, py - 1, px + 1, py + 1),
                                 fill=(int(red), int(green), int(blue)))
        if ego:
            for metre in range(1, math.floor(radius) + 1):
                r = metre / size * scale
                draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline="#d8dce0", width=1)
                draw.text((cx + 5, cy - r + 2), f"{metre}m", fill="#6b737c", font=small)
            draw.line((cx, cy, cx, 34), fill="#a2b7c5", width=1)
        else:
            for axis in (0, 1):
                for metre in range(math.ceil(bounds[axis] * size / 2) * 2,
                                   math.floor(bounds[axis + 2] * size / 2) * 2 + 1, 2):
                    value = metre / size
                    a = [value, bounds[1]] if axis == 0 else [bounds[0], value]
                    b = [value, bounds[3]] if axis == 0 else [bounds[2], value]
                    draw.line([pixel(a), pixel(b)], fill="#e7e9ec", width=1)
        for segment in data["walls"]:
            uv = [[(p[i] - data["origin"][i]) / size for i in (0, 1)] for p in segment]
            draw.line([pixel(p) for p in uv], fill="#262a2f", width=2)
        if len(trajectory or []) > 1:
            breaks = set(trajectory_breaks)
            for index, (a, b) in enumerate(zip(trajectory, trajectory[1:]), 1):
                if index not in breaks:
                    draw.line([pixel(a), pixel(b)], fill="#8fc6ed", width=2)
        highlights = set(highlight or [])
        for identifier in highlights - indexed.keys():
            warnings.append(f"{identifier} does not exist")
        markers, label_items = [], []
        for entity in entities:
            point = location(entity)
            if point is None or not inside(point):
                continue
            px, py = pixel(point)
            structural = entity["type"] != "object"
            marker_radius = 6 if structural else 3
            markers.append({"point": [px, py], "radius": marker_radius, "id": entity["id"], "type": entity["type"]})
            label = {"id": entity["id"], "name": entity["name"],
                     "both": f"{entity['id']} {entity['name']}", "none": ""}[labels]
            if not label:
                continue
            selected_font = bold if structural else font
            box = draw.textbbox((0, 0), label, font=selected_font)
            label_items.append({"id": entity["id"], "type": entity["type"], "text": label,
                                "point": [px, py], "size": [box[2] - box[0] + 6, box[3] - box[1] + 4],
                                "priority": 0 if structural else 1})
        for point in located_points:
            if not inside(point["uv"]):
                continue
            px, py = pixel(point["uv"])
            markers.append({"point": [px, py], "radius": 9, "id": point["label"], "type": "locate"})
            box = draw.textbbox((0, 0), point["label"], font=bold)
            label_items.append({"id": point["label"], "type": "locate", "text": point["label"],
                                "point": [px, py], "size": [box[2] - box[0] + 6, box[3] - box[1] + 4],
                                "priority": -1})
        obstacles = [[m["point"][0] - r, m["point"][1] - r,
                      m["point"][0] + r, m["point"][1] + r]
                     for m in markers for r in [10 if m["id"] in highlights else m["radius"] + 1]]
        if inside(pose["pos"]):
            px, py = pixel(pose["pos"])
            obstacles.append([px - 15, py - 15, px + 15, py + 15])
        if ego:
            obstacles.append([width - 110, 34, width - 38, 110])
        self.last_label_layout, missing = layout_labels(label_items, plot_bounds, obstacles)
        warnings.extend(f"{identifier} label could not be placed" for identifier in missing)
        for label in self.last_label_layout:
            draw.line([tuple(point) for point in label["leader"]], fill=COLORS[label["type"]], width=1)
        for label in self.last_label_layout:
            selected_font = bold if label["type"] != "object" else font
            rect = label["rect"]
            box = draw.textbbox((0, 0), label["text"], font=selected_font)
            draw.rectangle(rect, fill="white")
            draw.text((rect[0] + 3 - box[0], rect[1] + 2 - box[1]), label["text"],
                      fill=COLORS[label["type"]], font=selected_font)
        for marker in markers:
            px, py = marker["point"]
            marker_radius, kind = marker["radius"], marker["type"]
            color = COLORS[kind]
            if kind == "locate":
                continue
            if kind == "room":
                draw.rectangle((px - marker_radius, py - marker_radius, px + marker_radius, py + marker_radius),
                               fill=color, outline="white", width=1)
            elif kind == "door":
                draw.polygon([(px, py - marker_radius - 1), (px + marker_radius + 1, py),
                              (px, py + marker_radius + 1), (px - marker_radius - 1, py)], fill=color)
            else:
                draw.ellipse((px - marker_radius, py - marker_radius, px + marker_radius, py + marker_radius), fill=color)
            if marker["id"] in highlights:
                draw.ellipse((px - 10, py - 10, px + 10, py + 10), outline="#e65045", width=2)
        if inside(pose["pos"]):
            px, py = pixel(pose["pos"])
            angle = 0. if ego else math.radians(pose["heading"])
            points = [(px + math.sin(angle + offset) * radius, py - math.cos(angle + offset) * radius)
                      for offset, radius in ((0, 12), (2.5, 8), (-2.5, 8))]
            draw.polygon(points, fill="#e53935")
        for marker in markers:
            if marker["type"] == "locate":
                px, py = marker["point"]
                draw.ellipse((px - 6, py - 6, px + 6, py + 6), outline=COLORS["locate"], width=2)
                draw.line((px - 9, py, px + 9, py), fill=COLORS["locate"], width=2)
                draw.line((px, py - 9, px, py + 9), fill=COLORS["locate"], width=2)
        # Keep geometry outside the requested crop from leaking into margins.
        draw.rectangle((0, 0, 37, height), fill="white")
        draw.rectangle((width - 37, 0, width, height), fill="white")
        draw.rectangle((0, 0, width, 33), fill="white")
        draw.rectangle((0, height - 36, width, height), fill="white")
        if ego:
            nx, ny = width - 69, 70
            dx, dy = -sine * 22, -cosine * 22
            draw.line((nx, ny, nx + dx, ny + dy), fill="#26313d", width=2)
            draw.polygon([(nx + dx, ny + dy),
                          (nx + dx * .65 - dy * .2, ny + dy * .65 + dx * .2),
                          (nx + dx * .65 + dy * .2, ny + dy * .65 - dx * .2)], fill="#26313d")
            draw.text((nx + dx - 4, ny + dy - 18), "N", fill="#26313d", font=small)
        else:
            for axis in (0, 1):
                for metre in range(math.ceil(bounds[axis] * size / 2) * 2,
                                   math.floor(bounds[axis + 2] * size / 2) * 2 + 1, 2):
                    value = metre / size
                    px, py = pixel([value, bounds[1]] if axis == 0 else [bounds[0], value])
                    draw.text((px - 8, 15) if axis == 0 else (3, py - 5), f"{metre}m", fill="#92979d", font=small)
        bar_m = min(2., max(.1, (2 * radius if ego else span_u * size) / 4))
        x, y = 40, height - 19
        draw.line((x, y, x + bar_m / size * scale, y), fill="#26313d", width=3)
        draw.text((x + bar_m / size * scale + 6, y - 8),
                  f"{bar_m:g} m  |  {'heading ↑' if ego else 'N ↑'}", fill="#26313d", font=small)
        output = self.workspace / "observations/maps"
        output.mkdir(parents=True, exist_ok=True)
        counter = max((int(p.stem[2:]) for p in output.glob("m_*.png") if p.stem[2:].isdigit()), default=0) + 1
        path = output / f"m_{counter:04d}.png"
        image.save(path)
        result = {"map": path.relative_to(self.workspace).as_posix(), "view_id": path.stem,
                  "bounds": bounds, "pixel_to_uv": affine, "plot_bounds": plot_bounds,
                  "image_size": [width, height], "warnings": warnings}
        with (output / "index.jsonl").open("a") as stream:
            stream.write(json.dumps({**result, "width": width, "height": height,
                                     "view": view, "pose": pose}, ensure_ascii=False) + "\n")
        return result
