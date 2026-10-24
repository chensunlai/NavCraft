"""Public wall/memory views with reversible image-to-map coordinates."""

import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .memory import MemoryStore, location


def _font(size):
    for candidate in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                      "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


class MapRenderer:
    def __init__(self, workspace, memory=None):
        self.workspace = Path(workspace)
        self.memory = memory if memory is not None else MemoryStore(self.workspace)

    def render(self, pose, trajectory, view="explored", labels="id", highlight=None, *, overlay_points=None):
        if labels not in {"id", "name", "both", "none"}:
            raise ValueError("labels must be id, name, both, or none")
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

            def pixel(point):
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
                points = list(trajectory or []) + [pose["pos"]]
            elif isinstance(view, dict) and set(view) == {"around"}:
                ids = view["around"]
                if not isinstance(ids, list) or not ids:
                    raise ValueError("around must be a nonempty ID list")
                for identifier in ids:
                    if identifier not in indexed:
                        warnings.append(f"{identifier} does not exist")
                    elif location(indexed[identifier]) is not None:
                        entity = indexed[identifier]
                        points.extend(entity["segment"] if entity["type"] == "door" else [location(entity)])
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

            def pixel(point):
                return (38 + (point[0] - bounds[0]) * scale,
                        34 + (point[1] - bounds[1]) * scale)

        plot_bounds = [38., 34., float(width - 38), float(height - 42)]
        image = Image.new("RGB", (width, height), "#ffffff")
        draw = ImageDraw.Draw(image)
        font, small = _font(13), _font(11)

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
            draw.line([pixel(p) for p in trajectory], fill="#8fc6ed", width=2)
        occupied = []
        colors = ["#276cbd", "#4d8953", "#8e57b1", "#bc6a36", "#2c8992", "#bd5887"]
        highlights = set(highlight or [])
        for identifier in highlights - indexed.keys():
            warnings.append(f"{identifier} does not exist")
        for entity in entities:
            point = location(entity)
            if point is None or not inside(point):
                continue
            px, py = pixel(point)
            color_key = entity.get("room") or (entity["id"] if entity["type"] == "room" else "unknown")
            color = colors[int(hashlib.sha256(color_key.encode()).hexdigest()[:8], 16) % len(colors)] if color_key != "unknown" else "#79818a"
            if entity["type"] == "door":
                draw.line([pixel(p) for p in entity["segment"]], fill="#eb8d25", width=5)
            elif entity["type"] == "object":
                draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color)
            if entity["id"] in highlights:
                draw.ellipse((px - 10, py - 10, px + 10, py + 10), outline="#e65045", width=2)
            label = {"id": entity["id"], "name": entity["name"],
                     "both": f"{entity['id']} {entity['name']}", "none": ""}[labels]
            if not label:
                continue
            box = draw.textbbox((0, 0), label, font=font)
            tw, th = box[2] - box[0], box[3] - box[1]
            placed = False
            for dx, dy in ((7, -8), (7, 8), (-tw - 7, -8), (-tw / 2, -th - 9)):
                rect = [px + dx, py + dy, px + dx + tw + 2, py + dy + th + 3]
                if rect[0] < 38 or rect[1] < 34 or rect[2] > width - 38 or rect[3] > height - 42:
                    continue
                if any(rect[0] < other[2] and rect[2] > other[0] and rect[1] < other[3] and rect[3] > other[1] for other in occupied):
                    continue
                draw.rectangle(rect, fill="white")
                draw.text((rect[0], rect[1] - box[1]), label, fill=color, font=font)
                occupied.append(rect)
                placed = True
                break
            if not placed:
                warnings.append(f"{entity['id']} label could not be placed")
        if inside(pose["pos"]):
            px, py = pixel(pose["pos"])
            angle = 0. if ego else math.radians(pose["heading"])
            points = [(px + math.sin(angle + offset) * radius, py - math.cos(angle + offset) * radius)
                      for offset, radius in ((0, 12), (2.5, 8), (-2.5, 8))]
            draw.polygon(points, fill="#e53935")
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
