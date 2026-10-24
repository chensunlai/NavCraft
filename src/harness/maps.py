"""North-up views of architectural walls and the agent's current memory."""

import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .memory import MemoryStore, location


def _font(size):
    for candidate in ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


class MapRenderer:
    def __init__(self, workspace):
        self.workspace = Path(workspace)

    def render(self, pose, trajectory, view="explored", labels="id", highlight=None):
        if labels not in {"id", "name", "both", "none"}:
            raise ValueError("labels must be id, name, both, or none")
        data = json.loads((self.workspace / "scene/map.json").read_text())
        size = float(data["size_m"])
        if not math.isfinite(size) or size <= 0:
            raise ValueError("map size_m must be positive")
        entities = MemoryStore(self.workspace).entities
        indexed = {e["id"]: e for e in entities}
        warnings = [f"{e['id']} 没有位置" for e in entities if location(e) is None]
        points, padding = [], 2 / size
        if view == "full":
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
                        warnings.append(f"{identifier} 不存在")
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
        if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            raise ValueError("requested view is outside the map canvas")
        span_u, span_v = bounds[2] - bounds[0], bounds[3] - bounds[1]
        scale = 692 / max(span_u, span_v)
        width, height = round(span_u * scale) + 76, round(span_v * scale) + 76
        image = Image.new("RGB", (width, height), "#ffffff")
        draw = ImageDraw.Draw(image)
        font, small = _font(13), _font(11)

        def pixel(point):
            return (38 + (point[0] - bounds[0]) * scale,
                    34 + (point[1] - bounds[1]) * scale)

        def inside(point):
            return bounds[0] <= point[0] <= bounds[2] and bounds[1] <= point[1] <= bounds[3]

        for axis in (0, 1):
            for metre in range(math.ceil(bounds[axis] * size / 2) * 2,
                               math.floor(bounds[axis + 2] * size / 2) * 2 + 1, 2):
                value = metre / size
                a = [value, bounds[1]] if axis == 0 else [bounds[0], value]
                b = [value, bounds[3]] if axis == 0 else [bounds[2], value]
                draw.line([pixel(a), pixel(b)], fill="#e7e9ec", width=1)
                px, py = pixel(a)
                draw.text((px + 2, 15 if axis == 0 else py + 2), f"{metre}m", fill="#92979d", font=small)
        for segment in data["walls"]:
            uv = [[(p[i] - data["origin"][i]) / size for i in (0, 1)] for p in segment]
            draw.line([pixel(p) for p in uv], fill="#262a2f", width=2)
        if len(trajectory or []) > 1:
            draw.line([pixel(p) for p in trajectory], fill="#8fc6ed", width=2)
        occupied = []
        colors = ["#276cbd", "#4d8953", "#8e57b1", "#bc6a36", "#2c8992", "#bd5887"]
        highlights = set(highlight or [])
        for identifier in highlights - indexed.keys():
            warnings.append(f"{identifier} 不存在")
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
                warnings.append(f"{entity['id']} 标签未显示")
        if inside(pose["pos"]):
            px, py = pixel(pose["pos"])
            angle = math.radians(pose["heading"])
            points = [(px + math.sin(angle + offset) * radius, py - math.cos(angle + offset) * radius)
                      for offset, radius in ((0, 12), (2.5, 8), (-2.5, 8))]
            draw.polygon(points, fill="#e53935")
        # Keep geometry outside the requested crop from leaking into margins.
        draw.rectangle((0, 0, 37, height), fill="white")
        draw.rectangle((width - 37, 0, width, height), fill="white")
        draw.rectangle((0, 0, width, 33), fill="white")
        draw.rectangle((0, height - 36, width, height), fill="white")
        for axis in (0, 1):
            for metre in range(math.ceil(bounds[axis] * size / 2) * 2,
                               math.floor(bounds[axis + 2] * size / 2) * 2 + 1, 2):
                value = metre / size
                px, py = pixel([value, bounds[1]] if axis == 0 else [bounds[0], value])
                draw.text((px - 8, 15) if axis == 0 else (3, py - 5), f"{metre}m", fill="#92979d", font=small)
        bar_m = min(2., max(.1, span_u * size / 4))
        x, y = 40, height - 19
        draw.line((x, y, x + bar_m / size * scale, y), fill="#26313d", width=3)
        draw.text((x + bar_m / size * scale + 6, y - 8), f"{bar_m:g} m  |  N ↑", fill="#26313d", font=small)
        output = self.workspace / "observations/maps"
        output.mkdir(parents=True, exist_ok=True)
        counter = max((int(p.stem[2:]) for p in output.glob("m_*.png") if p.stem[2:].isdigit()), default=0) + 1
        path = output / f"m_{counter:04d}.png"
        image.save(path)
        return {"map": path.relative_to(self.workspace).as_posix(), "bounds": bounds, "warnings": warnings}
