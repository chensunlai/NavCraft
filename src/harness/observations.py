"""Publish only RGB and map-relative pose; simulator annotations stay private."""

import hashlib
import json
import math
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont


def _pixel_grid(image, source_bounds=None):
    """Draw original source-pixel coordinates on a copy, including after a crop."""
    width, height = image.size
    bounds = source_bounds if source_bounds is not None else [0, 0, width, height]
    if (not isinstance(bounds, (list, tuple)) or len(bounds) != 4
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in bounds)
            or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]):
        raise ValueError("source_bounds must be original pixel [left,top,right,bottom]")
    overlay = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    def label(position, text):
        box = draw.textbbox((0, 0), text, font=font)
        x = min(max(2, position[0]), max(2, width - box[2] - 2))
        y = min(max(2, position[1]), max(2, height - box[3] - 2))
        draw.text((x, y), text, font=font, fill="white", stroke_width=1, stroke_fill="black")

    for axis, extent in ((0, width), (1, height)):
        for value in range(math.ceil(bounds[axis] / 80) * 80, math.ceil(bounds[axis + 2]), 80):
            pixel = (value - bounds[axis]) * extent / (bounds[axis + 2] - bounds[axis])
            line = (pixel, 0, pixel, height) if axis == 0 else (0, pixel, width, pixel)
            draw.line(line, fill=(0, 0, 0, 100), width=3)
            draw.line(line, fill=(0, 229, 255, 170), width=1)
            label((pixel + 3, 3) if axis == 0 else (3, max(17, pixel + 3)),
                  f"{'x' if axis == 0 else 'y'}={value}")
    label((3, height - 14), "source pixels")
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def map_pose(pose, transform):
    x, _, z = pose["position"]
    qx, qy, qz, qw = pose["rotation_xyzw"]
    fx = -2 * (qx * qz + qw * qy)
    fz = -(1 - 2 * (qx * qx + qy * qy))
    origin, size = transform["origin"], transform["size_m"]
    return {"pos": [round((x - origin[0]) / size, 4), round((z - origin[1]) / size, 4)],
            "heading": round(math.degrees(math.atan2(fx, -fz)) % 360, 2)}


class Observations:
    def __init__(self, workspace, transform):
        self.workspace, self.transform = Path(workspace), transform
        self.directory = self.workspace / "observations"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.directory / "index.jsonl"
        self.items = {}
        if self.index_path.exists():
            self.items = {row["frame"]: row for row in
                          map(json.loads, self.index_path.read_text().splitlines())}
        for fid, row in self.items.items():
            if not re.fullmatch(r"f_\d{4,}", fid) or set(row) - {"frame", "step", "pose", "path", "kind", "group", "yaw", "sha256"}:
                raise ValueError("Invalid public observation index")
            path = (self.workspace / row["path"]).resolve()
            if not path.is_relative_to(self.directory.resolve()) or not path.is_file():
                raise ValueError("Observation image is outside the public image directory")
            if row.get("pose") is not None and set(row["pose"]) != {"pos", "heading"}:
                raise ValueError("Only public map pose may appear in an observation index")
        self.counter = max((int(key[2:]) for key in self.items), default=0)

    def add(self, image, *, step=None, pose=None, kind="observation", **metadata):
        self.counter += 1
        fid = f"f_{self.counter:04d}"
        relative = f"observations/{fid}.jpg"
        with Image.open(image) as source:
            source.convert("RGB").save(self.workspace / relative, quality=85)
        row = {"frame": fid, "step": step, "pose": pose, "path": relative,
               "kind": kind, **metadata}
        row["sha256"] = hashlib.sha256((self.workspace / relative).read_bytes()).hexdigest()
        self.items[fid] = row
        with self.index_path.open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def capture(self, observation, step, **metadata):
        return self.add(observation.rgb, step=step,
                        pose=map_pose(observation.pose, self.transform), **metadata)

    def get(self, fid):
        if fid not in self.items:
            raise ValueError(f"Unknown frame: {fid}")
        return self.items[fid]

    def grid_image(self, source, destination, source_bounds=None):
        with Image.open(self.workspace / source) as image:
            result = _pixel_grid(image, source_bounds)
        target = self.workspace / destination
        if target.resolve() == (self.workspace / source).resolve():
            raise ValueError("A grid image must be a new derived image")
        target.parent.mkdir(parents=True, exist_ok=True)
        result.save(target, quality=85)
        return destination

    def collage(self, members, destination, captions=None, *, grid=False, source_bounds=None):
        if not members or (captions is not None and len(captions) != len(members)):
            raise ValueError("A collage requires members with matching captions")
        if source_bounds is not None and len(source_bounds) != len(members):
            raise ValueError("source_bounds must match the collage members")
        width, height = 640, 480
        columns = min(3, len(members))
        rows = math.ceil(len(members) / columns)
        result = Image.new("RGB", (columns * width, rows * (height + 26)), "white")
        draw = ImageDraw.Draw(result)
        for i, path in enumerate(members):
            x, y = (i % columns) * width, (i // columns) * (height + 26)
            with Image.open(self.workspace / path) as image:
                original_size = image.size
                if grid:
                    image = _pixel_grid(image, source_bounds[i] if source_bounds is not None else None)
                image.thumbnail((width, height))
                result.paste(image, (x, y + 26))
            caption = captions[i] if captions else str(i + 1)
            if grid:
                caption += f" | image {original_size[0]}x{original_size[1]}; grid=source px"
            draw.text((x + 5, y + 6), caption, fill="black")
        target = self.workspace / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        result.save(target, quality=85)
        return destination
