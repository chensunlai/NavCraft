"""Publish only RGB and map-relative pose; simulator annotations stay private."""

import hashlib
import json
import math
from pathlib import Path
import re

from PIL import Image, ImageDraw


from src.common.coordinates import map_pose


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
        self.views_path = self.directory / "views.jsonl"
        self.views = {row["image"]: row for row in
                      map(json.loads, self.views_path.read_text().splitlines())} if self.views_path.exists() else {}

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

    def image(self, reference):
        """Resolve only registered public images, including crops and map views."""
        if reference in self.items:
            return {"image": reference, **self.items[reference]}
        if reference in self.views:
            row = self.views[reference]
        else:
            path = self.directory / "maps/index.jsonl"
            maps = {r["view_id"]: r for r in map(json.loads, path.read_text().splitlines())} if path.exists() else {}
            if reference not in maps:
                raise ValueError(f"Unknown image: {reference}")
            row = {"image": reference, "path": maps[reference]["map"], "kind": "map"}
        path = (self.workspace / row["path"]).resolve()
        if not path.is_relative_to(self.directory.resolve()) or not path.is_file():
            raise ValueError("Image is outside the public image directory")
        return row

    def resolve_pixel(self, reference, pixel):
        """Map a displayed pixel to its original frame or map view."""
        row = self.image(reference)
        with Image.open(self.workspace / row["path"]) as image:
            width, height = image.size
        if (not isinstance(pixel, (list, tuple)) or len(pixel) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in pixel)
                or not 0 <= pixel[0] < width or not 0 <= pixel[1] < height):
            raise ValueError("pixel_out_of_bounds")
        if reference not in self.views:
            return reference, list(pixel)
        for tile in row["sources"]:
            left, top, right, bottom = tile["bounds"]
            if left <= pixel[0] < right and top <= pixel[1] < bottom:
                mapped = [a * pixel[0] + b * pixel[1] + c for a, b, c in tile["pixel_to_source"]]
                return self.resolve_pixel(tile["image"], mapped)
        raise ValueError("pixel_is_in_label_or_padding")

    def _save_view(self, image, sources):
        number = max((int(key[2:]) for key in self.views), default=0) + 1
        identifier = f"i_{number:04d}"
        relative = f"observations/derived/{identifier}.jpg"
        (self.workspace / relative).parent.mkdir(exist_ok=True)
        image.save(self.workspace / relative, quality=85)
        row = {"image": identifier, "path": relative, "kind": "derived",
               "size": list(image.size), "sources": sources}
        self.views[identifier] = row
        with self.views_path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        return row

    def crop(self, reference, bounds):
        row = self.image(reference)
        if (not isinstance(bounds, list) or len(bounds) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in bounds)
                or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]):
            raise ValueError("crop must be normalized [left,top,right,bottom]")
        with Image.open(self.workspace / row["path"]) as source:
            w, h = source.size
            left, top = int(bounds[0] * w), int(bounds[1] * h)
            right, bottom = max(left + 1, int(bounds[2] * w)), max(top + 1, int(bounds[3] * h))
            image = source.crop((left, top, right, bottom))
        return self._save_view(image, [{"image": reference, "bounds": [0, 0, *image.size],
                                       "pixel_to_source": [[1, 0, left], [0, 1, top]]}])

    def grid(self, references):
        if not references:
            raise ValueError("A grid requires images")
        width, height = 640, 480
        columns = min(3, len(references))
        result = Image.new("RGB", (columns * width, math.ceil(len(references) / columns) * (height + 26)), "white")
        draw, sources = ImageDraw.Draw(result), []
        for i, reference in enumerate(references):
            row = self.image(reference)
            x, y = (i % columns) * width, (i // columns) * (height + 26) + 26
            with Image.open(self.workspace / row["path"]) as source:
                image = source.copy()
                original = image.size
                image.thumbnail((width, height))
                result.paste(image, (x, y))
            sx, sy = original[0] / image.width, original[1] / image.height
            sources.append({"image": reference, "bounds": [x, y, x + image.width, y + image.height],
                            "pixel_to_source": [[sx, 0, -sx * x], [0, sy, -sy * y]]})
            caption = reference + (f" | right {row['yaw']} deg" if "yaw" in row else "")
            draw.text((x + 5, y - 20), caption, fill="black")
        return self._save_view(result, sources)
