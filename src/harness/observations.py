"""Publish only RGB and map-relative pose; simulator annotations stay private."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

from PIL import Image, ImageDraw


from src.common.coordinates import map_pose
from .geometry import validate_pixel


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

    @staticmethod
    def _valid_pose(pose):
        return (isinstance(pose, dict) and set(pose) == {"pos", "heading"}
                and isinstance(pose["pos"], list) and len(pose["pos"]) == 2
                and all(type(value) in (int, float) and math.isfinite(value)
                        for value in [*pose["pos"], pose["heading"]]))

    def panorama(self, group):
        """Return one complete, immutable scene panorama at its capture pose."""
        if not isinstance(group, str) or not group.startswith("p_"):
            raise ValueError("Expected a panorama group ID (p_)")
        frames = [row for row in self.items.values() if row.get("group") == group]
        if (len(frames) != 6 or any(row.get("kind") != "observation" for row in frames)
                or any(type(row.get("yaw")) not in (int, float) for row in frames)
                or sorted(row["yaw"] for row in frames) != list(range(0, 360, 60))):
            raise ValueError(f"Panorama {group} must contain six scene observations at yaw 0,60,120,180,240,300")
        frames.sort(key=lambda row: row["yaw"])
        pose, step = frames[0].get("pose"), frames[0].get("step")
        if not self._valid_pose(pose):
            raise ValueError(f"Panorama {group} requires a valid capture pose")
        for row in frames:
            current = row.get("pose")
            if (not self._valid_pose(current)
                    or current["pos"] != pose["pos"] or row.get("step") != step):
                raise ValueError(f"Panorama {group} requires valid poses at one position and step")
        return deepcopy({"group": group, "frames": frames, "pose": pose, "step": step})

    def current_panorama(self, current):
        """Reuse only a complete panorama with the current step and body pose."""
        if (not isinstance(current, dict) or not self._valid_pose(current.get("pose"))
                or type(current.get("step")) is not int or current["step"] < 0
                or current.get("kind", "observation") != "observation"):
            return None
        groups = dict.fromkeys(row["group"] for row in reversed(list(self.items.values()))
                               if isinstance(row.get("group"), str))
        for group in groups:
            try:
                panorama = self.panorama(group)
            except ValueError:
                continue
            if panorama["step"] == current.get("step") and panorama["pose"] == current.get("pose"):
                return group
        return None

    def history(self, limit=10, offset=0):
        """List scene captures chronologically, grouping complete panoramas."""
        if type(limit) is not int or not 1 <= limit <= 300 or type(offset) is not int or offset < 0:
            raise ValueError("limit must be 1-300; offset must be a nonnegative integer")
        events, seen = [], set()
        for row in self.items.values():
            if row.get("kind") != "observation":
                continue
            group = row.get("group")
            if group in seen:
                continue
            if group:
                try:
                    panorama = self.panorama(group)
                except ValueError:
                    panorama = None
                if panorama is not None:
                    seen.add(group)
                    events.append({"image": group, "group": group, "kind": "panorama",
                                   "frames": [frame["frame"] for frame in panorama["frames"]],
                                   "pose": panorama["pose"], "step": panorama["step"]})
                    continue
            events.append({"image": row["frame"], "kind": "observation", "frames": [row["frame"]],
                           "pose": deepcopy(row.get("pose")), "step": row.get("step")})
        return {"events": events[offset:offset + limit], "total": len(events),
                "next_offset": offset + limit if offset + limit < len(events) else None}

    def image(self, reference):
        """Resolve only registered public images, including crops and map views."""
        if reference in self.items:
            row = {"image": reference, **self.items[reference]}
        elif reference in self.views:
            row = self.views[reference]
        elif isinstance(reference, str) and reference.startswith("p_"):
            panorama = self.panorama(reference)
            frames = [frame["frame"] for frame in panorama["frames"]]
            row = next((view for view in self.views.values()
                        if [source["image"] for source in view["sources"]] == frames), None)
            if row is None:
                row = self.grid(frames)
            row = {**row, "group": reference, "kind": "panorama",
                   "pose": panorama["pose"], "step": panorama["step"]}
        else:
            path = self.directory / "maps/index.jsonl"
            maps = {r["view_id"]: r for r in map(json.loads, path.read_text().splitlines())} if path.exists() else {}
            if reference not in maps:
                raise ValueError(f"Unknown image: {reference}")
            row = {"image": reference, "path": maps[reference]["map"], "kind": "map"}
        path = (self.workspace / row["path"]).resolve()
        if not path.is_relative_to(self.directory.resolve()) or not path.is_file():
            raise ValueError("Image is outside the public image directory")
        with Image.open(path) as image:
            return {**row, "size": list(image.size)}

    def resolve_pixel(self, reference, pixel):
        """Map a displayed pixel to its original frame or map view."""
        row = self.image(reference)
        validate_pixel(pixel, *row["size"], reference)
        if "sources" not in row:
            return reference, list(pixel)
        for tile in row["sources"]:
            left, top, right, bottom = tile["bounds"]
            if left <= pixel[0] < right and top <= pixel[1] < bottom:
                mapped = [a * pixel[0] + b * pixel[1] + c for a, b, c in tile["pixel_to_source"]]
                return self.resolve_pixel(tile["image"], mapped)
        raise ValueError(f"pixel_is_in_label_or_padding: image={reference}; select inside an image tile, "
                         "or reopen its original frame ID")

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
            caption = f"view {i + 1}" + (f" | right {row['yaw']} deg" if "yaw" in row else "")
            draw.text((x + 5, y - 20), caption + f" | detail: {reference}", fill="black")
        return self._save_view(result, sources)
