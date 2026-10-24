"""Geometry for selected RGB pixels and saved map views; no semantic inference."""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .memory import observation_index


def _records(path, key):
    if not path.exists():
        return {}
    return {row[key]: row for row in map(json.loads, path.read_text().splitlines()) if row}


def _pixel(pixel, width, height):
    if (not isinstance(pixel, (list, tuple)) or len(pixel) != 2
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in pixel)):
        raise ValueError("pixel must be a finite [x, y] pair")
    x, y = pixel
    if not 0 <= x < width or not 0 <= y < height:
        raise ValueError("pixel_out_of_bounds")
    return x, y


class Geometry:
    def __init__(self, workspace, private_dir, transform):
        self.workspace, self.private = Path(workspace), Path(private_dir)
        self.transform = transform

    def load_frame(self, frame):
        """Return metric optical-Z depth and its private camera/body metadata."""
        if not isinstance(frame, str):
            raise ValueError("frame must be an original frame ID")
        if frame not in observation_index(self.workspace):
            raise ValueError(f"Unknown frame: {frame}")
        cameras = _records(self.private / "depth/index.jsonl", "frame")
        if frame not in cameras:
            raise ValueError("depth_unavailable")
        camera = cameras[frame]
        directory = (self.private / "depth").resolve()
        path = (directory / camera["depth"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Invalid private depth path")
        with Image.open(path) as image:
            depth = np.asarray(image, dtype=float) * camera["depth_scale"]
        if depth.shape != (camera["height"], camera["width"]):
            raise ValueError("Depth dimensions disagree with camera metadata")
        intrinsic = np.asarray(camera["K"], dtype=float)
        transform = np.asarray(camera["world_from_optical"], dtype=float)
        if (intrinsic.shape != (3, 3) or transform.shape != (4, 4)
                or not np.isfinite(intrinsic).all() or not np.isfinite(transform).all()
                or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
                or not np.allclose(intrinsic[2], [0, 0, 1])
                or not np.allclose(transform[3], [0, 0, 0, 1])):
            raise ValueError("Invalid camera matrices")
        return depth, camera

    @staticmethod
    def _project(camera, pixels, depths):
        intrinsic = np.asarray(camera["K"], dtype=float)
        transform = np.asarray(camera["world_from_optical"], dtype=float)
        rays = np.linalg.solve(intrinsic, np.column_stack([pixels, np.ones(len(pixels))]).T).T
        optical = rays * depths[:, None]
        return optical @ transform[:3, :3].T + transform[:3, 3]

    def _uv(self, points):
        return (points[:, [0, 2]] - self.transform["origin"]) / self.transform["size_m"]

    @staticmethod
    def _depth_valid(depth, camera):
        return np.isfinite(depth) & (depth > camera["depth_min"]) & (depth < camera["depth_max"])

    def resolve_frame_point(self, frame, pixel):
        """Return an unrounded world point for internal motion and memory operations."""
        depth, camera = self.load_frame(frame)
        x, y = _pixel(pixel, camera["width"], camera["height"])
        ix, iy = min(round(x), camera["width"] - 1), min(round(y), camera["height"] - 1)
        value = depth[iy, ix]
        if not self._depth_valid(value, camera):
            raise ValueError("invalid_depth_or_range")
        point = self._project(camera, np.array([[ix, iy]]), np.array([value]))[0]
        if not np.isfinite(point).all():
            raise ValueError("invalid_projection")
        uv = self._uv(point[None])[0]
        if np.any(uv < -1e-10) or np.any(uv > 1 + 1e-10):
            raise ValueError("outside_canvas")
        floor_limits = camera.get("floor_y_range")
        if floor_limits and not floor_limits[0] <= point[1] <= floor_limits[1]:
            raise ValueError("outside_floor")
        return point.tolist(), camera

    def resolve_frame(self, frame, pixel):
        try:
            values, camera = self.resolve_frame_point(frame, pixel)
        except ValueError as error:
            return {"valid": False, "reason": str(error)}
        point = np.asarray(values)
        displacement = point[[0, 2]] - np.asarray(camera["body_position"])[[0, 2]]
        forward = np.asarray(camera["body_forward"])[[0, 2]]
        if not np.isfinite(forward).all() or np.linalg.norm(forward) == 0:
            raise ValueError("Invalid capture body direction")
        right = np.array([-forward[1], forward[0]])
        bearing = math.degrees(math.atan2(float(displacement @ right), float(displacement @ forward)))
        return {"uv": np.clip(self._uv(point[None])[0], 0, 1).round(4).tolist(),
                "distance": round(float(np.linalg.norm(displacement)), 3),
                "bearing": round(bearing, 2),
                "height": round(float(point[1] - camera["floor_height"]), 3), "valid": True}

    def resolve_map(self, view_id, pixel):
        if not isinstance(view_id, str):
            raise ValueError("view_id must be a saved map view ID")
        views = _records(self.workspace / "observations/maps/index.jsonl", "view_id")
        if view_id not in views:
            raise ValueError(f"Unknown map view: {view_id}")
        view = views[view_id]
        x, y = _pixel(pixel, view["width"], view["height"])
        left, top, right, bottom = view["plot_bounds"]
        if not left <= x <= right or not top <= y <= bottom:
            raise ValueError("pixel_outside_map_plot")
        uv = np.asarray(view["pixel_to_uv"], dtype=float) @ [x, y, 1]
        if not np.isfinite(uv).all() or np.any(uv < -1e-10) or np.any(uv > 1 + 1e-10):
            raise ValueError("outside_canvas")
        return {"uv": np.clip(uv, 0, 1).round(4).tolist()}

    def locate(self, source, points):
        if not isinstance(points, list) or not 1 <= len(points) <= 10:
            raise ValueError("points must contain 1–10 pixel pairs")
        if not isinstance(source, str) or not source.startswith(("f_", "m_")):
            raise ValueError("source must be an original frame or saved map view ID")
        resolve = self.resolve_frame if source.startswith("f_") else self.resolve_map
        results = []
        for point in points:
            try:
                result = resolve(source, point)
            except ValueError as error:
                result = {"valid": False, "reason": str(error)}
            results.append({"pixel": point, **result})
        return {"source": source, "points": results}

    def overlay(self, frames, max_range=6, minheight=.1, maxheight=1.8):
        """Return at most 50k [u,v,r,g,b] rows from actual RGB-D observations."""
        if not isinstance(frames, list) or not 1 <= len(frames) <= 6:
            raise ValueError("overlay must contain 1–6 original frame IDs")
        rows = []
        public = observation_index(self.workspace)
        for frame in frames:
            depth, camera = self.load_frame(frame)
            y, x = np.mgrid[0:camera["height"]:4, 0:camera["width"]:4]
            pixels = np.column_stack([x.ravel(), y.ravel()])
            depths = depth[y, x].ravel()
            valid = self._depth_valid(depths, camera)
            pixels, depths = pixels[valid], depths[valid]
            points = self._project(camera, pixels, depths)
            heights = points[:, 1] - camera["floor_height"]
            ranges = np.linalg.norm(points[:, [0, 2]] - np.asarray(camera["body_position"])[[0, 2]], axis=1)
            uv = self._uv(points)
            valid = (np.isfinite(points).all(axis=1) & (heights >= minheight) & (heights <= maxheight)
                     & (ranges <= max_range) & (uv >= 0).all(axis=1) & (uv <= 1).all(axis=1))
            path = (self.workspace / public[frame]["path"]).resolve()
            if not path.is_relative_to((self.workspace / "observations").resolve()):
                raise ValueError("Observation image is outside the public image directory")
            with Image.open(path) as image:
                if image.size != (camera["width"], camera["height"]):
                    raise ValueError("RGB dimensions disagree with depth")
                rgb = np.asarray(image.convert("RGB"))[pixels[:, 1], pixels[:, 0]]
            rows.append(np.column_stack([uv[valid], rgb[valid]]))
        result = np.concatenate(rows) if rows else np.empty((0, 5))
        if len(result) > 50_000:
            result = result[np.linspace(0, len(result) - 1, 50_000, dtype=int)]
        return result
