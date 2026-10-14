"""Private world-to-map calibration for the one-time initial pose hint.

Online map annotations use image coordinates only. This calibration must not be
used to update an actor's estimate after initialization.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class MapAsset:
    scene_id: str
    path: Path
    coordinate_frame: str = "habitat_world; XYZ, Y up"


@dataclass(frozen=True)
class MapPanel:
    level_id: str
    height: float
    # SVG coordinates = scale * authored horizontal coordinate + offset.
    x_scale: float
    x_offset: float
    y_scale: float
    y_offset: float
    clip: tuple[float, float, float, float]


def _line_fit(source: list[float], target: list[float]) -> tuple[float, float]:
    average = sum(source) / len(source)
    mean_target = sum(target) / len(target)
    variance = sum((value - average) ** 2 for value in source)
    if variance < 1e-10:
        raise ValueError("Map calibration has no spatial extent")
    scale = (
        sum((a - average) * (b - mean_target) for a, b in zip(source, target))
        / variance
    )
    offset = mean_target - scale * average
    if max(abs(scale * a + offset - b) for a, b in zip(source, target)) > 0.02:
        raise ValueError("SVG polygons disagree with scene geometry")
    return scale, offset


def _polygon(path: ET.Element) -> list[list[float]]:
    number = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    return [
        [float(x), float(y)]
        for x, y in re.findall(rf"[ML]\s*({number})[ ,]+({number})", path.get("d", ""))
    ]


class MapCalibration:
    """Validate archived polygons against SVG panels, including floor insets.

    Using polygons, rather than treating nominal bounds as rendered axis limits,
    also handles archives whose schematic bounds have small plotting margins.
    """

    def __init__(self, scene_directory: Path):
        raw = scene_directory / "raw"
        scene = json.loads((raw / "scene.json").read_text())
        regions = json.loads((raw / "regions.json").read_text())["regions"]
        svg = ET.parse(raw / "floor_plan.svg").getroot()
        ns = {"svg": "http://www.w3.org/2000/svg"}
        self.viewbox = tuple(float(v) for v in svg.get("viewBox", "").split())
        if len(self.viewbox) != 4 or min(self.viewbox[2:]) <= 0:
            raise ValueError("Map SVG needs a valid viewBox")
        clips = {
            clip.get("id"): clip.find("svg:rect", ns)
            for clip in svg.findall(".//svg:clipPath", ns)
        }
        axes = [
            group
            for group in svg.findall(".//svg:g", ns)
            if group.get("id", "").startswith("axes_")
            and any(path.get("clip-path") for path in group.findall(".//svg:path", ns))
        ]
        levels = scene["levels"]
        if len(axes) != len(levels):
            raise ValueError("Map SVG panels and scene levels disagree")
        self.panels: list[MapPanel] = []
        for level, axis in zip(levels, axes):
            polygons = [region for region in regions if region["level"] == level["id"]]
            paths = [
                path
                for path in axis.findall(".//svg:path", ns)
                if path.get("clip-path") and "fill: none" not in path.get("style", "")
            ]
            pairs = []
            for index, region in enumerate(polygons):
                group = axis.find(f".//svg:g[@id='region-{region['id']}']", ns)
                path = group.find("svg:path", ns) if group is not None else paths[index]
                points = _polygon(path)
                original = region["polygon"]
                if len(points) != len(original):
                    raise ValueError("Map region polygon does not match SVG")
                pairs.extend(zip(original, points))
            if not pairs:
                raise ValueError("Map panel has no calibration polygons")
            xs, xo = _line_fit([a[0] for a, _ in pairs], [b[0] for _, b in pairs])
            ys, yo = _line_fit([a[1] for a, _ in pairs], [b[1] for _, b in pairs])
            if xs <= 0 or ys >= 0 or not math.isclose(xs, -ys, rel_tol=0.015):
                raise ValueError("Map axes must preserve X,-Z orientation and scale")
            clip_id = paths[0].get("clip-path", "")[5:-1]
            rect = clips.get(clip_id)
            if rect is None:
                raise ValueError("Map panel has no clipping rectangle")
            rx, ry, rw, rh = [
                float(rect.get(key)) for key in ("x", "y", "width", "height")
            ]
            vx, vy, vw, vh = self.viewbox
            if (
                rx < vx
                or ry < vy
                or rx + rw > vx + vw + 0.01
                or ry + rh > vy + vh + 0.01
            ):
                raise ValueError("Map panel falls outside image")
            self.panels.append(
                MapPanel(
                    level["id"],
                    float(level["height_m"]),
                    xs,
                    xo,
                    ys,
                    yo,
                    (rx, ry, rx + rw, ry + rh),
                )
            )

    def initial_arrow(
        self, pose: dict
    ) -> tuple[str, tuple[float, float], tuple[float, float]]:
        position = pose["position"]
        rotation = pose["rotation_xyzw"]
        if (
            len(position) != 3
            or len(rotation) != 4
            or not all(math.isfinite(float(value)) for value in [*position, *rotation])
        ):
            raise ValueError("Invalid initial pose")
        candidates = [
            candidate
            for candidate in self.panels
            if candidate.clip[0] - 0.02
            <= candidate.x_scale * position[0] + candidate.x_offset
            <= candidate.clip[2] + 0.02
            and candidate.clip[1] - 0.02
            <= candidate.y_scale * -position[2] + candidate.y_offset
            <= candidate.clip[3] + 0.02
        ]
        if not candidates:
            raise ValueError("Initial position lies outside archived map panels")
        panel = min(
            candidates, key=lambda candidate: abs(candidate.height - position[1])
        )
        qx, qy, qz, qw = [float(value) for value in rotation]
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm < 1e-8:
            raise ValueError("Invalid initial quaternion")
        qx, qy, qz, qw = [value / norm for value in (qx, qy, qz, qw)]
        forward_x = -2 * (qx * qz + qw * qy)
        forward_minus_z = 1 - 2 * (qx * qx + qy * qy)
        if math.hypot(forward_x, forward_minus_z) < 1e-8:
            raise ValueError("Initial heading has no horizontal component")
        vx, vy, vw, vh = self.viewbox

        def point(x, minus_z):
            return (
                (panel.x_scale * x + panel.x_offset - vx) / vw,
                (panel.y_scale * minus_z + panel.y_offset - vy) / vh,
            )

        origin = point(position[0], -position[2])
        heading = point(position[0] + forward_x, -position[2] + forward_minus_z)
        if not all(0 <= value <= 1 for value in origin):
            raise ValueError("Initial position lies outside archived map")
        return panel.level_id, origin, heading
