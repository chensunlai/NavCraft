"""Read-only public navigation helpers; no simulator or private state access."""

import json
import math
from pathlib import Path


def _workspace():
    # This file is copied to the sandbox's scene/navlib.py.
    return Path(__file__).resolve().parent.parent


def _map():
    return json.loads((_workspace() / "scene/map.json").read_text())


def uv_to_m(uv):
    """Convert map UV to world X/Z metres using the public calibration."""
    data = _map()
    return [data["origin"][i] + float(uv[i]) * data["size_m"] for i in range(2)]


def m_to_uv(xy):
    data = _map()
    return [(float(xy[i]) - data["origin"][i]) / data["size_m"] for i in range(2)]


def bearing_distance(pose, uv):
    """Return bearing_degrees (right positive) and distance_m."""
    size = _map()["size_m"]
    dx, dz = [(uv[i] - pose["pos"][i]) * size for i in range(2)]
    angle = (math.degrees(math.atan2(dx, -dz)) - pose["heading"] + 180) % 360 - 180
    return {"bearing_degrees": angle, "distance_m": math.hypot(dx, dz)}


def project(pose, bearing_deg, distance_m):
    angle = math.radians(pose["heading"] + bearing_deg)
    distance = distance_m / _map()["size_m"]
    return [pose["pos"][0] + math.sin(angle) * distance,
            pose["pos"][1] - math.cos(angle) * distance]


def frame(fid):
    for line in (_workspace() / "observations/index.jsonl").read_text().splitlines():
        item = json.loads(line)
        if item["frame"] == fid:
            return item
    raise KeyError(fid)


def memory():
    return json.loads((_workspace() / "memory/entities.json").read_text())


def pose():
    return json.loads((_workspace() / "scene/state.json").read_text())["pose"]
