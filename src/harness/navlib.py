"""Read-only public navigation helpers; no simulator or private state access."""

import json
import math
from coordinates import world_to_uv, uv_to_world, bearing_distance as _bearing_distance
from pathlib import Path


def _workspace():
    # This file is copied to the sandbox's scene/navlib.py.
    return Path(__file__).resolve().parent.parent


def _map():
    return json.loads((_workspace() / "scene/map.json").read_text())


def uv_to_m(uv):
    """Convert map UV to world X/Z metres using the public calibration."""
    point = uv_to_world(uv, _map())
    return [point[0], point[2]]


def m_to_uv(xz):
    return world_to_uv([xz[0], 0, xz[1]], _map())


def bearing_distance(pose, uv):
    """Return bearing_degrees (right positive) and distance_m."""
    return _bearing_distance(pose, uv, _map())


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
