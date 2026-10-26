"""Shared navigation coordinates.

Habitat positions are [X,Y,Z] metres, Y-up. A map stores [X,Z] metres;
UV grows right/down from origin, heading zero faces -Z and turns right-positive.
Rendering changes the view only, never the stored calibration.
"""

import math


def _finite(values, length, name):
    if (not isinstance(values, (list, tuple)) or len(values) != length
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
        raise ValueError(f"{name} must contain {length} finite numbers")
    return values


def world_to_uv(point, transform):
    """Habitat [X,Y,Z] metres to unrounded map [u,v]."""
    return [(point[axis] - transform["origin"][i]) / transform["size_m"]
            for i, axis in enumerate((0, 2))]


def uv_to_world(uv, transform, height=None):
    """Map [u,v] to Habitat [X,Y,Z], on the selected floor unless specified."""
    x, z = [transform["origin"][i] + uv[i] * transform["size_m"] for i in (0, 1)]
    return [x, transform["floor_height"] if height is None else height, z]


def map_pose(pose, transform):
    """Publish a Habitat body pose; quaternion order is [x,y,z,w]."""
    qx, qy, qz, qw = pose["rotation_xyzw"]
    fx = -2 * (qx * qz + qw * qy)
    fz = -(1 - 2 * (qx * qx + qy * qy))
    return {"pos": [round(v, 4) for v in world_to_uv(pose["position"], transform)],
            "heading": round(math.degrees(math.atan2(fx, -fz)) % 360, 2) % 360}


def bearing_distance(pose, uv, transform):
    """Direction relative to the current body (right-positive), and metres."""
    dx, dz = [(uv[i] - pose["pos"][i]) * transform["size_m"] for i in (0, 1)]
    distance = math.hypot(dx, dz)
    angle = ((math.degrees(math.atan2(dx, -dz)) - pose["heading"] + 180) % 360 - 180
             if distance > 1e-9 else 0.)
    return {"bearing_degrees": angle, "distance_m": distance}


def validate_map(value):
    """Validate the public, single-floor geometry at a loading boundary."""
    if not isinstance(value, dict) or set(value) != {"origin", "size_m", "floor_height", "walls"}:
        raise ValueError("Public map requires exactly origin, size_m, floor_height and walls")
    _finite(value["origin"], 2, "Map origin")
    size, height = value["size_m"], value["floor_height"]
    _finite([size, height], 2, "Map size and floor height")
    if size <= 0:
        raise ValueError("Map size_m must be positive")
    if not isinstance(value["walls"], list):
        raise ValueError("Map walls must be a list")
    for segment in value["walls"]:
        if not isinstance(segment, list) or len(segment) < 2:
            raise ValueError("Map walls must contain at least two points")
        for point in segment:
            _finite(point, 2, "Wall point")
            if any(not origin - 1e-8 <= coordinate <= origin + size + 1e-8
                   for coordinate, origin in zip(point, value["origin"])):
                raise ValueError("Wall point is outside the calibrated canvas")
    return value
