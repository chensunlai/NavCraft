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


def xz_to_uv(x, z, transform):
    """Metric map coordinates to UV; accepts scalars or NumPy coordinate columns."""
    return [(coordinate - origin) / transform["size_m"]
            for coordinate, origin in zip((x, z), transform["origin"])]


def world_to_uv(point, transform):
    """Habitat [X,Y,Z] metres to unrounded map [u,v]."""
    return xz_to_uv(point[0], point[2], transform)


def uv_to_world(uv, transform, height=None):
    """Map [u,v] to Habitat [X,Y,Z], on the selected floor unless specified."""
    x, z = [transform["origin"][i] + uv[i] * transform["size_m"] for i in (0, 1)]
    return [x, transform["floor_height"] if height is None else height, z]


def body_forward(rotation_xyzw):
    """Rotate Habitat's local -Z forward axis into the world (unit XYZW quaternion)."""
    qx, qy, qz, qw = rotation_xyzw
    return [-2 * (qx * qz + qw * qy), 2 * (qw * qx - qy * qz),
            -(1 - 2 * (qx * qx + qy * qy))]


def heading_from_forward(forward):
    """Horizontal map heading: zero faces -Z, right-positive, in [0,360)."""
    fx, _, fz = _finite(forward, 3, "Body forward")
    if math.hypot(fx, fz) < 1e-12:
        raise ValueError("Body forward has no horizontal direction")
    return math.degrees(math.atan2(fx, -fz)) % 360


def map_pose(pose, transform):
    """Publish a Habitat body pose; round only at the public boundary."""
    heading = heading_from_forward(body_forward(pose["rotation_xyzw"]))
    return {"pos": [round(v, 4) for v in world_to_uv(pose["position"], transform)],
            "heading": round(heading, 2) % 360}


def relative_xz(dx, dz, heading):
    """Metric horizontal displacement to body-relative bearing and distance."""
    distance = math.hypot(dx, dz)
    angle = ((math.degrees(math.atan2(dx, -dz)) - heading + 180) % 360 - 180
             if distance > 1e-9 else 0.)
    return {"bearing_degrees": angle, "distance_m": distance}


def bearing_distance(pose, uv, transform):
    """Direction relative to the current body (right-positive), and metres."""
    dx, dz = [(uv[i] - pose["pos"][i]) * transform["size_m"] for i in (0, 1)]
    return relative_xz(dx, dz, pose["heading"])


def project_uv(pose, bearing_deg, distance_m, transform):
    """Body-relative horizontal bearing/distance to global UV."""
    angle = math.radians(pose["heading"] + bearing_deg)
    distance = distance_m / transform["size_m"]
    return [pose["pos"][0] + math.sin(angle) * distance,
            pose["pos"][1] - math.cos(angle) * distance]


def apply_affine(matrix, point):
    """Apply a 2x3 display transform without rounding or clipping."""
    x, y = point
    return [a * x + b * y + c for a, b, c in matrix]


def invert_affine(matrix):
    """Invert a 2x3 display transform; view changes never alter global UV."""
    (a, b, c), (d, e, f) = matrix
    det = a * e - b * d
    if not math.isfinite(det) or det == 0:
        raise ValueError("Display transform must be finite and invertible")
    return [[e / det, -b / det, (b * f - e * c) / det],
            [-d / det, a / det, (d * c - a * f) / det]]


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
