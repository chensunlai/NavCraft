"""Camera and motion measurements on private Habitat coordinates."""
import numpy as np
from PIL import Image


def project(world, camera, world_from_optical):
    """Return (pixel x, pixel y, optical Z in metres), or None outside the frame."""
    matrix = np.asarray(world_from_optical)
    p = matrix[:3, :3].T @ (np.asarray(world) - matrix[:3, 3])
    if not np.all(np.isfinite(p)) or p[2] <= 0:
        return None
    q = np.asarray(camera["K"]) @ p
    x, y = q[:2] / q[2]
    if not (0 <= x < camera["width"] and 0 <= y < camera["height"]):
        return None
    return float(x), float(y), float(p[2])


def unproject(pixel, depth_m, camera, world_from_optical):
    if not np.isfinite(depth_m) or not camera.get("depth_min", 0) < depth_m < camera.get("depth_max", np.inf):
        return None
    optical = np.linalg.solve(camera["K"], [*pixel, 1]) * depth_m
    matrix = np.asarray(world_from_optical)
    return (matrix[:3, :3] @ optical + matrix[:3, 3]).tolist()


def read_depth(path, camera):
    with Image.open(path) as image:
        raw = np.asarray(image)
    result = raw.astype(float) * camera["depth_scale"]
    result[(raw == camera["depth_invalid_value"]) | (result <= camera["depth_min"]) |
           (result >= camera["depth_max"])] = np.nan
    return result


def visible_point(world, depth, camera, world_from_optical, tolerance=0.08):
    p = project(world, camera, world_from_optical)
    if p is None:
        return None
    x, y, z = p
    col, row = int(x+0.5), int(y+0.5)
    if not (1 <= row < depth.shape[0]-1 and 1 <= col < depth.shape[1]-1):
        return None
    sample = depth[row-1:row+2, col-1:col+2]
    consistent = np.isfinite(sample) & (np.abs(sample-z) <= tolerance)
    if consistent.sum() < 5 or not consistent[1, 1]:
        return None
    return [col, row]


def crossed_segment(start, end, segment, endpoint_margin=0.05):
    """True only for a proper finite XZ crossing away from the jambs."""
    p = np.asarray([start[0], start[2]])
    q = np.asarray([end[0], end[2]])
    a, b = np.asarray(segment)
    r, s = q-p, b-a
    length = np.linalg.norm(s)
    if length <= 2*endpoint_margin:
        return False
    matrix = np.column_stack((r, -s))
    if abs(np.linalg.det(matrix)) < 1e-9:
        return False
    t, u = np.linalg.solve(matrix, a-p)
    return bool(0 < t <= 1 and endpoint_margin < u*length < length-endpoint_margin)
