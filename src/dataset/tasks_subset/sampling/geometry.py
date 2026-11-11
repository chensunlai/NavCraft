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


def project_opening_band(frame, opening, floor_height):
    """Project a middle-height band locating a passage; this does not measure its width."""
    a, b = opening["segment_xz"]
    corners = [[a[0], floor_height + .25, a[1]], [b[0], floor_height + .25, b[1]],
               [b[0], floor_height + 1.65, b[1]], [a[0], floor_height + 1.65, a[1]]]
    projected = [project(point, frame["camera"], frame["world_from_optical"]) for point in corners]
    if any(point is None for point in projected):
        return None
    width, height = frame["camera"]["width"], frame["camera"]["height"]
    if any(not (8 <= x <= width - 8 and 8 <= y <= height - 8) for x, y, _ in projected):
        return None
    outline = [[int(round(point[0])), int(round(point[1]))] for point in projected]
    if (any(not (8 <= x <= width - 8 and 8 <= y <= height - 8) for x, y in outline)
            or abs((outline[0][0] + outline[3][0] - outline[1][0] - outline[2][0]) / 2) < 30):
        return None
    return outline


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


def floor_patch_fraction(depth, camera, world_from_optical, pixel, floor_height, radius=9):
    """Fraction of the marked disk supported by floor, including occluded/invalid rays.

    A ray through a wheel or chair can hit floor while the displayed mark covers the
    obstacle. Require a visible patch, rather than accepting that isolated ray.
    """
    x, y = map(int, pixel)
    if not (radius <= x < depth.shape[1] - radius and radius <= y < depth.shape[0] - radius):
        return 0.
    yy, xx = np.mgrid[y-radius:y+radius+1, x-radius:x+radius+1]
    disk = (xx-x)**2 + (yy-y)**2 <= radius**2
    sample = depth[y-radius:y+radius+1, x-radius:x+radius+1][disk]
    rays = np.linalg.solve(camera["K"], np.array([xx[disk], yy[disk], np.ones(len(sample))]))
    matrix = np.asarray(world_from_optical)
    heights = (matrix[1, :3] @ rays) * sample + matrix[1, 3]
    return float((np.isfinite(heights) & (np.abs(heights-floor_height) <= .06)).mean())


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


def measure_horizontal_floor_patch(depth, camera, world_from_optical, pixel, floor_heights,
                                   body_height, radius=9):
    """Measure a locally horizontal real surface near one declared floor."""
    x,y = map(int,pixel)
    if not radius <= x < depth.shape[1]-radius or not radius <= y < depth.shape[0]-radius:
        return None
    yy,xx = np.mgrid[y-radius:y+radius+1,x-radius:x+radius+1]
    disk=(xx-x)**2+(yy-y)**2<=radius**2
    z=depth[y-radius:y+radius+1,x-radius:x+radius+1][disk]
    rays=np.linalg.solve(camera["K"],np.array([xx[disk],yy[disk],np.ones(len(z))]))
    matrix=np.asarray(world_from_optical)
    heights=(matrix[1,:3]@rays)*z+matrix[1,3]
    valid=np.isfinite(heights)&np.isfinite(z)&(z>0)
    if valid.sum()<.9*len(z):
        return None
    median=float(np.median(heights[valid]))
    matched=[h for h in floor_heights if abs(h-median)<=.10]
    residuals=np.abs(heights-median)
    support=valid&(residuals<=.025)
    if len(matched)!=1 or abs(median-body_height)>.10 or support.mean()<.9:
        return None
    points=(matrix[:3,:3]@(rays[:,support]*z[support])).T+matrix[:3,3]
    centered=points-points.mean(axis=0)
    _,singular,axes=np.linalg.svd(centered,full_matrices=False)
    if singular[1]<.01:
        return None
    normal=axes[-1]
    tilt=float(np.degrees(np.arccos(np.clip(abs(normal[1]),0,1))))
    # A small patch on a vertical furniture face can also have a narrow Y range.
    # Its fitted surface must itself be horizontal, not merely near floor height.
    if tilt>15:
        return None
    return {"height_m":median,"support_count":int(support.sum()),"sample_count":len(z),
            "support_fraction":float(support.mean()),
            "median_abs_residual_m":float(np.median(residuals[valid])),
            "max_inlier_residual_m":float(residuals[support].max()),
            "normal_world":normal.tolist(),"tilt_degrees":tilt,
            "declared_floor_height":matched[0],"body_height":body_height,
            "method":"depth_world_y_horizontal_disk","pixel":[x,y],"radius":radius}
