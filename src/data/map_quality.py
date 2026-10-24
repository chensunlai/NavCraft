"""Independent CPU checks for a wall raster in Habitat X/Z coordinates.

``origin`` is the outer top-left corner ``[xmin, zmin]``. Columns increase
with world X, rows with world Z; integer pixel coordinates denote centres.
The validator never repairs a map or treats a long NavMesh detour as proof
of a missing wall: furniture can cause the same signal.
"""

from collections import deque
import math

import numpy as np


def _coordinates(points, origin, resolution):
    points = np.asarray(points, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    if points.shape[-1:] != (2,) or origin.shape != (2,):
        raise ValueError("Expected X/Z points and an [xmin, zmin] origin")
    if not np.isfinite(points).all() or not np.isfinite(origin).all():
        raise ValueError("Coordinates must be finite")
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be positive and finite")
    return points, origin


def world_to_pixel(points, origin, resolution):
    """Return continuous ``[column, row]`` coordinates at exact metre spacing."""
    points, origin = _coordinates(points, origin, resolution)
    return (points - origin) / resolution - 0.5


def pixel_to_world(pixels, origin, resolution):
    """Return X/Z coordinates of pixel centres, without a Z-axis flip."""
    pixels, origin = _coordinates(pixels, origin, resolution)
    return origin + (pixels + 0.5) * resolution


def erode_one(mask):
    """Remove one complete 3x3 boundary ring, including the canvas boundary."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("A raster must be two dimensional")
    padded = np.pad(mask, 1)
    result = mask.copy()
    for row in range(3):
        for col in range(3):
            result &= padded[row:row + mask.shape[0], col:col + mask.shape[1]]
    return result


def _components(mask):
    """Four-connected labels; a span flood fill keeps large open maps cheap."""
    height, width = mask.shape
    labels = np.full(mask.shape, -1, dtype=np.int32)
    count = 0
    for row in range(height):
        for col in np.flatnonzero(mask[row] & (labels[row] < 0)):
            if labels[row, col] >= 0:
                continue
            queue = deque([(row, int(col))])
            while queue:
                r, c = queue.popleft()
                if not mask[r, c] or labels[r, c] >= 0:
                    continue
                left = c
                while left and mask[r, left - 1] and labels[r, left - 1] < 0:
                    left -= 1
                right = c + 1
                while right < width and mask[r, right] and labels[r, right] < 0:
                    right += 1
                labels[r, left:right] = count
                for neighbor in (r - 1, r + 1):
                    if 0 <= neighbor < height:
                        available = mask[neighbor, left:right] & (labels[neighbor, left:right] < 0)
                        starts = np.flatnonzero(available & np.r_[True, ~available[:-1]])
                        queue.extend((neighbor, left + int(start)) for start in starts)
            count += 1
    return labels, count


def traversability_report(walls, navmesh, *, max_overlap=0.005):
    """Check overlap and whether walls split each *original* NavMesh island.

    Connectivity is measured in the full wall complement, not the already
    eroded/cleared NavMesh. A short wall can be bypassed; a canvas-wide wall
    separating cells of one original island fails even if overlap is tiny.
    """
    walls, navmesh = np.asarray(walls, dtype=bool), np.asarray(navmesh, dtype=bool)
    if walls.ndim != 2 or walls.shape != navmesh.shape or not walls.size:
        raise ValueError("walls and navmesh must be nonempty, equally shaped 2D rasters")
    if not 0 <= max_overlap <= 1:
        raise ValueError("max_overlap must be in [0, 1]")
    interior = erode_one(navmesh)
    interior_count = int(interior.sum())
    overlap = int(np.count_nonzero(walls & interior))
    fraction = overlap / interior_count if interior_count else None
    nav_labels, nav_count = _components(navmesh)
    free_labels, _ = _components(~walls)
    components = []
    for identifier in range(nav_count):
        original = nav_labels == identifier
        available = original & ~walls
        reached = np.unique(free_labels[available]).tolist()
        components.append({
            "id": identifier,
            "navmesh_pixels": int(original.sum()),
            "wall_pixels": int(np.count_nonzero(original & walls)),
            "wall_complement_components": reached,
            "split": len(reached) > 1,
            "fully_blocked": not reached,
        })
    reasons = []
    if not nav_count:
        reasons.append("empty_navmesh")
    elif not interior_count:
        reasons.append("empty_navmesh_after_one_pixel_erosion")
    if fraction is not None and fraction > max_overlap:
        reasons.append("wall_overlap_exceeds_limit")
    if any(item["split"] for item in components):
        reasons.append("wall_splits_original_navmesh_component")
    if any(item["fully_blocked"] for item in components):
        reasons.append("original_navmesh_component_fully_blocked")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "navmesh_pixels": int(navmesh.sum()),
        "eroded_navmesh_pixels": interior_count,
        "wall_overlap_pixels": overlap,
        "wall_overlap_fraction": fraction,
        "max_wall_overlap_fraction": max_overlap,
        "erosion": "one_pixel_3x3",
        "original_component_count": nav_count,
        "components": components,
    }


def _supercover(a, b, shape):
    """All cells touched by a segment, including either side of exact corners."""
    delta = b - a
    times = [0.0, 1.0]
    for axis in (0, 1):
        if abs(delta[axis]) > 1e-12:
            low, high = sorted((a[axis], b[axis]))
            for boundary in range(math.ceil(low), math.floor(high) + 1):
                value = (boundary - a[axis]) / delta[axis]
                if 0 < value < 1:
                    times.append(value)
    times = sorted(set(times))
    probes = times + [(left + right) / 2 for left, right in zip(times, times[1:])]
    cells = set()
    for value in probes:
        point = a + value * delta
        axes = []
        for coordinate in point:
            nearest = round(float(coordinate))
            axes.append((nearest - 1, nearest) if abs(coordinate - nearest) < 1e-9
                        else (math.floor(float(coordinate)),))
        for col in axes[0]:
            for row in axes[1]:
                if 0 <= row < shape[0] and 0 <= col < shape[1]:
                    cells.add((row, col))
    return cells


def line_intersects_wall(walls, a, b, *, origin, resolution):
    """Test X/Z endpoints against every wall cell touched by their closed line."""
    walls = np.asarray(walls, dtype=bool)
    if walls.ndim != 2 or not walls.size:
        raise ValueError("walls must be a nonempty 2D raster")
    points = world_to_pixel([a, b], origin, resolution) + 0.5
    extent = np.array([walls.shape[1], walls.shape[0]])
    if np.any(points < -1e-9) or np.any(points > extent + 1e-9):
        raise ValueError("Pair endpoints lie outside the wall canvas")
    return any(walls[cell] for cell in _supercover(points[0], points[1], walls.shape))


def leak_report(walls, pairs, *, origin, resolution):
    """Flag actual snapped 3D point pairs needing visual/geometric adjudication.

    Each input contains ``a``, ``b`` and finite ``geodesic_m`` from the current
    NavMesh. A flagged pair remains unresolved; it is not automatically fixed.
    """
    records = []
    for index, pair in enumerate(pairs):
        points = np.asarray([pair["a"], pair["b"]], dtype=np.float64)
        geodesic = float(pair["geodesic_m"])
        if points.shape != (2, 3) or not np.isfinite(points).all():
            raise ValueError("Leak pairs require two finite snapped 3D points")
        if not math.isfinite(geodesic) or geodesic < 0:
            raise ValueError("Leak pair geodesic_m must be finite and nonnegative")
        euclidean = float(np.linalg.norm(points[1] - points[0]))
        intersects = line_intersects_wall(walls, points[0, [0, 2]], points[1, [0, 2]],
                                          origin=origin, resolution=resolution)
        threshold = 1.5 * euclidean + 1.0
        flagged = not intersects and geodesic > threshold
        records.append({
            "index": index, "a": points[0].tolist(), "b": points[1].tolist(),
            "euclidean_m": euclidean, "geodesic_m": geodesic,
            "detour_threshold_m": threshold, "line_intersects_wall": intersects,
            "flagged": flagged, "review": "unresolved" if flagged else "not_flagged",
            "reason": "detour_without_wall_may_be_furniture" if flagged else None,
        })
    flagged_count = sum(record["flagged"] for record in records)
    return {"pass": bool(records) and not flagged_count,
            "pair_count": len(records), "flagged_count": flagged_count,
            "flagged_fraction": flagged_count / len(records) if records else None,
            "reasons": ["no_geodesic_pairs"] if not records else [], "pairs": records}


def alignment_report(points, *, origin, resolution, shape, raster_pixels=None):
    """Check exact pixel spacing and independently supplied raster coordinates.

    A coordinate round trip alone cannot prove a raster is aligned. Callers
    must supply the actual rasterizer's pixel positions to pass this check.
    """
    expected = world_to_pixel(points, origin, resolution).reshape(-1, 2)
    if not len(expected) or len(shape) != 2 or min(shape) < 1:
        raise ValueError("Alignment requires points and a positive 2D raster shape")
    limit = math.sqrt(0.5)
    nearest = np.rint(expected)
    extent = np.array([shape[1] - 1, shape[0] - 1])
    nearest = np.clip(nearest, 0, extent)
    roundtrip = pixel_to_world(nearest, origin, resolution)
    original = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    quantization = np.linalg.norm((roundtrip - original) / resolution, axis=1)
    raster_error = None
    if raster_pixels is not None:
        actual = np.asarray(raster_pixels, dtype=np.float64)
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise ValueError("raster_pixels must match the finite construction points")
        raster_error = float(np.linalg.norm(actual - expected, axis=1).max())
    maximum = float(quantization.max())
    return {
        "pass": maximum <= limit + 1e-9 and raster_error is not None and raster_error <= limit + 1e-9,
        "roundtrip_pass": maximum <= limit + 1e-9,
        "roundtrip_max_error_px": maximum,
        "raster_max_error_px": raster_error,
        "raster_alignment_verified": raster_error is not None,
        "limit_px": limit,
        "canvas_size_m": [shape[1] * resolution, shape[0] * resolution],
        "axes": "right:+X, down:+Z",
    }


def validate_map(walls, navmesh, *, origin, resolution, pairs,
                 construction_points, raster_pixels=None):
    """Combine the independent checks without hiding missing alignment evidence."""
    walls = np.asarray(walls, dtype=bool)
    traversal = traversability_report(walls, navmesh)
    leaks = leak_report(walls, pairs, origin=origin, resolution=resolution)
    alignment = alignment_report(construction_points, origin=origin,
                                 resolution=resolution, shape=walls.shape,
                                 raster_pixels=raster_pixels)
    return {"pass": traversal["pass"] and leaks["pass"] and alignment["pass"],
            "traversability": traversal, "leaks": leaks, "alignment": alignment}
