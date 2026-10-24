"""Propose portals, verify geometry, and partition a floor through real portals.

Semantic doors/windows and gaps are candidates only.  Acceptance also requires
an injected straight NavMesh-crossing check and a clear raw-mesh height profile.
Pixel rows increase world Z; columns increase world X (image up is -Z).
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def _segment(value):
    segment = np.asarray(value, dtype=float)
    if segment.shape != (2, 2) or not np.isfinite(segment).all():
        raise ValueError("Opening segment must contain two finite world-XZ points")
    width = float(np.linalg.norm(segment[1] - segment[0]))
    if width <= 1e-8:
        raise ValueError("Opening segment must have nonzero width")
    return segment, width


def _semantic_candidates(categories, floor_height, min_width, max_width):
    result = []
    for category in ("door", "window"):
        triangles = np.asarray(categories.get(category, []), dtype=float).reshape(-1, 3, 3)
        if not len(triangles):
            continue
        keep = ((triangles[:, :, 1].max(1) >= floor_height + .2)
                & (triangles[:, :, 1].min(1) <= floor_height + 2.2))
        points = triangles[keep][:, :, [0, 2]].reshape(-1, 2)
        if not len(points):
            continue
        # Connected projection components avoid merging every door in the scene.
        # Their dimensions are only a proposal, not a claim about a traversable gap.
        origin = points.min(0) - .2
        cells = np.floor((points - origin) / .05).astype(int)
        image = Image.new("1", tuple(cells.max(0) + 1), 0)
        draw = ImageDraw.Draw(image)
        for triangle in cells.reshape(-1, 3, 2):
            draw.polygon([tuple(point) for point in triangle], fill=1)
        occupancy = np.asarray(image, dtype=bool)
        connected = ndimage.binary_dilation(occupancy, iterations=2)
        labels, count = ndimage.label(connected)
        membership = labels[cells[:, 1], cells[:, 0]]
        for identifier in range(1, count + 1):
            component = np.unique(points[membership == identifier], axis=0)
            if len(component) < 2:
                continue
            center = component.mean(0)
            _, _, axes = np.linalg.svd(component - center, full_matrices=False)
            tangent = axes[0]
            distances = (component - center) @ tangent
            width = float(np.ptp(distances))
            if min_width <= width <= max_width:
                result.append({"segment": [list(center + tangent * distances.min()),
                                           list(center + tangent * distances.max())],
                               "width_m": width,
                               "sources": [{"kind": "semantic_projection", "category": category}]})
    return result


def _gap_candidates(raw_slices, min_width, max_width):
    result = []
    angle_step, line_step = math.radians(10), .08
    for height, segments in raw_slices.items():
        segments = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
        if not len(segments):
            continue
        delta = segments[:, 1] - segments[:, 0]
        lengths = np.linalg.norm(delta, axis=1)
        keep = lengths >= .005
        segments, delta = segments[keep], delta[keep]
        angles = np.arctan2(delta[:, 1], delta[:, 0]) % math.pi
        angle_ids = np.rint(angles / angle_step).astype(int) % 18
        for angle_id in np.unique(angle_ids):
            tangent = np.array([math.cos(angle_id * angle_step), math.sin(angle_id * angle_step)])
            normal = np.array([-tangent[1], tangent[0]])
            chosen = segments[angle_ids == angle_id]
            offsets = chosen.mean(1) @ normal
            line_ids = np.rint(offsets / line_step).astype(int)
            for line_id in np.unique(line_ids):
                lines = chosen[line_ids == line_id]
                offset = float(np.median(lines.mean(1) @ normal))
                intervals = np.sort(lines @ tangent, axis=1)
                intervals = intervals[np.argsort(intervals[:, 0])]
                runs = []
                for start, end in intervals:
                    if runs and start <= runs[-1][1] + .08:
                        runs[-1][1] = max(runs[-1][1], float(end))
                    else:
                        runs.append([float(start), float(end)])
                for first, second in zip(runs, runs[1:]):
                    width = second[0] - first[1]
                    if (min_width <= width <= max_width and first[1] - first[0] >= .2
                            and second[1] - second[0] >= .2):
                        result.append({"segment": [(tangent * first[1] + normal * offset).tolist(),
                                                   (tangent * second[0] + normal * offset).tolist()],
                                       "width_m": float(width),
                                       "sources": [{"kind": "raw_slice_gap", "relative_height_m": float(height)}]})
    return result


def opening_candidates(category_triangles, raw_slices, floor_height, *, min_width=.45, max_width=1.6):
    """Return deduplicated world-XZ gap proposals, including nonpassable ones."""
    if not all(math.isfinite(value) for value in (floor_height, min_width, max_width)) or not 0 < min_width <= max_width:
        raise ValueError("Opening widths and floor height must be finite and ordered")
    candidates = _semantic_candidates(category_triangles, floor_height, min_width, max_width)
    candidates += _gap_candidates(raw_slices, min_width, max_width)
    merged = []
    for candidate in candidates:
        segment, width = _segment(candidate["segment"])
        center, tangent = segment.mean(0), (segment[1] - segment[0]) / width
        duplicate = None
        for old in merged:
            prior, prior_width = _segment(old["segment"])
            prior_tangent = (prior[1] - prior[0]) / prior_width
            if (np.linalg.norm(center - prior.mean(0)) <= .2 and abs(width - prior_width) <= .25
                    and abs(float(tangent @ prior_tangent)) >= math.cos(math.radians(20))):
                duplicate = old
                break
        if duplicate is None:
            merged.append({**candidate, "id": f"O{len(merged) + 1:03d}"})
        else:
            duplicate["sources"].extend(source for source in candidate["sources"] if source not in duplicate["sources"])
    return merged


def _height_coverage(segment, slices, *, distance=.035):
    """Fraction of the portal interior occupied by a raw surface at each height."""
    segment, width = _segment(segment)
    inset = min(.10, width * .2)
    fractions = np.linspace(inset / width, 1 - inset / width, max(4, math.ceil(width / .025)))
    points = segment[0] + fractions[:, None] * (segment[1] - segment[0])
    low, high = points.min(0) - distance, points.max(0) + distance
    coverage = {}
    for height, values in slices.items():
        lines = np.asarray(values, dtype=float).reshape(-1, 2, 2)
        keep = (lines.max(1) >= low).all(1) & (lines.min(1) <= high).all(1)
        lines = lines[keep]
        if not len(lines):
            coverage[str(height)] = 0.
            continue
        delta = lines[:, 1] - lines[:, 0]
        denominator = np.maximum(np.sum(delta ** 2, axis=1), 1e-16)
        offsets = points[:, None, :] - lines[None, :, 0, :]
        t = np.clip(np.sum(offsets * delta, axis=-1) / denominator, 0, 1)
        nearest = lines[None, :, 0, :] + t[:, :, None] * delta
        distances = np.linalg.norm(points[:, None, :] - nearest, axis=-1)
        coverage[str(height)] = float((distances.min(1) <= distance).mean())
    return coverage


def verify_openings(candidates, raw_slices, passable):
    """Require a true straight crossing and at least two clear raw height cuts.

    ``passable(segment)`` must return ``{"passable": bool, ...}`` with the actual
    two-sided NavMesh probes in its evidence. A detouring geodesic path does not
    prove that a window, closed door, or scan gap can be crossed directly.
    """
    result = []
    for candidate in candidates:
        segment, width = _segment(candidate["segment"])
        coverage = _height_coverage(segment, raw_slices)
        # The 2.0m cut may hit a valid door's lintel; lower cuts decide body clearance.
        body = [value for height, value in coverage.items() if float(height) <= 1.7 + 1e-6]
        closed_count, clear_count = sum(value > .35 for value in body), sum(value <= .2 for value in body)
        try:
            navigation = passable(segment.tolist())
            if not isinstance(navigation, dict) or type(navigation.get("passable")) is not bool:
                raise ValueError("passable callback must return a dict containing a boolean passable")
        except Exception as error:
            navigation = {"passable": None, "error": f"{type(error).__name__}: {error}"}
        if navigation["passable"] is False or closed_count >= 2:
            status = "rejected"
            reason = "no_direct_navmesh_crossing" if navigation["passable"] is False else "raw_surfaces_block_portal"
        elif navigation["passable"] is True and clear_count >= 2:
            status, reason = "accepted", "direct_crossing_and_clear_raw_height_profile"
        else:
            status, reason = "needs_review", "insufficient_independent_geometry_evidence"
        result.append({**candidate, "segment": segment.tolist(), "width_m": width, "status": status,
                       "evidence": {"reason": reason, "raw_height_coverage": coverage,
                                    "clear_body_heights": clear_count, "blocked_body_heights": closed_count,
                                    "navmesh": navigation}})
    return result


def _partition_closure(item, walls, origin, resolution):
    """Anchor an internal seal to existing walls without widening the portal."""
    segment, width = _segment(item["segment"])
    tangent = (segment[1] - segment[0]) / width
    normal = np.array([-tangent[1], tangent[0]])
    maximum, raster_tolerance = .20, .75 * resolution
    targets = np.array([0., width])
    evidence = {"method": "bounded_collinear_existing_wall_anchors",
                "rasterization": "rint((world-origin)/resolution-.5)",
                "max_endpoint_extension_m": maximum,
                "max_normal_offset_m": raster_tolerance,
                "candidate_segment_used": False,
                "extent_verification_scope": "existing_wall_raster_anchors_only",
                "physical_door_width_inferred": False,
                "wall_raster_modified": False}
    if item.get("candidate_segment") is not None:
        try:
            candidate, candidate_width = _segment(item["candidate_segment"])
            positions = np.sort((candidate - segment[0]) @ tangent)
            aligned = (abs(tangent @ ((candidate[1] - candidate[0]) / candidate_width))
                       >= math.cos(math.radians(5)))
            contains = positions[0] <= 1e-8 and positions[1] >= width - 1e-8
            bounded = positions[0] >= -maximum - 1e-8 and positions[1] <= width + maximum + 1e-8
            collinear = np.abs((candidate - segment[0]) @ normal).max() <= resolution / 2
            if aligned and contains and bounded and collinear:
                targets = positions
                evidence["candidate_segment_used"] = True
            else:
                evidence["candidate_segment_ignored_reason"] = "not_collinear_containing_and_bounded"
        except (ValueError, TypeError):
            evidence["candidate_segment_ignored_reason"] = "invalid_segment"

    anchors, offsets = [], []
    for index, endpoint in enumerate(segment):
        radius = math.ceil((maximum + raster_tolerance) / resolution) + 1
        col, row = np.rint((endpoint - origin) / resolution - .5).astype(int)
        left, right = max(0, col - radius), min(walls.shape[1], col + radius + 1)
        top, bottom = max(0, row - radius), min(walls.shape[0], row + radius + 1)
        cells = (np.argwhere(walls[top:bottom, left:right]) + [top, left])[:, ::-1]
        world = origin + (cells + .5) * resolution
        along, across = (world - segment[0]) @ tangent, (world - segment[0]) @ normal
        # A rasterized wall centre may fall up to one pixel inside the stated
        # endpoint. The outward extension is bounded in world metres.
        low, high = (-maximum, raster_tolerance) if index == 0 else (width - raster_tolerance, width + maximum)
        eligible = ((along >= low - 1e-8) & (along <= high + 1e-8)
                    & (np.abs(across) <= raster_tolerance + 1e-8))
        if not eligible.any():
            evidence.update(reason="missing_existing_wall_anchor", anchored_endpoint_count=len(anchors))
            return {"partition_segment": None, "partition_extent_verified": False,
                    "partition_evidence": evidence}, None
        selected = np.flatnonzero(eligible)
        best = selected[np.lexsort((np.abs(across[selected]), np.abs(along[selected] - targets[index])))[0]]
        anchors.append(cells[best])
        offsets.append([float(along[best]), float(across[best])])
    anchors = np.asarray(anchors)
    extent = origin + (anchors + .5) * resolution
    evidence.update(reason="both_ends_anchored_to_existing_wall_pixels",
                    anchor_pixels=anchors.tolist(),
                    endpoint_extension_m=[max(0., -offsets[0][0]), max(0., offsets[1][0] - width)],
                    normal_offsets_m=[value[1] for value in offsets])
    return {"partition_segment": extent.tolist(), "partition_extent_verified": True,
            "partition_evidence": evidence}, anchors


def region_partition(domain, walls, accepted_openings, origin, resolution, *, min_area_m2=.1):
    """Temporarily close verified portals to partition a building-floor domain.

    ``domain`` must be an independently determined floor-interior raster, not
    NavMesh island IDs. Labels are zero outside and positive for connected
    regions. Components below ``min_area_m2`` become zero, without being merged
    into neighbors; surviving IDs are continuous geometry-region candidates.
    An open-plan bypass keeps both portal sides in one region.
    Virtual seals require bounded collinear anchors on the original wall
    raster. Their recorded extent is an internal partition aid, never a change
    to ``opening.segment`` or a measurement of physical door width.
    Origin is the world-XZ upper-left *pixel corner*, with rows increasing Z.
    """
    domain, walls = np.asarray(domain, dtype=bool), np.asarray(walls, dtype=bool)
    origin = np.asarray(origin, dtype=float)
    if domain.ndim != 2 or walls.shape != domain.shape or origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError("Need matching 2D masks and a finite world-XZ origin")
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be positive and finite")
    if not math.isfinite(min_area_m2) or min_area_m2 < 0:
        raise ValueError("min_area_m2 must be finite and nonnegative")
    barrier = Image.fromarray(walls.astype(np.uint8) * 255)
    draw = ImageDraw.Draw(barrier)
    accepted = []
    for item in accepted_openings:
        if item.get("status") != "accepted":
            continue
        extent, anchors = _partition_closure(item, walls, origin, resolution)
        accepted.append({**item, **extent})
        if anchors is not None:
            draw.line([tuple(point) for point in anchors], fill=255, width=1)
    free = domain & ~np.asarray(barrier, dtype=bool)
    components, count = ndimage.label(free, structure=ndimage.generate_binary_structure(2, 1))
    sizes = np.bincount(components.reshape(-1), minlength=count + 1)
    keep = sizes * resolution ** 2 + 1e-12 >= min_area_m2
    keep[0] = False
    identifiers = np.zeros(count + 1, dtype=np.int32)
    identifiers[keep] = np.arange(1, int(keep.sum()) + 1)
    labels = identifiers[components]
    regions = []
    for identifier, bounds in enumerate(ndimage.find_objects(labels), start=1):
        if bounds is None:
            continue
        local = labels[bounds] == identifier
        rows, cols = np.nonzero(local)
        rows, cols = rows + bounds[0].start, cols + bounds[1].start
        regions.append({"id": f"R{identifier:03d}", "label": identifier,
                        "area_m2": float(len(rows) * resolution ** 2),
                        "center_xz": (origin + np.array([cols.mean() + .5, rows.mean() + .5]) * resolution).tolist(),
                        "status": "geometry_region_candidate", "minimum_area_m2": min_area_m2,
                        "source": "floor_domain_partitioned_by_walls_and_verified_portals"})

    def label_at(point):
        col, row = np.floor((point - origin) / resolution).astype(int)
        if 0 <= row < labels.shape[0] and 0 <= col < labels.shape[1]:
            return int(components[row, col]), int(labels[row, col])
        return 0, 0

    openings = []
    for item in accepted:
        segment, width = _segment(item["segment"])
        normal = np.array([-(segment[1, 1] - segment[0, 1]), segment[1, 0] - segment[0, 0]]) / width
        center, sides = segment.mean(0), []
        for sign in (-1, 1):
            values = [label_at(center + sign * normal * distance)
                      for distance in np.arange(max(.10, 2 * resolution), 1.01, resolution)]
            # The first real component on this side owns the assignment. If it
            # was discarded as a tiny fragment, keep the side unknown instead
            # of jumping over it to a more distant region.
            sides.append(next((label for component, label in values if component > 0), 0))
        status = ("unassigned" if 0 in sides else "split" if sides[0] != sides[1] else "open_plan_or_bypass")
        openings.append({**item, "regions": [f"R{value:03d}" if value else None for value in sides],
                         "partition_status": status})
    return labels, regions, openings
