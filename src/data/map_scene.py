"""Private Habitat rendering and navigation probes used by map construction.

All poses and depths here are builder-only evidence, never navigation inputs.
Run each scene in a fresh process so CUDA/EGL uses only physical GPU 0 or 1.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from itertools import combinations

import numpy as np
from PIL import Image, ImageDraw


NAVMESH = {"agent_radius": .18, "agent_height": .88,
           "agent_max_climb": .2, "cell_height": .05}
FLOOR_BAND_M = .35
SIGNIFICANT_ISLAND_M2 = .5


def _triangle_areas(triangles):
    return np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1) / 2


def _height_range(floor, floor_range=None):
    bounds = np.asarray(floor_range if floor_range is not None
                        else [floor - FLOOR_BAND_M, floor + FLOOR_BAND_M], dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
        raise ValueError("floor_range must be a finite [minY, maxY] with minY < maxY")
    return bounds


def _height_in_range(values, floor, floor_range=None):
    bounds = _height_range(floor, floor_range)
    values = np.asarray(values)
    return np.isfinite(values) & (values >= bounds[0]) & (values <= bounds[1])


def _floor_selection(triangles, floor, origin, side, floor_range=None):
    """Audit the floor filter before raster clipping can hide lost geometry."""
    bounds = _height_range(floor, floor_range)
    selected = _height_in_range(triangles[:, :, 1], floor, floor_range).all(axis=1)
    kept = triangles[selected]
    areas = _triangle_areas(triangles)
    low = kept[:, :, [0, 2]].reshape(-1, 2).min(0) if len(kept) else None
    high = kept[:, :, [0, 2]].reshape(-1, 2).max(0) if len(kept) else None
    outside = bool(len(kept) and ((low < origin - 1e-6).any()
                                  or (high > origin + side + 1e-6).any()))

    # NavMesh vertices may repeat at polygon/tile boundaries. Match world
    # edges, rather than vertex indices, to find a slope cut by the band.
    def edges(triangle):
        vertices = [tuple(point) for point in np.rint(triangle / .0001).astype(np.int64)]
        return [tuple(sorted((vertices[a], vertices[b]))) for a, b in ((0, 1), (1, 2), (2, 0))]

    border = {edge for triangle in kept for edge in edges(triangle)}
    adjacent = [i for i in np.flatnonzero(~selected)
                if any(edge in border for edge in edges(triangles[i]))]
    nearby = np.asarray(adjacent, dtype=int)
    surfaces = []
    for index in adjacent:
        triangle = triangles[index]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        length = float(np.linalg.norm(normal))
        normal_y = abs(float(normal[1])) / length if length > 1e-12 else None
        inclination = math.degrees(math.acos(min(1., normal_y))) if normal_y is not None else None
        surfaces.append({"triangle_index": int(index), "area_m2": float(areas[index]),
                         "normal_y_abs": normal_y, "inclination_degrees": inclination,
                         "y_range": [float(triangle[:, 1].min()), float(triangle[:, 1].max())]})
    slopes_only = bool(surfaces) and all(item["inclination_degrees"] is not None
                                        and item["inclination_degrees"] > 10 for item in surfaces)
    return selected, {
        "floor_height": float(floor), "height_band_m": FLOOR_BAND_M,
        "height_range_m": bounds.tolist(),
        "height_range_source": "explicit_episode_height_group" if floor_range is not None else "default_floor_band",
        "total_triangle_count": len(triangles), "selected_triangle_count": int(selected.sum()),
        "excluded_triangle_count": int((~selected).sum()),
        "total_area_m2": float(areas.sum()), "selected_area_m2": float(areas[selected].sum()),
        "selected_bbox_xz": [low.tolist(), high.tolist()] if len(kept) else None,
        "canvas_bbox_xz": [origin.tolist(), (origin + side).tolist()],
        "outside_canvas": outside,
        "adjacent_excluded_triangle_count": len(adjacent),
        "adjacent_excluded_area_m2": float(areas[nearby].sum()),
        "adjacent_excluded_y_range": [float(triangles[nearby, :, 1].min()),
                                       float(triangles[nearby, :, 1].max())] if adjacent else None,
        "adjacent_excluded_surfaces": surfaces,
        "boundary_classification": "bounded_stair_or_slope" if slopes_only else
                                   "excluded_flat_floor" if adjacent else "no_adjacent_exclusions",
        "requires_floor_review": bool(adjacent) and not slopes_only,
        "scope": "all native NavMesh islands, restricted by the declared absolute height range",
    }


def _camera_alignment(origin, side, size, floor, projection, world_from_camera):
    """Project asymmetric world controls through the actual camera matrices."""
    fractions = np.array([[.13, .21], [.79, .34], [.28, .87], [.64, .71], [.5, .5]])
    xz = np.asarray(origin) + fractions * side
    world = np.column_stack((xz[:, 0], np.full(len(xz), floor), xz[:, 1], np.ones(len(xz))))
    clip = (np.asarray(projection) @ np.linalg.inv(world_from_camera) @ world.T).T
    valid = np.isfinite(clip).all() and np.all(np.abs(clip[:, 3]) > 1e-12)
    if not valid:
        return {"pass": False, "reason": "invalid_camera_projection", "max_error_px": None}
    ndc = clip[:, :2] / clip[:, 3:4]
    actual = np.column_stack(((ndc[:, 0] + 1) * size / 2 - .5,
                              (1 - ndc[:, 1]) * size / 2 - .5))
    expected = fractions * size - .5
    error = np.linalg.norm(actual - expected, axis=1)
    return {"pass": bool(error.max() <= 1.), "max_error_px": float(error.max()), "limit_px": 1.,
            "method": "world controls projected through actual sensor transform and projection",
            "controls": [{"world_xz": point.tolist(), "expected_pixel": target.tolist(),
                          "camera_pixel": observed.tolist(), "error_px": float(distance)}
                         for point, target, observed, distance in zip(xz, expected, actual, error)]}


def orthographic_metric_depth(native_depth, projection):
    """Recover metric camera-forward Z from Habitat's orthographic observation.

    Habitat's depth reader applies its perspective-only unprojection to the
    orthographic OpenGL buffer as well. Its returned value q is therefore
    b/(d+a), with a=(P22-1)/2 and b=P23/2, rather than metres. Undo that step,
    then solve the actual orthographic projection for camera Z. Native zero
    denotes a patched far plane / missing hit and remains NaN.

    The input must be the native orthographic depth observation, not normalized
    buffer depth or an already-correct metric array. Perspective depth sensors
    already return metric Z and must not use this conversion.
    """
    matrix = np.asarray(projection, dtype=np.float64)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(matrix[2, :2], 0) or matrix[2, 2] >= 0):
        raise ValueError("Need an actual orthographic OpenGL projection matrix")
    scale, offset = matrix[2, 2], matrix[2, 3]
    near, far = (offset + 1) / scale, (offset - 1) / scale
    if not 0 < near < far:
        raise ValueError("Orthographic projection must have positive near and far distances")
    native = np.asarray(native_depth, dtype=np.float64)
    buffer = np.full(native.shape, np.nan)
    valid = np.isfinite(native) & (native != 0)
    np.divide(offset / 2, native, out=buffer, where=valid)
    buffer -= (scale - 1) / 2
    valid &= (buffer >= -1e-6) & (buffer < 1)
    metric = (offset - (2 * buffer - 1)) / scale
    metric = np.where(valid & np.isfinite(metric), metric, np.nan)
    return metric.astype(np.float32)


def select_gpu(gpu):
    if type(gpu) is not int or gpu not in (0, 1):
        raise ValueError("Map construction is restricted to physical GPU 0 or 1")
    selected = str(gpu)
    if os.environ.get("CUDA_VISIBLE_DEVICES", selected) != selected:
        raise ValueError("Launch a fresh scene worker with CUDA_VISIBLE_DEVICES=" + selected)
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")


class MapScene:
    def __init__(self, scene, *, origin, side, gpu=1, resolution=.05, floor_range=None):
        self.floor_range = _height_range(0, floor_range).tolist() if floor_range is not None else None
        select_gpu(gpu)
        import habitat_sim
        import magnum as mn

        self.habitat = habitat_sim
        self.origin, self.side = np.asarray(origin, dtype=float), float(side)
        self.resolution, self.gpu = float(resolution), gpu
        self.size = int(round(self.side / self.resolution))
        self.navmesh_audit, self.sample_audit, self.pair_audit = {}, {}, {}
        self._floor_islands = []
        native = habitat_sim.SimulatorConfiguration()
        native.scene_id, native.gpu_device_id = str(Path(scene).resolve(strict=True)), 0
        agent = habitat_sim.agent.AgentConfiguration()
        agent.height, agent.radius = .88, .18
        sensors = []
        for name, kind, topdown in (
                ("rgb", habitat_sim.SensorType.COLOR, False),
                ("depth", habitat_sim.SensorType.DEPTH, False),
                ("topdown", habitat_sim.SensorType.COLOR, True),
                ("topdepth", habitat_sim.SensorType.DEPTH, True)):
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid, spec.sensor_type = name, kind
            spec.sensor_subtype = (habitat_sim.SensorSubType.ORTHOGRAPHIC if topdown
                                   else habitat_sim.SensorSubType.PINHOLE)
            spec.resolution = [self.size, self.size] if topdown else [480, 640]
            spec.position = mn.Vector3(0, 0 if topdown else .88, 0)
            spec.orientation = mn.Vector3(-math.pi / 2, 0, 0) if topdown else mn.Vector3(0)
            spec.near, spec.far = .02, 100
            if topdown:
                # Habitat's ortho_scale is a zoom factor, not a world width.
                spec.ortho_scale = 1 / self.side
            else:
                spec.hfov = 79
            sensors.append(spec)
        agent.sensor_specifications = sensors
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(native, [agent]))
        self.agent = self.sim.get_agent(0)
        try:
            settings = habitat_sim.NavMeshSettings()
            settings.set_defaults()
            for key, value in NAVMESH.items():
                setattr(settings, key, value)
            if not self.sim.recompute_navmesh(self.sim.pathfinder, settings):
                raise RuntimeError("Recomputing the Lite NavMesh failed")
            self.sim.seed(19)
            self.vertices = np.asarray(self.sim.pathfinder.build_navmesh_vertices(), dtype=float)
            self.indices = np.asarray(self.sim.pathfinder.build_navmesh_vertex_indices(), dtype=int).reshape(-1, 3)
        except BaseException:
            self.close()
            raise

    def close(self):
        if getattr(self, "sim", None) is not None:
            self.sim.close()
            self.sim = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def floor(self, episode_start):
        """Snap a real training episode start, then use its NavMesh height."""
        point = np.asarray(self.sim.pathfinder.snap_point(np.asarray(episode_start, dtype=np.float32)))
        if (not np.isfinite(point).all() or abs(point[1] - episode_start[1]) > .5
                or not _height_in_range(point[1], episode_start[1], self.floor_range)):
            raise ValueError("Episode start cannot be associated with this NavMesh floor")
        return float(point[1]), point.tolist()

    def navmesh_mask(self, floor):
        triangles = self.vertices[self.indices]
        selected, self.navmesh_audit = _floor_selection(triangles, floor, self.origin, self.side, self.floor_range)
        triangles = triangles[selected]
        if not len(triangles):
            raise ValueError("No Lite NavMesh triangles near the episode floor")
        if self.navmesh_audit["outside_canvas"]:
            raise ValueError("Selected-floor NavMesh extends outside the map canvas; refusing silent clipping")
        islands = []
        for identifier in range(self.sim.pathfinder.num_islands):
            vertices = np.asarray(self.sim.pathfinder.build_navmesh_vertices(identifier), dtype=float)
            indices = np.asarray(self.sim.pathfinder.build_navmesh_vertex_indices(identifier), dtype=int).reshape(-1, 3)
            values = vertices[indices]
            keep = _height_in_range(values[:, :, 1], floor, self.floor_range).all(axis=1)
            if keep.any():
                islands.append({"id": identifier, "floor_area_m2": float(_triangle_areas(values[keep]).sum()),
                                "floor_triangle_count": int(keep.sum())})
        self._floor_islands = islands
        self.navmesh_audit.update({"native_island_count": int(self.sim.pathfinder.num_islands),
                                   "selected_floor_island_count": len(islands), "islands": islands})
        image = Image.new("1", (self.size, self.size))
        draw = ImageDraw.Draw(image)
        pixels = (triangles[:, :, [0, 2]] - self.origin) / self.resolution - .5
        for polygon in np.rint(pixels).astype(int):
            draw.polygon([tuple(point) for point in polygon], fill=1)
        return np.asarray(image, dtype=bool)

    def _pose(self, point, yaw=0):
        from habitat_sim.utils.common import quat_from_angle_axis
        state = self.habitat.AgentState()
        state.position = np.asarray(point, dtype=np.float32)
        state.rotation = quat_from_angle_axis(math.radians(yaw), np.array([0., 1., 0.]))
        self.agent.set_state(state)

    def _capture(self, names):
        # Capture only requested sensors; an RGB view should not render the
        # entire orthographic floor a second time.
        result = {}
        for name in names:
            sensor = self.sim._sensors[name]
            sensor.draw_observation()
            result[name] = np.asarray(sensor.get_observation()).copy()
        return result

    def topdown(self, floor):
        camera_height = floor + 2.05  # inside a typical ceiling, looking down
        self._pose([self.origin[0] + self.side / 2, camera_height,
                    self.origin[1] + self.side / 2])
        observations = self._capture(("topdown", "topdepth"))
        sensor = self.sim._sensors["topdown"]._sensor_object
        projection = np.asarray(sensor.render_camera.projection_matrix)
        transform = np.asarray(sensor.node.absolute_transformation(), dtype=float)
        width = 2 / abs(float(projection[0, 0]))
        if abs(width - self.side) > .001:
            raise ValueError(f"Orthographic camera width {width} != map width {self.side}")
        depth_sensor = self.sim._sensors["topdepth"]._sensor_object
        depth_projection = np.asarray(depth_sensor.render_camera.projection_matrix, dtype=float)
        depth_transform = np.asarray(depth_sensor.node.absolute_transformation(), dtype=float)
        if not np.allclose(projection, depth_projection) or not np.allclose(transform, depth_transform):
            raise ValueError("Topdown RGB and depth cameras must have matching projection and pose")
        if not np.allclose(depth_transform[:3, 2], [0, 1, 0], atol=1e-6):
            raise ValueError("Topdown depth camera must point vertically down")
        native_depth = observations["topdepth"].squeeze()
        depth = orthographic_metric_depth(native_depth, depth_projection)
        height = float(depth_transform[1, 3]) - depth - floor
        alignment = _camera_alignment(self.origin, self.side, self.size, floor, projection, transform)
        return observations["topdown"][:, :, :3], height, {
            "camera_y": camera_height, "world_width_m": width,
            "projection": projection.tolist(), "axes": "right:+X, down:+Z",
            "world_from_camera": transform.tolist(), "image_size": [self.size, self.size],
            "alignment": alignment,
            "depth_projection": depth_projection.tolist(),
            "depth": {"native_value_type": "habitat_perspective_unprojection_of_orthographic_buffer",
                      "metric_value_type": "camera_forward_z_metres",
                      "conversion": "d=(P23/2)/native-(P22-1)/2; metric_z=(P23-(2*d-1))/P22",
                      "near_m": float((depth_projection[2, 3] + 1) / depth_projection[2, 2]),
                      "far_m": float((depth_projection[2, 3] - 1) / depth_projection[2, 2]),
                      "native_missing_hit": 0, "invalid_height": "NaN"},
            "depth_to_relative_height": "depth_camera_world_y - metric_camera_forward_z_metres - floor_y",
        }

    def view(self, point, yaw, output):
        """Save RGB, metric depth, and optical camera matrix for reprojection."""
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        self._pose(point, yaw)
        observations = self._capture(("rgb", "depth"))
        rgb, depth = observations["rgb"][:, :, :3], observations["depth"].squeeze()
        Image.fromarray(rgb).save(output.with_suffix(".jpg"), quality=90)
        np.save(output.with_suffix(".npy"), depth)
        transform = np.asarray(self.sim._sensors["rgb"]._sensor_object.node.absolute_transformation(), dtype=float)
        optical = transform @ np.diag([1, -1, -1, 1])
        focal = 640 / (2 * math.tan(math.radians(79) / 2))
        return {"rgb": str(output.with_suffix(".jpg")), "depth": str(output.with_suffix(".npy")),
                "position": list(map(float, point)), "yaw_degrees": float(yaw),
                "intrinsics": [[focal, 0, 319.5], [0, focal, 239.5], [0, 0, 1]],
                "world_from_optical": optical.tolist()}

    def points(self, floor, count=80):
        if not self._floor_islands or self.navmesh_audit.get("floor_height") != floor:
            self.navmesh_mask(floor)
        if type(count) is not int or count < 1:
            raise ValueError("Sample count must be a positive integer")
        self._sample_floor = float(floor)
        islands = [{**item, "significant": item["floor_area_m2"] >= SIGNIFICANT_ISLAND_M2}
                   for item in self._floor_islands]
        for item in islands:
            item["requested"] = 4 if item["significant"] else 2
            item["sampled"] = 0
        target = max(count, sum(item["requested"] for item in islands))
        for _ in range(target - sum(item["requested"] for item in islands)):
            item = max(islands, key=lambda item: item["floor_area_m2"] / item["requested"])
            item["requested"] += 1
        points = []
        for item in islands:
            for _ in range(item["requested"] * 200):
                point = np.asarray(self.sim.pathfinder.get_random_navigable_point(
                    max_tries=100, island_index=item["id"]), dtype=float)
                if np.isfinite(point).all() and _height_in_range(point[1], floor, self.floor_range):
                    if not points or min(np.linalg.norm(point[[0, 2]] - p[[0, 2]]) for p in points) > .15:
                        points.append(point)
                        item["sampled"] += 1
                if item["sampled"] >= item["requested"]:
                    break
        uncovered = [item["id"] for item in islands if item["significant"] and item["sampled"] < 4]
        self.sample_audit = {"requested_total": count, "actual_total": len(points),
                             "effective_target": target, "height_band_m": FLOOR_BAND_M,
                             "height_range_m": _height_range(floor, self.floor_range).tolist(),
                             "significant_island_area_m2": SIGNIFICANT_ISLAND_M2,
                             "islands": islands, "undercovered_significant_islands": uncovered,
                             "pass": not uncovered}
        if len(points) < 4:
            raise ValueError("Too few actual navigable points on selected floor")
        return np.stack(points)

    def pairs(self, points, count=120, floor=None):
        points = np.asarray(points, dtype=float)
        floor = float(floor if floor is not None else getattr(self, "_sample_floor", np.median(points[:, 1])))
        groups = {}
        for index, point in enumerate(points):
            groups.setdefault(int(self.sim.pathfinder.get_island(point)), []).append(index)
        rng, pending, result = np.random.default_rng(193), {}, []
        for identifier, indices in groups.items():
            candidates = [(i, j) for i, j in combinations(indices, 2)
                          if np.linalg.norm(points[i, [0, 2]] - points[j, [0, 2]]) >= 1]
            rng.shuffle(candidates)
            pending[identifier] = candidates
        counts = {identifier: 0 for identifier in groups}
        failed, out_of_floor = 0, 0
        while len(result) < count and any(pending.values()):
            # Round-robin islands before using another pair from a large one.
            identifier = min((key for key in pending if pending[key]), key=lambda key: (counts[key], key))
            i, j = pending[identifier].pop()
            a, b = points[[i, j]]
            path = self.habitat.ShortestPath()
            path.requested_start, path.requested_end = a, b
            if self.sim.pathfinder.find_path(path) and math.isfinite(path.geodesic_distance):
                route = np.asarray(path.points, dtype=float)
                y_range = [float(route[:, 1].min()), float(route[:, 1].max())]
                outside = not bool(_height_in_range(route[:, 1], floor, self.floor_range).all())
                result.append({"a": a.tolist(), "b": b.tolist(),
                               "geodesic_m": float(path.geodesic_distance),
                               "reference_path": route.tolist(), "source_island_id": identifier,
                               "path_y_range": y_range, "out_of_floor_band": outside})
                counts[identifier] += 1
                out_of_floor += int(outside)
            else:
                failed += 1
        self.pair_audit = {"requested": count, "actual": len(result), "failed_queries": failed,
                           "out_of_floor_band": out_of_floor, "floor_height": floor,
                           "height_band_m": FLOOR_BAND_M,
                           "height_range_m": _height_range(floor, self.floor_range).tolist(),
                           "islands": [{"id": key, "sampled_points": len(groups[key]), "pairs": counts[key]}
                                       for key in groups]}
        return result

    def passage(self, segment, floor):
        """Require the robot to traverse a candidate doorway on straight probes."""
        a, b = np.asarray(segment, dtype=float)
        direction = b - a
        width = np.linalg.norm(direction)
        if width < .4:
            return {"pass": False, "reason": "insufficient_width"}
        normal = np.array([-direction[1], direction[0]]) / width
        attempts = []
        for fraction in (.35, .5, .65):
            midpoint = a + fraction * direction
            ends = [midpoint + sign * .45 * normal for sign in (-1, 1)]
            snapped = [np.asarray(self.sim.pathfinder.snap_point([p[0], floor, p[1]]), dtype=float) for p in ends]
            valid = all(np.isfinite(p).all() and np.linalg.norm(p[[0, 2]] - e) < .2
                        and _height_in_range(p[1], floor, self.floor_range) for p, e in zip(snapped, ends))
            if not valid:
                continue
            end = np.asarray(self.sim.pathfinder.try_step_no_sliding(snapped[0], snapped[1]))
            success = (np.linalg.norm(end[[0, 2]] - snapped[1][[0, 2]]) < .06
                       and bool(_height_in_range(end[1], floor, self.floor_range)))
            attempts.append({"start": snapped[0].tolist(), "requested_end": snapped[1].tolist(),
                             "actual_end": end.tolist(), "pass": bool(success)})
        return {"pass": any(p["pass"] for p in attempts), "probes": attempts,
                "reason": None if any(p["pass"] for p in attempts) else "no_robot_width_straight_passage"}
