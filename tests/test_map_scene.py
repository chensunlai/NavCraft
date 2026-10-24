from types import SimpleNamespace

import numpy as np
import pytest

from src.data.map_scene import (MapScene, _camera_alignment, _floor_selection,
                                _height_in_range, orthographic_metric_depth)


def _orthographic_camera(origin, side, height):
    transform = np.array([[1, 0, 0, origin[0] + side / 2],
                          [0, 0, 1, height],
                          [0, -1, 0, origin[1] + side / 2],
                          [0, 0, 0, 1]], dtype=float)
    projection = np.diag([2 / side, 2 / side, -.02, 1.])
    return projection, transform


def test_actual_camera_alignment_detects_reversed_z_and_wrong_center():
    origin, side, size = np.array([-3., -7.]), 20., 400
    projection, transform = _orthographic_camera(origin, side, 2.05)
    correct = _camera_alignment(origin, side, size, 0., projection, transform)
    assert correct["pass"]
    assert correct["max_error_px"] < 1e-10
    reversed_projection = projection.copy()
    reversed_projection[1, 1] *= -1
    assert not _camera_alignment(origin, side, size, 0., reversed_projection, transform)["pass"]
    transform[0, 3] += .1
    assert _camera_alignment(origin, side, size, 0., projection, transform)["max_error_px"] == pytest.approx(2.)


def _projection(side=10., near=.02, far=100.):
    return np.array([[2 / side, 0, 0, 0], [0, 2 / side, 0, 0],
                     [0, 0, -2 / (far - near), -(far + near) / (far - near)],
                     [0, 0, 0, 1]], dtype=np.float32)


def _habitat_native_orthographic(metric, projection):
    # Project explicit camera-space points into the OpenGL depth buffer, then
    # reproduce Habitat's documented perspective-only depth reader.
    metric = np.asarray(metric, dtype=np.float32)
    points = np.stack([np.zeros_like(metric), np.zeros_like(metric), -metric, np.ones_like(metric)], axis=-1)
    clip = points @ projection.T
    buffer = (clip[..., 2] / clip[..., 3] + 1) / 2
    a = np.float32((projection[2, 2] - np.float32(1)) / np.float32(2))
    b = np.float32(projection[2, 3] / np.float32(2))
    return b / (buffer + a)


@pytest.mark.parametrize("near,far", [(.02, 100.), (.1, 20.)])
def test_orthographic_metric_depth_recovers_three_known_plane_heights(near, far):
    projection = _projection(near=near, far=far)
    camera_y = 2.05
    plane_y = np.array([0., .65, 1.3])
    native = _habitat_native_orthographic(camera_y - plane_y, projection)
    assert np.ptp(native) < .2  # This compression caused the false near-1m height map.
    metric = orthographic_metric_depth(native, projection)
    np.testing.assert_allclose(camera_y - metric, plane_y, atol=2e-5)


def test_orthographic_conversion_handles_actual_near_far_and_negative_native_encoding():
    projection = _projection(near=.3, far=80.)
    distances = np.array([.3, 1., 4., 25., 60., 79.])
    native = _habitat_native_orthographic(distances, projection)
    assert (native < 0).any()  # Negative native values can encode valid distant surfaces.
    np.testing.assert_allclose(orthographic_metric_depth(native, projection), distances, atol=2e-5)


def test_orthographic_far_plane_and_invalid_observations_do_not_become_fake_floor():
    recovered = orthographic_metric_depth(np.array([0., np.nan, np.inf, -np.inf]), _projection())
    assert np.isnan(recovered).all()
    perspective = _projection()
    perspective[3] = [0, 0, -1, 0]
    with pytest.raises(ValueError, match="orthographic"):
        orthographic_metric_depth([1.], perspective)


def test_topdown_preserves_rgb_and_reports_correct_metric_depth_conversion():
    scene = MapScene.__new__(MapScene)
    scene.origin, scene.side, scene.size = np.array([-5., -5.]), 10., 3
    scene._pose = lambda *args: None
    projection = _projection()
    _, transform = _orthographic_camera(scene.origin, scene.side, 2.05)
    camera = SimpleNamespace(render_camera=SimpleNamespace(projection_matrix=projection),
                             node=SimpleNamespace(absolute_transformation=lambda: transform))
    sensor = SimpleNamespace(_sensor_object=camera)
    scene.sim = SimpleNamespace(_sensors={"topdown": sensor, "topdepth": sensor})
    expected = np.tile([0., .65, 1.3], (3, 1))
    native = _habitat_native_orthographic(2.05 - expected, projection)
    native[0, 0] = 0
    rgb = np.full((3, 3, 4), 127, dtype=np.uint8)
    scene._capture = lambda names: {"topdown": rgb, "topdepth": native}
    actual_rgb, height, metadata = scene.topdown(0.)
    np.testing.assert_array_equal(actual_rgb, rgb[:, :, :3])
    assert np.isnan(height[0, 0])
    np.testing.assert_allclose(height[1:], expected[1:], atol=2e-5)
    assert metadata["alignment"]["pass"]
    assert metadata["depth"]["metric_value_type"] == "camera_forward_z_metres"
    assert metadata["depth_projection"] == projection.tolist()


def test_floor_audit_detects_clipping_before_rasterization():
    triangles = np.array([[[0, 0, 0], [2, 0, 0], [0, 0, 2]],
                          [[3, 3, 3], [4, 3, 3], [3, 3, 4]]], dtype=float)
    selected, audit = _floor_selection(triangles, 0., np.array([0., 0.]), 1.)
    assert selected.tolist() == [True, False]
    assert audit["outside_canvas"]
    assert audit["selected_area_m2"] == 2
    assert audit["total_area_m2"] == 2.5
    assert audit["selected_bbox_xz"] == [[0, 0], [2, 2]]


def test_floor_audit_identifies_bounded_stair_or_slope():
    triangles = np.array([[[0, 0, 0], [1, 0, 0], [0, .2, 1]],
                          [[1, 0, 0], [0, .2, 1], [1, .45, 1]],
                          [[3, 1.2, 3], [4, 1.2, 3], [3, 1.2, 4]]], dtype=float)
    selected, audit = _floor_selection(triangles, 0., np.array([-1., -1.]), 10.)
    assert selected.tolist() == [True, False, False]
    assert not audit["requires_floor_review"]
    assert audit["boundary_classification"] == "bounded_stair_or_slope"
    assert audit["adjacent_excluded_surfaces"][0]["inclination_degrees"] > 10
    assert audit["adjacent_excluded_triangle_count"] == 1
    assert audit["adjacent_excluded_y_range"] == [0., .45]


def test_explicit_range_includes_same_floor_zero_and_point_four_but_not_upper_floor():
    triangles = np.array([[[0, 0, 0], [1, 0, 0], [0, 0, 1]],
                          [[1, .4, 0], [2, .4, 0], [1, .4, 1]],
                          [[1, 1.6, 0], [2, 1.6, 0], [1, 1.6, 1]]], dtype=float)
    selected, audit = _floor_selection(triangles, .4, np.array([-1., -1.]), 10., [-.2, .7])
    assert selected.tolist() == [True, True, False]
    assert audit["height_range_m"] == [-.2, .7]
    assert audit["height_range_source"] == "explicit_episode_height_group"
    default, _ = _floor_selection(triangles, .4, np.array([-1., -1.]), 10.)
    assert default.tolist() == [False, True, False]


def test_adjacent_nearly_horizontal_surface_cut_by_range_requires_review():
    triangles = np.array([[[0, .3, 0], [1, .3, 0], [0, .3, 1]],
                          [[1, .3, 0], [0, .3, 1], [1, .4, 1]]], dtype=float)
    _, audit = _floor_selection(triangles, 0., np.array([-1., -1.]), 10.)
    assert audit["requires_floor_review"]
    assert audit["boundary_classification"] == "excluded_flat_floor"
    assert audit["adjacent_excluded_surfaces"][0]["inclination_degrees"] < 10
    assert audit["adjacent_excluded_area_m2"] > 0


def test_common_height_range_rejects_other_levels_and_invalid_bounds():
    assert _height_in_range([0., .4, 1.6, -2.3], .4, [-.2, .7]).tolist() == [True, True, False, False]
    with pytest.raises(ValueError, match="floor_range"):
        _height_in_range([0.], 0., [1., 0.])


class _Pathfinder:
    def __init__(self):
        self.indices = {0: 0, 1: 0}

    def get_random_navigable_point(self, max_tries, island_index):
        index = self.indices[island_index]
        self.indices[island_index] += 1
        return [island_index * 10 + (index % 3), 0., (index // 3) % 3]

    def get_island(self, point):
        return int(point[0] >= 10)

    def find_path(self, query):
        query.points = [query.requested_start,
                        (query.requested_start + query.requested_end) / 2 + [0, .5, 0],
                        query.requested_end]
        query.geodesic_distance = float(np.linalg.norm(query.requested_end - query.requested_start) + 1)
        return True


def _fake_scene():
    scene = MapScene.__new__(MapScene)
    scene.floor_range = None
    scene._floor_islands = [{"id": 0, "floor_area_m2": 100.}, {"id": 1, "floor_area_m2": 1.}]
    scene.navmesh_audit = {"floor_height": 0.}
    scene.sim = SimpleNamespace(pathfinder=_Pathfinder())
    scene.habitat = SimpleNamespace(ShortestPath=SimpleNamespace)
    return scene


def test_sampling_covers_small_significant_island_before_area_weighting():
    scene = _fake_scene()
    points = scene.points(0., count=12)
    assert points.shape == (12, 3)
    assert np.count_nonzero(points[:, 0] >= 10) >= 4
    assert scene.sample_audit["pass"]
    assert not scene.sample_audit["undercovered_significant_islands"]


def test_pairs_are_stratified_and_mark_paths_leaving_floor():
    scene = _fake_scene()
    points = scene.points(0., count=12)
    pairs = scene.pairs(points, count=4)
    assert {pair["source_island_id"] for pair in pairs} == {0, 1}
    assert all(pair["out_of_floor_band"] for pair in pairs)
    assert all(pair["path_y_range"] == [0., .5] for pair in pairs)
    assert scene.pair_audit["out_of_floor_band"] == 4
    assert [item["pairs"] for item in scene.pair_audit["islands"]] == [2, 2]


def test_pairs_use_explicit_range_instead_of_old_fixed_band():
    scene = _fake_scene()
    scene.floor_range = [-.2, .7]
    points = scene.points(0., count=12)
    pairs = scene.pairs(points, count=4)
    assert not any(pair["out_of_floor_band"] for pair in pairs)
    assert scene.sample_audit["height_range_m"] == [-.2, .7]
    assert scene.pair_audit["height_range_m"] == [-.2, .7]
