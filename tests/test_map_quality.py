import math

import numpy as np
import pytest

from src.data.map_quality import (
    alignment_report, erode_one, leak_report, line_intersects_wall,
    pixel_to_world, traversability_report, validate_map, world_to_pixel,
)


def test_exact_resolution_and_z_increases_down():
    origin = [-2, -4]
    points = [[-1.975, -3.975], [-1.925, -3.875]]
    pixels = world_to_pixel(points, origin, .05)
    assert np.allclose(pixels, [[0, 0], [1, 2]])
    assert np.allclose(pixel_to_world(pixels, origin, .05), points)
    assert np.allclose(pixel_to_world([[0, 0], [1, 1]], origin, .05)[1] - points[0], [.05, .05])


def test_empty_navmesh_is_failure_not_zero_overlap_success():
    report = traversability_report(np.zeros((8, 8)), np.zeros((8, 8)))
    assert not report["pass"]
    assert report["wall_overlap_fraction"] is None
    assert report["reasons"] == ["empty_navmesh"]


def test_erode_one_removes_full_boundary_ring():
    mask = np.ones((5, 5), dtype=bool)
    mask[1, 1] = False
    assert erode_one(mask).sum() == 5


def test_real_wall_split_fails_even_below_overlap_threshold():
    navmesh = np.zeros((600, 600), dtype=bool)
    navmesh[2:-2, 2:-2] = True
    walls = np.zeros_like(navmesh)
    walls[:, 300] = True
    report = traversability_report(walls, navmesh)
    assert report["wall_overlap_fraction"] < .005
    assert not report["pass"]
    assert report["components"][0]["split"]


def test_open_door_preserves_original_navmesh_connectivity():
    navmesh = np.zeros((30, 30), dtype=bool)
    navmesh[3:-3, 3:-3] = True
    walls = np.zeros_like(navmesh)
    walls[:, 15] = True
    walls[13:17, 15] = False
    report = traversability_report(walls, navmesh, max_overlap=1)
    assert report["pass"]
    assert not report["components"][0]["split"]


def test_short_wall_can_be_bypassed_in_full_wall_complement():
    navmesh = np.zeros((20, 20), dtype=bool)
    navmesh[5:15, 5:15] = True
    walls = np.zeros_like(navmesh)
    walls[5:15, 10] = True
    report = traversability_report(walls, navmesh, max_overlap=1)
    assert report["pass"]
    assert report["components"][0]["wall_complement_components"] == [0]


def test_original_islands_are_checked_separately():
    navmesh = np.zeros((20, 20), dtype=bool)
    navmesh[2:8, 2:8] = True
    navmesh[12:18, 12:18] = True
    walls = np.zeros_like(navmesh)
    walls[:, 10] = True
    report = traversability_report(walls, navmesh)
    assert report["pass"]
    assert report["original_component_count"] == 2


def test_wall_overlap_is_measured_on_eroded_original_navmesh():
    navmesh = np.ones((10, 10), dtype=bool)
    walls = np.zeros_like(navmesh)
    walls[0] = True
    assert traversability_report(walls, navmesh)["pass"]
    walls[5, 5] = True
    report = traversability_report(walls, navmesh)
    assert report["wall_overlap_pixels"] == 1
    assert report["wall_overlap_fraction"] == 1 / 64
    assert not report["pass"]


def test_supercover_includes_corner_touch_and_rejects_outside_endpoints():
    walls = np.zeros((5, 5), dtype=bool)
    walls[0, 1] = True
    assert line_intersects_wall(walls, [.5, .5], [4.5, 4.5], origin=[0, 0], resolution=1)
    assert not line_intersects_wall(walls, [.5, 2.5], [4.5, 2.5], origin=[0, 0], resolution=1)
    with pytest.raises(ValueError, match="outside"):
        line_intersects_wall(walls, [-.1, 2], [4, 2], origin=[0, 0], resolution=1)


def test_leak_pairs_distinguish_detour_crossing_wall_and_no_detour():
    walls = np.zeros((8, 8), dtype=bool)
    walls[:, 4] = True
    pairs = [
        {"a": [1.5, 0, 1.5], "b": [1.5, 0, 6.5], "geodesic_m": 10},
        {"a": [1.5, 0, 1.5], "b": [6.5, 0, 1.5], "geodesic_m": 10},
        {"a": [1.5, 0, 1.5], "b": [1.5, 0, 6.5], "geodesic_m": 5},
    ]
    report = leak_report(walls, pairs, origin=[0, 0], resolution=1)
    assert report["flagged_count"] == 1
    assert report["flagged_fraction"] == 1 / 3
    assert [item["flagged"] for item in report["pairs"]] == [True, False, False]
    assert report["pairs"][0]["review"] == "unresolved"
    assert "furniture" in report["pairs"][0]["reason"]
    with pytest.raises(ValueError, match="finite"):
        leak_report(walls, [{**pairs[0], "geodesic_m": math.inf}], origin=[0, 0], resolution=1)


def test_alignment_does_not_claim_roundtrip_proves_raster_alignment():
    points = [[.025, .025], [.475, .925], [.249, .501]]
    report = alignment_report(points, origin=[0, 0], resolution=.05, shape=(20, 20))
    assert report["roundtrip_pass"]
    assert not report["pass"]
    assert not report["raster_alignment_verified"]
    assert report["canvas_size_m"] == [1, 1]
    pixels = np.rint(world_to_pixel(points, [0, 0], .05))
    good = alignment_report(points, origin=[0, 0], resolution=.05,
                            shape=(20, 20), raster_pixels=pixels)
    assert good["pass"]
    assert good["roundtrip_max_error_px"] <= math.sqrt(.5)
    pixels[:, 1] = 19 - pixels[:, 1]
    flipped = alignment_report(points, origin=[0, 0], resolution=.05,
                               shape=(20, 20), raster_pixels=pixels)
    assert not flipped["pass"]
    assert flipped["raster_max_error_px"] > 1


def test_combined_check_needs_pair_and_independent_alignment_evidence():
    walls = np.zeros((10, 10), dtype=bool)
    navmesh = np.ones_like(walls)
    points = [[1.5, 1.5], [8.5, 8.5]]
    pairs = [{"a": [1.5, 0, 1.5], "b": [8.5, 0, 8.5], "geodesic_m": 10}]
    report = validate_map(walls, navmesh, origin=[0, 0], resolution=1, pairs=pairs,
                          construction_points=points, raster_pixels=[[1, 1], [8, 8]])
    assert report["pass"]
    assert not validate_map(walls, navmesh, origin=[0, 0], resolution=1, pairs=[],
                            construction_points=points)["pass"]
