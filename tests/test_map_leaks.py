"""Flag review retains uncertainty and never edits a map or original scores."""

import numpy as np
import pytest

from src.data.map_leaks import review_leak_pairs


def panel(x, low, high):
    return np.array([[[x, low, 0], [x, high, 0], [x, high, 2]],
                     [[x, low, 0], [x, high, 2], [x, low, 2]]], dtype=float)


def review(triangles, *, domain=None, walls=None, pair=None):
    domain = np.ones((40, 40), dtype=bool) if domain is None else domain
    walls = np.zeros_like(domain) if walls is None else walls
    pair = {"a": [.25, 0, 1], "b": [3.75, 0, 1], "geodesic_m": 10., "flagged": True} if pair is None else pair
    return review_leak_pairs(triangles, [pair], 0, [0, 0], 4, domain, walls)[0]


def test_low_table_geometry_is_reviewed_without_calling_it_a_table():
    triangles = np.concatenate([panel(1.3, .5, .94), panel(2.0, .5, .94)])
    walls = np.zeros((40, 40), dtype=bool)
    original = walls.copy()
    result = review(triangles, walls=walls)
    assert result["classification"] == "low_obstacle_detour"
    assert result["review"] == "reviewed_false_flag"
    assert result["floor_domain_support_fraction"] == 1
    assert result["raw_intersection_counts"]["0.9"] == 2
    assert result["low_obstacle_proof"]["interior_upper_cut_hits"] == 0
    assert result["low_obstacle_proof"]["high_guard_distance_height_segments"] == []
    assert not result["rule"]["semantic_labels_used"]
    assert "table" not in str(result)
    np.testing.assert_array_equal(walls, original)


def test_high_wall_is_not_reviewed_as_a_low_obstacle():
    result = review(panel(2, 0, 2.6))
    assert result["classification"] == "unresolved"
    assert result["review"] == "unresolved"
    assert result["low_obstacle_proof"]["interior_upper_cut_hits"] > 0
    assert result["low_obstacle_proof"]["high_guard_distance_height_segments"]


def test_sloped_surface_between_cut_heights_triggers_exact_section_guard():
    low = panel(1, 0, .8)
    sloped = np.array([[[2, 1.05, 0], [2, 1.25, 2], [2.2, 1.05, 0]]])
    result = review(np.concatenate([low, sloped]))
    assert result["low_obstacle_proof"]["interior_upper_cut_hits"] == 0
    assert result["low_obstacle_proof"]["high_guard_distance_height_segments"]
    assert result["classification"] == "unresolved"


def test_unobserved_floor_void_stays_unresolved_even_with_low_surfaces():
    domain = np.ones((40, 40), dtype=bool)
    domain[:, 9:32] = False
    result = review(panel(2, 0, .9), domain=domain)
    assert result["floor_domain_support_fraction"] < .7
    assert result["classification"] == "unobserved_floor_gap"
    assert result["review"] == "unresolved"
    assert "not_disambiguated" in result["reason"]


def test_no_geometry_intersection_is_not_automatically_a_false_flag():
    result = review(np.empty((0, 3, 3)))
    assert result["classification"] == "unresolved"
    assert sum(result["raw_intersection_counts"].values()) == 0


def test_existing_wall_is_not_reclassified_as_low_false_flag():
    walls = np.zeros((40, 40), dtype=bool)
    walls[:, 20] = True
    result = review(panel(1.0, 0, .9), walls=walls)
    assert result["line_intersects_existing_wall"]
    assert result["classification"] == "unresolved"


def test_triangle_vertex_boundary_is_preserved_and_duplicate_hits_coalesced():
    # The .5m slice touches the triangle's vertex; report it once even when a
    # duplicate scan face is present, without invalid zero-length intersections.
    triangle = np.array([[[2, .1, 0], [2, .5, 1], [2, .1, 2]]])
    result = review(np.concatenate([triangle, triangle]))
    assert result["raw_intersection_counts"]["0.2"] == 1
    # A lone plane/vertex touch has no positive-length cut at its top, which is
    # explicit rather than silently creating a barrier at that height.
    assert result["raw_intersection_counts"]["0.5"] == 0
    assert result["classification"] == "low_obstacle_detour"


def test_coplanar_chord_triangle_cannot_hide_a_high_surface():
    high = np.array([[[1, 0, 1], [2, 2.2, 1], [3, 0, 1]]])
    result = review(high)
    assert result["classification"] == "unresolved"
    assert result["low_obstacle_proof"]["high_guard_distance_height_segments"]
    assert any(hit["kind"] == "collinear_overlap" for hit in result["raw_intersections"]["0.2"])


def test_explicit_flagged_indices_or_flagged_records_are_required():
    pairs = [{"a": [.25, 0, 1], "b": [3.75, 0, 1], "geodesic_m": 10.}]
    args = (panel(2, 0, .9), pairs, 0, [0, 0], 4, np.ones((40, 40)), np.zeros((40, 40)))
    assert review_leak_pairs(*args) == []
    assert len(review_leak_pairs(*args, flagged_indices=[0])) == 1
    with pytest.raises(ValueError, match="unique valid"):
        review_leak_pairs(*args, flagged_indices=[0, 0])


def test_pair_endpoint_touch_is_reported_but_not_low_obstacle_evidence():
    result = review(panel(.25, 0, .9))
    assert result["raw_intersection_counts"]["0.2"] == 1
    assert result["low_obstacle_proof"]["interior_low_cut_hits"] == 0
    assert result["classification"] == "unresolved"


def offset_panel(z, low, high):
    return np.array([[[1.5, low, z], [2., low, z], [2., high, z]],
                     [[1.5, low, z], [2., high, z], [1.5, high, z]]])


def test_low_obstacle_at_robot_radius_edge_explains_clearance_flag():
    result = review(offset_panel(1.18, .3, .6))
    assert result["classification"] == "low_obstacle_clearance_detour"
    assert result["review"] == "reviewed_false_flag"
    assert sum(result["raw_intersection_counts"].values()) == 0
    footprint = result["footprint_review"]
    assert footprint["radius_m"] == .18
    assert footprint["low_obstacle_offsets_m"] == [.18]
    assert footprint["boundary_touch_only"]
    assert len(footprint["parallel_chords"]) == 11
    assert not footprint["action_replayed"]
    assert not footprint["movement_feasibility_claimed"]


def test_side_high_wall_blocks_clearance_explanation():
    result = review(np.concatenate([offset_panel(1.18, .3, .6), offset_panel(.9, 0, 2.5)]))
    assert result["classification"] == "unresolved"
    footprint = result["footprint_review"]
    assert not footprint["nonarchitectural_clearance_evidence"]
    assert footprint["continuous_band_high_surface_polygons_distance_offset_height"]
    assert any(row["high_guard_distance_height_segments"] for row in footprint["parallel_chords"])


def test_high_surface_between_parallel_chords_is_not_missed():
    # This narrow panel lies between sampled offsets+.05 and+.10 and between
    # the1.3/0.9height cuts. Exact continuous-strip clipping still finds it.
    hidden = offset_panel(1.071, 1.11, 1.20)
    result = review(np.concatenate([offset_panel(1.18, .3, .6), hidden]))
    assert result["classification"] == "unresolved"
    footprint = result["footprint_review"]
    assert not any(row["high_guard_distance_height_segments"] for row in footprint["parallel_chords"])
    assert footprint["continuous_band_high_surface_polygons_distance_offset_height"]


def test_footprint_floor_void_blocks_clearance_explanation():
    domain = np.ones((40, 40), dtype=bool)
    domain[11, :] = False  # Centerline is supported; the+.18m side is not.
    result = review(offset_panel(1.18, .3, .6), domain=domain)
    assert result["floor_domain_support_fraction"] == 1.
    assert result["classification"] == "unresolved"
    assert not result["footprint_review"]["nonarchitectural_clearance_evidence"]


def test_clear_footprint_without_a_raw_obstacle_remains_unknown():
    result = review(np.empty((0, 3, 3)))
    assert result["classification"] == result["review"] == "unresolved"
    assert result["footprint_review"]["low_obstacle_offsets_m"] == []
    assert not result["footprint_review"]["nonarchitectural_clearance_evidence"]
