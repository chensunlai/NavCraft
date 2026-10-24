"""Wall fitting preserves geometry, gaps, curvature and source evidence."""

import numpy as np
import pytest

from src.data.map_wall_fit import fit_wall_primitives


def segments(points):
    points = np.asarray(points, dtype=float)
    return np.stack([points[:-1], points[1:]], axis=1)


def test_noisy_long_straight_wall_becomes_one_supported_line():
    rng = np.random.default_rng(19)
    points = np.column_stack([np.linspace(0, 8, 65), rng.normal(0, .003, 65)])
    result = fit_wall_primitives(segments(points))
    assert len(result["primitives"]) == 1
    wall = result["primitives"][0]
    assert wall["type"] == "line"
    assert wall["length_m"] == pytest.approx(8, abs=.01)
    assert wall["source_segment_indices"] == list(range(64))
    assert wall["fitting"]["max_residual_m"] < .015
    assert result["metrics"]["source_length_coverage"] == 1


def test_four_duplicate_height_slices_merge_before_endpoint_chains():
    wall = segments(np.column_stack([np.linspace(0, 8, 65), np.zeros(65)]))
    result = fit_wall_primitives(np.tile(wall, (4, 1, 1)))
    assert len(result["primitives"]) == 1
    assert result["primitives"][0]["length_m"] == pytest.approx(8)
    assert result["primitives"][0]["source_segment_indices"] == list(range(256))
    assert result["metrics"]["prepared_segment_count"] == 1


def test_noisy_differently_sampled_height_cuts_consolidate_same_wall():
    rng = np.random.default_rng(3)
    layers = []
    for count in (51, 63, 69, 57):
        points = np.column_stack([np.linspace(0, 8, count), rng.normal(0, .002, count)])
        layers.append(segments(points))
    result = fit_wall_primitives(np.concatenate(layers))
    assert len(result["primitives"]) == 1
    assert result["primitives"][0]["length_m"] == pytest.approx(8, abs=.01)
    assert result["metrics"]["source_length_coverage"] == 1


def test_overlapping_collinear_fragments_preserve_every_support_index():
    wall = np.array([[[0, 0], [2, 0]], [[1, 0], [3, 0]], [[2.5, 0], [5, 0]],
                     [[0, 0], [1, 0]], [[4, 0], [5, 0]]], dtype=float)
    result = fit_wall_primitives(wall)
    assert len(result["primitives"]) == 1
    assert result["primitives"][0]["source_segment_indices"] == list(range(5))
    assert result["primitives"][0]["length_m"] == pytest.approx(5)


def test_actual_door_gap_is_not_joined_even_with_duplicate_layers():
    left = segments(np.column_stack([np.linspace(0, 2, 17), np.zeros(17)]))
    right = segments(np.column_stack([np.linspace(2.9, 5, 17), np.zeros(17)]))
    result = fit_wall_primitives(np.tile(np.concatenate([left, right]), (4, 1, 1)))
    assert len(result["primitives"]) == 2
    endpoints = [np.asarray(wall["endpoints"])[:, 0] for wall in result["primitives"]]
    assert sorted((float(value.min()), float(value.max())) for value in endpoints) == [(0, 2), (2.9, 5)]


def test_oblique_wall_is_not_snapped_to_manhattan_axes():
    points = np.column_stack([np.linspace(0, 5, 41), np.linspace(1, 6, 41)])
    result = fit_wall_primitives(segments(points))
    assert len(result["primitives"]) == 1
    direction = np.diff(result["primitives"][0]["endpoints"], axis=0)[0]
    assert direction[1] / direction[0] == pytest.approx(1)
    assert not result["metrics"]["manhattan_snapping"]


def test_true_quarter_circle_keeps_arc_instead_of_a_straight_chord():
    angles = np.linspace(0, np.pi / 2, 49)
    points = np.column_stack([2 * np.cos(angles) + 3, 2 * np.sin(angles) - 1])
    result = fit_wall_primitives(segments(points))
    assert len(result["primitives"]) == 1
    arc = result["primitives"][0]
    assert arc["type"] == "arc"
    np.testing.assert_allclose(arc["center"], [3, -1], atol=1e-7)
    assert arc["radius_m"] == pytest.approx(2)
    assert abs(arc["sweep_degrees"]) == pytest.approx(90)
    assert arc["length_m"] == pytest.approx(np.pi)
    assert arc["fitting"]["circle_to_line_rms_ratio"] < .01
    assert arc["source_segment_indices"] == list(range(48))


def test_duplicate_quarter_circle_layers_still_keep_curvature():
    angles = np.linspace(-np.pi / 2, 0, 49)
    circle = segments(np.column_stack([2 * np.cos(angles), 2 * np.sin(angles)]))
    result = fit_wall_primitives(np.tile(circle, (4, 1, 1)))
    assert len(result["primitives"]) == 1
    assert result["primitives"][0]["type"] == "arc"
    assert abs(result["primitives"][0]["sweep_degrees"]) == pytest.approx(90)
    assert result["metrics"]["covered_segment_count"] == len(circle) * 4


def test_l_corner_is_two_walls_not_a_diagonal_or_circle():
    wall = np.concatenate([segments([[0, 0], [1, 0], [2, 0], [3, 0]]),
                           segments([[3, 0], [3, 1], [3, 2], [3, 3]])])
    result = fit_wall_primitives(wall)
    assert len(result["primitives"]) == 2
    assert all(item["type"] == "line" for item in result["primitives"])
    for item in result["primitives"]:
        delta = np.diff(item["endpoints"], axis=0)[0]
        assert min(abs(delta)) < 1e-8


@pytest.mark.parametrize("separation", [.02, .2])
def test_parallel_walls_are_not_collapsed_to_one_average_line(separation):
    first = segments(np.column_stack([np.linspace(0, 4, 33), np.zeros(33)]))
    second = first + [0, separation]
    result = fit_wall_primitives(np.concatenate([first, second]))
    assert len(result["primitives"]) == 2
    assert sorted(round(np.asarray(item["endpoints"])[:, 1].mean(), 4) for item in result["primitives"]) == [0, separation]


def test_degenerate_and_short_support_is_reported_not_invented():
    result = fit_wall_primitives([[[0, 0], [0, 0]], [[1, 1], [1.1, 1]]])
    assert result["primitives"] == []
    assert result["metrics"]["degenerate_segment_indices"] == [0]
    assert result["metrics"]["discarded_segment_indices"] == [1]
    assert result["metrics"]["source_length_coverage"] == 0
    assert fit_wall_primitives(np.empty((0, 2, 2)))["primitives"] == []


def test_invalid_geometry_or_limits_are_explicit():
    with pytest.raises(ValueError, match="finite"):
        fit_wall_primitives(np.full((1, 2, 2), np.nan))
    with pytest.raises(ValueError, match="positive"):
        fit_wall_primitives([[[0, 0], [1, 0]]], max_gap=0)
