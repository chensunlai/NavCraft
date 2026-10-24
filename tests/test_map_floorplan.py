"""Independent surface evidence must precede architectural vector rendering."""
import numpy as np

from src.data.map_floorplan import (RawWallEvidence, primitive_points, split_primitive,
                                    support_with_wall_axes, verify_primitive)


def wall(x0, x1, z, height=2.5):
    return np.array([[[x0, 0, z], [x1, 0, z], [x1, height, z]],
                     [[x0, 0, z], [x1, height, z], [x0, height, z]]], dtype=float)


def line(a, b):
    return {"id": "candidate", "type": "line", "start": a, "end": b,
            "endpoints": [a, b]}


def test_semantic_candidate_without_raw_surface_is_rejected():
    evidence = RawWallEvidence(wall(0, 3, 2), 0, [-1, -1], 5, np.zeros((100, 100), bool))
    accepted, result = verify_primitive(line([0, 0], [3, 0]), evidence)
    assert accepted == []
    assert not result["accepted"]
    assert result["raw_supported_fraction"] == 0


def test_independent_raw_wall_supports_a_fitted_long_line():
    evidence = RawWallEvidence(wall(0, 3, 0), 0, [-1, -1], 5, np.zeros((100, 100), bool))
    accepted, result = verify_primitive(line([0, 0], [3, 0]), evidence)
    assert len(accepted) == 1 and result["accepted"]
    assert result["retained_supported_fraction"] == 1
    assert accepted[0]["type"] == "line"


def test_measured_passage_is_carved_as_vector_intervals_not_pixel_erasure():
    raw = np.concatenate([wall(0, 1, 0), wall(2, 3, 0)])
    nav = np.zeros((100, 100), bool)
    nav[10:30, 45:55] = True  # centre X=1.5, crossing world Z=0
    evidence = RawWallEvidence(raw, 0, [-1, -1], 5, nav)
    accepted, result = verify_primitive(line([0, 0], [3, 0]), evidence)
    assert len(accepted) == 2
    assert result["carved_passages"]
    assert accepted[0]["end"][0] < 1.3 and accepted[1]["start"][0] > 1.7
    assert accepted[0]["start"][1] == accepted[0]["end"][1] == 0


def test_true_arc_sampling_does_not_turn_into_straight_chord():
    points = primitive_points({"type": "arc", "center": [2, 3], "radius_m": 2,
        "start_angle_degrees": 0, "end_angle_degrees": 90, "sweep_degrees": 90})
    np.testing.assert_allclose(np.linalg.norm(points - [2, 3], axis=1), 2, atol=1e-10)
    assert points[len(points) // 2, 1] > 4.3


def test_full_circle_survives_passage_splitting_and_keeps_arc_parameters():
    arc = {"id": "round_wall", "type": "arc", "center": [2, 3], "radius_m": 2,
           "start_angle_degrees": 0, "end_angle_degrees": 360, "sweep_degrees": 360}
    points = primitive_points(arc)
    whole = split_primitive(arc, points, np.ones(len(points), dtype=bool))
    assert len(whole) == 1
    assert abs(whole[0]["length_m"] - 4 * np.pi) < 1e-10
    half = split_primitive(arc, points, np.arange(len(points)) <= len(points) // 2)[0]
    assert half["sweep_degrees"] < 181
    assert abs(half["length_m"] - abs(half["sweep_radians"]) * 2) < 1e-10


def test_door_lintel_does_not_supply_wall_body_support():
    lintel = wall(0, 3, 0, height=.5)
    lintel[:, :, 1] += 2
    raw = np.concatenate([wall(0, 1, 0), wall(2, 3, 0), lintel])
    nav = np.zeros((100, 100), bool)
    nav[10:30, 45:55] = True
    evidence = RawWallEvidence(raw, 0, [-1, -1], 5, nav)
    accepted, result = verify_primitive(line([0, 0], [3, 0]), evidence)
    assert len(accepted) == 2 and result["carved_passages"]
    assert not evidence.support([[1.3, 0], [1.7, 0]])["pass"]


def test_explicit_circle_samples_one_radius_and_a_carved_piece_becomes_an_arc():
    circle = {"id": "round_wall", "type": "circle", "center": [2, 3], "radius_m": 2.}
    points = primitive_points(circle)
    np.testing.assert_allclose(np.linalg.norm(points - [2, 3], axis=1), 2, atol=1e-10)
    whole = split_primitive(circle, points, np.ones(len(points), dtype=bool))
    assert whole[0]["type"] == "circle"
    partial = split_primitive(circle, points, np.arange(len(points)) <= len(points) // 2)
    assert partial[0]["type"] == "arc"
    assert partial[0]["sweep_degrees"] < 181


def test_measured_two_face_wall_supports_its_axis_but_not_unknown_extensions():
    raw = np.concatenate([wall(0, 3, 0), wall(0, 3, .3)])
    evidence = RawWallEvidence(raw, 0, [-1, -1], 5, np.zeros((100, 100), bool))
    axis = {"id": "axis", "start": [0, .15], "end": [3, .15], "wall_thickness_m": .3}
    middle = [[1, .15], [2, .15]]
    assert not evidence.support(middle)["pass"]
    checked = support_with_wall_axes(middle, evidence, [axis])
    assert checked["pass"] and checked["measured_wall_axis_ids"] == ["axis"]
    assert not support_with_wall_axes([[3.2, .15], [3.4, .15]], evidence, [axis])["pass"]
    assert not support_with_wall_axes([[1, .6], [2, .6]], evidence, [axis])["pass"]


def test_near_parallel_fits_can_select_one_independent_raw_surface():
    evidence = RawWallEvidence(wall(0, 3, 0), 0, [-1, -1], 5, np.zeros((100, 100), bool))
    checked = evidence.same_surface(line([0, -.005], [3, -.005]),
                                    line([.5, .008], [2, .008]))
    assert checked["pass"]
    assert all(item["same_nearest_raw_cut_fraction"] == 1 for item in checked["height_checks"])


def test_separate_raw_planes_two_centimetres_apart_are_not_duplicates():
    raw = np.concatenate([wall(0, 3, 0), wall(0, 3, .02)])
    evidence = RawWallEvidence(raw, 0, [-1, -1], 5, np.zeros((100, 100), bool))
    checked = evidence.same_surface(line([0, 0], [3, 0]), line([.5, .02], [2, .02]))
    assert not checked["pass"]
    assert checked["reason"] == "multiple_raw_surface_sheets"


def test_two_inaccurate_fits_selecting_one_nearest_plane_still_cannot_merge_two_sheets():
    raw = np.concatenate([wall(0, 3, 0), wall(0, 3, .02)])
    for triangles in (raw, raw[::-1]):
        evidence = RawWallEvidence(triangles, 0, [-1, -1], 5, np.zeros((100, 100), bool))
        for offsets in ((0, .009), (.01, .015)):
            assert not evidence.same_surface(line([0, offsets[0]], [3, offsets[0]]),
                                            line([.5, offsets[1]], [2, offsets[1]]))["pass"]


def test_narrow_second_surface_between_grid_points_is_continuously_detected():
    # The second sheet deliberately falls between the 0.1m sampled columns.
    raw = np.concatenate([wall(0, 3, 0)] + [wall(x + .04, x + .06, .02)
                                          for x in np.arange(0, 3, .1)])
    evidence = RawWallEvidence(raw, 0, [-1, -1], 5, np.zeros((100, 100), bool))
    checked = evidence.same_surface(line([0, 0], [3, 0]), line([0, .009], [3, .009]))
    assert not checked["pass"] and checked["reason"] == "multiple_raw_surface_sheets"


def test_submillimetre_tessellation_does_not_hide_a_separate_continuous_plane():
    raw = np.concatenate([wall(0, .3, 0)] + [wall(x, x + .00075, .02)
                                           for x in np.arange(0, .3, .00075)])
    evidence = RawWallEvidence(raw, 0, [-1, -1], 5, np.zeros((100, 100), bool))
    checked = evidence.same_surface(line([0, 0], [.3, 0]), line([0, .009], [.3, .009]))
    assert not checked["pass"] and checked["reason"] == "multiple_raw_surface_sheets"


def test_one_height_or_unobserved_gap_does_not_establish_same_surface():
    raw = wall(0, 3, 0, height=.4)
    evidence = RawWallEvidence(raw, 0, [-1, -1], 5, np.zeros((100, 100), bool))
    assert not evidence.same_surface(line([0, 0], [3, 0]), line([.5, .01], [2, .01]))["pass"]
    assert not evidence.same_surface(line([0, 0], [1, 0]), line([2, .01], [3, .01]))["pass"]
