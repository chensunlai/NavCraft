"""Independent surface evidence must precede architectural vector rendering."""
import numpy as np

from src.data.map_floorplan import RawWallEvidence, primitive_points, split_primitive, verify_primitive


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
