"""Recovery requires raw wall-plane proof and cannot seal real passages."""

from copy import deepcopy

import numpy as np
import pytest

from src.data.map_wall_recovery import raw_gap_support, recover_collinear_gaps


def wall(identifier, start, end):
    return {"id": identifier, "type": "line", "start": start, "end": end,
            "endpoints": [start, end], "length_m": float(np.linalg.norm(np.array(end) - start))}


def neighbours(offset=0.):
    return [wall("left", [0, 0], [1, 0]), wall("right", [2, offset], [3, offset])]


def wall_cuts(*, start=1., end=2., offset=0., heights=(.3, .6, 1.8, 2.1)):
    return {height: np.array([[[start, offset], [end, offset]]]) for height in heights}


def clear_navmesh(segment):
    return {"pass": True, "source": "fixture walkable raster", "segment": segment}


def recover(items=None, *, slices=None, **kwargs):
    slices = wall_cuts() if slices is None else slices
    return recover_collinear_gaps(neighbours() if items is None else items,
                                 raw_support=lambda segment: raw_gap_support(segment, slices),
                                 not_blocks_navmesh=clear_navmesh, **kwargs)


def test_measured_window_wall_plane_is_recovered_without_modifying_jambs():
    original = neighbours()
    before = deepcopy(original)
    result = recover(original, semantic_priors=[{"id": "window-7", "category": "window",
                                               "segment": [[1.1, 0], [1.9, 0]]}])
    assert original == before
    assert result["primitives"][:2] == before
    assert len(result["recovered"]) == 1
    new = result["primitives"][2]
    assert new["type"] == "line"
    assert new["endpoints"] == [[1., 0.], [2., 0.]]
    assert new["length_m"] == pytest.approx(1)
    assert new["recovery"]["semantic_prior_ids"] == ["window-7"]
    evidence = new["recovery"]["evidence"]
    assert evidence["semantic_priors_are_geometric_proof"] is False
    assert evidence["raw_support"]["supported_fraction"] == 1
    assert evidence["navmesh"]["pass"] is True


def test_true_portal_is_protected_even_when_high_lintel_has_full_support():
    result = recover(slices=wall_cuts(heights=(2.1, 2.4)),
                     portals=[{"id": "doorway", "status": "accepted", "segment": [[1, 0], [2, 0]]}])
    assert result["primitives"] == neighbours()
    assert result["rejected"][0]["reason"] == "protected_door_or_portal"
    assert "raw_support" not in result["rejected"][0]["evidence"]


def test_semantic_closed_door_is_not_relabelled_as_a_wall():
    result = recover(semantic_priors=[{"id": "closed-door", "category": "door",
                                      "segment": [[1.1, .01], [1.9, .01]]}])
    assert not result["recovered"]
    assert result["rejected"][0]["reason"] == "protected_door_or_portal"


def test_clear_door_without_prior_cannot_be_closed_from_lintel_alone():
    result = recover(slices=wall_cuts(heights=(2., 2.3, 2.4)))
    assert not result["recovered"]
    assert result["rejected"][0]["reason"] == "insufficient_aligned_multilevel_raw_support"


def test_window_semantics_cannot_fill_an_unsupported_scan_hole():
    result = recover(slices={.3: [], 1.8: []},
                     semantic_priors=[{"id": "window", "category": "window", "segment": [[1, 0], [2, 0]]}])
    assert not result["recovered"]
    evidence = result["rejected"][0]["evidence"]["raw_support"]
    assert evidence["supported_fraction"] == 0
    assert not evidence["pass"]


def test_near_parallel_offset_wall_is_not_connected_even_with_surface_evidence():
    result = recover(neighbours(.04))
    assert not result["recovered"]
    assert result["rejected"][0]["reason"] == "adjacent_walls_not_on_same_plane"


def test_parallel_furniture_surface_and_transverse_hits_do_not_support_the_gap():
    offset = raw_gap_support([[1, 0], [2, 0]], wall_cuts(offset=.04))
    transverse = raw_gap_support([[1, 0], [2, 0]],
                                {height: [[[1.5, -.1], [1.5, .1]]] for height in (.3, 1.8)})
    assert not offset["pass"]
    assert not transverse["pass"]
    assert offset["supported_fraction"] == transverse["supported_fraction"] == 0


def test_low_furniture_and_close_height_duplicates_are_insufficient():
    assert not raw_gap_support([[1, 0], [2, 0]], wall_cuts(heights=(.3, .6, .9)))["pass"]
    assert not raw_gap_support([[1, 0], [2, 0]], wall_cuts(heights=(1.5, 1.6)))["pass"]
    assert raw_gap_support([[1, 0], [2, 0]], wall_cuts(heights=(.9, 1.7)))["pass"]
    with pytest.raises(ValueError, match="unique actual heights"):
        raw_gap_support([[1, 0], [2, 0]], {.3: [[[1, 0], [2, 0]]], "0.3": [[[1, 0], [2, 0]]]})


def test_exact_interval_coverage_cannot_hide_an_eleven_percent_scan_gap():
    result = raw_gap_support([[1, 0], [2, 0]], wall_cuts(end=1.89))
    assert result["supported_fraction"] == pytest.approx(.89)
    assert not result["pass"]
    ninety = raw_gap_support([[1, 0], [2, 0]], wall_cuts(end=1.9))
    assert ninety["pass"]


def test_two_disjoint_height_supports_do_not_add_to_a_fake_complete_wall():
    evidence = raw_gap_support([[1, 0], [2, 0]],
                               {.3: [[[1, 0], [1.55, 0]]], 1.8: [[[1.45, 0], [2, 0]]]})
    assert evidence["supported_fraction"] == pytest.approx(.1)
    assert not evidence["pass"]


def test_navmesh_veto_wins_despite_complete_raw_support():
    result = recover_collinear_gaps(neighbours(),
                raw_support=lambda segment: raw_gap_support(segment, wall_cuts()),
                not_blocks_navmesh=lambda segment: {"pass": False, "intersects_walkable_interior": True})
    assert not result["recovered"]
    assert result["rejected"][0]["reason"] == "would_block_navmesh"


def test_arc_and_short_frame_survive_and_existing_wall_is_not_bridged_twice():
    items = neighbours() + [wall("frame", [1.4, 0], [1.45, 0]),
                            {"id": "curve", "type": "arc", "center": [9, 9], "radius_m": 2,
                             "start_angle_degrees": 0, "end_angle_degrees": 90, "sweep_degrees": 90}]
    result = recover(items, slices=wall_cuts(start=0, end=3))
    assert result["primitives"][:len(items)] == items
    assert any(item["reason"] == "existing_collinear_wall_inside_gap" for item in result["rejected"])
    assert all(item["adjacent_primitive_ids"] != ["left", "right"] for item in result["recovered"])


def test_real_large_gap_and_loose_callback_are_not_accepted():
    too_large = [wall("left", [0, 0], [1, 0]), wall("right", [3.1, 0], [4, 0])]
    assert not recover(too_large)["recovered"]
    result = recover_collinear_gaps(neighbours(),
                raw_support=lambda segment: {"pass": True, "supported_fraction": 1,
                                             "height_support_fractions": {".3": 1, "1.8": 1}},
                not_blocks_navmesh=clear_navmesh)
    assert not result["recovered"]
    with pytest.raises(ValueError, match="explicit boolean"):
        recover_collinear_gaps(neighbours(), raw_support=lambda segment: {}, not_blocks_navmesh=clear_navmesh)


def test_oblique_wall_has_no_manhattan_assumption():
    angle = np.radians(37)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    items = [wall(item["id"], (rotation @ item["start"]).tolist(), (rotation @ item["end"]).tolist())
             for item in neighbours()]
    slices = {height: values @ rotation.T for height, values in wall_cuts().items()}
    result = recover(items, slices=slices)
    assert len(result["recovered"]) == 1
    np.testing.assert_allclose(result["primitives"][-1]["endpoints"], np.array([[1, 0], [2, 0]]) @ rotation.T)
