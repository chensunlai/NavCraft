"""Window frame geometry is distinct from a measured solid vertical wall."""

from copy import deepcopy

import numpy as np

from src.data.map_instances import SemanticInstance
from src.data.map_window_boundaries import build_window_boundaries


def quad(points):
    points = np.asarray(points, dtype=float)
    return points[[[0, 1, 2], [0, 2, 3]]]


def fixture(*, sill=True, header=True, jamb_height=1.6, jamb_depth=.2, incomplete_bottom=False):
    # World Y is up; the actual frame lies 12–20 cm behind the wall ends.
    pieces = []
    if sill:
        pieces.append(quad([[1, .6, .12], [2 if not incomplete_bottom else 1.75, .6, .12],
                            [2 if not incomplete_bottom else 1.75, .6, .2], [1, .6, .2]]))
    if header:
        pieces.append(quad([[1, 2.2, .12], [2, 2.2, .12], [2, 2.2, .2], [1, 2.2, .2]]))
    for x in (1, 2):
        pieces.append(quad([[x, .6, 0], [x, .6 + jamb_height, 0],
                            [x, .6 + jamb_height, jamb_depth], [x, .6, jamb_depth]]))
    raw = np.concatenate(pieces)
    semantic = quad([[1, .6, .2], [2, .6, .2], [2, 2.2, .2], [1, 2.2, .2]])
    window = SemanticInstance(7, "window", "window", 1, semantic,
                              np.array([[0, 0], [0, 1]]), {"mesh": "fixture.semantic.glb"})
    walls = [{"id": "A", "type": "line", "start": [0, 0], "end": [1, 0]},
             {"id": "B", "type": "line", "start": [2, 0], "end": [3, 0]}]
    return raw, walls, [window]


def blocked(segment):
    return {"pass": False, "probes": [{"pass": False, "start": [1.5, 0, -.4],
                                      "requested_end": [1.5, 0, .6], "actual_end": [1.5, 0, -.2]}]}


def build(values=None, **kwargs):
    return build_window_boundaries(*(fixture() if values is None else values), floor_height=0,
                                   passage=kwargs.get("passage", blocked),
                                   not_blocks_navmesh=kwargs.get("not_blocks_navmesh"))


def test_measured_frame_reconstructs_adjacent_wall_plane_as_explicit_inference():
    values = fixture()
    before = deepcopy(values[1])
    result = build(values)
    assert values[1] == before
    assert result["primitives"][:2] == before
    assert len(result["accepted"]) == 1
    candidate = result["accepted"][0]
    assert candidate["solid_wall_measured"] is False
    assert candidate["source_kind"] == "reconstructed_window_wall"
    assert candidate["evidence_kind"] == "structural_inference"
    assert candidate["glass_open_or_closed"] == "not_determined"
    assert len(candidate["proposed_primitives"]) == 1
    boundary = candidate["proposed_primitives"][0]
    assert boundary["start"] == [1, 0]
    assert boundary["end"] == [2, 0]
    assert candidate["evidence"]["best_frame_plane"]["joint_coverage"] > .99
    assert all(jamb["continuous_height_span_m"] >= 1.4 for jamb in candidate["evidence"]["jambs"])
    assert candidate["source_raw_face_ids"]
    assert boundary["evidence"]["sill"]["source_raw_face_ids"]
    assert boundary["evidence"]["header"]["source_raw_face_ids"]


def test_no_robot_probe_does_not_veto_closed_architectural_frame_inference():
    result = build(passage=lambda segment: {"pass": False, "probes": [], "reason": "no_outer_navmesh_point"})
    assert len(result["primitives"]) == 3
    assert len(result["accepted"]) == 1
    candidate = result["candidates"][0]
    assert candidate["status"] == "accepted"
    assert candidate["evidence"]["robot_crossing_status"] == "not_established"
    assert candidate["evidence"]["walkable_space_check"]["pass"] is None
    assert candidate["proposed_primitives"]


def test_actual_clear_crossing_rejects_window_boundary_despite_label_and_frame():
    result = build(passage=lambda segment: {"pass": True, "probes": [{"pass": True}]})
    assert result["primitives"] == fixture()[1]
    assert result["rejected"][0]["reason"] == "measured_direct_passage_exists"


def test_measured_walkable_space_overlap_vetoes_reconstruction_without_probes():
    result = build(passage=lambda segment: {"pass": False, "probes": []},
                   not_blocks_navmesh=lambda segment: {"pass": False, "intersects_walkable_interior": True})
    assert not result["accepted"]
    assert result["primitives"] == fixture()[1]
    assert result["rejected"][0]["reason"] == "would_block_measured_walkable_space"


def test_missing_sill_and_low_jamb_cannot_become_a_closed_window():
    result = build(fixture(sill=False))
    assert not result["accepted"]
    assert result["rejected"][0]["reason"] == "missing_full_width_raw_sill_or_header"
    result = build(fixture(jamb_height=.8))
    assert not result["accepted"]
    assert result["candidates"][0]["status"] == "needs_review"


def test_incomplete_sill_does_not_pass_from_semantic_window_alone():
    result = build(fixture(incomplete_bottom=True))
    assert not result["accepted"]
    assert result["rejected"][0]["reason"] == "missing_full_width_raw_sill_or_header"


def test_small_missing_sill_fragment_is_inferred_when_four_corners_close():
    raw, walls, windows = fixture()
    sill = np.concatenate([quad([[1, .6, .12], [1.46, .6, .12], [1.46, .6, .2], [1, .6, .2]]),
                           quad([[1.54, .6, .12], [2, .6, .12], [2, .6, .2], [1.54, .6, .2]])])
    result = build((np.concatenate([sill, raw[2:]]), walls, windows))
    assert len(result["accepted"]) == 1
    assert result["accepted"][0]["evidence"]["sill"]["projected_coverage"] < 1
    assert result["accepted"][0]["evidence_kind"] == "structural_inference"


def test_two_tall_sides_are_insufficient_when_they_do_not_meet_horizontal_ends():
    raw, walls, windows = fixture()
    raw[4:6, :, 0] += .12
    result = build((raw, walls, windows))
    assert not result["accepted"]
    rectangle = result["candidates"][0]["evidence"]["frame_rectangle"]
    assert all(check["pass"] for check in rectangle["vertical_checks"])
    assert not all(check["pass"] for check in rectangle["corner_checks"])


def test_adjacent_floor_window_cannot_fill_current_floor_wall_gap():
    raw, walls, windows = fixture()
    raw[:, :, 1] += 3.5
    semantic = windows[0].triangles.copy()
    semantic[:, :, 1] += 3.5
    old = windows[0]
    above = SemanticInstance(old.id, old.label, old.category, old.region_id, semantic, old.triangle_ids, old.source)
    result = build((raw, walls, [above]))
    assert not result["accepted"]
    assert result["candidates"][0]["reason"] == "requires_floor_or_window_type_confirmation"


def test_clerestory_is_retained_for_review_instead_of_being_called_no_window():
    raw, walls, windows = fixture()
    raw[:, :, 1] += 1.5
    old = windows[0]
    semantic = old.triangles.copy()
    semantic[:, :, 1] += 1.5
    high = SemanticInstance(old.id, old.label, old.category, old.region_id, semantic, old.triangle_ids, old.source)
    result = build((raw, walls, [high]))
    assert not result["accepted"]
    assert result["candidates"][0]["status"] == "needs_review"


def test_different_sill_header_recesses_still_define_a_connected_rectangle():
    raw, walls, windows = fixture(jamb_depth=.35)
    # Different depths are legitimate when both connect to the two side frames.
    raw[:2] = quad([[1, .6, .25], [2, .6, .25], [2, .6, .3], [1, .6, .3]])
    result = build((raw, walls, windows))
    assert len(result["accepted"]) == 1
    assert result["accepted"][0]["evidence"]["best_frame_plane"] is None
    assert result["accepted"][0]["evidence"]["frame_rectangle"]["pass"]


def test_depth_varying_sill_is_inferred_from_closed_frame_not_one_raw_plane():
    raw, walls, windows = fixture(jamb_depth=.35)
    sill = np.concatenate([quad([[1, .6, .12], [1.25, .6, .12], [1.25, .6, .2], [1, .6, .2]]),
                           quad([[1.25, .6, .25], [2, .6, .25], [2, .6, .3], [1.25, .6, .3]])])
    result = build((np.concatenate([sill, raw[2:]]), walls, windows))
    assert len(result["accepted"]) == 1
    candidate = result["candidates"][0]
    assert candidate["status"] == "accepted"
    assert candidate["proposed_primitives"]
    assert candidate["evidence"]["best_frame_plane"]["sill_coverage"] < .9
    assert candidate["inferred_boundary_spans_m"] == [[0., 1.]]


def test_arcs_are_preserved_and_semantics_do_not_produce_a_window_without_gap():
    raw, walls, windows = fixture()
    arc = {"id": "C", "type": "arc", "center": [9, 9], "radius_m": 2,
           "start_angle_degrees": 0, "end_angle_degrees": 90}
    result = build((raw, walls + [arc], windows))
    assert result["primitives"][2] == arc
    result = build((raw, walls, []))
    assert result["candidates"] == []
    assert result["primitives"] == walls


def test_already_reconstructed_window_is_not_duplicated_on_a_second_run():
    raw, walls, windows = fixture()
    first = build((raw, walls, windows))
    second = build((raw, first["primitives"], windows))
    assert not second["accepted"]
    assert second["primitives"] == first["primitives"]
