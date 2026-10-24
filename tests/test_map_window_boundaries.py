"""Window frame geometry is distinct from a measured solid vertical wall."""

from copy import deepcopy

import numpy as np

from src.data.map_instances import SemanticInstance
from src.data.map_window_boundaries import build_window_boundaries


def quad(points):
    points = np.asarray(points, dtype=float)
    return points[[[0, 1, 2], [0, 2, 3]]]


def fixture(*, sill=True, header=True, jamb_height=1.6, incomplete_bottom=False):
    # World Y is up; the actual frame lies 12–20 cm behind the wall ends.
    pieces = []
    if sill:
        pieces.append(quad([[1, .6, .12], [2 if not incomplete_bottom else 1.75, .6, .12],
                            [2 if not incomplete_bottom else 1.75, .6, .2], [1, .6, .2]]))
    if header:
        pieces.append(quad([[1, 2.2, .12], [2, 2.2, .12], [2, 2.2, .2], [1, 2.2, .2]]))
    for x in (1, 2):
        pieces.append(quad([[x, .6, 0], [x, .6 + jamb_height, 0],
                            [x, .6 + jamb_height, .2], [x, .6, .2]]))
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
                                   passage=kwargs.get("passage", blocked))


def test_measured_inset_frame_keeps_originals_and_adds_recess_connections():
    values = fixture()
    before = deepcopy(values[1])
    result = build(values)
    assert values[1] == before
    assert result["primitives"][:2] == before
    assert len(result["accepted"]) == 1
    candidate = result["accepted"][0]
    assert candidate["solid_wall_measured"] is False
    assert candidate["source_kind"] == "frame_supported_window_boundary"
    assert len(candidate["proposed_primitives"]) == 3
    boundary = candidate["proposed_primitives"][0]
    assert .119 < boundary["start"][1] < .201
    assert boundary["start"][1] == boundary["end"][1]
    assert candidate["evidence"]["best_frame_plane"]["joint_coverage"] > .99
    assert all(jamb["continuous_height_span_m"] >= 1.4 for jamb in candidate["evidence"]["jambs"])
    assert candidate["source_raw_face_ids"]
    assert boundary["evidence"]["sill"]["source_raw_face_ids"]
    assert boundary["evidence"]["header"]["source_raw_face_ids"]


def test_no_probe_is_unknown_and_preserves_candidate_geometry_for_rgb():
    result = build(passage=lambda segment: {"pass": False, "probes": [], "reason": "no_outer_navmesh_point"})
    assert result["primitives"] == fixture()[1]
    assert not result["accepted"]
    candidate = result["candidates"][0]
    assert candidate["status"] == "needs_review"
    assert candidate["interpretation"] == "inferred_window_boundary_pending_visual_confirmation"
    assert candidate["proposed_primitives"]


def test_actual_clear_crossing_rejects_window_boundary_despite_label_and_frame():
    result = build(passage=lambda segment: {"pass": True, "probes": [{"pass": True}]})
    assert result["primitives"] == fixture()[1]
    assert result["rejected"][0]["reason"] == "measured_direct_passage_exists"


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


def test_projection_coverage_is_not_confused_with_shared_frame_plane():
    raw, walls, windows = fixture()
    # Sill spans the full width, but most lies deeper than the header plane.
    raw[:2] = quad([[1, .6, .25], [2, .6, .25], [2, .6, .3], [1, .6, .3]])
    result = build((raw, walls, windows))
    assert not result["accepted"]
    assert result["candidates"][0]["reason"] == "no_common_recess_frame_depth"


def test_partial_shared_plane_keeps_inferred_line_and_explicit_missing_spans():
    raw, walls, windows = fixture()
    sill = np.concatenate([quad([[1, .6, .12], [1.25, .6, .12], [1.25, .6, .2], [1, .6, .2]]),
                           quad([[1.25, .6, .25], [2, .6, .25], [2, .6, .3], [1.25, .6, .3]])])
    result = build((np.concatenate([sill, raw[2:]]), walls, windows))
    assert not result["accepted"]
    candidate = result["candidates"][0]
    assert candidate["status"] == "needs_review"
    assert candidate["proposed_primitives"]
    assert candidate["evidence"]["best_frame_plane"]["sill_coverage"] < .9
    assert candidate["uncertain_boundary_spans_m"]


def test_arcs_are_preserved_and_semantics_do_not_produce_a_window_without_gap():
    raw, walls, windows = fixture()
    arc = {"id": "C", "type": "arc", "center": [9, 9], "radius_m": 2,
           "start_angle_degrees": 0, "end_angle_degrees": 90}
    result = build((raw, walls + [arc], windows))
    assert result["primitives"][2] == arc
    result = build((raw, walls, []))
    assert result["candidates"] == []
    assert result["primitives"] == walls
