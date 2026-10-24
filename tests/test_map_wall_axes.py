from copy import deepcopy

import numpy as np
import pytest

from src.data.map_wall_axes import merge_wall_faces


def line(name, z, start=0., end=3.):
    return {"id": name, "type": "line", "start": [start, z], "end": [end, z],
            "source_segment_indices": [1 if name.startswith("low") else 2]}


def face(z, sign, start=0., end=3., bottom=0., top=2.4):
    vertices = np.array([[start, bottom, z], [end, bottom, z], [end, top, z], [start, top, z]])
    triangles = vertices[[[0, 1, 2], [0, 2, 3]]]
    return triangles if sign > 0 else triangles[:, ::-1]


def support(_):
    return {"pass": True, "source": "measured fixture NavMesh"}


def cap(x, sign, span=.2):
    vertices = np.array([[x, 0, 0], [x, 0, span], [x, 2.4, span], [x, 2.4, 0]])
    triangles = vertices[[[0, 1, 2], [0, 2, 3]]]
    return triangles if sign < 0 else triangles[:, ::-1]


def merge(items, raw, **kwargs):
    return merge_wall_faces(items, raw, not_blocks_navmesh=support, **kwargs)


def test_outward_wall_faces_become_one_axis_with_single_line_rendering():
    original = [line("low", 0), line("high", .2)]
    before = deepcopy(original)
    result = merge(original, np.concatenate([face(0, -1), face(.2, 1)]))
    assert len(result["primitives"]) == len(result["merged"]) == 1
    axis = result["primitives"][0]
    assert axis["start"] == pytest.approx([0, .1]) and axis["end"] == pytest.approx([3, .1])
    assert axis["thickness_m"] is None
    assert axis["wall_thickness_m"] == pytest.approx(.2)
    assert axis["source_primitive_ids"] == ["low", "high"]
    assert axis["source_segment_indices"] == [1, 2]
    assert all(p["raw_evidence"]["low_face"]["outward_area_fraction"] == 1
               for p in result["merged"][0]["raw_pairs"])
    assert original == before


def test_narrow_corridor_inward_faces_cannot_be_merged_even_if_navmesh_has_no_cells():
    result = merge([line("low", 0), line("high", .3)], np.concatenate([face(0, 1), face(.3, -1)]))
    assert not result["merged"] and len(result["primitives"]) == 2
    assert result["pending"][0]["reason"] == "raw_area_or_outward_normals_not_confirmed"


def test_parallel_pair_near_vertical_direction_sign_boundary_is_still_recognised():
    items = [{"id": "low", "type": "line", "start": [0, 0], "end": [.01, 3]},
             {"id": "high", "type": "line", "start": [.2, 0], "end": [.19, 3]}]
    result = merge(items, np.concatenate([cap(0, -1, span=3), cap(.2, 1, span=3)]))
    assert len(result["merged"]) == 1
    assert np.allclose(np.array(result["primitives"][0]["endpoints"])[:, 0], .1)


def test_bed_cabinet_gap_and_low_furniture_are_not_walls():
    # Even accidentally supplied architectural lines cannot overcome inward
    # surface winding or insufficient common vertical support.
    for raw in [np.concatenate([face(0, 1), face(.2, -1)]),
                np.concatenate([face(0, -1, top=.5), face(.2, 1, top=.6)])]:
        result = merge([line("low", 0), line("high", .2)], raw)
        assert not result["merged"]


def test_raw_area_is_required_not_only_face_normal_or_bbox():
    raw = np.concatenate([face(0, -1, end=.15), face(.2, 1, end=.15)])
    result = merge([line("low", 0), line("high", .2)], raw)
    assert not result["merged"]


def test_navmesh_context_is_required_and_known_walkable_strip_vetoes_axis():
    items = [line("low", 0), line("high", .2)]
    raw = np.concatenate([face(0, -1), face(.2, 1)])
    missing = merge_wall_faces(items, raw)
    assert missing["pending"][0]["reason"] == "navmesh_context_required"
    blocked = merge_wall_faces(items, raw, not_blocks_navmesh=lambda _: {"pass": False})
    assert not blocked["merged"]
    assert blocked["pending"][0]["reason"] == "axis_would_occupy_measured_navmesh"


def test_real_door_gap_survives_as_two_axis_segments():
    items = [line("low-left", 0, 0, 1), line("high-left", .2, 0, 1),
             line("low-right", 0, 2, 3), line("high-right", .2, 2, 3)]
    raw = np.concatenate([face(z, sign, a, b) for z, sign in [(0, -1), (.2, 1)]
                          for a, b in [(0, 1), (2, 3)]])
    result = merge(items, raw, portals=[{"segment": [[1, .1], [2, .1]], "status": "accepted"}])
    assert len(result["primitives"]) == 2
    assert [(p["start"][0], p["end"][0]) for p in result["primitives"]] == [(0, 1), (2, 3)]


@pytest.mark.parametrize("offset", [0., .1, .2])
def test_known_portal_in_a_continuous_input_line_is_not_bridged(offset):
    raw = np.concatenate([face(0, -1), face(.2, 1)])
    result = merge([line("low", 0), line("high", .2)], raw,
                   portals=[{"segment": [[1, offset], [2, offset]], "status": "accepted"}])
    assert not result["merged"]
    assert result["pending"][0]["reason"] == "known_portal_would_be_occupied"


def test_single_face_tails_are_one_bounded_axis_with_local_thickness_unknown():
    items = [line("low", 0, 0, 4), line("high", .2, 1, 3)]
    raw = np.concatenate([face(0, -1, 0, 4), face(.2, 1, 1, 3)])
    result = merge(items, raw)
    assert len(result["primitives"]) == 1
    axis = result["primitives"][0]
    assert axis["start"] == [0, .1] and axis["end"] == [4, .1]
    assert axis["axis_evidence"]["paired_intervals_m"] == [[1., 3.]]
    assert axis["axis_evidence"]["single_face_intervals_m"] == [[0., 1.], [3., 4.]]


def test_fragmented_opposite_face_does_not_leave_a_parallel_tail():
    items = [line("low", 0, 0, 4), line("high-left", .2, 0, 1.5), line("high-right", .2, 2.5, 4)]
    raw = np.concatenate([face(0, -1, 0, 4), face(.2, 1, 0, 1.5), face(.2, 1, 2.5, 4)])
    result = merge(items, raw)
    assert len(result["primitives"]) == 1
    assert result["primitives"][0]["source_primitive_ids"] == [p["id"] for p in items]
    assert result["primitives"][0]["axis_evidence"]["single_face_intervals_m"] == [[1.5, 2.5]]


def test_measured_thickness_endcaps_become_axis_evidence_not_t_branches():
    items = [line("low", 0), line("high", .2),
             {"id": "left-cap", "type": "line", "start": [0, 0], "end": [0, .2]},
             {"id": "right-cap", "type": "line", "start": [3, 0], "end": [3, .2]}]
    raw = np.concatenate([face(0, -1), face(.2, 1), cap(0, -1), cap(3, 1)])
    result = merge(items, raw)
    assert len(result["primitives"]) == 1
    assert result["merged"][0]["cap_primitive_ids"] == ["left-cap", "right-cap"]
    assert set(result["primitives"][0]["source_primitive_ids"]) == {p["id"] for p in items}


def test_covered_cap_shape_is_not_claimed_as_measured_endcap_and_real_short_wall_is_kept():
    inside = {"id": "covered-edge", "type": "line", "start": [0, 0], "end": [0, .2]}
    outside = {"id": "real-short-wall", "type": "line", "start": [0, -.2], "end": [0, 0]}
    result = merge([line("low", 0), line("high", .2), inside, outside],
                   np.concatenate([face(0, -1), face(.2, 1)]))
    assert inside["id"] not in {p["id"] for p in result["primitives"]}
    assert outside["id"] in {p["id"] for p in result["primitives"]}
    assert result["merged"][0]["covered_primitive_ids"] == [inside["id"]]
    assert not result["merged"][0]["cap_primitive_ids"]


def test_different_height_surfaces_do_not_form_one_wall_and_split_level_is_not_global_floor_clipped():
    items = [line("low", 0), line("high", .2)]
    result = merge(items, np.concatenate([face(0, -1, bottom=0, top=1), face(.2, 1, bottom=2, top=3)]))
    assert not result["merged"]
    lower = np.concatenate([face(0, -1, bottom=-.65, top=1.75), face(.2, 1, bottom=-.65, top=1.75)])
    result = merge(items, lower)
    assert len(result["merged"]) == 1
    assert result["merged"][0]["raw_pairs"][0]["raw_evidence"]["support_height_band_m"] == [-.65, 1.75]


def test_single_sided_wall_arc_and_circle_pass_through_without_invented_thickness():
    arc = {"id": "curve", "type": "arc", "center": [1, 1], "radius_m": 1, "sweep_degrees": 90}
    circle = {"id": "circle", "type": "circle", "center": [2, 2], "radius_m": .4}
    result = merge([line("single", 0), arc, circle], face(0, -1))
    assert result["primitives"][0]["thickness_m"] is None
    assert result["primitives"][0]["axis_status"] == "unpaired_face_thickness_unmeasured"
    assert result["primitives"][1:] == [arc, circle]


def sparse_faces(end=3., top=2.4):
    # A real triangular patch on each side covers the full along/height extent,
    # but only half of its rectangular face. This is not missing raw evidence.
    return np.concatenate([face(0, -1, end=end, top=top)[:1],
                           face(.2, 1, end=end, top=top)[:1]])


def test_area_only_failure_can_be_reviewed_without_becoming_a_geometry_pass():
    items = [line("low", 0), line("high", .2)]
    assert not merge(items, sparse_faces())["merged"]
    calls = []

    def review(record):
        calls.append(record)
        assert record["source_primitive_ids"] == ["low", "high"]
        assert record["strict_geometry_pass"] is False
        assert record["raw_evidence"]["low_face"]["area_fraction"] == pytest.approx(.5)
        # A reviewer receives a snapshot, not mutable internal geometry proof.
        record["raw_evidence"]["low_face"]["outward_area_fraction"] = 0
        return {"pass": True, "evidence": {"source": "fixture local image review", "decision": "same_wall"}}

    result = merge(items, sparse_faces(), review_pair=review)
    assert len(calls) == len(result["merged"]) == len(result["primitives"]) == 1
    merged = result["merged"][0]
    assert merged["acceptance"] == "structural_inference"
    assert merged["strict_geometry_pass"] is False
    pair = merged["raw_pairs"][0]
    assert pair["raw_evidence"]["pass"] is False
    assert pair["raw_evidence"]["reason"] == "raw_area_or_outward_normals_not_confirmed"
    assert pair["raw_evidence"]["low_face"]["outward_area_fraction"] == 1
    assert pair["same_wall_review"]["evidence"]["source"] == "fixture local image review"
    axis = result["primitives"][0]
    assert axis["evidence_kind"] == "reviewed_same_wall_structural_inference"
    assert axis["axis_evidence"]["strict_geometry_pass"] is False


def test_strict_geometry_pass_does_not_need_a_review():
    def unexpected(_):
        pytest.fail("Strict geometry acceptance must not ask a reviewer")

    result = merge([line("low", 0), line("high", .2)],
                   np.concatenate([face(0, -1), face(.2, 1)]), review_pair=unexpected)
    assert result["merged"][0]["acceptance"] == "raw_geometry"


@pytest.mark.parametrize("case", ["inward_normals", "low_coverage", "low_height", "tiny_area", "absent_raw"])
def test_review_cannot_bypass_missing_or_contradictory_geometry(case):
    def unexpected(_):
        pytest.fail("Review must not bypass raw normal/coverage/height/area requirements")

    items = [line("low", 0), line("high", .2)]
    raw = {"inward_normals": np.concatenate([face(0, 1)[:1], face(.2, -1)[:1]]),
           "low_coverage": np.concatenate([face(0, -1, end=1.5), face(.2, 1, end=1.5)]),
           "low_height": sparse_faces(top=.6),
           "tiny_area": sparse_faces(end=.4, top=.8),
           "absent_raw": np.empty((0, 3, 3))}[case]
    if case == "tiny_area":
        items = [line("low", 0, end=.4), line("high", .2, end=.4)]
    result = merge(items, raw, review_pair=unexpected)
    assert not result["merged"] and len(result["primitives"]) == 2


def test_declined_review_preserves_original_failure_and_evidence():
    result = merge([line("low", 0), line("high", .2)], sparse_faces(),
                   review_pair=lambda _: {"pass": False, "evidence": {"source": "fixture", "decision": "unknown"}})
    assert not result["merged"]
    pending = result["pending"][0]
    assert pending["raw_evidence"]["pass"] is False
    assert pending["same_wall_review"]["evidence"]["decision"] == "unknown"


@pytest.mark.parametrize("veto", ["navmesh", "portal", "missing_navmesh"])
def test_review_confirmation_cannot_bypass_navmesh_or_known_portal(veto):
    kwargs = {"review_pair": lambda _: {"pass": True, "evidence": {"source": "fixture"}}}
    if veto == "portal":
        kwargs["portals"] = [{"segment": [[1, .1], [2, .1]], "status": "accepted"}]
    result = merge_wall_faces([line("low", 0), line("high", .2)], sparse_faces(),
                             not_blocks_navmesh=(None if veto == "missing_navmesh" else
                                                 lambda _: {"pass": veto != "navmesh"}), **kwargs)
    assert not result["merged"] and len(result["primitives"]) == 2
    assert result["pending"][0]["acceptance"] == "structural_inference"
    assert result["pending"][0]["strict_geometry_pass"] is False
    assert result["pending"][0]["reason"] in {"axis_would_occupy_measured_navmesh",
                                            "known_portal_would_be_occupied", "navmesh_context_required"}


@pytest.mark.parametrize("reply", [{"pass": 1, "evidence": "fixture"}, {"pass": True}, True])
def test_review_requires_explicit_boolean_and_traceable_evidence(reply):
    with pytest.raises(ValueError, match="review_pair"):
        merge([line("low", 0), line("high", .2)], sparse_faces(), review_pair=lambda _: reply)


def test_wholly_covered_fragment_is_consumed_with_sources_without_extending_axis():
    inside = {"id": "scan-fragment", "type": "line", "start": [.5, .22], "end": [2.5, .14],
              "source_segment_indices": [71, 72]}
    result = merge([line("low", 0), line("high", .2), inside],
                   np.concatenate([face(0, -1), face(.2, 1)]))
    assert len(result["primitives"]) == 1
    axis = result["primitives"][0]
    assert axis["start"] == pytest.approx([0, .1]) and axis["end"] == pytest.approx([3, .1])
    assert axis["thickness_m"] is None
    assert axis["source_primitive_ids"] == ["low", "high", "scan-fragment"]
    assert axis["source_segment_indices"] == [1, 2, 71, 72]
    proof = axis["axis_evidence"]["covered_fragment_evidence"][0]
    assert proof["axis_extended"] is False
    assert proof["source_role"] == "covered_wall_volume_scan_edge"
    assert proof["normal_fit_tolerance_m"] <= .025


@pytest.mark.parametrize("extra", [line("outside-slab", .23, .5, 2.5),
                                  {"id": "outside-extent", "type": "line",
                                   "start": [.5, .22], "end": [3.01, .14]},
                                  {"id": "short-partition", "type": "line",
                                   "start": [1, .1], "end": [1, .4]}])
def test_fragment_outside_physical_slab_or_finite_extent_is_not_deleted(extra):
    result = merge([line("low", 0), line("high", .2), extra],
                   np.concatenate([face(0, -1), face(.2, 1)]))
    assert extra["id"] in {p["id"] for p in result["primitives"]}
    assert not result["merged"][0]["covered_primitive_ids"]


def test_line_in_real_axis_gap_is_not_consumed_by_adjacent_wall_groups():
    items = [line("low-left", 0, 0, 1), line("high-left", .2, 0, 1),
             line("low-right", 0, 2, 3), line("high-right", .2, 2, 3),
             line("in-opening", .1, 1.2, 1.8), line("across-opening", .1, .5, 2.5)]
    raw = np.concatenate([face(z, sign, a, b) for z, sign in [(0, -1), (.2, 1)]
                          for a, b in [(0, 1), (2, 3)]])
    result = merge(items, raw, portals=[{"segment": [[1, .1], [2, .1]], "status": "accepted"}])
    ids = {p["id"] for p in result["primitives"]}
    assert {"in-opening", "across-opening"} <= ids
    assert not any(x["covered_primitive_ids"] for x in result["merged"])


def test_supported_independent_wall_pair_is_not_swallowed_by_an_overlapping_slab():
    # Even physically inconsistent overlapping input wall groups need their own
    # conflict review; coverage must not silently discard the smaller pair.
    items = [line("low-outer", 0), line("high-outer", .4),
             {"id": "inner-left", "type": "line", "start": [1, .04], "end": [1, .36]},
             {"id": "inner-right", "type": "line", "start": [1.2, .04], "end": [1.2, .36]}]
    inner = np.concatenate([cap(1, -1, span=.32), cap(1.2, 1, span=.32)])
    inner[:, :, 2] += .04
    raw = np.concatenate([face(0, -1), face(.4, 1),
                          inner])
    result = merge(items, raw)
    assert len(result["merged"]) == len(result["primitives"]) == 2
    assert not any(x["covered_primitive_ids"] for x in result["merged"])
    assert {"inner-left", "inner-right"} <= {s for p in result["primitives"]
                                            for s in p.get("source_primitive_ids", [p["id"]])}


def test_actual_dnw_fragment_exceeds_finite_axis_and_is_kept_without_hidden_tolerance():
    # Saved DNW CAD0001/CAD0003/CAD0031 coordinates, not a synthetic case tuned
    # to delete the real fragment. Its normal extent fits, but 4.11cm is outside
    # the finite axis, so there is no full-coverage justification for deletion.
    from src.data.map_wall_axes import _covered_fragments

    tangent = np.array([.0063473418071669, -4.516577304495379])
    tangent /= np.linalg.norm(tangent)
    if tangent[0] < 0:
        tangent = -tangent
    normal = np.array([-tangent[1], tangent[0]])
    axis = np.array([[8.631156974322094, 2.4586868021427506],
                     [8.63750431612926, -2.0578905023526284]])
    offset = float((axis @ normal).mean())
    thickness = .13692274687427286
    fragment = np.array([[8.709549156943426, -1.1208331140614216],
                         [8.728216295361197, -2.099035550043998]])
    intervals = [[float((axis @ tangent).min()), float((axis @ tangent).max())]]
    assert (fragment @ tangent).max() - intervals[0][1] == pytest.approx(.04127249, abs=1e-8)
    assert _covered_fragments({0: (fragment, tangent, .97838)}, set(), tangent, normal,
                              offset - thickness / 2, offset + thickness / 2, intervals, []) == []
