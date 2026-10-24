from copy import deepcopy

import numpy as np
import pytest

from src.data.map_wall_completion import complete_wall_gaps, wall_surface_evidence, _area


def wall(a, b, bottom=0., top=2.5):
    a, b = np.asarray(a), np.asarray(b)
    lower_a, lower_b = [a[0], bottom, a[1]], [b[0], bottom, b[1]]
    upper_a, upper_b = [a[0], top, a[1]], [b[0], top, b[1]]
    return np.asarray([[lower_a, lower_b, upper_b], [lower_a, upper_b, upper_a]], float)


def line(identifier, a, b, **extra):
    return {"id": identifier, "type": "line", "start": list(a), "end": list(b), **extra}


def allowed(segment):
    return {"pass": True, "source": "fixture_saved_navmesh", "checked_segment": segment}


def test_projected_polygon_union_counts_duplicate_and_overlapping_faces_once():
    triangle = np.asarray([[0., 0.], [2., 0.], [0., 2.]])
    square = np.asarray([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    assert _area([triangle, triangle], 0, 2, 0, 2) == pytest.approx(2.)
    assert _area([triangle, square], 0, 2, 0, 2) == pytest.approx(2.)
    # Different diagonal slopes cross inside the integration range.
    opposite = np.asarray([[2., 0.], [2., 2.], [0., 2.]])
    assert _area([triangle, opposite], 0, 2, 0, 2) == pytest.approx(4.)


def test_short_scanning_break_is_replaced_by_one_line_with_original_source_ids():
    original = [line("a", [0, 0], [1, 0], source_ids=["observed-a"], source_segment_indices=[0]),
                line("b", [1.2, 0], [2, 0], source_ids=["observed-b"], source_segment_indices=[1])]
    before = deepcopy(original)
    source = np.asarray([[[0, 0], [1, 0]], [[1.2, 0], [2, 0]]], float)
    result = complete_wall_gaps(original, wall([0, 0], [2, 0]), floor_height=0,
                               source_segments=source, not_blocks_navmesh=allowed)
    assert original == before
    assert len(result["accepted"]) == 1 and len(result["primitives"]) == 1
    value = result["primitives"][0]
    assert value["source_ids"] == ["observed-a", "observed-b"]
    assert value["source_segment_indices"] == [0, 1]
    assert value["length_m"] == pytest.approx(2)
    assert result["accepted"][0]["length_m"] == pytest.approx(.2)
    proof = result["accepted"][0]["raw_surface"]
    component = proof["components"][proof["selected_component_index"]]
    assert component["body_area_coverage"] == pytest.approx(1)
    assert component["anchor_area_coverage"] == pytest.approx([1, 1])
    assert set(component["raw_triangle_indices"]) == {0, 1}


def test_open_body_and_lintel_or_sill_cannot_be_closed_as_a_short_wall():
    fitted = [line("a", [0, 0], [1, 0]), line("b", [1.2, 0], [2, 0])]
    meshes = [np.concatenate([wall([0, 0], [1, 0]), wall([1.2, 0], [2, 0])]),
              np.concatenate([wall([0, 0], [2, 0], top=.6), wall([0, 0], [2, 0], bottom=1.7)])]
    for mesh in meshes:
        result = complete_wall_gaps(fitted, mesh, floor_height=0, not_blocks_navmesh=allowed)
        assert not result["accepted"]
        assert result["rejected"][0]["reason"] == "insufficient_connected_raw_wall_body"


def test_disconnected_parallel_sheets_cannot_combine_to_manufacture_a_wall_body():
    # Their projection fills the rectangle, but neither original mesh surface
    # connects the low and high body. A summed-area or nearest-height vote lies.
    raw = np.concatenate([wall([0, 0], [2, 0], top=1),
                          wall([0, .02], [2, .02], bottom=1)])
    checked = wall_surface_evidence([[.9, .01], [1.1, .01]], raw, floor_height=0)
    assert not checked["pass"]
    assert len(checked["components"]) == 2


def test_nearby_second_plane_and_tiny_between_sample_patch_are_explicitly_ambiguous():
    for second in (wall([0, .02], [2, .02]), wall([1.04, .02], [1.041, .02])):
        raw = np.concatenate([wall([0, 0], [2, 0]), second])
        checked = wall_surface_evidence([[1, 0], [1.2, 0]], raw, floor_height=0)
        assert not checked["pass"]
        assert checked["reason"] == "ambiguous_multiple_raw_surface_sheets"
        assert checked["sheet_conflicts"][0]["projected_overlap_m2"] > 0


def test_closed_door_geometry_and_verified_or_uncertain_portals_remain_open():
    fitted = [line("a", [0, 0], [1, 0]), line("b", [1.2, 0], [2, 0])]
    raw = wall([0, 0], [2, 0])
    for arguments in ({"semantic_priors": [{"id": "door", "category": "door", "segment": [[1, .02], [1.2, .02]]}]},
                      {"portals": [{"id": "opening", "status": "accepted", "segment": [[1.04, 0], [1.16, 0]]}]},
                      {"portals": [{"id": "opening", "status": "needs_review", "segment": [[1.04, 0], [1.16, 0]]}]}):
        result = complete_wall_gaps(fitted, raw, floor_height=0, not_blocks_navmesh=allowed, **arguments)
        assert not result["accepted"]
        assert any(item["reason"] == "protected_door_or_portal" for item in result["rejected"])


def test_real_navmesh_crossing_vetoes_an_otherwise_supported_completion():
    result = complete_wall_gaps([line("a", [0, 0], [1, 0]), line("b", [1.2, 0], [2, 0])],
                               wall([0, 0], [2, 0]), floor_height=0,
                               not_blocks_navmesh=lambda _: {"pass": False, "intersects_walkable_interior": True})
    assert not result["accepted"]
    assert result["rejected"][0]["reason"] == "would_block_navmesh"


def test_joint_fit_keeps_existing_corner_node_exact_and_limits_geometry_motion():
    fitted = [line("a", [0, 0], [1, 0]), line("b", [1.2, .01], [2, .01]),
              line("branch", [0, 0], [0, 1])]
    raw = np.concatenate([wall([0, 0], [2, 0]), wall([0, 0], [0, 1])])
    result = complete_wall_gaps(fitted, raw, floor_height=0, not_blocks_navmesh=allowed)
    assert len(result["accepted"]) == 1
    fitted_line = next(w for w in result["primitives"] if w["id"] != "branch")
    assert min(np.linalg.norm(np.asarray(fitted_line[at])) for at in ("start", "end")) < 1e-12
    assert result["accepted"][0]["joint_fit"]["max_original_line_displacement_m"] <= .03
    assert any(len(node["incidences"]) == 2 for node in result["nodes"])


def test_missing_corner_extends_the_existing_line_without_adding_overlapping_primitives():
    fitted = [line("a", [0, 0], [1, 0]), line("b", [1.2, 0], [1.2, 2])]
    raw = np.concatenate([wall([0, 0], [1.2, 0]), wall([1.2, 0], [1.2, 2])])
    result = complete_wall_gaps(fitted, raw, floor_height=0, not_blocks_navmesh=allowed)
    assert len(result["accepted"]) == 1
    assert result["accepted"][0]["kind"] == "missing_corner"
    assert len(result["primitives"]) == 2
    np.testing.assert_allclose(next(w for w in result["primitives"] if w["id"] == "a")["end"], [1.2, 0])
    assert any(node["point"] == [1.2, 0.] and node["kind"] == "junction" for node in result["nodes"])


def test_existing_full_coverage_is_not_added_again_and_inputs_are_immutable():
    fitted = [line("a", [0, 0], [1, 0]), line("covered", [1, 0], [1.2, 0]),
              line("b", [1.2, 0], [2, 0])]
    before = deepcopy(fitted)
    result = complete_wall_gaps(fitted, wall([0, 0], [2, 0]), floor_height=0, not_blocks_navmesh=allowed)
    assert not result["accepted"] and len(result["primitives"]) == 3
    assert fitted == before


def test_callback_protocol_and_source_index_errors_are_not_silent():
    fitted = [line("a", [0, 0], [1, 0]), line("b", [1.2, 0], [2, 0])]
    with pytest.raises(ValueError, match="explicit boolean"):
        complete_wall_gaps(fitted, wall([0, 0], [2, 0]), floor_height=0, not_blocks_navmesh=lambda _: True)
    fitted[0]["source_segment_indices"], fitted[1]["source_segment_indices"] = [0], [3]
    with pytest.raises(ValueError, match="source segment indices"):
        complete_wall_gaps(fitted, wall([0, 0], [2, 0]), floor_height=0,
                           source_segments=np.asarray([[[0, 0], [1, 0]]]), not_blocks_navmesh=allowed)


def test_semantic_door_protection_does_not_relabel_unexplained_nodes_as_verified_jambs():
    fitted = [line("a", [0, 0], [1, 0]), line("b", [1.2, 0], [2, 0])]
    result = complete_wall_gaps(fitted, wall([0, 0], [2, 0]), floor_height=0,
                               not_blocks_navmesh=allowed,
                               semantic_priors=[{"id": "door", "category": "door", "segment": [[1, 0], [1.2, 0]]}])
    assert len(result["unexplained_endpoints_before"]) == len(result["unexplained_endpoints"]) == 4
    assert not any(node["kind"] == "portal_jamb" for node in result["nodes"])


def test_source_joint_refit_cannot_move_an_old_wall_more_than_three_centimetres():
    # The source observes a plane 4 cm away from both inaccurate accepted fits.
    fitted = [line("a", [0, 0], [1, 0], source_segment_indices=[0]),
              line("b", [1.2, 0], [2, 0], source_segment_indices=[1])]
    observed = np.asarray([[[0, .04], [1, .04]], [[1.2, .04], [2, .04]]])
    result = complete_wall_gaps(fitted, wall([0, .04], [2, .04]), floor_height=0,
                               source_segments=observed, not_blocks_navmesh=allowed)
    assert not result["accepted"]
    assert result["rejected"][0]["reason"] == "joint_fit_residual_or_displacement_exceeded"
    assert result["rejected"][0]["joint_fit"]["max_original_line_displacement_m"] == pytest.approx(.04)


def test_wall_axes_are_not_treated_as_measured_surface_fits():
    fitted = [line("axis", [0, 0], [1, 0], representation="wall_axis"),
              line("b", [1.2, 0], [2, 0])]
    result = complete_wall_gaps(fitted, wall([0, 0], [2, 0]), floor_height=0, not_blocks_navmesh=allowed)
    assert not result["accepted"]


def test_joint_refit_keeps_an_existing_interior_t_junction_exact():
    fitted = [line("a", [0, 0], [1, 0]), line("b", [1.2, .01], [2, .01]),
              line("branch", [.5, 0], [.5, 1])]
    raw = np.concatenate([wall([0, 0], [2, 0]), wall([.5, 0], [.5, 1])])
    result = complete_wall_gaps(fitted, raw, floor_height=0, not_blocks_navmesh=allowed)
    assert len(result["accepted"]) == 1
    fit = result["accepted"][0]["joint_fit"]
    assert fit["fixed_existing_junctions"] == [[.5, 0.]]
    assert any(node["point"] == [.5, 0.] and len(node["incidences"]) == 2 for node in result["nodes"])
