"""CPU fixtures for coverage-preserving CAD canonicalization."""

from copy import deepcopy
import math

import numpy as np
import pytest

from src.data.map_wall_cad import canonicalize_walls, overlap_report


def line(identifier, start, end, **extra):
    return {"id": identifier, "type": "line", "start": start, "end": end, **extra}


def arc(identifier, start, sweep, radius=2., center=(0., 0.), **extra):
    return {"id": identifier, "type": "arc", "center": list(center), "radius_m": radius,
            "start_angle_degrees": start, "sweep_degrees": sweep, **extra}


def circle(identifier, radius=2., center=(0., 0.)):
    return {"id": identifier, "type": "circle", "center": list(center), "radius_m": radius}


def assert_line_span(value, start, end):
    np.testing.assert_allclose(value["start"], start, atol=1e-10)
    np.testing.assert_allclose(value["end"], end, atol=1e-10)


def test_overlapping_reversed_and_contained_lines_become_one_source_audited_segment():
    inputs = [line("a", [0., 0.], [2., 0.], source_segment_indices=[1, 2]),
              line("b", [3., 0.], [1., 0.], source_segment_indices=[2, 3]),
              line("contained", [.5, 0.], [1.5, 0.]),
              line("duplicate", [0., 0.], [2., 0.])]
    original = deepcopy(inputs)
    result = canonicalize_walls(inputs)
    assert len(result["primitives"]) == 1
    wall = result["primitives"][0]
    assert_line_span(wall, [0., 0.], [3., 0.])
    assert wall["source_ids"] == ["a", "b", "contained", "duplicate"]
    assert wall["source_segment_indices"] == [1, 2, 3]
    assert result["metrics"]["before"]["sum_length_m"] == 7.
    assert result["metrics"]["after"]["sum_length_m"] == 3.
    assert result["metrics"]["qualified"]
    assert result["metrics"]["added_gap_length_m"] == 0.
    assert inputs == original


def test_real_door_gap_is_preserved_even_on_one_infinite_line():
    result = canonicalize_walls([line("a", [0., 0.], [2., 0.]), line("b", [2.8, 0.], [5., 0.])])
    assert len(result["primitives"]) == 2
    assert result["metrics"]["added_gap_length_m"] == 0.
    assert all(not min(item["start"][0], item["end"][0]) < 2.4 < max(item["start"][0], item["end"][0])
               for item in result["primitives"])


@pytest.mark.parametrize("gap,expected", [(8e-7, 1), (1.1e-6, 2)])
def test_only_numerical_seams_join_by_default(gap, expected):
    result = canonicalize_walls([line("a", [0., 0.], [1., 0.]), line("b", [1. + gap, 0.], [2., 0.])])
    assert len(result["primitives"]) == expected
    assert result["metrics"]["added_gap_length_m"] == pytest.approx(gap if expected == 1 else 0.)


def test_nonzero_join_needs_explicit_evidence_and_preserves_that_evidence():
    calls = []

    def allow(segment):
        calls.append(segment)
        return {"pass": True, "source": "provided_raw_support", "coverage": 1.}

    inputs = [line("a", [0., 0.], [1., 0.]), line("b", [1.2, 0.], [2., 0.])]
    result = canonicalize_walls(inputs, allow_gap_join=allow, not_blocks_navmesh=lambda _: {"pass": True})
    assert len(result["primitives"]) == 1
    np.testing.assert_allclose(calls, [[[1., 0.], [1.2, 0.]]])
    assert result["metrics"]["added_gap_length_m"] == pytest.approx(.2)
    assert result["primitives"][0]["cad"]["joins"][0]["evidence"]["source"] == "provided_raw_support"
    rejected = canonicalize_walls(inputs, allow_gap_join=lambda _: {"pass": False})
    assert len(rejected["primitives"]) == 2
    assert rejected["metrics"]["added_gap_length_m"] == 0.


def test_projection_uses_an_existing_carrier_and_respects_the_residual_budget():
    inputs = [line("long", [0., 0.], [4., 0.], fitting={"max_residual_m": .01}),
              line("slight_fit_noise", [1., .002], [3., .018], fitting={"max_residual_m": .01})]
    result = canonicalize_walls(inputs)
    assert len(result["primitives"]) == 1
    assert_line_span(result["primitives"][0], [0., 0.], [4., 0.])
    assert result["metrics"]["max_projection_distance_m"] == pytest.approx(.018)
    assert result["metrics"]["added_gap_length_m"] == 0.
    inputs[0]["fitting"]["max_residual_m"] = inputs[1]["fitting"]["max_residual_m"] = .001
    assert len(canonicalize_walls(inputs)["primitives"]) == 2


def test_a_two_degree_limit_does_not_flatten_a_real_corner():
    theta = math.radians(3)
    result = canonicalize_walls([line("a", [0., 0.], [1., 0.]),
                                 line("b", [.9, 0.], [.9 + math.cos(theta), math.sin(theta)])])
    assert len(result["primitives"]) == 2


def test_different_parallel_faces_are_not_averaged_or_collapsed():
    result = canonicalize_walls([line("inside", [0., 0.], [3., 0.]),
                                 line("outside", [0., .01], [3., .01])])
    assert len(result["primitives"]) == 2
    assert {item["start"][1] for item in result["primitives"]} == {0., .01}
    assert not result["metrics"]["qualified"]
    assert result["metrics"]["unresolved_overlap_count"] == 0
    assert result["metrics"]["unresolved_near_line_overlap_count"] == 1


def test_two_centimetre_parallel_surfaces_remain_pending_without_same_surface_evidence():
    inputs = [line("first", [0., 0.], [3., 0.]), line("second", [.5, .02], [2.5, .02])]
    result = canonicalize_walls(inputs)
    assert len(result["primitives"]) == 2
    assert not result["metrics"]["qualified"]
    assert result["metrics"]["visible_overlap_count"] == 1
    conflict = next(item for item in result["conflicts"] if item["reason"] == "unresolved_near_line_overlap")
    assert conflict["max_normal_distance_m"] == pytest.approx(.02)
    assert conflict["shorter_span_coverage_fraction"] == pytest.approx(1.)
    assert not conflict["same_surface_verified"]
    assert conflict["within_projection_budget"]


def test_visible_near_coverage_outside_fit_budget_still_needs_review():
    result = canonicalize_walls([
        line("first", [0., 0.], [3., 0.], fitting={"max_residual_m": .001}),
        line("second", [.5, .025], [2.5, .025], fitting={"max_residual_m": .001})])
    conflict = next(item for item in result["conflicts"] if item["reason"] == "unresolved_near_line_overlap")
    assert not result["metrics"]["qualified"]
    assert not conflict["within_projection_budget"]
    assert conflict["fit_constrained_projection_budget_m"] == pytest.approx(.002)
    assert conflict["diagnostic_distance_limit_m"] == .03


@pytest.mark.parametrize("distance", [.03001, .14, .3])
def test_separate_parallel_walls_beyond_three_centimetres_are_not_near_duplicates(distance):
    result = canonicalize_walls([line("a", [0., 0.], [3., 0.]), line("b", [0., distance], [3., distance])])
    assert len(result["primitives"]) == 2
    assert result["metrics"]["qualified"]
    assert result["metrics"]["unresolved_near_line_overlap_count"] == 0


def test_point_junctions_do_not_claim_duplicate_coverage():
    perpendicular = canonicalize_walls([line("a", [-1., 0.], [1., 0.]), line("b", [0., -1.], [0., 1.])])
    assert perpendicular["metrics"]["qualified"]


def test_shared_automatic_junction_cannot_hide_long_acute_near_duplicate_coverage():
    # Both lines remain intact, but an automatically assigned common node
    # cannot turn their extensive subpixel double coverage into a CAD pass.
    acute = canonicalize_walls([
        line("a", [-1., 0.], [1., 0.], fitting={"max_residual_m": 0.}),
        line("b", [-1., -.01], [1., .01], fitting={"max_residual_m": 0.})],
        junctions=[{"id": "crossing", "point": [0., 0.]}])
    assert len(acute["primitives"]) == 2
    assert all("crossing" in item["interior_nodes"] for item in acute["primitives"])
    assert not acute["metrics"]["qualified"]
    assert acute["metrics"]["unresolved_near_line_overlap_count"] == 1
    assert acute["metrics"]["visible_overlap_count"] == 1


def test_a_genuine_acute_branch_with_only_small_longitudinal_overlap_is_preserved():
    result = canonicalize_walls([
        line("short", [0., 0.], [1., 0.], fitting={"max_residual_m": .001}),
        line("branch", [.9, 0.], [3.9, .05236], fitting={"max_residual_m": .001})],
        junctions=[{"id": "branch-node", "point": [.9, 0.]}])
    assert len(result["primitives"]) == 2
    assert result["metrics"]["qualified"]
    assert result["metrics"]["unresolved_near_line_overlap_count"] == 0


def test_small_endpoint_projection_overlap_is_not_a_long_duplicate_surface():
    result = overlap_report([line("a", [0., 0.], [3., 0.]), line("b", [2.9, .02], [5.9, .02])])
    assert result["qualified"]
    assert result["visible_overlap_count"] == 0


def test_read_only_overlap_diagnostic_preserves_ids_and_does_not_double_count_exact_overlap():
    inputs = [line("original-id-a", [0., 0.], [3., 0.]), line("original-id-b", [1., 0.], [2., 0.])]
    original = deepcopy(inputs)
    result = overlap_report(inputs)
    assert result["exact_overlap_count"] == 1
    assert result["near_line_overlap_count"] == 0
    assert inputs == original
    assert result["exact"][0]["canonical_ids"] == ["original-id-a", "original-id-b"]


def test_same_surface_raw_evidence_can_merge_only_the_bounded_parallel_overlap():
    calls = []
    def same_surface(first, second):
        calls.append([first["id"], second["id"]])
        return {"pass": True, "source": "independent_raw_same_facing_plane", "support_fraction": .98}
    inputs = [line("a", [0., 0.], [3., 0.]), line("b", [.5, .01], [2.5, .01])]
    original = deepcopy(inputs)
    result = canonicalize_walls(inputs, same_surface=same_surface,
                                 not_blocks_navmesh=lambda _: {"pass": True})
    assert len(result["primitives"]) == 1
    assert result["metrics"]["qualified"]
    assert result["metrics"]["added_gap_length_m"] == 0.
    assert result["metrics"]["max_projection_distance_m"] == pytest.approx(.01)
    assert calls == [["a", "b"]]
    union = next(item for item in result["decisions"] if item["action"] == "line_union")
    assert union["same_surface_checks"][0]["evidence"]["support_fraction"] == .98
    assert result["primitives"][0]["cad"]["same_surface_checks"] == union["same_surface_checks"]
    assert inputs == original


@pytest.mark.parametrize("evidence", [{"pass": False, "reason": "different_facing_planes"}, True])
def test_same_surface_rejection_or_invalid_return_keeps_parallel_surfaces_pending(evidence):
    result = canonicalize_walls([line("a", [0., 0.], [3., 0.]), line("b", [.5, .01], [2.5, .01])],
                                 same_surface=lambda *_: evidence)
    assert len(result["primitives"]) == 2
    assert not result["metrics"]["qualified"]
    check = next(item for item in result["decisions"] if item["action"] == "same_surface_check")
    assert check["evidence"]["pass"] is False


@pytest.mark.parametrize("first,second", [
    (line("a", [0., 0.], [3., 0.]), line("b", [.5, .025], [2.5, .025])),
    (line("a", [0., 0.], [3., 0.], fitting={"max_residual_m": .001}),
     line("b", [.5, .01], [2.5, .01], fitting={"max_residual_m": .001})),
    (line("a", [0., 0.], [1., 0.]), line("b", [1.2, .01], [3., .01])),
])
def test_same_surface_is_not_consulted_outside_budget_or_across_a_real_gap(first, second):
    def forbidden(*_):
        pytest.fail("same_surface must not relax distance, residual or gap restrictions")
    result = canonicalize_walls([first, second], same_surface=forbidden)
    assert len(result["primitives"]) == 2
    assert not any(item["action"] == "same_surface_check" for item in result["decisions"])


def test_every_offset_parallel_pair_needs_surface_permission_not_just_the_carrier():
    def same_surface(first, second):
        return {"pass": {first["id"], second["id"]} != {"b", "c"}, "source": "raw_pair_fixture"}
    result = canonicalize_walls([line("a", [0., 0.], [4., 0.]), line("b", [.5, .01], [3.5, .01]),
                                 line("c", [1., .02], [3., .02])], same_surface=same_surface)
    assert len(result["primitives"]) == 2
    assert not result["metrics"]["qualified"]
    assert not any({"b", "c"} <= set(item["source_ids"]) for item in result["primitives"])


def test_projected_endpoint_contact_does_not_close_an_actual_offset_gap():
    result = canonicalize_walls([line("a", [0., 0.], [2., 0.]),
                                 line("b", [2., .01], [3., .02])])
    assert len(result["primitives"]) == 2
    assert result["metrics"]["added_gap_length_m"] == 0.


def test_unrelated_carrier_support_indices_are_not_attached_to_another_run():
    result = canonicalize_walls([line("carrier", [10., 0.], [20., 0.], source_segment_indices=[999]),
                                 line("a", [0., 0.], [2., 0.]), line("b", [1., 0.], [3., 0.])])
    merged = next(item for item in result["primitives"] if item["source_ids"] == ["a", "b"])
    assert "source_segment_indices" not in merged
    assert "source_segment_count" not in merged


def test_parallel_faces_cannot_be_collapsed_transitively_via_a_third_fit():
    result = canonicalize_walls([line("carrier", [0., 0.], [10., 0.]),
                                 line("face-a", [1., .002], [3., .006]),
                                 line("face-b", [1., .012], [3., .016])])
    assert len(result["primitives"]) == 2
    assert not any({"face-a", "face-b"} <= set(item["source_ids"]) for item in result["primitives"])


def test_t_junctions_are_internal_nodes_of_one_long_wall_not_extra_segments():
    result = canonicalize_walls([
        line("left", [0., 0.], [1., 0.], start_node="end-left", end_node="T"),
        line("right", [1., 0.], [3., 0.], start_node="T", end_node="end-right"),
        line("branch", [1., 0.], [1., 2.], start_node="T")],
        junctions=[{"id": "other-T", "point": [2., 0.]}])
    horizontal = next(item for item in result["primitives"] if set(item["source_ids"]) == {"left", "right"})
    assert len(result["primitives"]) == 2
    assert set(horizontal["interior_nodes"]) == {"T", "other-T"}
    assert horizontal["start_node"] == "end-left"
    assert horizontal["end_node"] == "end-right"
    assert result["source_to_canonical"]["left"] == result["source_to_canonical"]["right"]


def test_failed_navigation_check_keeps_original_geometry_and_hard_overlap_conflict():
    inputs = [line("a", [0., 0.], [2., 0.]), line("b", [1., 0.], [3., 0.])]
    result = canonicalize_walls(inputs, not_blocks_navmesh=lambda _: {"pass": False, "reason": "walkable_crossing"})
    assert len(result["primitives"]) == 2
    spans = sorted((item["start"], item["end"]) for item in result["primitives"])
    assert spans == [([0., 0.], [2., 0.]), ([1., 0.], [3., 0.])]
    assert not result["metrics"]["qualified"]
    assert result["metrics"]["unresolved_overlap_count"] == 1
    assert {item["reason"] for item in result["conflicts"]} == {"not_blocks_navmesh_failed", "unresolved_overlap"}


def test_callback_errors_are_audited_and_cannot_silently_accept_a_merge():
    def broken(_):
        raise RuntimeError("fixture probe unavailable")

    result = canonicalize_walls([line("a", [0., 0.], [2., 0.]), line("b", [1., 0.], [3., 0.])],
                                 not_blocks_navmesh=broken)
    assert not result["metrics"]["qualified"]
    assert "RuntimeError" in result["conflicts"][0]["evidence"]["error"]


def test_overlapping_circular_arcs_union_actual_coverage_without_closing_the_circle():
    result = canonicalize_walls([arc("a", 0., 120.), arc("b", 60., 120.), arc("contained", 80., 20.)])
    assert len(result["primitives"]) == 1
    wall = result["primitives"][0]
    assert wall["type"] == "arc"
    assert wall["sweep_degrees"] == pytest.approx(180.)
    assert wall["radius_m"] == 2.
    assert wall["source_ids"] == ["a", "b", "contained"]


def test_clockwise_and_wraparound_arcs_use_the_same_unbroken_circular_interval():
    result = canonicalize_walls([arc("wrap", 350., 30.), arc("clockwise", 40., -40.)])
    assert len(result["primitives"]) == 1
    wall = result["primitives"][0]
    assert wall["type"] == "arc"
    assert wall["start_angle_degrees"] == pytest.approx(350.)
    assert wall["sweep_degrees"] == pytest.approx(50.)


def test_a_circle_requires_complete_original_angular_coverage():
    result = canonicalize_walls([arc("left", 0., 180.), arc("right", 180., 180.)])
    assert len(result["primitives"]) == 1
    assert result["primitives"][0]["type"] == "circle"
    assert result["primitives"][0]["radius_m"] == 2.
    assert result["metrics"]["after"]["circle_count"] == 1
    assert result["metrics"]["after"]["sum_length_m"] == pytest.approx(4 * math.pi)
    for missing_degrees in (1., 1e-7):
        partial = canonicalize_walls([arc("partial", 0., 360. - missing_degrees)])
        assert partial["primitives"][0]["type"] == "arc"


def test_angular_door_gap_remains_open_and_duplicate_circles_are_removed():
    partial = canonicalize_walls([arc("a", 0., 90.), arc("b", 91., 90.)])
    assert len(partial["primitives"]) == 2
    assert partial["metrics"]["added_gap_length_m"] == 0.
    result = canonicalize_walls([circle("one"), circle("duplicate"), arc("contained", 20., 50.)])
    assert len(result["primitives"]) == 1
    assert result["primitives"][0]["type"] == "circle"
    assert result["primitives"][0]["source_ids"] == ["contained", "duplicate", "one"]


def test_distinct_concentric_faces_or_shifted_circles_are_not_refitted_together():
    result = canonicalize_walls([circle("inner"), circle("outer", radius=2.2),
                                 circle("shifted", center=(.2, 0.))])
    assert len(result["primitives"]) == 3
    assert result["metrics"]["qualified"]


def test_b6_near_duplicate_arcs_get_one_joint_circle_with_analytic_displacement_bound():
    inputs = [arc("W0151", -152., 116., radius=.364, center=(28.72, 14.24), source_segment_indices=[1]),
              arc("W0521", -158., 128., radius=.345, center=(28.74, 14.24), source_segment_indices=[2]),
              arc("W0522", -151., 122., radius=.325, center=(28.75, 14.23), source_segment_indices=[3])]
    result = canonicalize_walls(inputs)
    assert len(result["primitives"]) == 1
    wall = result["primitives"][0]
    assert wall["type"] == "arc"  # Circular column evidence does not prove a full circle.
    assert wall["source_ids"] == ["W0151", "W0521", "W0522"]
    assert wall["source_segment_indices"] == [1, 2, 3]
    fit = wall["cad"]["joint_fit"]
    assert fit["verification"] == "analytic_extrema_over_each_entire_source_arc"
    assert fit["max_displacement_m"] <= .03
    assert fit["circle_parameter_shifts_within_5cm"]
    assert wall["cad"]["circle_parameters_refitted"]
    assert result["metrics"]["qualified"]
    # Independently densely sample the original curves, rather than only the
    # optimizer's sparse samples, to check its analytic guarantee.
    for source in inputs:
        angles = np.radians(np.linspace(source["start_angle_degrees"],
                                        source["start_angle_degrees"] + source["sweep_degrees"], 10001))
        points = np.asarray(source["center"]) + source["radius_m"] * np.c_[np.cos(angles), np.sin(angles)]
        error = abs(np.linalg.norm(points - wall["center"], axis=1) - wall["radius_m"])
        assert error.max() <= fit["source_displacements_m"][source["id"]] + 1e-9


def test_near_arc_fit_failure_is_a_hard_conflict_instead_of_silent_overlap():
    inputs = [arc("a", -152., 116., radius=.364, center=(28.72, 14.24)),
              arc("b", -158., 128., radius=.345, center=(28.74, 14.24)),
              arc("c", -151., 122., radius=.325, center=(28.75, 14.23))]
    result = canonicalize_walls(inputs, arc_max_displacement_m=.02)
    assert len(result["primitives"]) > 1
    assert not result["metrics"]["qualified"]
    assert result["metrics"]["unresolved_near_circular_overlap_count"] > 0


def test_joint_arc_fit_never_fills_the_angular_door_between_two_coverage_groups():
    result = canonicalize_walls([arc("a", 0., 80.), arc("a-copy", 0., 80., center=(.01, 0.)),
                                 arc("b", 100., 80.), arc("b-copy", 100., 80., center=(.01, 0.))])
    assert len(result["primitives"]) == 2
    assert all(item["type"] == "arc" for item in result["primitives"])
    assert result["metrics"]["added_gap_length_m"] == 0.
    assert result["metrics"]["qualified"]


def test_refitted_arcs_are_checked_along_the_curve_instead_of_a_long_chord():
    calls = []
    def navigation(segment):
        calls.append(segment)
        return {"pass": False, "reason": "fixture crossing"}
    result = canonicalize_walls([arc("a", 0., 90.), arc("b", 0., 90., center=(.01, 0.))],
                                 not_blocks_navmesh=navigation)
    assert len(result["primitives"]) == 2
    assert not result["metrics"]["qualified"]
    assert calls
    assert max(np.linalg.norm(np.asarray(segment)[1] - segment[0]) for segment in calls) <= .025 + 1e-8


def test_joint_projection_cannot_turn_an_original_angular_gap_into_a_complete_circle():
    inputs = [arc("a", 0., 179.9), arc("b", 179.8, 180.1, center=(0., .01))]
    result = canonicalize_walls(inputs)
    assert not any(item["type"] == "circle" for item in result["primitives"])
    assert not result["metrics"]["qualified"]
    assert any(item["reason"] == "circular_projection_would_close_original_gap" for item in result["conflicts"])


def test_repeated_canonicalization_keeps_source_ids_and_geometry():
    first = canonicalize_walls([line("a", [0., 0.], [2., 0.]), line("b", [1., 0.], [3., 0.])])
    second = canonicalize_walls(first["primitives"])
    assert first["primitives"][0]["source_ids"] == second["primitives"][0]["source_ids"] == ["a", "b"]
    assert_line_span(second["primitives"][0], [0., 0.], [3., 0.])


@pytest.mark.parametrize("items,kwargs", [
    ([line("same", [0, 0], [1, 0]), line("same", [1, 0], [2, 0])], {}),
    ([line("zero", [0, 0], [0, 0])], {}),
    ([circle("bad", radius=-1.)], {}),
    ([arc("too_long", 0., 361.)], {}),
    ([], {"distance_tolerance_m": .04}),
    ([], {"angle_tolerance_degrees": 3.}),
])
def test_invalid_geometry_or_excessive_tolerances_are_rejected(items, kwargs):
    with pytest.raises(ValueError):
        canonicalize_walls(items, **kwargs)


def test_empty_input_has_no_invented_geometry():
    result = canonicalize_walls([])
    assert result["primitives"] == []
    assert result["metrics"]["qualified"]
    assert result["metrics"]["after"]["sum_length_m"] == 0.
