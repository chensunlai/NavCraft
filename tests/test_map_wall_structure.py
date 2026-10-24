from copy import deepcopy

import numpy as np
import pytest

from src.data.map_wall_structure import reconnect_primitives, rasterize_primitives, structure_svg


def line(identifier, start, end):
    return {"id": identifier, "type": "line", "start": start, "end": end, "endpoints": [start, end]}


def support(segment):
    return {"pass": True, "source": "independent fixture geometry", "segment": segment}


def reconnect(items, **kwargs):
    return reconnect_primitives(items, raw_support=support, not_blocks_navmesh=support, **kwargs)


def test_l_corner_uses_supported_intersection_and_shared_endpoint():
    original = [line("A", [0, 0], [.9, 0]), line("B", [1, .1], [1, 1])]
    before = deepcopy(original)
    result = reconnect(original)
    a, b = result["primitives"]
    assert a["end"] == b["start"] == [1, 0]
    assert a["end_node"] == b["start_node"]
    assert a["endpoints"] == [a["start"], a["end"]]
    assert len(result["shared_junctions"]) == 1
    assert len(result["connections"]) == 1
    assert all(check["raw_support"]["pass"] for check in result["connections"][0]["checks"])
    assert original == before


def test_t_junction_keeps_carrier_geometry_and_splits_graph_edges():
    items = [line("A", [0, 0], [2, 0]), line("B", [1, 1], [1, .1])]
    result = reconnect(items)
    a, b = result["primitives"]
    assert a["start"] == [0, 0] and a["end"] == [2, 0]
    assert b["end"] == [1, 0]
    assert a["interior_nodes"] == [b["end_node"]]
    assert len(result["edges"]) == 3
    assert result["connections"][0]["kind"] == "endpoint_to_interior"


def test_near_parallel_walls_are_not_joined():
    items = [line("A", [0, 0], [1, 0]), line("B", [1.05, .1], [2.05, .1])]
    result = reconnect(items)
    assert not result["connections"]
    assert not result["shared_junctions"]
    assert result["primitives"][0]["end"] == [1, 0]


def test_protected_door_gap_stays_open_but_parallel_frame_is_allowed():
    portal = {"segment": [[.9, 0], [1.1, 0]], "status": "accepted"}
    blocked = reconnect([line("A", [0, 0], [.9, 0]), line("B", [1.1, 0], [2, 0])], portals=[portal])
    assert not blocked["connections"]
    assert blocked["rejected"][0]["reason"] == "protected_portal_interior"
    parallel = reconnect([line("A", [0, .1], [.9, .1]), line("B", [1.1, .1], [2, .1])], portals=[portal])
    assert parallel["connections"]


def test_unsupported_or_navmesh_blocking_gap_is_not_filled():
    items = [line("A", [0, 0], [.9, 0]), line("B", [1.1, 0], [2, 0])]
    for missing in ("raw", "nav"):
        result = reconnect_primitives(items, raw_support=(lambda _: {"pass": False}) if missing == "raw" else support,
                                      not_blocks_navmesh=(lambda _: {"pass": False}) if missing == "nav" else support)
        assert not result["connections"]
        assert result["rejected"][0]["reason"] == "independent_evidence_failed"
        assert result["primitives"][0]["end"] == [.9, 0]


def test_larger_collinear_gap_is_explicit_and_never_exceeds_shift_limit():
    result = reconnect([line("A", [0, 0], [.8, 0]), line("B", [1.2, 0], [2, 0])])
    assert not result["connections"]
    assert result["rejected"][0]["kind"] == "collinear_gap"
    assert result["rejected"][0]["reason"] == "endpoint_displacement_limit"


def test_arc_shape_is_immutable_and_rendered_as_a_curve():
    arc = {"id": "C", "type": "arc", "center": [0, 0], "radius_m": 1.,
           "start_angle_degrees": 0., "end_angle_degrees": 90., "sweep_degrees": 90.}
    result = reconnect([arc, line("A", [2, 0], [1.05, 0])])
    assert {key: result["primitives"][0][key] for key in arc} == arc
    assert result["primitives"][1]["end"] == [1.05, 0]
    mask = rasterize_primitives([arc], [-.5, -.5], 3.)
    assert mask.shape == (60, 60)
    for point in ([1., 0.], [0., 1.], [2 ** -.5, 2 ** -.5]):
        col, row = np.rint((np.array(point) + .5) / .05 - .5).astype(int)
        assert mask[max(0, row - 1):row + 2, max(0, col - 1):col + 2].any()
    assert not mask[10, 10]
    assert " A 1 1 " in structure_svg([arc], [-.5, -.5], 3.)


def test_evidence_callback_requires_an_explicit_pass_boolean():
    with pytest.raises(ValueError, match="explicit boolean"):
        reconnect_primitives([line("A", [0, 0], [.9, 0]), line("B", [1.1, 0], [2, 0])],
                             raw_support=lambda _: True, not_blocks_navmesh=support)


def test_short_primitive_cannot_collapse_when_both_endpoints_snap():
    result = reconnect([line("short", [-.1, 0], [.1, 0]), line("carrier", [0, -1], [0, 1])])
    short = result["primitives"][0]
    assert np.linalg.norm(np.asarray(short["end"]) - short["start"]) >= .05
    assert any(item["reason"] == "primitive_below_minimum_length" for item in result["rejected"])
    assert all(edge["start_node"] != edge["end_node"] for edge in result["edges"])


def test_merges_a_small_supported_endpoint_cluster():
    result = reconnect([
        line("a", [0, 0], [1, 0]),
        line("b", [0, .01], [0, 1]),
        line("c", [.012, 0], [1, 1]),
    ])
    assert any(item["kind"] == "near_endpoint_cluster" for item in result["connections"])
    assert result["primitives"][0]["start"] == result["primitives"][1]["start"]
    assert result["primitives"][1]["start"] == result["primitives"][2]["start"]


def test_endpoint_cluster_keeps_an_existing_junction_fixed():
    result = reconnect([line("a", [0, 0], [1, 0]),
                        line("b", [0, 0], [0, 1]),
                        line("c", [.01, .01], [1, 1])])
    a, b, c = result["primitives"]
    assert a["start"] == b["start"] == c["start"] == [0, 0]
    assert a["start_node"] == b["start_node"] == c["start_node"]


def test_endpoint_cluster_cannot_merge_two_distinct_fixed_junctions():
    items = [line("a", [0, 0], [1, 0]), line("b", [0, 0], [0, 1]),
             line("c", [.03, .01], [1, .98]), line("d", [.03, .01], [-1, 1.04])]
    result = reconnect(items)
    assert [item["endpoints"] for item in result["primitives"]] == [item["endpoints"] for item in items]
    assert result["primitives"][0]["start_node"] != result["primitives"][2]["start_node"]
    assert any(item["reason"] == "endpoint_already_assigned_to_another_junction"
               for item in result["rejected"])


def test_endpoint_cluster_cannot_rotate_a_carrier_off_an_existing_t_junction():
    result = reconnect([line("carrier", [0, 0], [2, 0]),
                        line("branch", [1, 0], [1, 1]),
                        line("near", [0, .01], [0, 1])])
    carrier, branch, _ = result["primitives"]
    assert carrier["start"] == [0, 0] and carrier["end"] == [2, 0]
    assert branch["start_node"] in carrier["interior_nodes"]
    assert any(item["reason"] == "would_break_existing_interior_junction"
               for item in result["rejected"])


def test_endpoint_cluster_does_not_merge_parallel_wall_faces():
    items = [line("front", [0, 0], [1, 0]), line("back", [0, .01], [1, .01])]
    result = reconnect(items)
    assert not result["connections"]
    assert [item["endpoints"] for item in result["primitives"]] == [item["endpoints"] for item in items]


def test_endpoint_cluster_never_contains_both_ends_of_one_primitive():
    result = reconnect([line("seed", [0, 0], [0, 1]),
                        line("short", [-.04, .01], [.04, .01])], min_line_length_m=.01)
    for connection in result["connections"]:
        if connection["kind"] == "near_endpoint_cluster":
            identifiers = [endpoint["primitive_id"] for endpoint in connection["endpoints"]]
            assert len(identifiers) == len(set(identifiers))
    assert np.linalg.norm(np.asarray(result["primitives"][1]["end"])
                          - result["primitives"][1]["start"]) >= .01


def test_endpoint_cluster_diameter_is_bounded_not_just_distance_to_seed():
    items = [line("middle", [.04, 0], [.04, 1]),
             line("left", [0, 0], [-1, 1]),
             line("right", [.08, 0], [1, 1])]
    result = reconnect(items)
    positions = {(item["id"], at): np.asarray(item[at]) for item in items for at in ("start", "end")}
    for connection in result["connections"]:
        if connection["kind"] == "near_endpoint_cluster":
            points = [positions[(endpoint["primitive_id"], endpoint["at"])]
                      for endpoint in connection["endpoints"]]
            assert all(np.linalg.norm(a - b) <= .05 + 1e-9 for a in points for b in points)


def test_endpoint_cluster_obeys_the_configured_displacement_limit():
    result = reconnect([line("a", [0, 0], [1, 0]),
                        line("b", [.04, .01], [.04, 1])], max_shift_m=.01)
    assert not any(item["kind"] == "near_endpoint_cluster" for item in result["connections"])
    assert all(item["maximum_shift_m"] <= .01 + 1e-9 for item in result["connections"])
    assert any(item["reason"] == "endpoint_displacement_limit" for item in result["rejected"])


def test_configured_minimum_length_is_preserved():
    result = reconnect([line("short", [-.1, 0], [.1, 0]), line("carrier", [0, -1], [0, 1])],
                       min_line_length_m=.2)
    assert not result["connections"]
    assert result["primitives"][0]["start"] == [-.1, 0]
    assert result["primitives"][0]["end"] == [.1, 0]


def test_numerically_duplicate_endpoints_across_hash_rounding_boundary_share_node():
    result = reconnect([line("A", [0, 0], [1.00000049, 0]),
                        line("B", [1.00000051, 0], [2, 0])])
    a, b = result["primitives"]
    assert a["end_node"] == b["start_node"]
    assert len(result["nodes"]) == 3
    assert len(result["shared_junctions"]) == 1
    assert len(result["edges"]) == 2


def test_t_junction_is_recomputed_after_earlier_carrier_adjustment():
    result = reconnect([line("A", [0, 0], [2, 0]),
                        line("C", [2.04, .01], [3, .01]),
                        line("B", [1, 1], [1, .1])])
    carrier, _, branch = result["primitives"]
    assert carrier["end"] == [2.02, .005]
    assert branch["end"][1] > 0
    assert branch["end_node"] in carrier["interior_nodes"]
    point = np.array(branch["end"])
    direction = np.array(carrier["end"]) - carrier["start"]
    assert abs(np.cross(direction, point - carrier["start"])) < 1e-10


def test_later_carrier_rotation_cannot_break_accepted_t_junction():
    result = reconnect([line("A", [0, 0], [2, 0]),
                        line("B", [1, 1], [1, .08]),
                        line("C", [2.2, .01], [3, .01])])
    carrier, branch, _ = result["primitives"]
    assert branch["end"] == [1, 0]
    assert carrier["end"] == [2, 0]
    assert branch["end_node"] in carrier["interior_nodes"]
    assert any(item["reason"] == "would_break_existing_interior_junction" for item in result["rejected"])


def test_zero_length_input_is_rejected_but_closed_circle_is_valid():
    with pytest.raises(ValueError, match="nonzero length"):
        reconnect([line("bad", [0, 0], [0, 0])])
    with pytest.raises(ValueError, match="nonzero length"):
        reconnect([{"id": "bad", "type": "arc", "center": [0, 0], "radius_m": 1,
                    "start_angle_degrees": 0, "end_angle_degrees": 0}])
    circle = {"id": "circle", "type": "arc", "center": [0, 0], "radius_m": 1,
              "start_angle_degrees": 0, "end_angle_degrees": 360, "sweep_degrees": 360}
    result = reconnect([circle])
    assert len(result["nodes"]) == 1
    assert not result["unexplained_endpoints"]
