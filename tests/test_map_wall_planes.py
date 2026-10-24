import numpy as np
import pytest

from src.data.map_wall_planes import refit_wall_planes


HEIGHTS = np.array([.5, .9, 1.3, 1.7, 2.3])


def face(z=0., lean=0., start=0., end=3., sign=1):
    vertices = np.array([[start, 0, z], [end, 0, z],
                         [end, 2.4, z + 2.4 * lean], [start, 2.4, z + 2.4 * lean]])
    result = vertices[[[0, 1, 2], [0, 2, 3]]]
    return result if sign > 0 else result[:, ::-1]


def line(name, z, indices, start=0., end=3.):
    return {"id": name, "type": "line", "start": [start, z], "end": [end, z],
            "source_segment_indices": indices}


def clear(_):
    return {"pass": True, "source": "fixture measured NavMesh"}


def tilted_case():
    source = np.array([[[0, .01 * h], [3, .01 * h]] for h in HEIGHTS])
    return [line("low-cut", .005, [0, 1, 2]), line("high-cut", .023, [2, 3, 4])], source


def run(items, raw, source, heights=HEIGHTS, nav=clear):
    return refit_wall_planes(items, raw, source, heights, floor_height=0., not_blocks_navmesh=nav)


def test_tilted_single_sheet_becomes_one_reference_height_line_from_actual_3d_points():
    items, source = tilted_case()
    result = run(items, face(lean=.01), source)
    assert len(result["accepted"]) == len(result["primitives"]) == 1
    value = result["primitives"][0]
    assert sorted([value["start"][0], value["end"][0]]) == pytest.approx([0, 3])
    assert value["start"][1] == pytest.approx(.012) and value["end"][1] == pytest.approx(.012)
    proof = result["accepted"][0]
    assert proof["max_raw_registration_distance_m"] < 1e-10
    assert proof["max_raw_plane_residual_m"] < 1e-10
    assert abs(proof["plane_normal_xyz"][1]) > .009
    assert proof["shared_raw_component_fractions"] == [1, 1]
    assert value["thickness_m"] is None and value["graph_requires_rebuild"] is True


@pytest.mark.parametrize("with_endcap", [False, True])
def test_two_same_facing_parallel_sheets_are_not_one_surface_even_when_mesh_connected(with_endcap):
    # A thin perpendicular endcap connects the raw mesh globally. It must not
    # connect the local same-directed-normal wall-face graph.
    raw = np.concatenate([face(), face(z=.02)])
    if with_endcap:
        vertices = np.array([[0, 0, 0], [0, 0, .02], [0, 2.4, .02], [0, 2.4, 0]])
        raw = np.concatenate([raw, vertices[[[0, 1, 2], [0, 2, 3]]]])
    source = np.array([[[0, z], [3, z]] for z in [0] * 3 + [.02] * 3])
    result = run([line("a", 0, [0, 1, 2]), line("b", .02, [3, 4, 5])], raw, source,
                 np.array([.5, 1.3, 2.3] * 2))
    assert not result["accepted"] and len(result["primitives"]) == 2
    assert result["rejected"][0]["reason"] == "independent_or_disconnected_raw_surfaces"


def test_parallel_sheets_cannot_connect_through_a_horizontal_header_or_far_corner():
    source = np.array([[[0, z], [3, z]] for z in [0] * 3 + [.02] * 3])
    items = [line("a", 0, [0, 1, 2]), line("b", .02, [3, 4, 5])]
    heights = np.array([.5, 1.3, 2.3] * 2)
    # A far endcap lies outside the accepted old span. The global mesh is
    # connected, but no permitted local same-facing path connects these fits.
    cap = np.array([[5, 0, 0], [5, 0, .02], [5, 2.4, .02], [5, 2.4, 0]])
    far = np.concatenate([face(end=5), face(z=.02, end=5), cap[[[0, 1, 2], [0, 2, 3]]]])
    assert not run(items, far, source, heights)["accepted"]
    pieces = []
    for z in [0., .02]:
        for bottom, top in [(0., 1.7), (1.7, 2.4)]:
            vertices = np.array([[0, bottom, z], [3, bottom, z], [3, top, z], [0, top, z]])
            pieces.append(vertices[[[0, 1, 2], [0, 2, 3]]])
    header = np.array([[0, 1.7, 0], [3, 1.7, 0], [3, 1.7, .02], [0, 1.7, .02]])
    raw = np.concatenate(pieces + [header[[[0, 1, 2], [0, 2, 3]]]])
    assert not run(items, raw, source, heights)["accepted"]


def test_shallow_measured_relief_can_be_simplified_but_is_not_claimed_exactly_planar():
    # One continuous wall has a 2cm shallow change over 4cm of along span.
    # Its connector is within 45deg and the bounded 5cm plane envelope.
    first, last = face(end=1), face(z=.02, start=1.04)
    vertices = np.array([[1, 0, 0], [1.04, 0, .02], [1.04, 2.4, .02], [1, 2.4, 0]])
    connector = vertices[[[0, 1, 2], [0, 2, 3]]]
    raw = np.concatenate([first, connector, last])
    source = np.array([[[0, 0], [1, 0]]] * 3 + [[[1.04, .02], [3, .02]]] * 3)
    items = [line("a", 0, [0, 1, 2], end=2), line("b", .02, [3, 4, 5], start=.5)]
    result = run(items, raw, source, np.array([.5, 1.3, 2.3] * 2))
    assert len(result["accepted"]) == 1
    assert result["accepted"][0]["acceptance"] == "bounded_connected_raw_3d_surface_simplification"
    assert result["accepted"][0]["max_raw_plane_residual_m"] > 0
    reordered = run(items, raw[::-1], source, np.array([.5, 1.3, 2.3] * 2))
    assert len(reordered["accepted"]) == 1
    assert np.allclose(reordered["primitives"][0]["endpoints"], result["primitives"][0]["endpoints"])


def test_opposite_normals_of_real_thin_wall_cannot_be_erased_by_small_plane_residual():
    raw = np.concatenate([face(), face(z=.02, sign=-1)])
    source = np.array([[[0, z], [3, z]] for z in [0] * 3 + [.02] * 3])
    result = run([line("a", 0, [0, 1, 2]), line("b", .02, [3, 4, 5])], raw, source,
                 np.array([.5, 1.3, 2.3] * 2))
    assert not result["accepted"]
    assert result["rejected"][0]["reason"] == "raw_plane_or_directed_normal_not_bounded"


def test_source_registration_is_independent_of_semantic_fit_and_triangle_order():
    items, source = tilted_case()
    good = run(items, face(lean=.01)[::-1], source)
    assert len(good["accepted"]) == 1
    bad = run(items, face(z=.03, lean=.01), source)
    assert not bad["accepted"]
    assert bad["rejected"][0]["reason"] == "source_cuts_do_not_register_to_raw_geometry"


def test_removed_source_extent_does_not_reenter_after_door_carving():
    items, source = tilted_case()
    source[:, 1, 0] = 7  # Original source IDs retain cuts beyond accepted ends.
    result = run(items, face(lean=.01), source)
    assert len(result["accepted"]) == 1
    assert sorted([result["primitives"][0]["start"][0], result["primitives"][0]["end"][0]]) == pytest.approx([0, 3])


def test_successful_3d_refits_on_each_side_of_opening_keep_the_real_gap():
    items = [line("left-a", .005, [0, 1, 2], 0, 1), line("left-b", .023, [2, 3, 4], 0, 1),
             line("right-a", .005, [0, 1, 2], 2, 3), line("right-b", .023, [2, 3, 4], 2, 3)]
    source = np.array([[[0, .01 * h], [3, .01 * h]] for h in HEIGHTS])
    raw = np.concatenate([face(lean=.01, start=0, end=1), face(lean=.01, start=2, end=3)])
    result = run(items, raw, source)
    assert len(result["accepted"]) == len(result["primitives"]) == 2
    coverage = sorted(sorted([p["start"][0], p["end"][0]]) for p in result["primitives"])
    assert np.allclose(coverage, [[0, 1], [2, 3]])


@pytest.mark.parametrize("nav", [None, lambda _: {"pass": False}])
def test_measured_3d_plane_never_bypasses_navmesh(nav):
    items, source = tilted_case()
    result = run(items, face(lean=.01), source, nav=nav)
    assert result["primitives"] == items and not result["accepted"]
    assert result["rejected"][0]["reason"] in {"navmesh_evidence_required", "projected_wall_blocks_navmesh"}


def test_projection_cannot_close_a_known_opening_even_if_navmesh_check_is_clear():
    items, source = tilted_case()
    result = refit_wall_planes(items, face(lean=.01), source, HEIGHTS, floor_height=0.,
                              not_blocks_navmesh=clear,
                              portals=[{"segment": [[1.2, .012], [1.8, .012]], "status": "accepted"}])
    assert not result["accepted"]
    assert result["rejected"][0]["reason"] == "projected_wall_crosses_protected_opening"


def test_sparse_height_or_missing_source_evidence_stays_unmodified():
    items, source = tilted_case()
    items[0]["source_segment_indices"] = []
    assert not run(items, face(lean=.01), source)["accepted"]
    items, source = tilted_case()
    result = run(items, face(lean=.01), source, np.full(5, 1.3))
    assert not result["accepted"]
    assert result["rejected"][0]["reason"] == "insufficient_measured_height_extent"
