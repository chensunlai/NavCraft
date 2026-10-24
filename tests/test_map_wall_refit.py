from copy import deepcopy

import numpy as np
import pytest

from src.data.map_wall_refit import refit_overlapping_lines


def line(identifier, y=0., start=0., end=2., sources=None):
    value = {"id": identifier, "type": "line", "start": [start, y], "end": [end, y]}
    if sources is not None:
        value["source_segment_indices"] = sources
    return value


def same_surface(first, second):
    return {"pass": True, "source": "independent fixture measured cuts", "ids": [first["id"], second["id"]]}


def clear(_):
    return {"pass": True, "source": "fixture measured NavMesh"}


def run(items, sources, **kwargs):
    return refit_overlapping_lines(items, sources, same_surface=kwargs.pop("same_surface", same_surface),
                                   not_blocks_navmesh=kwargs.pop("not_blocks_navmesh", clear), **kwargs)


def test_length_weighted_joint_original_fit_and_provenance_without_mutating_input():
    items = [line("long", sources=[0]), line("short", .02, .5, 1.5, [1])]
    items[0].update(start_node="old-start", end_node="old-end", cad={"node_projections": []})
    raw = [[[0, 0], [2, 0]], [[.5, .02], [1.5, .02]]]
    before = deepcopy(items)
    result = run(items, raw)
    assert len(result["primitives"]) == len(result["accepted"]) == 1
    fitted = result["primitives"][0]
    assert fitted["start"] == pytest.approx([0, .02 / 3])
    assert fitted["end"] == pytest.approx([2, .02 / 3])
    assert fitted["fitting"]["max_residual_m"] == pytest.approx(.04 / 3)
    assert fitted["source_segment_indices"] == [0, 1]
    assert fitted["source_primitive_ids"] == ["long", "short"]
    assert fitted["joint_refit"]["original_primitives"] == before
    assert fitted["joint_refit"]["same_surface_check"]["pass"] is True
    assert fitted["joint_refit"]["navmesh_checks"][0]["pass"] is True
    assert fitted["graph_requires_rebuild"] is True
    assert not {"start_node", "end_node", "cad"} & fitted.keys()
    assert items == before


@pytest.mark.parametrize("sources", [None, [], [2], [-1], [True]])
def test_missing_or_invalid_sources_cannot_be_replaced_by_a_surface_claim(sources):
    def unexpected(*_):
        pytest.fail("No independent evidence should be requested without original sources")

    result = run([line("a", sources=[0]), line("b", .01, sources=sources)],
                 [[[0, 0], [2, 0]]], same_surface=unexpected)
    assert len(result["primitives"]) == 2 and not result["accepted"]
    assert result["rejected"][0]["reason"] == "missing_or_invalid_original_source_indices"


@pytest.mark.parametrize("callback", [None, lambda *_: {"pass": False, "source": "different raw sheets"}])
def test_independent_raw_same_surface_check_is_mandatory(callback):
    items = [line("a", sources=[0]), line("b", .01, sources=[0])]
    result = run(items, [[[0, 0], [2, 0]]], same_surface=callback)
    assert result["primitives"] == items and not result["accepted"]
    assert "same_surface" in result["rejected"][0]["reason"]


@pytest.mark.parametrize("callback", [None, lambda _: {"pass": False, "source": "walkable cells"}])
def test_joint_fit_cannot_bypass_missing_or_failed_navmesh(callback):
    items = [line("a", sources=[0]), line("b", .01, sources=[0])]
    result = run(items, [[[0, 0], [2, 0]]], not_blocks_navmesh=callback)
    assert result["primitives"] == items and not result["accepted"]
    assert "navmesh" in result["rejected"][0]["reason"]


@pytest.mark.parametrize("raw", [
    [[[0, -.08], [2, -.08]], [[0, .08], [2, .08]]],  # raw residual, not mean residual
    [[[0, .10], [2, .10]], [[0, .10], [2, .10]]],   # no raw residual, excessive old-fit shift
])
def test_raw_maximum_residual_and_each_old_line_movement_are_bounded(raw):
    result = run([line("a", sources=[0]), line("b", .01, sources=[1])], raw)
    assert not result["accepted"] and len(result["primitives"]) == 2
    assert result["rejected"][0]["reason"] == "raw_residual_or_old_geometry_displacement_exceeded"


def test_source_extent_does_not_extend_old_coverage_and_door_gap_does_not_merge():
    # Raw source cuts extend beyond the accepted portions, e.g. after carving.
    items = [line("a", 0, 0, 1.5, [0]), line("b", .01, 1, 2, [1])]
    result = run(items, [[[0, .005], [3, .005]], [[0, .005], [3, .005]]])
    axis = result["primitives"][0]
    assert axis["start"] == pytest.approx([0, .005])
    assert axis["end"] == pytest.approx([2, .005])
    assert axis["joint_refit"]["coverage_scope"].startswith("union_of_projected_old_segments")
    disconnected = [line("left", 0, 0, 1, [0]), line("right", .01, 2, 3, [1])]
    result = run(disconnected, [[[0, .005], [3, .005]], [[0, .005], [3, .005]]])
    assert result["primitives"] == disconnected and not result["accepted"]


def test_successful_refits_on_both_sides_of_door_preserve_the_intervening_gap():
    items = [line("left-a", 0, 0, 1, [0]), line("left-b", .01, .1, 1.1, [1]),
             line("right-a", 0, 2, 3, [2]), line("right-b", .01, 2.1, 3.1, [3])]
    # Even source cuts running across the opening may only determine the fit,
    # never its finite coverage. Coverage comes from the four accepted pieces.
    raw = [[[0, .005], [4, .005]]] * 4
    result = run(items, raw)
    assert len(result["accepted"]) == len(result["primitives"]) == 2
    intervals = sorted((value["start"][0], value["end"][0]) for value in result["primitives"])
    assert np.allclose(intervals, [[0, 1.1], [2, 3.1]])


def test_chain_refit_checks_all_original_lines_and_keeps_prior_raw_evidence():
    items = [line("a", 0, sources=[0]), line("b", .01, sources=[1]), line("c", .02, sources=[2])]
    raw = [[[0, y], [2, y]] for y in [0, .01, .02]]
    result = run(items, raw)
    assert len(result["primitives"]) == 1
    fitted = result["primitives"][0]
    assert fitted["source_segment_indices"] == [0, 1, 2]
    assert len(fitted["joint_refit"]["same_surface_checks"]) == 2
    assert set(fitted["fitting"]["original_line_displacements_m"]) == {"a", "b", "c"}
    assert fitted["fitting"]["max_original_line_displacement_m"] <= .03


def test_curves_corners_and_wall_axes_are_not_refitted_as_surface_duplicates():
    arc = {"id": "arc", "type": "arc", "center": [0, 0], "radius_m": 1, "sweep_degrees": 90}
    corner = {"id": "corner", "type": "line", "start": [2, 0], "end": [2, 2], "source_segment_indices": [0]}
    items = [line("a", sources=[0]), corner, arc]
    result = run(items, [[[0, 0], [2, 0]]])
    assert result["primitives"] == items and not result["accepted"]
    items = [line("a", sources=[0]), {**line("axis", .01, sources=[0]), "representation": "wall_axis"}]
    result = run(items, [[[0, 0], [2, 0]]])
    assert result["primitives"] == items
    assert result["rejected"][0]["reason"] == "wall_axes_are_not_raw_surface_fits"


def test_source_ids_are_deduplicated_and_zero_length_sources_cannot_support_fit():
    items = [line("a", sources=[0]), line("b", .01, sources=[0, 0])]
    fitted = run(items, [[[0, .005], [2, .005]]])["primitives"][0]
    assert fitted["source_segment_indices"] == [0] and fitted["source_segment_count"] == 1
    result = run(items, [[[0, 0], [0, 0]]])
    assert not result["accepted"]
    assert result["rejected"][0]["reason"] == "degenerate_original_source_segment"


def test_callbacks_must_return_explicit_boolean_and_cannot_mutate_original_inputs():
    items = [line("a", sources=[0]), line("b", .01, sources=[0])]
    with pytest.raises(ValueError, match="explicit boolean"):
        run(items, [[[0, .005], [2, .005]]], same_surface=lambda *_: {"pass": 1})

    def mutating(first, _):
        first["start"][1] = 900
        return {"pass": True, "source": "fixture raw evidence"}

    result = run(items, [[[0, .005], [2, .005]]], same_surface=mutating)
    assert result["primitives"][0]["joint_refit"]["original_primitives"][0]["start"] == [0, 0]
    assert items[0]["start"] == [0, 0]
