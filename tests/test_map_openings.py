"""Portals require independent geometry and cannot invent separate rooms."""

import numpy as np
import pytest

from src.data.map_openings import opening_candidates, region_partition, verify_openings


def slices(closed=False):
    side_walls = [[[0., 0.], [2., 0.]], [[3., 0.], [5., 0.]]]
    door = [[[2., 0.], [3., 0.]]] if closed else []
    result = {height: np.asarray(side_walls + door) for height in (.9, 1.3, 1.7)}
    result[2.] = np.asarray([[[0., 0.], [5., 0.]]])  # overhead lintel
    return result


def candidate():
    return {"id": "O001", "segment": [[2., 0.], [3., 0.]], "sources": [{"kind": "test_candidate"}]}


def test_raw_geometry_gaps_propose_openings_without_semantics():
    candidates = opening_candidates({}, slices(), 0.)
    assert len(candidates) == 1
    assert candidates[0]["width_m"] == pytest.approx(1.)
    np.testing.assert_allclose(candidates[0]["segment"], [[2., 0.], [3., 0.]])
    assert len(candidates[0]["sources"]) == 3


def test_semantic_closed_door_is_only_a_candidate():
    triangles = np.array([[[2., 0., 0.], [3., 0., 0.], [3., 2.2, 0.]],
                          [[2., 0., 0.], [3., 2.2, 0.], [2., 2.2, 0.]]])
    candidates = opening_candidates({"door": triangles}, {}, 0.)
    assert len(candidates) == 1
    assert candidates[0]["sources"][0]["category"] == "door"
    assert "status" not in candidates[0]
    checked = verify_openings(candidates, slices(closed=True), lambda _: {"passable": False})
    assert checked[0]["status"] == "rejected"


def test_real_opening_accepts_direct_crossing_despite_high_lintel():
    calls = []

    def passable(segment):
        calls.append(segment)
        return {"passable": True, "probe_start": [2.5, 0, -.4], "probe_end": [2.5, 0, .4],
                "source": "native_try_step_no_sliding"}

    checked = verify_openings([candidate()], slices(), passable)
    assert calls == [candidate()["segment"]]
    assert checked[0]["status"] == "accepted"
    assert checked[0]["evidence"]["clear_body_heights"] == 3
    assert checked[0]["evidence"]["raw_height_coverage"]["2.0"] == 1.
    assert checked[0]["evidence"]["navmesh"]["source"] == "native_try_step_no_sliding"


def test_raw_closed_door_overrides_a_claimed_navmesh_crossing():
    checked = verify_openings([candidate()], slices(closed=True), lambda _: {"passable": True})
    assert checked[0]["status"] == "rejected"
    assert checked[0]["evidence"]["blocked_body_heights"] == 3
    assert checked[0]["evidence"]["reason"] == "raw_surfaces_block_portal"


def test_window_or_scan_gap_cannot_pass_on_semantics_or_clear_slices_alone():
    checked = verify_openings([candidate()], slices(), lambda _: {"passable": False, "reason": "window_sill"})
    assert checked[0]["status"] == "rejected"
    assert checked[0]["evidence"]["reason"] == "no_direct_navmesh_crossing"


def test_missing_or_ambiguous_geometry_is_not_forced_to_pass():
    assert verify_openings([candidate()], {}, lambda _: {"passable": True})[0]["status"] == "needs_review"
    bad_callback = verify_openings([candidate()], slices(), lambda _: {"path_found": True})
    assert bad_callback[0]["status"] == "needs_review"
    assert "callback" in bad_callback[0]["evidence"]["navmesh"]["error"]


def test_verified_door_temporarily_closed_to_split_floor_regions():
    domain = np.ones((40, 60), dtype=bool)
    walls = np.zeros_like(domain)
    walls[:, 30] = True
    walls[15:25, 30] = False
    portal = {"id": "O001", "segment": [[3., 1.5], [3., 2.5]], "status": "accepted"}
    labels, regions, portals = region_partition(domain, walls, [portal], [0., 0.], .1)
    assert len(regions) == 2
    assert labels[20, 15] != labels[20, 45]
    assert portals[0]["partition_status"] == "split"
    assert len(set(portals[0]["regions"])) == 2
    assert sum(region["area_m2"] for region in regions) == pytest.approx(23.6)
    assert all(region["source"] == "floor_domain_partitioned_by_walls_and_verified_portals" for region in regions)
    assert all(region["status"] == "geometry_region_candidate" and region["minimum_area_m2"] == .1
               for region in regions)


def test_open_plan_bypass_does_not_invent_two_rooms():
    domain = np.ones((40, 60), dtype=bool)
    walls = np.zeros_like(domain)
    walls[10:30, 30] = True
    walls[15:25, 30] = False
    portal = {"id": "O001", "segment": [[3., 1.5], [3., 2.5]], "status": "accepted"}
    _, regions, portals = region_partition(domain, walls, [portal], [0., 0.], .1)
    assert len(regions) == 1
    assert portals[0]["partition_status"] == "open_plan_or_bypass"
    assert portals[0]["regions"] == ["R001", "R001"]


def test_unverified_semantic_candidate_is_not_used_to_cut_region():
    domain = np.ones((40, 60), dtype=bool)
    portal = {"id": "O001", "segment": [[3., 0.], [3., 4.]], "status": "needs_review"}
    _, regions, portals = region_partition(domain, np.zeros_like(domain), [portal], [0., 0.], .1)
    assert len(regions) == 1
    assert portals == []


def test_tiny_fragments_are_removed_without_merging_and_ids_are_continuous():
    domain = np.zeros((30, 30), dtype=bool)
    domain[1, 1] = True  # .01m², encountered before either real region
    domain[3:8, 3:8] = True
    domain[10:15, 10:15] = True
    domain[20:22, 20:22] = True  # .04m²
    labels, regions, _ = region_partition(domain, np.zeros_like(domain), [], [0, 0], .1)
    assert [region["id"] for region in regions] == ["R001", "R002"]
    assert sorted(np.unique(labels)) == [0, 1, 2]
    assert labels[1, 1] == labels[20, 20] == 0
    assert labels[4, 4] == 1 and labels[11, 11] == 2
    assert sum(region["area_m2"] for region in regions) == pytest.approx(.5)
    # Fixture callers can deliberately keep small components, without changing
    # the production default or claiming tiny fragments are actual rooms.
    _, unfiltered, _ = region_partition(domain, np.zeros_like(domain), [], [0, 0], .1, min_area_m2=0)
    assert len(unfiltered) == 4


def test_filtered_portal_side_stays_unknown_instead_of_jumping_to_far_region():
    domain = np.zeros((30, 30), dtype=bool)
    domain[5:25, 2:10] = True
    domain[14:16, 12:15] = True  # .06m² on the door's right side
    domain[5:25, 17:25] = True  # further right but separated by absent floor
    portal = {"id": "O001", "segment": [[1.1, 1.4], [1.1, 1.6]], "status": "accepted"}
    _, regions, portals = region_partition(domain, np.zeros_like(domain), [portal], [0, 0], .1)
    assert len(regions) == 2
    assert portals[0]["partition_status"] == "unassigned"
    assert set(portals[0]["regions"]) == {"R001", None}


def test_two_real_rooms_and_portal_ids_survive_earlier_tiny_fragment():
    domain = np.zeros((50, 70), dtype=bool)
    domain[1, 1] = True
    domain[5:45, 5:65] = True
    walls = np.zeros_like(domain)
    walls[5:45, 35] = True
    walls[20:30, 35] = False
    portal = {"id": "O007", "segment": [[3.5, 2.0], [3.5, 3.0]], "status": "accepted"}
    labels, regions, portals = region_partition(domain, walls, [portal], [0, 0], .1)
    assert [region["id"] for region in regions] == ["R001", "R002"]
    assert labels[1, 1] == 0
    assert portals[0]["id"] == "O007"
    assert portals[0]["partition_status"] == "split"
    assert set(portals[0]["regions"]) == {"R001", "R002"}


def test_invalid_masks_and_segments_are_reported():
    with pytest.raises(ValueError, match="matching 2D"):
        region_partition(np.ones((3, 3)), np.ones((4, 4)), [], [0., 0.], .05)
    with pytest.raises(ValueError, match="nonzero"):
        verify_openings([{"segment": [[0., 0.], [0., 0.]]}], {}, lambda _: {"passable": True})
    with pytest.raises(ValueError, match="min_area_m2"):
        region_partition(np.ones((3, 3)), np.zeros((3, 3)), [], [0, 0], .05, min_area_m2=-1)
