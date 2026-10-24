import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from src.data.map_gap_audit import (audit_scene, line_gap_candidates, clear_gap_interval,
                                    same_opening, passed_probe_count, merge_audited_openings)


def line(identifier, start, end):
    return {"id": identifier, "type": "line", "start": start, "end": end}


def test_finds_only_a_collinear_endpoint_gap():
    result = line_gap_candidates([
        line("left", [0, 0], [2, 0]),
        line("right", [3, 0], [5, 0]),
        line("parallel", [0, .3], [2, .3]),
    ])
    assert len(result) == 1
    assert result[0]["primitives"] == ["left", "right"]
    assert abs(result[0]["width_m"] - 1) < 1e-8


def test_does_not_bridge_a_large_gap_or_a_perpendicular_corner():
    result = line_gap_candidates([
        line("large", [0, 0], [1, 0]), line("far", [3, 0], [4, 0]),
        line("vertical", [2, 0], [2, 2]),
    ])
    assert result == []


def test_scattered_clear_samples_do_not_claim_one_large_opening():
    class Evidence:
        trees = {.3: None, .6: None, .9: None, 1.2: None, 1.5: None}

        def profile(self, points):
            blocked = (np.arange(len(points)) // 4) % 2 == 0
            return np.tile(blocked[:, None], (1, 5))

    result = clear_gap_interval([[0, 0], [1, 0]], Evidence())
    assert result["raw_clear_body_fraction"] > .4
    assert result["clear_width_m"] < .15


def test_duplicate_openings_require_same_plane_and_overlap():
    portal = {"segment": [[0, 0], [1, 0]]}
    assert same_opening(portal, {"segment": [[.1, 0], [.9, 0]]})
    assert not same_opening(portal, {"segment": [[1.2, 0], [2.2, 0]]})
    assert not same_opening(portal, {"segment": [[0, 1], [1, 1]]})


def test_compressed_depth_preserves_values_and_updates_reference(tmp_path):
    from src.data.map_gap_audit import _compact_view
    depth = np.arange(12, dtype=np.float32).reshape(3, 4) / 7
    path = tmp_path / 'view.npy'
    np.save(path, depth)
    result = _compact_view({"depth": str(path), "rgb": "view.jpg"})
    assert not path.exists()
    assert result["depth"].endswith('.npz')
    np.testing.assert_array_equal(np.load(result["depth"])["depth"], depth)


def passage_at(segment, *, passed=True):
    a, b = np.asarray(segment, dtype=float)
    tangent = b - a
    normal = np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
    probes = []
    for fraction in (.35, .65):
        center = a + fraction * tangent
        start, end = [center + sign * .45 * normal for sign in (-1, 1)]
        probes.append({"start": [start[0], 0., start[1]],
                       "requested_end": [end[0], 0., end[1]],
                       "actual_end": [end[0], 0., end[1]], "pass": passed})
    return {"pass": passed, "probes": probes}


def test_passed_probes_must_be_distinct_complete_crossings_of_the_clear_span():
    segment = [[0., 0.], [1., 0.]]
    passage = passage_at(segment)
    assert passed_probe_count(passage, segment) == 2
    passage["probes"] = [passage["probes"][0]] * 3
    assert passed_probe_count(passage, segment) == 1
    assert passed_probe_count({"pass": True, "probes": [{"pass": True}] * 2}, segment) == 0
    assert passed_probe_count(passage_at([[2., 0.], [3., 0.]]), segment) == 0
    passage = passage_at(segment)
    passage["probes"][1]["actual_end"][2] += .2
    assert passed_probe_count(passage, segment) == 1
    passage["pass"] = False
    assert passed_probe_count(passage, segment) == 0


def accepted_gap(identifier="G001", segment=None):
    segment = segment or [[0., 0.], [1., 0.]]
    return {"id": identifier, "status": "accepted", "segment": segment, "width_m": 1.,
            "evidence": {"clear_segment": [[.2, 0.], [.8, 0.]], "clear_width_m": .6,
                         "passage": passage_at([[.2, 0.], [.8, 0.]]), "passed_probe_count": 2}}


def test_duplicate_keeps_one_opening_and_preserves_each_audit_evidence():
    existing = [{"id": "O016", "status": "accepted", "segment": [[0., 0.], [1., 0.]],
                 "evidence": {"reason": "prior_actual_probe"}}]
    original = deepcopy(existing)
    audited = [accepted_gap(), accepted_gap("G002")]
    merged, added = merge_audited_openings(existing, audited, "run/gap_audit.json")
    assert added == 0
    assert len(merged) == 1
    assert existing == original
    assert merged[0]["evidence"] == original[0]["evidence"]
    assert [item["candidate_id"] for item in merged[0]["gap_audit_evidence"]] == ["G001", "G002"]
    assert all(item["duplicate_of"] == "O016" for item in audited)
    assert merged[0]["segment"] == [[.2, 0.], [.8, 0.]]
    assert merged[0]["initial_geometry"]["segment"] == original[0]["segment"]


def test_duplicate_reanchors_to_measured_cad_axis_not_the_old_surface_plane():
    prior = {"id": "door", "status": "accepted", "segment": [[0, .08], [1, .08]],
             "width_m": 1., "evidence": {"kind": "original_surface_probe"}}
    merged, added = merge_audited_openings([prior], [accepted_gap()], "cad/audit.json")
    assert added == 0 and merged[0]["id"] == "door"
    assert merged[0]["segment"] == [[.2, 0.], [.8, 0.]]
    assert merged[0]["initial_geometry"] == {"segment": prior["segment"], "width_m": 1.}
    assert merged[0]["evidence"] == prior["evidence"]
    assert merged[0]["gap_audit_evidence"][0]["path"] == "cad/audit.json"


def test_publish_measured_clear_span_and_allocate_unused_id_without_dropping_reviews():
    prior = {"id": "A001", "status": "needs_review", "segment": [[3., 0.], [4., 0.]]}
    gap = accepted_gap()
    merged, added = merge_audited_openings([prior], [gap], "run/gap_audit.json")
    assert added == 1
    assert merged[0] == prior
    assert merged[1]["id"] == "A002"
    assert merged[1]["segment"] == [[.2, 0.], [.8, 0.]]
    assert merged[1]["candidate_segment"] == [[0., 0.], [1., 0.]]
    assert merged[1]["width_m"] == .6


def _audit_fixture(tmp_path, monkeypatch, *, blocked_heights=0, passed=True, repeated_probe=False):
    import src.data.map_gap_audit as module
    source, fitted, output = [tmp_path / name for name in ("source", "fitted", "audit")]
    (source / "raw_vote").mkdir(parents=True)
    fitted.mkdir()
    baseline = {"scene": "fixture.glb", "floor_height": 0., "origin": [0., 0.], "side_m": 5.}
    (source / "raw_vote/meta.json").write_text(json.dumps(baseline))
    walls = np.zeros((100, 100), dtype=bool)
    walls[:40, 50] = walls[60:, 50] = True
    Image.fromarray(np.full((100, 100), 255, dtype=np.uint8)).save(source / "navmesh.png")
    Image.fromarray(np.where(walls, 0, 255).astype(np.uint8)).save(fitted / "walls.png")
    primitives = [line("left", [2.5, 0.], [2.5, 2.]), line("right", [2.5, 3.], [2.5, 5.])]
    (fitted / "wall_geometry.json").write_text(json.dumps({"primitives": primitives}))
    (fitted / "openings.json").write_text("[]")
    (fitted / "regions.json").write_text("[]")
    (fitted / "meta.json").write_text("{}")

    class Evidence:
        trees = {.3: None, .6: None, .9: None, 1.2: None, 1.5: None}

        def __init__(self, *args):
            pass

        def profile(self, points):
            profile = np.zeros((len(points), 5), dtype=bool)
            profile[:, :blocked_heights] = True
            return profile

    class Scene:
        def __init__(self, *args, **kwargs):
            self.sim = SimpleNamespace(pathfinder=SimpleNamespace(
                snap_point=lambda point: np.full(3, np.nan)))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def passage(self, segment, floor):
            result = passage_at(segment, passed=passed)
            if repeated_probe:
                result["probes"] = [result["probes"][0]] * 3
            return result

    monkeypatch.setattr(module, "MapScene", Scene)
    monkeypatch.setattr(module, "RawWallEvidence", Evidence)
    monkeypatch.setattr(module, "load_scene_geometry", lambda *args, **kwargs: SimpleNamespace(triangles=[]))
    monkeypatch.setattr(module, "floor_domain", lambda *args: (np.ones_like(walls), {}))
    return source, fitted, output


def test_partitioned_region_references_are_the_published_openings(tmp_path, monkeypatch):
    source, fitted, output = _audit_fixture(tmp_path, monkeypatch)
    prior = [{"id": "review", "status": "needs_review", "segment": [[0., 0.], [1., 0.]]}]
    (fitted / "openings.json").write_text(json.dumps(prior))
    audited = audit_scene(source, fitted, output)
    openings = json.loads((fitted / "openings.json").read_text())
    regions = json.loads((fitted / "regions.json").read_text())
    portal = next(item for item in openings if item["status"] == "accepted")
    assert portal["partition_status"] == "split"
    assert len(set(portal["regions"])) == 2
    assert set(portal["regions"]) <= {item["id"] for item in regions}
    assert next(item for item in openings if item["id"] == "review") == prior[0]
    assert audited[0]["opening_id"] == portal["id"]
    assert json.loads((output / "gap_audit.json").read_text())["accepted_count"] == 1
    assert json.loads((fitted / "meta.json").read_text())["counts"]["audited_openings"] == 1


def test_duplicate_scene_audit_has_consistent_zero_new_count_and_updated_regions(tmp_path, monkeypatch):
    source, fitted, output = _audit_fixture(tmp_path, monkeypatch)
    prior = [{"id": "O016", "status": "accepted", "segment": [[2.5, 2.], [2.5, 3.]],
              "regions": ["stale", "stale"], "partition_status": "unassigned",
              "evidence": {"reason": "prior_actual_probe"}}]
    (fitted / "openings.json").write_text(json.dumps(prior))
    audit_scene(source, fitted, output)
    audit = json.loads((output / "gap_audit.json").read_text())
    meta = json.loads((fitted / "meta.json").read_text())["gap_audit"]
    openings = json.loads((fitted / "openings.json").read_text())
    assert audit["accepted_count"] == meta["accepted_count"] == 0
    assert audit["accepted_total_count"] == meta["accepted_total_count"] == 1
    assert len(openings) == 1
    assert openings[0]["id"] == "O016"
    assert openings[0]["partition_status"] == "split"
    assert "stale" not in openings[0]["regions"]
    assert openings[0]["gap_audit_evidence"][0]["candidate_id"] == "G001"
    assert json.loads((fitted / "openings.before_gap_audit.json").read_text()) == prior


@pytest.mark.parametrize("blocked_heights,passed,repeated_probe", [
    (2, True, False),  # Window sill/frame blocks body cuts despite upper clearance.
    (0, False, False),  # Raw air does not establish a walkable opening.
    (0, True, True),  # One successful probe copied three times is not two probes.
])
def test_window_or_nonwalkable_gap_is_not_published_as_a_door(
        tmp_path, monkeypatch, blocked_heights, passed, repeated_probe):
    source, fitted, output = _audit_fixture(tmp_path, monkeypatch, blocked_heights=blocked_heights,
                                           passed=passed, repeated_probe=repeated_probe)
    audited = audit_scene(source, fitted, output)
    assert audited[0]["status"] == "needs_review"
    assert json.loads((fitted / "openings.json").read_text()) == []
