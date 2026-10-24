import json

from src.data.map_structure_audit import audit_fitted, near_endpoint_clusters


def test_near_endpoint_clusters_are_reported_without_merging_geometry():
    primitives = [
        {"id": "a", "type": "line", "start": [0, 0], "end": [1, 0]},
        {"id": "b", "type": "line", "start": [.02, .01], "end": [0, 1]},
    ]
    clusters = near_endpoint_clusters(primitives)
    assert len(clusters) == 1
    assert len(clusters[0]["endpoints"]) == 2
    assert primitives[0]["start"] == [0, 0]


def test_offset_overlapping_lines_are_not_hidden_by_a_zero_exact_overlap_count(tmp_path):
    primitives = [
        {"id": "long", "type": "line", "start": [0, 0], "end": [3, 0]},
        {"id": "copy", "type": "line", "start": [1, .02], "end": [2, .02]},
    ]
    for name, value in {"wall_geometry": {"primitives": primitives}, "meta": {"scene": "fixture"},
                        "openings": [], "regions": []}.items():
        (tmp_path / (name + ".json")).write_text(json.dumps(value))
    checked = audit_fitted(tmp_path)
    assert checked["status"] == "pending_structural_review"
    assert checked["counts"]["overlapping_wall_spans"] == 1
    assert checked["wall_coverage"]["exact_overlap_count"] == 0
    assert checked["wall_coverage"]["near_line_overlap_count"] == 1


def test_stored_failed_curve_fit_remains_pending_even_without_exact_overlap(tmp_path):
    for name, value in {"wall_geometry": {"primitives": [], "cad": {"qualified": False,
                           "unresolved_near_circular_overlap_count": 1}},
                        "meta": {}, "openings": [], "regions": []}.items():
        (tmp_path / (name + ".json")).write_text(json.dumps(value))
    checked = audit_fitted(tmp_path)
    assert checked["wall_coverage"]["qualified"]
    assert checked["status"] == "pending_structural_review"


def test_perpendicular_opening_at_the_same_center_does_not_explain_a_wall_gap(tmp_path):
    primitives = [
        {"id": "lower", "type": "line", "start": [0, 0], "end": [0, 1]},
        {"id": "upper", "type": "line", "start": [0, 2], "end": [0, 3]},
    ]
    values = {"wall_geometry": {"primitives": primitives}, "meta": {}, "regions": [],
              "openings": [{"id": "wrong", "status": "accepted", "segment": [[-.5, 1.5], [.5, 1.5]]}]}
    for name, value in values.items():
        (tmp_path / (name + ".json")).write_text(json.dumps(value))
    checked = audit_fitted(tmp_path)
    assert checked["counts"]["unresolved_gaps"] == 1
    assert checked["gaps"][0]["opening_record"] is None
