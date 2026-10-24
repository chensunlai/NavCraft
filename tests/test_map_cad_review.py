import numpy as np
import pytest
from PIL import Image
import hashlib
import json

from src.data.map_cad_review import review_point_tasks, validate_jamb_points, wall_pair_reviewer


def evidence(tmp_path):
    rgb, depth = tmp_path / "rgb.png", tmp_path / "depth.npz"
    Image.new("RGB", (101, 101), "white").save(rgb)
    np.savez_compressed(depth, depth=np.full((101, 101), 2., dtype=np.float32))
    pose = np.diag([1., -1., -1., 1.])
    pose[1, 3] = 1.
    return {"rgb": str(rgb), "depth": str(depth),
            "intrinsics": [[100., 0, 50], [0, 100., 50], [0, 0, 1]],
            "world_from_optical": pose.tolist()}


def answer():
    return {"visibility": "complete", "points": [
        {"label": "left", "xy_normalized": [.25, 1.]},
        {"label": "right", "xy_normalized": [.75, 1.]}], "reason": "Visible ground jambs"}


def test_ground_jambs_use_original_resolution_and_compressed_depth(tmp_path):
    result = validate_jamb_points(answer(), evidence(tmp_path), [[-.5, -2], [.5, -2]], 0)
    assert result["pass"]
    np.testing.assert_allclose([point["world_xyz"] for point in result["points"]],
                               [[-.5, 0, -2], [.5, 0, -2]])
    assert result["maps_modified"] is False


def test_points_on_another_door_or_above_ground_are_not_accepted(tmp_path):
    view = evidence(tmp_path)
    wrong_door = validate_jamb_points(answer(), view, [[4.5, -2], [5.5, -2]], 0)
    assert not wrong_door["pass"] and not wrong_door["near_target"]
    proposed = answer()
    for point in proposed["points"]:
        point["xy_normalized"][1] = .5
    high = validate_jamb_points(proposed, view, [[-.5, -2], [.5, -2]], 0)
    assert not high["pass"] and not high["grounded"]


def test_occluded_missing_or_duplicate_jambs_cannot_become_a_full_span(tmp_path):
    view = evidence(tmp_path)
    proposed = answer()
    proposed["visibility"] = "partial"
    assert not validate_jamb_points(proposed, view, [[-.5, -2], [.5, -2]], 0)["pass"]
    proposed = answer()
    proposed["points"][1]["label"] = "left"
    assert validate_jamb_points(proposed, view, [[-.5, -2], [.5, -2]], 0)["reason"] == "duplicate_or_missing_jamb_label"


def test_outside_coordinates_are_rejected_without_guessing_a_resize(tmp_path):
    proposed = answer()
    proposed["points"][0]["xy_normalized"][0] = 1.4
    with pytest.raises(ValueError, match="within"):
        validate_jamb_points(proposed, evidence(tmp_path), [[-.5, -2], [.5, -2]], 0)


def test_batch_keeps_occluded_answers_and_geometry_failures_separate(tmp_path):
    class LocalClient:
        def request(self, prompt, images, **kwargs):
            proposed = answer()
            if kwargs["metadata"]["task_id"] == "occluded":
                proposed.update(visibility="not_visible", points=[])
            return {"ok": True, "parsed": proposed}

    view = evidence(tmp_path)
    tasks = [{"id": name, "hint": "The opening ahead", "view": view,
              "target_segment": [[-.5, -2], [.5, -2]], "floor_height": 0}
             for name in ["visible", "occluded"]]
    result = review_point_tasks(tasks, tmp_path / "review", client=LocalClient())
    assert result["valid_structured_answers"] == 2
    assert result["locally_consistent_proposals"] == 1
    assert result["maps_modified"] is False


def test_saved_same_wall_decision_cannot_apply_to_changed_geometry_or_wrong_backend():
    walls = [{"id": key, "type": "line", "start": [0, z], "end": [2, z]}
             for key, z in [("A", 0.), ("B", .2)]]
    digest = hashlib.sha256(json.dumps(walls, sort_keys=True).encode()).hexdigest()
    response = {"ok": True, "response_model": "Qwen3.8-27B", "record_path": "audited-call/record.json",
                "parsed": {"relation": "same_wall_faces", "visible_wall_evidence": True, "reason": "Visible wall"}}
    record = {"task_id": "wall-pair", "source_primitive_ids": ["A", "B"],
              "geometry_sha256": digest, "response": response}
    pair = {"source_primitive_ids": ["A", "B"]}
    assert wall_pair_reviewer([record], walls)(pair)["pass"]
    changed = [{**walls[0], "end": [3, 0.]}, walls[1]]
    assert not wall_pair_reviewer([record], changed)(pair)["pass"]
    response["response_model"] = "unverified-backend"
    assert not wall_pair_reviewer([record], walls)(pair)["pass"]
    with pytest.raises(ValueError, match="duplicate"):
        wall_pair_reviewer([record, record], walls)
