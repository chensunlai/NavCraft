import json

import numpy as np
import pytest

from src.bench.subtasks.scoring import direction_label, match_entities, score_sample


def scored(task, label, answer="", **outcome):
    return score_sample({"task_id": task}, label,
                        {"answer": answer, "submitted": True, "actions": 0, **outcome})


@pytest.fixture
def truth():
    instance = {"id": "chair-1", "category": "chair", "center": [5., .5, 5.], "size": [1., 1., 1.]}
    return {"instances": [instance], "target_instance": "chair-1",
            "transform": {"origin": [0., 0.], "size_m": 10.},
            "frame_visibility": {"f_0001": ["chair-1"]}}


def object_record(identifier="O01", **fields):
    return {"id": identifier, "type": "object", "category": "chair", "name": "chair",
            "point": [.5, .5], "description": "", "evidence": ["f_0001"], **fields}


def test_pixel_uses_actual_mask_and_bound_frame(tmp_path):
    mask = np.zeros((480, 640), dtype=np.uint16)
    mask[180:240, 250:310] = 42
    path = tmp_path / "semantic.npy"
    np.save(path, mask)
    label = {"frame": "f_0101", "mask_file": str(path), "instance_id": 42}
    assert scored("5a", label, '{"frame":"f_0101","pixel":[275,210]}')["success"]
    assert scored("5c", label, "```json\n[275,210]\n```")["success"]
    assert not scored("5a", label, 'f_0102 [275,210]')["success"]
    assert not scored("5a", label, "[400,200]")["success"]
    assert scored("5a", label, "[640,240]")["reason"] == "pixel_out_of_bounds"
    # A frame number or narrative number must never become a pixel coordinate.
    assert scored("5a", label, "In f_0101 it is near 275 and 210.")["reason"] == "format_error"
    assert scored("5a", label, "[true,210]")["reason"] == "format_error"


def test_absence_and_qa_immutability():
    label = {"answer_kind": "none"}
    assert scored("5a", label, "none")["success"]
    assert scored("5a", label, "null")["success"]
    assert not scored("5a", label, "[100,100]")["success"]
    assert scored("5a", label, "none", actions=1)["reason"] == "qa_state_changed"
    assert scored("5a", label, "none", submitted=False)["reason"] == "not_submitted"


def test_entity_correspondence_accepts_all_frozen_answers():
    label = {"accepted_ids": ["O01", "O35"]}
    assert scored("6a", label, "O01")["success"]
    assert scored("6a", label, '{"id":"O35"}')["success"]
    assert not scored("6a", label, "O99")["success"]
    assert scored("6a", {"accepted_ids": ["new"]}, "new")["success"]
    assert scored("6a", label, "object 1")["reason"] == "format_error"


@pytest.mark.parametrize("answer", ["yes", "true", '{"same_instance":true}', '"yes"'])
def test_same_instance_formats(answer):
    assert scored("6c", {"same_instance": True}, answer)["success"]
    assert not scored("6c", {"same_instance": False}, answer)["success"]


def test_same_instance_does_not_treat_truthy_text_as_yes():
    assert scored("6c", {"same_instance": False}, "no")["success"]
    assert scored("6c", {"same_instance": True}, "probably yes")["reason"] == "format_error"
    assert scored("6c", {"same_instance": True}, "1")["reason"] == "format_error"


def test_eight_way_direction_and_clockwise_sign():
    assert [direction_label(a) for a in range(0, 360, 45)] == [
        "front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left"]
    assert direction_label(-45) == "front-left"
    assert direction_label(22.499) == "front"
    assert direction_label(22.5) == "front-right"
    assert scored("13d", {"direction": "front-left"}, '{"direction":"front-left"}')["success"]
    assert scored("13d", {"direction": "front"}, "north")["reason"] == "format_error"


def test_surface_point_inside_large_footprint_is_valid(truth):
    truth["instances"][0]["size"] = [4., 1., 2.]
    entity = object_record("O88", point=[.69, .5])
    result = match_entities([entity], truth["instances"], truth["transform"], truth["frame_visibility"])
    assert result["bindings"] == {"O88": "chair-1"}
    assert not result["unmatched"]


def test_matching_respects_evidence_and_preserves_ambiguity(truth):
    entity = object_record(evidence=["f_9999"])
    result = match_entities([entity], truth["instances"], truth["transform"], truth["frame_visibility"])
    assert result["unmatched"] == ["O01"]
    second = {**truth["instances"][0], "id": "chair-2", "center": [5.5, .5, 5.]}
    result = match_entities([object_record()], [*truth["instances"], second], truth["transform"],
                            {"f_0001": ["chair-1", "chair-2"]})
    assert not result["bindings"]
    assert result["ambiguous"] == [{"id": "O01", "candidates": ["chair-1", "chair-2"]}]


def test_new_memory_accepts_arbitrary_id_and_requires_new_target(truth):
    assert scored("1b", truth, entities=[object_record("O99")])["success"]
    assert scored("1b", truth, entities=[])["reason"] == "missing_target_memory"
    truth["baseline_bindings"] = {"O01": "chair-1"}
    assert scored("1b", truth, entities=[object_record()])["reason"] == "target_not_created"


def test_existing_instance_must_not_be_duplicated(truth):
    truth["baseline_bindings"] = {"O01": "chair-1"}
    assert scored("1n", truth, entities=[object_record(description="A better, correct description")])["success"]
    result = scored("1n", truth, entities=[object_record(), object_record("O72")])
    assert result["reason"] == "duplicate_target_memory"
    assert result["metrics"]["target_records"] == 2


def test_unknown_new_memory_is_review_not_automatic_hallucination(truth):
    result = scored("1b", truth, entities=[object_record(), object_record("O02", category="lamp")])
    assert result["reason"] == "memory_requires_review"
    assert result["metrics"]["unverified_new"] == ["O02"]
    result = scored("1b", truth, entities=[object_record(), {"id": "R01", "type": "room"}])
    assert result["reason"] == "memory_requires_review"


def test_stacked_instances_use_current_surface_height_and_known_identity():
    instances = [{"id": "lower-shelf", "category": "shelf", "center": [5., 4.5, 5.], "size": [1., .4, 1.]},
                 {"id": "upper-shelf", "category": "shelf", "center": [5., 5.5, 5.], "size": [1., .4, 1.]}]
    transform = {"origin": [0., 0.], "size_m": 10., "floor_height": 4.}
    visibility = {"f_0001": ["lower-shelf", "upper-shelf"]}
    entity = object_record("O01", category="shelf")
    source = {"O01": {"uv": [.5, .5], "height": 1.5, "frame": "f_0001"}}
    result = match_entities([entity], instances, transform, visibility, location_sources=source)
    assert result["bindings"] == {"O01": "upper-shelf"}
    # Prior ID identity resolves an unchanged record, but never an arbitrary new ID.
    result = match_entities([entity, {**entity, "id": "O22"}], instances, transform, visibility,
                            baseline_bindings={"O01": "upper-shelf"})
    assert result["bindings"] == {"O01": "upper-shelf"}
    assert result["ambiguous"] == [{"id": "O22", "candidates": ["lower-shelf", "upper-shelf"]}]
    # Fresh contradictory geometry wins over the old binding.
    source["O01"]["height"] = .5
    result = match_entities([entity], instances, transform, visibility, location_sources=source,
                            baseline_bindings={"O01": "upper-shelf"})
    assert result["bindings"] == {"O01": "lower-shelf"}
    # A height from an older location cannot disambiguate the current point.
    source["O01"]["uv"] = [.6, .5]
    result = match_entities([entity], instances, transform, visibility, location_sources=source)
    assert not result["bindings"] and result["ambiguous"]
    source["O01"].update(uv=[.5, .5], frame="f_9999")
    result = match_entities([entity], instances, transform, visibility, location_sources=source)
    assert not result["bindings"] and result["ambiguous"]


def test_missing_or_duplicate_target_takes_precedence_over_unverified_room(truth):
    room = {"id": "R01", "type": "room"}
    assert scored("1b", truth, entities=[room])["reason"] == "missing_target_memory"
    result = scored("1b", truth, entities=[object_record(), object_record("O02"), room])
    assert result["reason"] == "duplicate_target_memory"
    assert scored("1b", truth, entities=[object_record(), room])["reason"] == "memory_requires_review"


def test_memory_scoring_uses_actual_height_without_requiring_new_id_to_match_baseline():
    truth = {"instances": [{"id": "lower", "category": "shelf", "center": [5., .5, 5.], "size": [1., .4, 1.]},
                            {"id": "upper", "category": "shelf", "center": [5., 1.5, 5.], "size": [1., .4, 1.]}],
             "target_instance": "lower", "transform": {"origin": [0., 0.], "size_m": 10., "floor_height": 0.},
             "frame_visibility": {"f_0001": ["lower", "upper"]}, "baseline_bindings": {"O01": "upper"}}
    original = object_record(category="shelf")
    new = object_record("O77", category="shelf")
    sources = {"O01": {"uv": [.5, .5], "height": 1.5, "frame": "f_0001"},
               "O77": {"uv": [.5, .5], "height": .5, "frame": "f_0001"}}
    assert scored("1b", truth, entities=[original, new], location_sources=sources)["success"]
    # Updating the existing upper record to the lower shelf is not creating it.
    sources["O01"]["height"] = .5
    assert scored("1b", truth, entities=[original], location_sources=sources)["reason"] == "target_not_created"


def test_baseline_preservation_compares_physical_coverage_not_record_ids(truth):
    truth["instances"].append({"id": "chair-2", "category": "chair", "center": [7., .5, 5.], "size": [1., 1., 1.]})
    truth["frame_visibility"]["f_0001"].append("chair-2")
    truth["baseline_bindings"] = {"O01": "chair-1", "O02": "chair-2"}
    # O02 still exists, but overwriting its position lost the old chair memory.
    result = scored("1n", truth, entities=[object_record(), object_record("O02", point=[.9, .9])])
    assert result["reason"] == "unrelated_memory_lost"
    assert result["metrics"]["missing_baseline"] == ["O02"]
    assert result["metrics"]["missing_baseline_instances"] == ["chair-2"]
    # A renamed or merged record preserves the physical chair even if O02 is gone.
    result = scored("1n", truth, entities=[object_record(), object_record("O77", point=[.7, .5])])
    assert result["success"]
    assert not result["metrics"]["missing_baseline"]
    assert not result["metrics"]["missing_baseline_instances"]
    # Several baseline IDs referring to one physical chair need only one record.
    truth["baseline_bindings"]["O03"] = "chair-2"
    result = scored("1n", truth, entities=[object_record(), object_record("O77", point=[.7, .5])])
    assert result["success"]


def test_approach_uses_geodesic_region_and_allows_effective_sliding():
    sample = {"task_id": "14b", "limits": {"max_actions": 24, "max_path_length": 4.}}
    label = {"approach_tolerance": .35}
    outcome = {"submitted": True, "distance_to_approach": .2, "actions": 9,
               "path_length": 1.8, "collision": True, "blocked": False}
    assert score_sample(sample, label, outcome)["success"]
    assert not score_sample(sample, label, {**outcome, "distance_to_approach": 7.})["success"]
    assert score_sample(sample, label, {**outcome, "path_length": 4.01})["reason"] == "movement_budget"
    assert score_sample(sample, label, {**outcome, "actions": 25})["reason"] == "movement_budget"
    assert score_sample(sample, label, {**outcome, "submitted": False})["reason"] == "not_submitted"
    assert score_sample(sample, label, {**outcome, "distance_to_approach": None})["reason"] == "no_reachable_approach"


def test_execution_fault_is_distinct_from_task_failure(truth):
    assert scored("1b", truth, reason="execution_uncertain", entities=[object_record()])["reason"] == "execution_error"


def test_unimplemented_tasks_are_not_silently_successful():
    with pytest.raises(ValueError, match="Unsupported task"):
        scored("11b", {}, "front")
