"""The campaign must use complete instance goals, never evidence-camera proxies."""

from copy import deepcopy
import gzip
import json

import pytest

from src.dataset.qa_campaign import goals as module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def catalog_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "TARGETS", {
        "O042": ("卧室电视", "R01", [42]),
        "O108": ("双台盆", "R03", [108, 109, 110]),
    })
    raw = tmp_path / "data/scene_maps/hm3d/val/00829-QaLdnwvtxbs/raw/objects.json"
    write_json(raw, {"objects": [{"id": f"O{i:03d}", "evidence_frames": [f"frame-{i}"]}
                                 for i in [42, 108, 109, 110]]})
    path = tmp_path / module.OFFICIAL
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt") as f:
        json.dump({"goals_by_category": {"tv": [
            {"object_id": 42, "object_name": "tv_42", "position": [1, 1, 1],
             "view_points": [{"agent_state": {"position": p}}
                             for p in [[1, 0, 1], [2, 0, 1], [3, 0, 1]]]},
            {"object_id": 196, "object_name": "tv_196", "position": [-5, 1, 1],
             "view_points": [{"agent_state": {"position": [-5, 0, 1]}}]},
        ]}}, f)
    custom = {"scene_id": module.SCENE, "method": {"grid_m": .1, "minimum_visible_pixels": 308},
              "environment": {"agent_radius": .18}, "targets": {"O108": {
                  "semantic_instance_ids": [108, 109, 110], "room_id": "R03",
                  "positions": [[0, 0, 1], [0, 0, 2]], "candidate_count": 3,
                  "accepted_count": 2, "viewpoints": [
                      {"position": p, "accepted": True, "target_pixels": 400}
                      for p in [[0, 0, 1], [0, 0, 2]]
                  ]}}}
    write_json(tmp_path / module.GENERATED, custom)
    return tmp_path, custom


def test_preserve_all_official_views_of_only_the_requested_instance(catalog_repo):
    repo, _ = catalog_repo
    catalog = module.load_goal_catalog(repo)["targets"]
    assert catalog["O042"]["positions"] == [[1, 0, 1], [2, 0, 1], [3, 0, 1]]
    assert catalog["O042"]["source_type"] == "official_instance_viewpoints"
    assert catalog["O108"]["semantic_instance_ids"] == [108, 109, 110]
    assert catalog["O108"]["evidence_ids"] == ["frame-108", "frame-109", "frame-110"]
    assert catalog["O108"]["viewpoint_count"] == 2


def test_missing_custom_goals_do_not_silently_fall_back(catalog_repo):
    repo, _ = catalog_repo
    (repo / module.GENERATED).unlink()
    with pytest.raises(FileNotFoundError, match="not an acceptable fallback"):
        module.load_goal_catalog(repo)


@pytest.mark.parametrize("bad", [[], [[1, 2]], [[1, 0, float("nan")]]])
def test_reject_unusable_goal_positions(catalog_repo, bad):
    repo, custom = catalog_repo
    custom["targets"]["O108"]["positions"] = bad
    write_json(repo / module.GENERATED, custom)
    with pytest.raises(ValueError, match="accepted viewpoints"):
        module.load_goal_catalog(repo)


def test_reject_cross_instance_custom_annotation(catalog_repo):
    repo, custom = catalog_repo
    custom["targets"]["O108"]["semantic_instance_ids"] = [243]
    write_json(repo / module.GENERATED, custom)
    with pytest.raises(ValueError, match="identity mismatch"):
        module.load_goal_catalog(repo)


@pytest.mark.parametrize("corruption", ["count", "position", "visibility", "rejected"])
def test_require_matching_visibility_receipts(catalog_repo, corruption):
    repo, custom = catalog_repo
    item = custom["targets"]["O108"]
    if corruption == "count": item["accepted_count"] = 1
    if corruption == "position": item["viewpoints"][0]["position"] = [99, 0, 99]
    if corruption == "visibility": item["viewpoints"][0]["target_pixels"] = 0
    if corruption == "rejected": item["viewpoints"][0]["accepted"] = False
    write_json(repo / module.GENERATED, custom)
    with pytest.raises(ValueError, match="verification records disagree"):
        module.load_goal_catalog(repo)


def test_completion_updates_route_target_without_changing_thresholds_or_input():
    target = {"positions": [[1, 0, 1], [2, 0, 1]], "object_ids": ["O175"],
              "source_type": "rendered_custom_viewpoints", "label": "餐桌"}
    catalog = {"targets": {"O175": target}, "environment": {"agent_radius": .18}}
    scenario = {"environment": {"agent_radius": .18}, "rounds": [{
        "kind": "navigation", "positions": [[99, 0, 99]], "instruction": "经过餐桌",
        "success_distance": 1, "private_verification": {"target_object_id": "O175"},
        "route_events": [{"id": "dining_table", "kind": "waypoint", "positions": [[99, 0, 99]], "radius": 1.5}],
    }]}
    original = deepcopy(scenario)
    completed = module.complete_scenario(scenario, catalog)
    spec = completed["rounds"][0]
    assert scenario == original
    assert spec["positions"] == spec["route_events"][0]["positions"] == target["positions"]
    assert spec["instruction"] == "经过餐桌"
    assert spec["success_distance"] == 1 and spec["route_events"][0]["radius"] == 1.5
    assert completed["goal_catalog_version"] == 2
    scenario["environment"]["agent_radius"] = .3
    with pytest.raises(ValueError, match="environment mismatch"):
        module.complete_scenario(scenario, catalog)
