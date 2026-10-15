"""Offline scoring consumes frozen dataset files, never dataset builders."""

import builtins
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from src.bench.goal_audit import audit, load_prepared_dataset, _validate_rescore_scenario


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def prepared(tmp_path):
    dataset = tmp_path / "dataset"
    target = {"id": "O175", "label": "餐桌", "room_id": "R01", "object_ids": ["O175"],
              "positions": [[1, 0, 0], [1.25, 0, 0]], "source_type": "rendered_custom_viewpoints",
              "source": "/nonexistent/offline-only-source.json"}
    environment = {"agent_height": .88, "agent_radius": .18, "navmesh_cell_height": .05}
    catalog = {"schema_version": 2, "scene_id": "test", "environment": environment,
               "generation_provenance": {"navmesh_sha256": hashlib.sha256(b"mesh").hexdigest()},
               "targets": {"O175": target}}
    spec = {"id": "table", "kind": "navigation", "instruction": "走到餐桌旁。",
            "max_actions": 24, "max_requests": 12, "seconds": 600, "success_distance": 1,
            "positions": target["positions"], "snap_viewpoints": False,
            "private_verification": {"target_object_id": "O175", "target_object_ids": ["O175"],
                                     "target": {k: v for k, v in target.items() if k != "positions"}},
            "route_events": [{"kind": "waypoint", "id": "dining_table", "radius": .2,
                              "positions": target["positions"], "target_object_id": "O175",
                              "viewpoint_source": target["source_type"]}]}
    scenario = {"schema_version": 3, "goal_catalog_version": 2,
                "protocol": "navcraft-map-free-12-round-goal-catalog-v2",
                "scene_id": "/nonexistent/test.basis.glb", "environment": environment,
                "start_position": [0, 0, 0], "start_rotation": [0, 0, 0, 1], "seconds": 720,
                "private_rooms": [{"id": "R01", "polygon": [[-1, -1], [2, -1], [2, 1], [-1, 1]]}],
                "rounds": [spec]}
    write_json(dataset / "scenario.json", scenario)
    write_json(dataset / "catalog.json", catalog)
    return dataset, scenario, catalog


def forbid_generation_and_habitat(monkeypatch):
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name == "habitat_sim" or name.startswith("src.dataset.qa_campaign"):
            raise AssertionError(f"Unexpected dependency during frozen dataset loading: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)


def test_prepared_pair_loads_without_raw_scene_cache_or_generation_sources(prepared, monkeypatch):
    dataset, scenario, catalog = prepared
    forbid_generation_and_habitat(monkeypatch)
    before = {p.name: p.read_bytes() for p in dataset.iterdir()}
    assert load_prepared_dataset(dataset) == (scenario, catalog)
    assert {p.name: p.read_bytes() for p in dataset.iterdir()} == before


@pytest.mark.parametrize("missing", ["scenario.json", "catalog.json"])
def test_missing_prepared_file_fails_before_habitat_and_never_rebuilds(prepared, tmp_path, monkeypatch, missing):
    dataset, _, _ = prepared
    (dataset / missing).unlink()
    forbid_generation_and_habitat(monkeypatch)
    with pytest.raises(FileNotFoundError, match=missing):
        audit(tmp_path / "missing-run", tmp_path / "audit", tmp_path / "missing.navmesh", dataset)
    assert not (dataset / missing).exists()


@pytest.mark.parametrize("corruption", ["protocol", "scene", "environment", "identity", "positions",
                                         "empty", "nan", "metadata", "waypoint"])
def test_prepared_pair_rejects_inconsistent_or_invalid_data(prepared, corruption):
    dataset, scenario, catalog = prepared
    spec, target = scenario["rounds"][0], catalog["targets"]["O175"]
    if corruption == "protocol": scenario["protocol"] = "different-protocol"
    if corruption == "scene": catalog["scene_id"] = "other"
    if corruption == "environment": catalog["environment"] = {"agent_radius": .3}
    if corruption == "identity": target["id"] = "O243"
    if corruption == "positions": spec["positions"] = [[9, 0, 9]]
    if corruption == "empty": target["positions"] = []
    if corruption == "nan": target["positions"] = [[float("nan"), 0, 0]]
    if corruption == "metadata": spec["private_verification"]["target"]["label"] = "水槽"
    if corruption == "waypoint": spec["route_events"][0]["positions"] = [[9, 0, 9]]
    write_json(dataset / "scenario.json", scenario)
    write_json(dataset / "catalog.json", catalog)
    with pytest.raises(ValueError):
        load_prepared_dataset(dataset)


def original_scenario(scenario):
    original = deepcopy(scenario)
    original.update(schema_version=2, protocol="navcraft-map-free-12-round-exploratory-campaign")
    original.pop("goal_catalog_version")
    spec = original["rounds"][0]
    spec.update(positions=[[2, 0, 0]], snap_viewpoints=True)
    for key in ("target", "target_object_ids"):
        spec["private_verification"].pop(key)
    event = spec["route_events"][0]
    event["positions"] = [[2, 0, 0]]
    for key in ("target_object_id", "viewpoint_source"):
        event.pop(key)
    return original


@pytest.mark.parametrize("field", ["id", "kind", "instruction", "success_distance", "max_actions",
                                   "max_requests", "seconds", "route", "environment"])
def test_posthoc_audit_cannot_change_original_task_definitions(prepared, field):
    _, scenario, _ = prepared
    original = original_scenario(scenario)
    changed = deepcopy(scenario)
    if field == "route": changed["rounds"][0]["route_events"][0]["radius"] = 5
    elif field == "environment": changed["environment"]["agent_radius"] = .3
    else: changed["rounds"][0][field] = "changed"
    with pytest.raises(ValueError, match="changed archived"):
        _validate_rescore_scenario(original, changed)
    _validate_rescore_scenario(original, scenario)


def test_audit_only_scores_prepared_data_and_snapshots_the_exact_inputs(prepared, tmp_path, monkeypatch):
    dataset, scenario, catalog = prepared
    run, output, navmesh = tmp_path / "run", tmp_path / "audit", tmp_path / "mesh.navmesh"
    navmesh.write_bytes(b"mesh")
    original = original_scenario(scenario)
    write_json(run / "private/scenario.json", original)
    archived = {"rounds": [{"round": 1, "id": "table", "result": {
        "trajectory": [[0, 0, 0], [1, 0, 0]], "subtasks": [{"reason": "stop"}],
        "metrics": {"navigation_error": 1, "path_length": 1},
        "independent": {"passed": False, "route": {"passed": False}}}}]}
    write_json(run / "result.json", archived)

    class FakePathFinder:
        nav_mesh_settings = SimpleNamespace(agent_height=.88, agent_radius=.18, cell_height=.05)

        def load_nav_mesh(self, path):
            return Path(path).read_bytes() == b"mesh"

        def find_path(self, query):
            query.geodesic_distance = min(math.dist(query.requested_start, p) for p in query.requested_ends)
            return True

    monkeypatch.setitem(sys.modules, "habitat_sim", SimpleNamespace(
        PathFinder=FakePathFinder, MultiGoalShortestPath=SimpleNamespace))
    original_import = builtins.__import__

    def no_preparation(name, *args, **kwargs):
        if name.startswith("src.dataset.qa_campaign"):
            raise AssertionError("Evaluation must not invoke dataset preparation")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_preparation)
    before = {p.name: p.read_bytes() for p in dataset.iterdir()}
    report = audit(run, output, navmesh, dataset)
    assert (report["old_passed"], report["new_passed"]) == (0, 1)
    assert report["rounds"][0]["new_distance"] == 0
    assert json.loads((output / "scenario-v2.json").read_text()) == scenario
    assert json.loads((output / "catalog.json").read_text()) == catalog
    assert json.loads((run / "private/scenario.json").read_text()) == original
    assert json.loads((run / "result.json").read_text()) == archived
    assert {p.name: p.read_bytes() for p in dataset.iterdir()} == before


def test_audit_refuses_to_overwrite_its_prepared_dataset(prepared, tmp_path):
    dataset, _, _ = prepared
    with pytest.raises(ValueError, match="overwrite the prepared dataset"):
        audit(tmp_path / "run", dataset, tmp_path / "mesh.navmesh", dataset)
