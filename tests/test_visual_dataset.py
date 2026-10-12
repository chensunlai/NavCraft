"""Frozen private data and route generation do not require models or Habitat."""

import math

import pytest

from src.common.io import write_json
from src.common.register import episode_registry
from src.dataset.episodes import load_builtin_sources
from src.dataset.episodes.base import EpisodeSpec
from src.dataset.episodes.manifest import ManifestEpisodeSource
from src.dataset.prepare_visual import (
    FixtureValidator,
    RoomArchive,
    balanced_selection,
    instruction_episodes,
    save_manifest,
)


def specification(identifier="r2r:synthetic_hm3d:fixture:000"):
    return EpisodeSpec(
        id=identifier,
        source_id="manifest:fixture:0",
        benchmark="r2r",
        scene_id="fixture.glb",
        start_position=[0, 0, 0],
        start_rotation=[0, 0, 0, 1],
        goals=[
            {
                "modality": "instruction",
                "instruction": "Pass the hall.",
                "positions": [[8, 0, 0]],
            }
        ],
        source_file="fixture",
        route_checks=[{"label": "hall", "position": [4, 0, 0], "radius": 1.5}],
        provenance="synthetic_hm3d_instruction",
    )


def test_manifest_round_trip_private_fields_and_exact_selection(tmp_path):
    first, second = specification(), specification("r2r:synthetic_hm3d:fixture:001")
    second.source_id = "manifest:fixture:1"
    summary = save_manifest(
        tmp_path, "instruction_test", [first, second], validated=True, seed=0
    )
    source = ManifestEpisodeSource(
        {"name": "manifest", "root": str(tmp_path), "manifest": "instruction_test.json"}
    )
    selected = source.load([second.id, first.id])
    assert [row.to_dict() for row in selected] == [second.to_dict(), first.to_dict()]
    assert summary["count"] == 2 and summary["goals"] == 2
    assert len(summary["sha256"]) == 64
    load_builtin_sources()
    assert episode_registry.get("manifest") is ManifestEpisodeSource


def test_unvalidated_fixture_manifest_cannot_be_used_by_default(tmp_path):
    save_manifest(tmp_path, "fixture", [specification()], validated=False, seed=0)
    config = {"name": "manifest", "root": str(tmp_path), "manifest": "fixture.json"}
    with pytest.raises(ValueError, match="validation"):
        ManifestEpisodeSource(config).load()
    assert (
        ManifestEpisodeSource({**config, "allow_unvalidated": True}).load()[0].id
        == specification().id
    )


def test_manifest_rejects_missing_duplicate_or_noncanonical_selection(tmp_path):
    episode = specification()
    save_manifest(tmp_path, "tasks", [episode], validated=True, seed=0)
    source = ManifestEpisodeSource(
        {"name": "manifest", "root": str(tmp_path), "manifest": "tasks.json"}
    )
    with pytest.raises(ValueError, match="not found"):
        source.load(["missing"])
    with pytest.raises(ValueError, match="not a canonical"):
        source.load([episode.source_id])
    with pytest.raises(ValueError, match="duplicate"):
        source.load([episode.id, episode.id])


def archive_fixture(tmp_path):
    raw = tmp_path / "raw"
    rooms, frames = [], []
    for index, (low, high, center) in enumerate([(0, 3, 1), (3, 7, 5), (7, 10, 9)]):
        rooms.append(
            {
                "id": f"R{index}",
                "level": "L1",
                "kind": "room",
                "name": f"room {index}",
                "polygon": [[low, -1], [high, -1], [high, 1], [low, 1]],
                "description": f"Visible landmark {index}.",
                "evidence_frames": [f"F{index}"],
            }
        )
        frames.append({"id": f"F{index}", "pose": {"position": [center, 0, 0]}})
    write_json(
        raw / "scene.json",
        {"scene_id": "fixture", "levels": [{"id": "L1", "height_m": 0}]},
    )
    write_json(raw / "rooms.json", {"rooms": rooms})
    write_json(raw / "evidence.json", {"frames": frames})
    write_json(raw / "regions.json", {"stair_connections": []})
    write_json(
        raw / "doors.json",
        {
            "doors": [
                {"level": "L1", "connects": ["R0", "R1"], "status": "open"},
                {"level": "L1", "connects": ["R1", "R2"], "status": "open"},
            ]
        },
    )
    return RoomArchive(tmp_path)


def test_synthetic_routes_have_correct_turns_ordered_checks_and_nontrivial_starts(
    tmp_path,
):
    archive = archive_fixture(tmp_path)
    rejected = []
    episodes = instruction_episodes(archive, FixtureValidator(), 2, 0, rejected)
    again = instruction_episodes(archive, FixtureValidator(), 2, 0, [])
    assert [row.to_dict() for row in episodes] == [row.to_dict() for row in again]
    for index, episode in enumerate(episodes):
        assert episode.provenance == "synthetic_hm3d_instruction"
        assert episode.benchmark == "r2r"
        assert math.dist(episode.start_position, episode.goals[0]["positions"][0]) >= 5
        assert episode.route_checks[0]["label"] == "room 1"
        assert episode.route_checks[0]["position"][0] == pytest.approx(5, abs=0.26)
        yaw = 2 * math.atan2(episode.start_rotation[1], episode.start_rotation[3])
        action = -math.pi / 2 if index == 0 else math.pi / 2
        dx = episode.goals[0]["positions"][0][0] - episode.start_position[0]
        assert -math.sin(yaw + action) == pytest.approx(math.copysign(1, dx))


def test_unreachable_candidates_are_logged_before_freezing(tmp_path):
    archive = archive_fixture(tmp_path)

    class NoPath(FixtureValidator):
        def path(self, start, end):
            return None

    rejected = []
    with pytest.raises(ValueError, match="Only 0"):
        instruction_episodes(archive, NoPath(), 1, 0, rejected)
    assert rejected and all(row["reason"] == "unreachable" for row in rejected)


def test_selection_balances_object_categories_and_preserves_full_goals():
    episodes = []
    for category in ["chair", "bed", "sofa"]:
        for index in range(3):
            row = specification(f"objnav:val:fixture:{category}:{index}")
            row.benchmark = "objnav"
            episodes.append(row)
    selected = balanced_selection(episodes, 6, 0)
    assert [row.id.split(":")[-2] for row in selected] == ["bed", "chair", "sofa"] * 2
    assert all(row in episodes for row in selected)
    with pytest.raises(ValueError, match="Cannot freeze"):
        balanced_selection(episodes, 10, 0)


def test_synthetic_constraint_metrics_are_ordered_and_label_reference_correctly():
    from src.environment.tasks.base import VLNCETask

    episode = specification()
    episode.gt_path = [[0, 0, 0], [4, 0, 0], [8, 0, 0]]
    episode.route_checks = [
        {"label": "first", "position": [2, 0, 0], "radius": 0.5},
        {"label": "second", "position": [4, 0, 0], "radius": 0.5},
    ]
    task = VLNCETask({})
    task.reset(episode, [0, 0, 0], 8)
    task.on_step([4, 0, 0], 4)
    task.on_step([2, 0, 0], 6)
    task.finish([8, 0, 0], 0)
    scores = task.metrics()
    assert scores["success"] == 1
    assert scores["route_constraint_completion"] == 0.5
    assert scores["route_constraints_satisfied"] == 0
    assert scores["ndtw_reference"] == "synthetic_path"
    assert scores["provenance"] == "synthetic_hm3d_instruction"

    task.reset(episode, [0, 0, 0], 8)
    task.on_step([2, 0, 0], 6)
    task.on_step([4, 0, 0], 4)
    task.finish([8, 0, 0], 0)
    assert task.metrics()["route_constraints_satisfied"] == 1
