"""Dataset protocol and selection tests with small on-disk Habitat fixtures."""

import gzip
import json
from pathlib import Path

import pytest

from src.dataset.episodes import GOATEpisodeSource, ObjectNavEpisodeSource, VLNCEEpisodeSource


def write_gzip(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(value, stream)


def raw_episode(episode_id: int = 1, **extra) -> dict:
    return {
        "episode_id": episode_id, "scene_id": "hm3d_v0.2/val/scene/scene.basis.glb",
        "start_position": [0, 0, 0], "start_rotation": [0, 0, 0, 1], **extra,
    }


def instance(object_id: int, position: list, **extra) -> dict:
    return {"object_id": object_id, "view_points": [{"agent_state": {"position": position}}], **extra}


def test_objnav_canonical_index_and_exact_selection(tmp_path):
    write_gzip(tmp_path / "val/content/scene.json.gz", {
        "episodes": [raw_episode(1, object_category="bed"), raw_episode(1, object_category="chair")],
        "goals_by_category": {
            "scene.basis.glb_bed": [instance(1, [2, 0, 0])],
            "scene.basis.glb_chair": [instance(2, [3, 0, 0]), instance(3, [4, 0, 0])],
        },
    })
    source = ObjectNavEpisodeSource({"name": "objnav", "root": str(tmp_path), "split": "val"})
    episode = source.load(["objnav:val:scene:chair:1"])[0]
    assert episode.source_id == "1"
    assert episode.scene_id == "hm3d/val/scene/scene.basis.glb"
    assert episode.goals[0]["positions"] == [[3, 0, 0], [4, 0, 0]]
    with pytest.raises(ValueError, match="ambiguous"):
        source.load(["1"])
    with pytest.raises(ValueError, match="duplicate"):
        source.load([episode.id, episode.id])
    with pytest.raises(ValueError, match="not found"):
        source.load(["objnav:val:scene:chair:0"])


def test_limit_applied_before_shard_read_and_unselected_goals_not_decoded(tmp_path):
    write_gzip(tmp_path / "val/content/scene.json.gz", {
        "episodes": [raw_episode(1, object_category="bed"), raw_episode(2, object_category="bad")],
        "goals_by_category": {"scene.basis.glb_bed": [instance(1, [2, 0, 0])]},
    })
    broken = tmp_path / "val/content/other.json.gz"
    broken.write_bytes(b"not gzip")
    source = ObjectNavEpisodeSource({
        "name": "objnav", "root": str(tmp_path), "split": "val", "limit": 1,
        "ids": ["objnav:val:scene:bed:0", "objnav:val:other:chair:0"],
    })
    assert source.load()[0].id == "objnav:val:scene:bed:0"


def test_selection_file_order_and_conflicts(tmp_path):
    write_gzip(tmp_path / "val_unseen/val_unseen.json.gz", {
        "episodes": [
            raw_episode(i, instruction={"instruction_text": f"Walk {i}"}, goals=[{"position": [i, 0, 0]}])
            for i in [1, 2]
        ],
    })
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"episode_ids": ["r2r:val_unseen:2", "r2r:val_unseen:1"]}))
    config = {"name": "r2r", "root": str(tmp_path), "selection_file": str(selection)}
    assert [episode.source_id for episode in VLNCEEpisodeSource(config).load()] == ["2", "1"]
    with pytest.raises(ValueError, match="either ids or selection_file"):
        VLNCEEpisodeSource({**config, "ids": ["r2r:val_unseen:1"]}).load()
    with pytest.raises(ValueError, match="positive integer"):
        VLNCEEpisodeSource({**config, "limit": 0}).load()


def test_rxr_role_language_and_missing_gt_are_explicit(tmp_path):
    write_gzip(tmp_path / "val_unseen/val_unseen_guide.json.gz", {
        "episodes": [raw_episode(
            7, instruction={"instruction_text": " Turn right. ", "language": "en-US"},
            goals=[{"position": [2, 0, 0]}], reference_path=[[0, 0, 0], [2, 0, 0]],
        )],
    })
    config = {"name": "rxr", "root": str(tmp_path), "role": "guide"}
    episode = VLNCEEpisodeSource(config).load()[0]
    assert episode.id == "rxr:val_unseen:guide:7"
    assert episode.language == "en-US"
    assert episode.goals[0]["instruction"] == "Turn right."
    assert episode.gt_path == []
    assert episode.reference_path == [[0, 0, 0], [2, 0, 0]]
    write_gzip(tmp_path / "val_unseen/val_unseen_guide_gt.json.gz", {"7": {"locations": [[1, 0, 0]]}})
    assert VLNCEEpisodeSource(config).load()[0].gt_path == [[1, 0, 0]]


def test_goat_modalities_child_categories_and_description_skip(tmp_path):
    camera = {"position": [2, 1, 0], "rotation": [0, 0, 0, 1], "hfov": 90, "image_dimensions": [100, 200]}
    chair = instance(3, [2, 0, 0], children_object_categories=["armchair"],
                     lang_desc="A red chair.", image_goals=[camera])
    write_gzip(tmp_path / "val_unseen/content/scene.json.gz", {
        "episodes": [raw_episode(7, tasks=[
            ["chair", "object", 3], ["chair", "description", 3],
            ["chair", "image", 3, 0], ["table", "description", 4],
        ])],
        "goals": {
            "scene.basis.glb_chair": [chair],
            "scene.basis.glb_armchair": [instance(9, [4, 0, 0])],
            "scene.basis.glb_table": [instance(4, [5, 0, 0], lang_desc="one two three four five")],
        },
    })
    config = {"name": "goat", "root": str(tmp_path), "goat_max_description_words": 3}
    episode = GOATEpisodeSource(config).load(["goat:val_unseen:scene:7"])[0]
    assert [goal["modality"] for goal in episode.goals] == ["object", "description", "image"]
    assert episode.goals[0]["positions"] == [[2, 0, 0], [4, 0, 0]]
    assert episode.goals[1]["positions"] == [[2, 0, 0]]
    assert episode.goals[2]["image_goal"] == camera
    assert episode.goals[2]["source_index"] == 2
    assert episode.skipped_goals == 1
    assert len(GOATEpisodeSource({**config, "goat_max_description_words": None}).load()[0].goals) == 4


def test_goat_invalid_image_index_and_all_goals_filtered(tmp_path):
    path = tmp_path / "val_unseen/content/scene.json.gz"
    document = {
        "episodes": [raw_episode(7, tasks=[["chair", "image", 3, 1]])],
        "goals": {"scene.basis.glb_chair": [instance(3, [2, 0, 0], image_goals=[], lang_desc="long description")]},
    }
    write_gzip(path, document)
    config = {"name": "goat", "root": str(tmp_path)}
    with pytest.raises(ValueError, match="Invalid GOAT image index"):
        GOATEpisodeSource(config).load()
    document["episodes"][0]["tasks"] = [["chair", "description", 3]]
    write_gzip(path, document)
    with pytest.raises(ValueError, match="no goals after preprocessing"):
        GOATEpisodeSource({**config, "goat_max_description_words": 1}).load()
