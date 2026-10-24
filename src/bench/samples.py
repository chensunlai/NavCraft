"""Small immutable pilot plan from the existing public-benchmark selections."""

import hashlib
import json
from pathlib import Path

import yaml

from src.dataset.episodes import create_source

SOURCES = {
    "objnav": {"root": "data/datasets/objectnav/hm3d/v2", "split": "val",
               "selection_file": "data/samples/objnav100/selected/objnav100.json"},
    "r2r": {"root": "data/datasets/r2r", "split": "val_unseen",
            "selection_file": "data/samples/vlnce100/selected/r2r100.json"},
    "goat": {"root": "data/datasets/goat_bench/hm3d/v1", "split": "val_unseen",
             "selection_file": "data/samples/goat20/selected/goat20.json",
             "goat_max_description_words": 55},
}


def read_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_episodes(repo, ids):
    """Reject unknown/non-frozen IDs and preserve the explicit dispatch order."""
    repo = Path(repo)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Episode IDs must be nonempty and unique")
    loaded = {}
    for family in dict.fromkeys(i.split(":")[0] for i in ids):
        source = SOURCES[family]
        frozen = json.loads((repo / source["selection_file"]).read_text())
        selected = [i for i in ids if i.split(":")[0] == family]
        if not set(selected).issubset(frozen):
            raise ValueError(f"IDs outside frozen {family} selection: {set(selected) - set(frozen)}")
        reader = create_source({"name": family, **source}, repo)
        loaded.update((e.id, e) for e in reader.load(selected))
    return [loaded[i] for i in ids]


def map_directory(repo, episode):
    scene = Path(episode.scene_id).name.split(".")[0]
    height_mm = round(episode.start_position[1] * 1000)
    return Path(repo) / "cache/wall_maps" / f"{scene}-{height_mm}"


def plan(repo, config):
    repo = Path(repo)
    if config["environment"]["gpus"] != [0, 1]:
        raise ValueError("The pilot simulator GPU pool must be [0, 1]")
    episodes = load_episodes(repo, config["episodes"])
    hashes = {family: hashlib.sha256((repo / SOURCES[family]["selection_file"]).read_bytes()).hexdigest()
              for family in {e.benchmark for e in episodes}}
    jobs = []
    for i, episode in enumerate(episodes):
        scene = repo / config["environment"]["scenes"] / episode.scene_id.removeprefix("data/scene_datasets/")
        if not scene.is_file():
            raise FileNotFoundError(scene)
        jobs.append({"index": i, "episode_id": episode.id, "benchmark": episode.benchmark,
                     "scene": str(scene.resolve()), "floor_height": episode.start_position[1],
                     "map": str(map_directory(repo, episode) / "map.json"),
                     "planned_goals": len(episode.goals)})
    return {"selection_sha256": hashes, "dispatch_order": config["episodes"], "jobs": jobs}
