"""Small immutable pilot plan from the existing public-benchmark selections."""

import hashlib
import json
from pathlib import Path

from src.common.scenes import resolve_scene

from src.dataset.episodes import create_source
from src.dataset.maps import map_directory as scene_map_directory, load_wall_map

SOURCES = {
    "objnav": {"root": "data/datasets/objectnav/hm3d/v2", "split": "val",
               "selection_file": "data/samples/objnav100/selected/objnav100.json"},
    "r2r": {"root": "data/datasets/r2r", "split": "val_unseen",
            "selection_file": "data/samples/vlnce100/selected/r2r100.json"},
    "goat": {"root": "data/datasets/goat_bench/hm3d/v1", "split": "val_unseen",
             "selection_file": "data/samples/goat20/selected/goat20.json",
             "goat_max_description_words": 55},
}


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


def map_directory(repo, episode, root="data/scene_maps"):
    return scene_map_directory(Path(repo) / root, episode.scene_id)


def plan(repo, config):
    repo = Path(repo)
    if config["environment"]["gpus"] != [0, 1]:
        raise ValueError("The pilot simulator GPU pool must be [0, 1]")
    episodes = load_episodes(repo, config["episodes"])
    hashes = {family: hashlib.sha256((repo / SOURCES[family]["selection_file"]).read_bytes()).hexdigest()
              for family in {e.benchmark for e in episodes}}
    jobs = []
    maps = config["maps"]
    for i, episode in enumerate(episodes):
        scene = resolve_scene(repo / config["environment"]["scenes"], episode.scene_id)
        directory = map_directory(repo, episode, maps["root"])
        info, error = None, None
        try:
            _, info = load_wall_map(directory, episode.start_position, maps.get("layer", "walls"))
            expected = Path(episode.scene_id).name.split(".")[0]
            if info["scene_id"] != expected:
                raise ValueError(f"Expected scene {expected}, map declares {info['scene_id']}")
        except (OSError, ValueError, KeyError, TypeError) as exception:
            error = str(exception)
        jobs.append({"index": i, "episode_id": episode.id, "benchmark": episode.benchmark,
                     "scene": str(scene.resolve()), "floor_height": episode.start_position[1],
                     "map": str(directory), "map_layer": maps.get("layer", "walls"),
                     "map_source": info, "map_error": error,
                     "planned_goals": len(episode.goals)})
    return {"selection_sha256": hashes, "dispatch_order": config["episodes"], "jobs": jobs}
