"""Small immutable pilot plan from the existing public-benchmark selections."""

import hashlib
from pathlib import Path

from src.common.scenes import resolve_scene
from src.harness.tools import Toolsets
from src.common.config import merge_config, read_config

from src.dataset.episodes import create_source, read_selection
from src.dataset.maps import map_directory as scene_map_directory, load_wall_map


def load_episodes(repo, ids, sources):
    """Reject unknown/non-frozen IDs and preserve the explicit dispatch order."""
    repo = Path(repo)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Episode IDs must be nonempty and unique")
    loaded = {}
    for family in dict.fromkeys(i.split(":")[0] for i in ids):
        source = sources[family]
        frozen = read_selection(repo / source["selection_file"])
        selected = [i for i in ids if i.split(":")[0] == family]
        if not set(selected).issubset(frozen):
            raise ValueError(f"IDs outside frozen {family} selection: {set(selected) - set(frozen)}")
        reader = create_source({"name": family, **source}, repo)
        loaded.update((e.id, e) for e in reader.load(selected))
    return [loaded[i] for i in ids]


def map_directory(repo, episode, root="data/scene_maps"):
    return scene_map_directory(Path(repo) / root, episode.scene_id)


def plan(repo, config, model_names):
    repo = Path(repo)
    if config["environment"]["gpus"] != [0, 1]:
        raise ValueError("The pilot simulator GPU pool must be [0, 1]")
    episodes = load_episodes(repo, config["episodes"], config["sources"])
    hashes = {family: hashlib.sha256((repo / config["sources"][family]["selection_file"]).read_bytes()).hexdigest()
              for family in {e.benchmark for e in episodes}}
    jobs = []
    maps = config.get("maps", {})
    models = {name: merge_config(read_config(repo / config["models"][name]), config.get("model_overrides", {}))
              for name in model_names}
    node_only = all(Toolsets.resolve(model["toolsets"]).memory_type == "node" for model in models.values())
    for i, episode in enumerate(episodes):
        scene = resolve_scene(repo / config["environment"]["scenes"], episode.scene_id)
        directory = None if node_only else map_directory(repo, episode, maps.get("root", "data/scene_maps"))
        info, error = None, None
        try:
            if not node_only:
                _, info = load_wall_map(directory, episode.start_position, maps.get("layer", "walls"))
                expected = Path(episode.scene_id).name.split(".")[0]
                if info["scene_id"] != expected:
                    raise ValueError(f"Expected scene {expected}, map declares {info['scene_id']}")
        except (OSError, ValueError, KeyError, TypeError) as exception:
            error = str(exception)
        jobs.append({"index": i, "episode_id": episode.id, "episode": episode.to_dict(), "benchmark": episode.benchmark,
                     "scene": str(scene.resolve()), "floor_height": episode.start_position[1],
                     "map": None if node_only else str(directory), "map_layer": maps.get("layer", "walls"),
                     "map_condition": "blank" if node_only else "walls",
                     "map_source": info, "map_error": error,
                     "planned_goals": len(episode.goals)})
    return {"models": models, "selection_sha256": hashes, "dispatch_order": config["episodes"], "jobs": jobs}
