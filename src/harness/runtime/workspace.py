"""Public workspace initialization and copying of navigation snapshot assets."""

import json
from pathlib import Path
import shutil

from src.common.coordinates import validate_map
from src.common.io import write_json


PRIVATE_ASSETS = ("depth", "location_sources.json", "memory_used_ids.json")


def prepare_workspace(workspace, private, environment, scene_map, memory_type, state):
    for name in ("scene", "observations", "memory", "scratch"):
        (workspace / name).mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    restoring = any(key in state for key in ("workspace", "session_file", "history", "memory"))
    if ("memory_type" in state or restoring) and state.get("memory_type", "entity") != memory_type:
        raise ValueError("Snapshot memory_type does not match the requested mode; start a new session")
    if state.get("environment"):
        environment.restore_state(state["environment"])
    elif state.get("pose"):
        environment.restore_pose(state["pose"])
    if memory_type == "entity":
        shutil.copyfile(scene_map, workspace / "scene/map.json")
    elif state.get("workspace"):
        shutil.copyfile(Path(state["workspace"]) / "scene/map.json", workspace / "scene/map.json")
    else:
        x, height, z = environment.pose()["position"]
        write_json(workspace / "scene/map.json", {
            "origin": [x - 10, z - 10], "size_m": 20,
            "floor_height": height, "walls": [],
        })
    transform = json.loads((workspace / "scene/map.json").read_text())
    validate_map(transform)
    if memory_type == "node" and transform["walls"]:
        raise ValueError("Node memory requires a public map without walls")
    shutil.copyfile(Path(__file__).parents[1] / "spatial/navlib.py", workspace / "scene/navlib.py")
    shutil.copyfile(Path(__file__).parents[2] / "common/coordinates.py", workspace / "scene/coordinates.py")
    if state.get("workspace"):
        for name in ("scene", "observations", "memory", "scratch"):
            source = Path(state["workspace"]) / name
            if source.exists():
                shutil.copytree(source, workspace / name, dirs_exist_ok=True, symlinks=True)
    if state.get("private_dir"):
        for name in ("depth", "location_sources.json", "memory_used_ids.json"):
            source = Path(state["private_dir"]) / name
            if source.is_dir():
                shutil.copytree(source, private / name, dirs_exist_ok=True)
            elif source.is_file():
                shutil.copyfile(source, private / name)
    if state.get("observations"):
        shutil.copytree(state["observations"], workspace / "observations", dirs_exist_ok=True)
    memory_name = "entities.json" if memory_type == "entity" else "nodes.json"
    other_name = "nodes.json" if memory_type == "entity" else "entities.json"
    if (workspace / "memory" / other_name).exists():
        raise ValueError("Workspace contains the other memory representation")
    if "memory" in state or not (workspace / "memory" / memory_name).exists():
        empty_memory = [] if memory_type == "entity" else {"nodes": [], "rooms": {}}
        write_json(workspace / "memory" / memory_name, state.get("memory", empty_memory))
    restored_map = json.loads((workspace / "scene/map.json").read_text())
    if restored_map != transform:
        raise ValueError("Snapshot map does not match the requested scene map")
    return transform


def copy_snapshot_assets(workspace, private, destination):
    destination.mkdir(parents=True)
    shutil.copytree(workspace, destination / "workspace", symlinks=True)
    for name in PRIVATE_ASSETS:
        source = private / name
        if source.is_dir():
            shutil.copytree(source, destination / "private" / name)
        elif source.is_file():
            (destination / "private").mkdir(exist_ok=True)
            shutil.copyfile(source, destination / "private" / name)

def read_snapshot(path):
    """Resolve a saved snapshot without exposing its private files to the model."""
    path = Path(path)
    path = path / "state.json" if path.is_dir() else path
    state = json.loads(path.read_text())
    for name in ("workspace", "private_dir"):
        state[name] = str((path.parent / state[name]).resolve())
    if state.get("session_file"):
        state["session_file"] = str((path.parent / state["session_file"]).resolve())
    return state
