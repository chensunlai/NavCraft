"""Canonical asset identifiers relative to the configured scene root."""

from pathlib import Path, PurePosixPath


# Dataset releases use these two historical spellings in their source records.
# Normalize once while loading, rather than repeating replacements in runtimes.
def canonical_scene_id(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Scene ID must be a nonempty dataset-relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Scene ID must be dataset-relative: {value}")
    value = value.removeprefix("data/scene_datasets/")
    if value.startswith("hm3d_v0.2/"):
        value = "hm3d/" + value[len("hm3d_v0.2/"):]
    path = PurePosixPath(value)
    if len(path.parts) < 2:
        raise ValueError("Scene ID must include the dataset and asset path")
    return path.as_posix()


def resolve_scene(root, scene_id):
    path = Path(root) / canonical_scene_id(scene_id)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
