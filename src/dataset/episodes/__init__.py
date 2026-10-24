"""Official frozen sample loading; no simulator or Harness dependency."""

from pathlib import Path

from .base import EpisodeSource, EpisodeSpec, read_selection
from .readers import GOATEpisodeSource, ObjectNavEpisodeSource, VLNCEEpisodeSource


def create_source(config: dict, repo: Path) -> EpisodeSource:
    options = dict(config)
    for key in ("root", "selection_file", "gt_path"):
        if options.get(key):
            options[key] = str((Path(repo) / options[key]).resolve())
    factories = {"objnav": ObjectNavEpisodeSource, "r2r": VLNCEEpisodeSource,
                 "rxr": VLNCEEpisodeSource, "goat": GOATEpisodeSource}
    return factories[options["name"]](options)


__all__ = ["EpisodeSpec", "EpisodeSource", "create_source", "read_selection"]
