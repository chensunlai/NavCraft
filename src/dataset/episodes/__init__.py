"""Episode readers; registration is explicit and requires no Habitat import."""

from .base import EpisodeSource, EpisodeSpec, read_selection
from .readers import GOATEpisodeSource, ObjectNavEpisodeSource, VLNCEEpisodeSource

_loaded = False


def load_builtin_sources() -> None:
    global _loaded
    if _loaded:
        return
    from src.common.register import episode_registry

    episode_registry.register("objectnav")(ObjectNavEpisodeSource)
    episode_registry.register("vlnce")(VLNCEEpisodeSource)
    episode_registry.register("goat")(GOATEpisodeSource)
    _loaded = True


__all__ = [
    "EpisodeSource", "EpisodeSpec", "read_selection", "load_builtin_sources",
    "ObjectNavEpisodeSource", "VLNCEEpisodeSource", "GOATEpisodeSource",
]
