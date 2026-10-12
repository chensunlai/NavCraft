"""Frozen, private episode manifests, independent of simulator imports."""

from pathlib import Path

from src.common.io import read_json

from .base import EpisodeSource, EpisodeSpec


class ManifestEpisodeSource(EpisodeSource):
    def _read(self, requested: list[str] | None) -> list[EpisodeSpec]:
        path = Path(self.config.get("manifest", "episodes.json"))
        if not path.is_absolute() and not path.is_file():
            path = self.root / path
        document = read_json(path)
        if document.get("schema_version") != 1:
            raise ValueError("Unsupported frozen episode manifest schema")
        if not document.get("validated") and not self.config.get(
            "allow_unvalidated", False
        ):
            raise ValueError("Manifest has not passed environment validation")
        result = []
        for raw in document["episodes"]:
            if not self._wanted(raw["id"], raw["source_id"], requested):
                continue
            episode = EpisodeSpec(**raw)
            if episode.benchmark not in {"objnav", "r2r", "rxr", "goat"}:
                raise ValueError(
                    f"Unsupported benchmark in manifest: {episode.benchmark}"
                )
            result.append(episode)
        return result
