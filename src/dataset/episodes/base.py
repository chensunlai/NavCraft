"""Private episode specifications and exact dataset selection."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.common.io import read_json


@dataclass
class EpisodeSpec:
    """Initialization and scoring data; do not serialize this into model input."""

    id: str
    source_id: str
    benchmark: str
    scene_id: str
    start_position: list[float]
    start_rotation: list[float]
    goals: list[dict[str, Any]]
    source_file: str
    language: str | None = None
    reference_path: list = field(default_factory=list)
    gt_path: list = field(default_factory=list)
    skipped_goals: int = 0
    route_checks: list[dict[str, Any]] = field(default_factory=list)
    provenance: str = "official"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_selection(path: str | Path) -> list[str]:
    """Read a JSON list/episode_ids object or one canonical ID per text line."""
    path = Path(path)
    if path.suffix in {".json", ".gz"}:
        value = read_json(path)
        if isinstance(value, dict):
            value = value["episode_ids"]
    else:
        value = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    return validate_selection(value)


def validate_selection(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("Episode selection must be a nonempty list of canonical IDs")
    if not all(
        isinstance(item, str) and item.strip() == item and item for item in value
    ):
        raise ValueError("Episode selection requires nonempty canonical string IDs")
    if len(value) != len(set(value)):
        raise ValueError("Episode selection contains duplicate IDs")
    return value.copy()


class EpisodeSource(ABC):
    """Read benchmark episodes without exposing simulator/runtime dependencies."""

    def __init__(self, config: dict[str, Any]):
        self.config = dict(config)
        self.root = Path(self.config["root"])
        self.name = self.config["name"]
        self.split = self.config.get(
            "split", "val" if self.name == "objnav" else "val_unseen"
        )

    def load(self, selection: list[str] | None = None) -> list[EpisodeSpec]:
        requested = selection
        if requested is None:
            ids = self.config.get("ids")
            selection_file = self.config.get("selection_file")
            if ids is not None and selection_file is not None:
                raise ValueError("Set either ids or selection_file, not both")
            requested = (
                read_selection(selection_file) if selection_file is not None else ids
            )
        if requested is not None:
            requested = validate_selection(requested)
        limit = self.config.get("limit")
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("limit must be a positive integer")
        # Bound the selected shard set before loading large GOAT/ObjectNav files.
        if requested is not None and limit is not None:
            requested = requested[:limit]
        episodes = self._read(requested)
        canonical = {episode.id: episode for episode in episodes}
        if len(canonical) != len(episodes):
            raise ValueError("Dataset contains duplicate canonical episode IDs")
        if requested is not None:
            selected = []
            for identifier in requested:
                if identifier not in canonical:
                    raw_matches = [
                        episode.id
                        for episode in episodes
                        if episode.source_id == identifier
                    ]
                    if len(raw_matches) > 1:
                        detail = "ambiguous source ID; use a canonical ID"
                    elif raw_matches:
                        detail = f"not a canonical ID; use {raw_matches[0]!r}"
                    else:
                        detail = "not found"
                    raise ValueError(f"Episode {identifier!r}: {detail}")
                selected.append(canonical[identifier])
        else:
            selected = episodes[:limit] if limit is not None else episodes
        if not selected:
            raise ValueError("Dataset contains no evaluable episodes")
        return selected

    @abstractmethod
    def _read(self, requested: list[str] | None) -> list[EpisodeSpec]:
        """Decode selected episodes and their private goal metadata."""

    @staticmethod
    def _wanted(identifier: str, source_id: str, requested: list[str] | None) -> bool:
        return requested is None or identifier in requested or source_id in requested

    def _shards(self, requested: list[str] | None) -> list[Path]:
        directory = self.root / self.split / "content"
        paths = sorted(directory.glob("*.json.gz"))
        if not paths:
            raise FileNotFoundError(f"No {self.name} content shards under {directory}")
        prefix = f"{self.name}:{self.split}:"
        if requested and all(identifier.startswith(prefix) for identifier in requested):
            shards = {identifier.split(":")[2] for identifier in requested}
            paths = [
                path for path in paths if path.name.removesuffix(".json.gz") in shards
            ]
        return paths
