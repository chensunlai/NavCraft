"""Reserved structural-map input; no renderer or semantic map is implemented."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MapAsset:
    scene_id: str
    path: Path
    coordinate_frame: str = "habitat_world; XYZ, Y up"
