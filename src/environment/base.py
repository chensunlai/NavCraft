"""Private environment records; the Harness constructs model-facing messages."""

from dataclasses import dataclass, field
from pathlib import Path

from src.common.actions import ACTIONS as ACTIONS, TRANSLATION_ACTIONS as TRANSLATION_ACTIONS


@dataclass(frozen=True)
class MotionResult:
    collision: bool = False
    blocked: bool = False
    auto_yaw_delta_degrees: float = 0.0
    samples: list[dict] = field(default_factory=list)

    @property
    def positions(self) -> list[list[float]]:
        return [sample["position"] for sample in self.samples]


@dataclass(frozen=True)
class Observation:
    frame_id: str
    rgb: Path
    depth: Path | None
    pose: dict
    camera: dict
