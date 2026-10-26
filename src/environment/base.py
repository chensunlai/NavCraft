"""Private environment records; the Harness constructs model-facing messages."""

from dataclasses import dataclass, field
import math
from pathlib import Path

import numpy as np
from PIL import Image

from src.common.actions import ACTIONS as ACTIONS, TRANSLATION_ACTIONS as TRANSLATION_ACTIONS

DEPTH_SCALE = 0.001
DEPTH_MIN = 0.01
DEPTH_MAX = 10.0


def save_depth(path: Path, values) -> None:
    """Save optical-axis depth in millimetres; zero denotes invalid/far pixels."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values) & (values >= DEPTH_MIN) & (values < DEPTH_MAX)
    encoded = np.zeros(values.shape, dtype=np.uint16)
    encoded[valid] = np.rint(values[valid] / DEPTH_SCALE).astype(np.uint16)
    Image.fromarray(encoded).save(path)


def turn_action(degrees) -> str:
    """Validate one native right-positive turn, without inventing a new action."""
    if isinstance(degrees, bool) or not isinstance(degrees, (int, float)) or not math.isfinite(degrees) or not 0 < abs(degrees) <= 30:
        raise ValueError("Turn degrees must be finite and satisfy 0 < abs(degrees) <= 30")
    return "turn_right" if degrees > 0 else "turn_left"


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
