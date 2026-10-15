"""Public navigation contracts; evaluator annotations never enter transitions."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

AtomicAction = str
TRANSLATION_ACTIONS = ("move_forward", "move_left", "move_right", "move_backward")


@dataclass(frozen=True)
class MotionResult:
    """One atomic action; absolute microstep states stay in private records."""

    collision: bool = False
    blocked: bool = False
    auto_yaw_delta_degrees: float = 0.0
    samples: list[dict] = field(default_factory=list)

    @property
    def positions(self) -> list[list[float]]:
        return [sample["position"] for sample in self.samples]

    def public_dict(self) -> dict:
        return {"collision": self.collision, "blocked": self.blocked,
                "auto_yaw_delta_degrees": self.auto_yaw_delta_degrees}


@dataclass(frozen=True)
class Observation:
    frame_id: str
    rgb: Path
    depth: Path | None
    pose: dict
    camera: dict

    def public_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "rgb": str(self.rgb),
            "depth": str(self.depth) if self.depth else None,
            "pose": self.pose,
            "camera": self.camera,
        }


@dataclass(frozen=True)
class PublicTask:
    goal_id: int
    modality: str
    instruction: str
    image: Path | None = None

    def public_dict(self) -> dict:
        return {
            "goal_id": self.goal_id,
            "modality": self.modality,
            "instruction": self.instruction,
            "image": str(self.image) if self.image else None,
        }


@dataclass(frozen=True)
class Transition:
    observation: Observation
    task: PublicTask | None
    feedback: dict
    goal_changed: bool = False
    terminated: bool = False
    truncated: bool = False
    reason: str | None = None

    def public_dict(self) -> dict:
        return {
            "observation": self.observation.public_dict(),
            "task": self.task.public_dict() if self.task else None,
            "feedback": self.feedback,
            "goal_changed": self.goal_changed,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "reason": self.reason,
        }


class NavigationInputError(ValueError):
    """Rejected navigation input; no atomic action has been executed."""


class Environment(Protocol):
    def reset(self, episode) -> Transition: ...
    def observe(self) -> Observation: ...
    def capture_current(self) -> Observation: ...
    def panorama(self) -> list[Observation]: ...
    def step(self, action: AtomicAction) -> Transition: ...
    def finish_goal(self) -> Transition: ...
    def finalize(self, reason: str) -> None: ...
    def metrics(self) -> dict: ...
    def close(self) -> None: ...
