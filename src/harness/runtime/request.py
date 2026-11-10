"""One user request and its execution budget; independent of Agent transport."""

from dataclasses import asdict, dataclass, field
import math


@dataclass(frozen=True)
class PromptLimits:
    max_requests: int = 150
    max_actions: int = 500
    seconds: float = 3600
    max_path_length: float | None = None
    max_turns: int | None = None
    max_observations: int | None = None

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in (self.max_requests, self.max_actions)):
            raise ValueError("Request and action limits must be positive integers")
        if not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError("seconds must be finite and positive")
        if self.max_path_length is not None and (not math.isfinite(self.max_path_length) or self.max_path_length <= 0):
            raise ValueError("max_path_length must be finite and positive")
        if self.max_observations is not None and (type(self.max_observations) is not int or self.max_observations < 0):
            raise ValueError("max_observations must be a nonnegative integer")
        if self.max_turns is not None and (type(self.max_turns) is not int or self.max_turns < 1):
            raise ValueError("max_turns must be a positive integer")


@dataclass(frozen=True)
class PromptResult:
    prompt_id: str
    submitted: bool
    answer: str
    reason: str
    requests: int
    actions: int
    record: str
    elapsed_seconds: float
    error: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass
class RequestState:
    text: str = ""
    target_frames: list[str] = field(default_factory=list)
    submitted: bool = False
    answer: str = ""
    stop_reason: str | None = None
    action_end: int | None = None
    path_end: float | None = None
    observation_end: int | None = None
