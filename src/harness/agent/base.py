"""Model and runtime contracts, independent of navigation implementation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any] | str


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass
class ToolResult:
    call_id: str
    ok: bool
    data: dict[str, Any]
    images: list[Path] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False


@dataclass
class ModelRequest:
    request_id: str
    items: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    timeout_seconds: float | None = None


@dataclass
class ModelTurn:
    items: list[dict[str, Any]]
    calls: list[ToolCall]
    usage: dict[str, Any] = field(default_factory=dict)
    status: str = "completed"


class ModelBackend(ABC):
    @abstractmethod
    def complete(self, request: ModelRequest) -> ModelTurn:
        """Return complete protocol items, including opaque reasoning items."""

    def close(self) -> None:
        """Release connections; stateless backends need no cleanup."""


@dataclass
class AgentLimits:
    max_requests: int = 32
    timeout_seconds: float = 1200
    max_context_chars: int = 1_000_000


@dataclass
class AgentResult:
    reason: str
    requests: int
    usage: dict[str, Any]
    elapsed_seconds: float
    error: str | None = None
