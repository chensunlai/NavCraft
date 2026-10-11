"""Explicitly loaded navigation tools; calls are routed through a session."""

from src.harness.agent.base import ToolCall, ToolResult, ToolSpec


def load_builtin_tools() -> None:
    from . import navigation  # noqa: F401


class ToolSet:
    def __init__(self, session):
        self.session = session

    def specs(self) -> list[ToolSpec]:
        return [tool.spec() for tool in self.session.tools.values()]

    def call(self, call: ToolCall) -> ToolResult:
        return self.session.execute(call)
