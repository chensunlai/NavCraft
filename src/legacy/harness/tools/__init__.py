"""Explicitly loaded navigation tools; calls are routed through a session."""

from src.legacy.harness.agent.base import ToolCall, ToolResult, ToolSpec


def load_builtin_tools() -> None:
    from . import files, navigation  # noqa: F401


class ToolSet:
    def __init__(self, session):
        self.session = session

    def specs(self) -> list[ToolSpec]:
        return [tool.spec() for tool in self.session.tools.values()]

    def call(self, call: ToolCall) -> ToolResult:
        return self.session.execute(call)

    def save_summary(self, text: str) -> str | None:
        if self.session.assets is not None:
            return self.session.assets.write_summary(text)
        return None
