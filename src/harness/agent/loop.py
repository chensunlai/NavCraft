"""Synchronous tool loop with explicit termination and append-only context."""

from dataclasses import asdict
import time
from typing import Any, Protocol

from src.common.register import runtime_registry

from .base import (
    AgentLimits,
    AgentResult,
    ModelBackend,
    ModelRequest,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from .context import AgentContext


class RuntimeTools(Protocol):
    def specs(self) -> list[ToolSpec]: ...

    def call(self, call: ToolCall) -> ToolResult: ...


class RuntimeTrace(Protocol):
    def emit(self, event: str, **fields: Any) -> None: ...


def _add_usage(total: dict[str, Any], current: dict[str, Any]) -> None:
    for key, value in current.items():
        if isinstance(value, dict):
            nested = total.setdefault(key, {})
            if isinstance(nested, dict):
                _add_usage(nested, value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            total[key] = total.get(key, 0) + value


@runtime_registry.register("agent")
class AgentRuntime:
    def __init__(self, model: ModelBackend, config: dict[str, Any]) -> None:
        self.model = model
        self.config = dict(config)
        self.context: AgentContext | None = None

    def run(
        self,
        initial: list[dict[str, Any]],
        tools: RuntimeTools,
        trace: RuntimeTrace,
        limits: AgentLimits,
    ) -> AgentResult:
        if (
            limits.max_requests <= 0
            or limits.timeout_seconds <= 0
            or limits.max_context_chars <= 0
        ):
            raise ValueError("Agent budgets must all be positive")
        started = time.monotonic()
        self.context = AgentContext(initial)
        specs = [spec.to_dict() for spec in tools.specs()]
        usage: dict[str, Any] = {}
        requests = 0

        def finish(reason: str, error: str | None = None) -> AgentResult:
            result = AgentResult(
                reason, requests, usage, time.monotonic() - started, error
            )
            trace.emit("agent_end", **asdict(result))
            return result

        while requests < limits.max_requests:
            remaining = limits.timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                return finish("timeout")
            if self.context.size_chars > limits.max_context_chars:
                return finish("context_budget")
            requests += 1
            request = ModelRequest(
                f"request-{requests:06d}", self.context.snapshot(), specs, remaining
            )
            trace.emit(
                "model_request",
                request_id=request.request_id,
                items=request.items,
                tools=request.tools,
                timeout_seconds=remaining,
                options=(
                    self.model.request_options()
                    if hasattr(self.model, "request_options")
                    else {}
                ),
            )
            try:
                turn = self.model.complete(request)
            except Exception as exc:
                # A failed episode remains observable; runner proceeds to other episodes.
                return finish("model_error", f"{type(exc).__name__}: {exc}")
            _add_usage(usage, turn.usage)
            self.context.append_output(turn.items)
            trace.emit(
                "model_response",
                request_id=request.request_id,
                items=turn.items,
                calls=[asdict(call) for call in turn.calls],
                usage=turn.usage,
                status=turn.status,
            )
            if time.monotonic() - started >= limits.timeout_seconds:
                return finish("timeout")
            if turn.status != "completed":
                return finish("model_incomplete", f"Response status: {turn.status}")
            if not turn.calls:
                # Completion text has no authority to issue navigation STOP.
                return finish("model_finished")
            stop_reason: str | None = None
            for call in turn.calls:
                if stop_reason is not None:
                    result = ToolResult(
                        call.call_id,
                        False,
                        {"error": "Tool not executed because the episode already ended"},
                    )
                    trace.emit("tool_skipped", call=asdict(call), reason=stop_reason)
                elif time.monotonic() - started >= limits.timeout_seconds:
                    stop_reason = "timeout"
                    result = ToolResult(call.call_id, False, {"error": "Agent time budget exhausted"})
                    trace.emit("tool_skipped", call=asdict(call), reason=stop_reason)
                else:
                    try:
                        result = tools.call(call)
                    except Exception as exc:
                        return finish("environment_error", f"{type(exc).__name__}: {exc}")
                    if result.terminated:
                        stop_reason = "terminated"
                    elif result.truncated:
                        stop_reason = "truncated"
                self.context.append_tool_result(result)
                trace.emit(
                    "tool_message",
                    request_id=request.request_id,
                    call_id=result.call_id,
                    ok=result.ok,
                    data=result.data,
                    images=[str(path) for path in result.images],
                    terminated=result.terminated,
                    truncated=result.truncated,
                )
            if stop_reason:
                return finish(stop_reason)
        return finish("request_budget")

    def close(self) -> None:
        self.model.close()
