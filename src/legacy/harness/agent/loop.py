"""Synchronous tool loop with explicit termination and context maintenance."""

from dataclasses import asdict
import time
from typing import Any, Protocol

from src.common.register import runtime_registry

from .base import (
    AgentLimits,
    AgentResult,
    ModelBackend,
    ModelRequest,
    ModelTurn,
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


_SUMMARY_INSTRUCTION = """Summarize the entire completed navigation transcript above,
including the latest RGB and tool feedback, not only the older prefix.
Return concise plain text, not tool calls. Preserve the current target and instruction
constraints, estimated floor/location/heading and their uncertainty, route progress,
landmark identities, collisions or failures, unresolved contradictions, and exact image
IDs/file references supporting conclusions. Include the earlier summary when present.
Separate observations from hypotheses. A planned action, guessed room, or tentative
object identity is not a confirmed observation or arrival. Do not promote plans or
new visual guesses into confirmed objects, rooms, or reached-goal facts. Confirmation
requires explicit supporting evidence; retain unresolved identities and locations as
hypotheses with their uncertainty. Do not invent positions or use any hidden
simulator state. Images attached to initial messages and tool results were already
provided to the model without a view_images call. Absence of view_images does not
mean a map/image was unavailable, never seen, or never inspected. Describe explicit
interpretations and actions supported by the transcript; if no interpretation was
stated, say only that no explicit interpretation was recorded. Do not invent whether
an image was attended to internally. Recent complete turns will be retained separately
in their original protocol form. Use them to resolve the timeline and current state,
but do not override them with unsupported summary claims. This summary will
replace the older transcript, so retain facts needed to continue navigation."""


def _summary_text(turn: ModelTurn) -> str:
    parts = []
    for item in turn.items:
        if item.get("type") != "message" or item.get("role") != "assistant":
            continue
        for part in item.get("content", []):
            if isinstance(part, dict) and part.get("type") == "output_text":
                parts.append(str(part.get("text", "")))
    return "\n".join(parts).strip()


@runtime_registry.register("agent")
class AgentRuntime:
    def __init__(self, model: ModelBackend, config: dict[str, Any]) -> None:
        self.model = model
        self.config = dict(config)
        self.context: AgentContext | None = None
        policy = config.get("context")
        self.context_policy = None
        if policy is not None:
            if not isinstance(policy, dict):
                raise ValueError("harness.context must be a mapping")
            self.context_policy = {
                "image_prune_after": 10,
                "image_keep": 10,
                "compact_tokens": 100000,
                "compact_chars": 400000,
                "keep_turns": 5,
                **policy,
            }
            for key, value in self.context_policy.items():
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    raise ValueError(f"Context policy {key} must be a positive integer")
            if (
                self.context_policy["image_keep"]
                > self.context_policy["image_prune_after"]
            ):
                raise ValueError("image_keep must not exceed image_prune_after")

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

        def complete(
            items: list[dict], tool_specs: list[dict], *, purpose: str = "navigation"
        ) -> ModelTurn:
            nonlocal requests
            requests += 1
            remaining = limits.timeout_seconds - (time.monotonic() - started)
            request = ModelRequest(
                f"request-{requests:06d}", items, tool_specs, remaining
            )
            trace.emit(
                "model_request",
                request_id=request.request_id,
                items=request.items,
                tools=request.tools,
                timeout_seconds=remaining,
                purpose=purpose,
                options=(
                    self.model.request_options()
                    if hasattr(self.model, "request_options")
                    else {}
                ),
            )
            request_started = time.monotonic()
            turn = self.model.complete(request)
            _add_usage(usage, turn.usage)
            trace.emit(
                "model_response",
                request_id=request.request_id,
                items=turn.items,
                calls=[asdict(call) for call in turn.calls],
                usage=turn.usage,
                status=turn.status,
                purpose=purpose,
                latency_seconds=time.monotonic() - request_started,
            )
            return turn

        while requests < limits.max_requests:
            if time.monotonic() - started >= limits.timeout_seconds:
                return finish("timeout")
            if self.context.size_chars > limits.max_context_chars:
                return finish("context_budget")
            try:
                turn = complete(self.context.snapshot(), specs)
            except Exception as exc:
                return finish("model_error", f"{type(exc).__name__}: {exc}")
            self.context.append_output(turn.items)
            if time.monotonic() - started >= limits.timeout_seconds:
                return finish("timeout")
            if turn.status != "completed":
                return finish("model_incomplete", f"Response status: {turn.status}")
            if not turn.calls:
                # Completion text has no authority to issue navigation STOP.
                return finish("model_finished")
            stop_reason: str | None = None
            goal_changed = False
            for call in turn.calls:
                skip_reason = stop_reason or ("goal_changed" if goal_changed else None)
                if skip_reason:
                    result = ToolResult(
                        call.call_id,
                        False,
                        {
                            "error": (
                                "Tool not executed because the target changed"
                                if goal_changed and not stop_reason
                                else "Tool not executed because the episode already ended"
                            ),
                        },
                    )
                    trace.emit("tool_skipped", call=asdict(call), reason=skip_reason)
                elif time.monotonic() - started >= limits.timeout_seconds:
                    stop_reason = "timeout"
                    result = ToolResult(
                        call.call_id, False, {"error": "Agent time budget exhausted"}
                    )
                    trace.emit("tool_skipped", call=asdict(call), reason=stop_reason)
                else:
                    try:
                        result = tools.call(call)
                    except Exception as exc:
                        return finish(
                            "environment_error", f"{type(exc).__name__}: {exc}"
                        )
                    goal_changed = goal_changed or result.goal_changed
                    if result.terminated:
                        stop_reason = "terminated"
                    elif result.truncated:
                        stop_reason = "truncated"
                self.context.append_tool_result(result)
                trace.emit(
                    "tool_message",
                    request_id=f"request-{requests:06d}",
                    call_id=result.call_id,
                    ok=result.ok,
                    data=result.data,
                    images=[str(path) for path in result.images],
                    terminated=result.terminated,
                    truncated=result.truncated,
                    goal_changed=result.goal_changed,
                    image_roles=result.image_roles,
                )
            if stop_reason:
                return finish(stop_reason)
            policy = self.context_policy
            if policy is None:
                continue
            pruned = self.context.prune_images(
                policy["image_prune_after"], policy["image_keep"],
                max_images=policy.get("image_max"),
            )
            if pruned is not None:
                trace.emit(
                    "context_pruned", **pruned, size_chars=self.context.size_chars
                )
            input_tokens = turn.usage.get("input_tokens")
            due = (
                input_tokens >= policy["compact_tokens"]
                if isinstance(input_tokens, (int, float))
                else self.context.size_chars >= policy["compact_chars"]
            )
            older_history = (
                self.context.compaction_source(policy["keep_turns"]) if due else None
            )
            if older_history is None:
                continue
            if requests >= limits.max_requests:
                return finish("request_budget")
            if time.monotonic() - started >= limits.timeout_seconds:
                return finish("timeout")
            before = self.context.size_chars
            # The cutoff controls what is replaced, not what the summary may see.
            # Latest RGB and feedback must remain visible to avoid stale-state claims.
            summary_input = [
                *self.context.snapshot(),
                {"role": "user", "content": _SUMMARY_INSTRUCTION},
            ]
            if len(str(summary_input)) > limits.max_context_chars:
                return finish("context_budget")
            try:
                summarized = complete(summary_input, [], purpose="context_compaction")
                if time.monotonic() - started >= limits.timeout_seconds:
                    return finish("timeout")
                summary = _summary_text(summarized)
                if summarized.status != "completed" or summarized.calls or not summary:
                    raise ValueError(
                        "Compaction response must be completed plain text without tool calls"
                    )
                saver = getattr(tools, "save_summary", None)
                reference = saver(summary) if saver else None
                self.context.compact(summary, policy["keep_turns"], reference)
                trace.emit(
                    "context_compacted",
                    summary=summary,
                    resource=reference,
                    before_chars=before,
                    after_chars=self.context.size_chars,
                    keep_turns=policy["keep_turns"],
                )
            except Exception as exc:
                # Original transcript remains available; failure is never a fabricated summary.
                message = f"{type(exc).__name__}: {exc}"
                trace.emit("context_compaction_failed", error=message)
                return finish("context_compaction_error", message)
        return finish("request_budget")

    def close(self) -> None:
        self.model.close()
