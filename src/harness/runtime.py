"""Persistent Python API for successive prompts to one Pi Agent session."""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import threading
import time
import uuid

from .config import core_options
from .core_client import CoreClient
from .session import _write
from .trace import TraceWriter


class SessionBusyError(RuntimeError):
    """A prompt is still running; its receipts must settle before another call."""


@dataclass(frozen=True)
class PromptLimits:
    max_requests: int
    seconds: float = 600

    def __post_init__(self):
        if type(self.max_requests) is not int or self.max_requests <= 0:
            raise ValueError("max_requests must be a positive integer")
        if not isinstance(self.seconds, (int, float)) or not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError("seconds must be positive and finite")


@dataclass(frozen=True)
class PromptResult:
    prompt_id: str
    answer: str
    reason: str
    requests: int
    total_requests: int
    record: str
    elapsed_seconds: float
    trace_range: dict
    error: str | None = None
    last: dict | None = None
    usage: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class _Cancellation:
    def __init__(self, local, external):
        self.local, self.external = local, external

    def is_set(self):
        return self.local.is_set() or bool(self.external and self.external.is_set())


class HarnessSession:
    """Own the Core and tool resources; prompt completion never closes them.

    Navigation callbacks execute in the prompt caller's thread, as before.
    Bench code owns task preparation and scoring; this class accepts messages.
    """

    def __init__(self, repo, config, directory, workspace, system_prompt, tools, *,
                 max_requests, trace=None, on_event=None, cancel_event=None, compact=True):
        self.repo, self.directory = Path(repo).resolve(), Path(directory).resolve()
        self.workspace = Path(workspace).resolve()
        self.private = self.directory / "private"
        self.private.mkdir(parents=True, exist_ok=True)
        self.tools = tools
        self.default_limits = PromptLimits(max_requests)
        self.trace = trace or TraceWriter(self.private)
        self._owns_trace = trace is None
        self.on_event = on_event
        self._lock = threading.Lock()
        self._busy = self._closed = self._broken = False
        self._abort = threading.Event()
        self._cancel = _Cancellation(self._abort, cancel_event)
        self.tools.context.cancel_event = self._cancel
        if self.tools.context.bridge is not None:
            self.tools.context.bridge.cancel_event = self._cancel
        self._requests = 0
        self._execution_fault = False
        self.close_status = None
        self.cleanup_error = None
        self.client = None
        try:
            options = core_options(config, self.repo, self.directory, self.workspace,
                                   system_prompt, tools.specs(), max_requests,
                                   tools.checkpoint(), compact=compact)
            self.trace.redact_with(options["auth"]["apiKey"])
            self.client = CoreClient(self.repo, self.private)
            receipt = self.client.request("init", options)
            self.session_id = receipt["session_id"]
        except Exception:
            self.close()
            raise

    @property
    def requests(self):
        ledger = self.private / "requests.json"
        if ledger.exists():
            return len(json.loads(ledger.read_text())["attempts"])
        return self._requests

    def _idle(self):
        if self._closed:
            raise RuntimeError("Session is closed")
        if self._busy:
            raise SessionBusyError("A prompt is already running")
        if getattr(self.tools.context.bridge, "execution_uncertain", False) is True:
            self._broken = True
        if self._broken:
            raise RuntimeError("Session did not settle cleanly; close it before further use")

    def _event(self, message, callback=None):
        self.trace.accept(message)
        if message.get("event", {}).get("type") == "turn_start":
            self.tools.turn_start()
        if self.on_event:
            self.on_event(message)
        if callback and callback is not self.on_event:
            callback(message)

    def update_checkpoint(self):
        """Publish caller-prepared public state without appending a user message."""
        with self._lock:
            self._idle()
            return self.client.request("set_checkpoint", {"checkpoint": self.tools.checkpoint()})

    def _execute_tool(self, call_id, name, args):
        reply = self.tools.call(call_id, name, args)
        bridge = self.tools.context.bridge
        if getattr(bridge, "execution_uncertain", False) is True:
            self._execution_fault = self._broken = True
            self.trace.emit("tool_execution_uncertain", {"call_id": call_id, "name": name})
            # Keep the actual tool feedback, but collect the remaining receipts
            # before returning control to a caller that must stop this run.
            reply = {**reply, "control": {**reply.get("control", {}),
                                          "stop_remaining_tools": True, "end_run": True}}
        return reply

    def prompt(self, text, images=(), limits=None, on_event=None):
        if not isinstance(text, str):
            raise ValueError("prompt text must be a string")
        limits = limits or self.default_limits
        if not isinstance(limits, PromptLimits):
            raise TypeError("limits must be PromptLimits")
        images = list(images)
        with self._lock:
            self._idle()
            before = self.requests
            start_offset = self.trace.position
            self._busy = True
            self._abort.clear()
        prompt_id = uuid.uuid4().hex
        record = f"private/prompts/{prompt_id}.json"
        started = time.monotonic()
        settled = {"reason": "runtime_error", "answer": ""}
        try:
            self.trace.emit("prompt_start", {"prompt_id": prompt_id, "session_id": self.session_id,
                            "text": text, "images": images, "requests_before": before,
                            "limits": asdict(limits)})
            self.client.request("configure", {"maxRequests": before + limits.max_requests})
            self.client.request("set_checkpoint", {"checkpoint": self.tools.checkpoint()})
            settled = self.client.run_prompt(text, images, self._execute_tool, limits.seconds,
                                            on_event=lambda message: self._event(message, on_event),
                                            cancel_event=self._cancel, prompt_id=prompt_id)
            self._requests = settled.get("requests", self._requests)
            if self._execution_fault:
                settled = {**settled, "reason": "runtime_error",
                           "error": "Environment execution has no reliable receipt; no further prompt is allowed"}
            if settled.get("reason") in {"runtime_error", "tool_timeout", "tool_result_invalid"}:
                self._broken = True
        except Exception as error:
            # The old answer is never used when this prompt has no settled result.
            settled = {"reason": "runtime_error", "answer": "",
                       "error": f"{type(error).__name__}: {error}"}
            self._broken = True
        finally:
            try:
                total = self.requests
                data = {"prompt_id": prompt_id, "answer": settled.get("answer") or "",
                        "reason": settled.get("reason", "runtime_error"),
                        "requests": total - before, "total_requests": total,
                        "record": record, "elapsed_seconds": time.monotonic() - started,
                        "error": settled.get("error"), "last": settled.get("last"),
                        "usage": (settled.get("last") or {}).get("usage", {})}
                self.trace.emit("prompt_end", data)
                data["trace_range"] = {"path": "private/trace.jsonl",
                                       "start_byte": start_offset, "end_byte": self.trace.position}
                data = self.trace.sanitize(data)
                _write(self.directory / record, data)
                result = PromptResult(**data)
            finally:
                with self._lock:
                    self._busy = False
        return result

    def abort(self):
        """Request cancellation; the prompt pump still collects its final receipts."""
        with self._lock:
            if not self._closed and self._busy:
                self._abort.set()

    def close(self):
        with self._lock:
            if self._closed:
                return
            if self._busy:
                raise SessionBusyError("Abort and wait for the prompt to return before closing")
            self._closed = True
        errors = []
        bridge = self.tools.context.bridge
        if bridge is not None:
            try:
                bridge.close()
            except Exception as error:
                errors.append(f"Environment close: {error}")
        if self.client:
            try:
                self.client.close()
                self.close_status = self.client.close_status
                self.client.drain_events(self.trace.accept)
            except Exception as error:
                errors.append(f"Core close: {error}")
        self.cleanup_error = "; ".join(errors) or None
        if self._owns_trace:
            self.trace.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def open_session(repo, config, directory, workspace, system_prompt, tools, **options):
    """Open once; callers may submit any number of successive prompts."""
    return HarnessSession(repo, config, directory, workspace, system_prompt, tools, **options)
