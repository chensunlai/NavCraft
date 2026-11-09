"""Persistent Pi session around an independently usable navigation executor."""

from dataclasses import asdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
import uuid

from src.common.io import write_json
from .config import core_options
from .core_client import CoreClient
from .executor import NavigationExecutor
from .request import PromptLimits, PromptResult
from .workspace import read_snapshot
from .service_retry import charged_requests, wait_seconds


class HarnessSession:
    def __init__(self, repo, config, directory, environment, scene_map=None, initial_state=None,
                 *, client_factory=CoreClient):
        self.repo, self.directory = Path(repo).resolve(), Path(directory).resolve()
        self.config = config
        self.client = None
        self.session_id = None
        self._closed = self._busy = False
        self._lock = threading.Lock()
        self.navigation = NavigationExecutor(
            directory, environment, scene_map, initial_state=initial_state,
            memory_type=config.get("memory_type", "entity"), tool_tier=config.get("tool_tier", "basic"),
            goto_planner=config.get("goto_planner", "local"))
        nav = self.navigation
        try:
            options = core_options(config, self.repo, self.directory, nav.workspace, nav.tool_specs)
            state = initial_state or {}
            if state.get("session_file"):
                session_file = nav.private / "restored-session.jsonl"
                shutil.copyfile(state["session_file"], session_file)
                options["sessionFile"] = str(session_file)
            elif state.get("history"):
                options["initialMessages"] = deepcopy(state["history"])
            nav.trace._secrets = (options["auth"]["apiKey"],)
            source_files = []
            for folder in ("harness", "environment", "bench", "common", "dataset/episodes"):
                source_files.extend((self.repo / "src" / folder).rglob("*.py"))
            source_files.extend((self.repo / "src/agent-core").glob("*.ts"))
            source_files.append(self.repo / "src/agent-core/pi/packages/coding-agent/src/core/compaction/compaction.ts")
            digest = hashlib.sha256()
            for file in sorted(source_files):
                digest.update(file.relative_to(self.repo).as_posix().encode() + b"\0" + file.read_bytes())
            nav.trace.emit("application_started", {
                "model": config["model"], "reasoning_effort": config.get("reasoning_effort", "high"),
                "source_sha256": digest.hexdigest(),
                "prompt_sha256": hashlib.sha256(options["systemPrompt"].encode()).hexdigest()})
            self.client = client_factory(self.repo, nav.private)
            self.session_id = self.client.request("init", options, timeout=120)["session_id"]
        except BaseException:
            self.close()
            raise

    def snapshot(self, destination):
        """Export paired Pi history and navigation assets at an idle request boundary."""
        with self._lock:
            if self._busy or self._closed or self.navigation.broken:
                raise RuntimeError("Cannot snapshot a busy, closed or uncertain session")
            exported = self.client.request("export_state")
            state = self.navigation.snapshot(destination)
            state["history"] = exported["messages"]
            if exported.get("session_file"):
                private = Path(destination) / "private"
                private.mkdir(exist_ok=True)
                shutil.copyfile(exported["session_file"], private / "pi-session.jsonl")
                state["session_file"] = "private/pi-session.jsonl"
            write_json(Path(destination) / "state.json", state)
            return state

    def _event(self, value, callback):
        self.navigation.trace.accept(value)
        if callback:
            callback(value)

    def prompt(self, text, images=(), limits=None, on_event=None):
        limits = limits or PromptLimits()
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Expected nonempty user request")
        if not isinstance(limits, PromptLimits):
            raise TypeError("limits must be PromptLimits")
        nav = self.navigation
        with self._lock:
            if self._busy or self._closed or nav.broken:
                raise RuntimeError("Session is busy, closed or has uncertain execution")
            refs, targets = [nav.current["path"]], []
            captions = [f"Image 1: [CURRENT {nav.current['frame']}]"]
            for ref in images:
                row = nav.observations.get(ref) if ref in nav.observations.items else nav.observations.add(ref, kind="target")
                refs.append(row["path"])
                targets.append(row["frame"])
                captions.append(f"Image {len(refs)}: [TARGET {row['frame']}]")
            nav.begin_request(text, targets, limits)
            self._busy = True
        started, prompt_id = time.monotonic(), uuid.uuid4().hex
        before = wait_before = None
        action_start = nav.environment.steps
        settled = {}
        reason, error_text = "runtime_error", None
        try:
            before, wait_before = charged_requests(nav.private), wait_seconds(nav.private)
            model_text = text + "\nCurrent map pose: " + json.dumps(nav.current["pose"]) + "\n" + "\n".join(captions)
            nav.trace.emit("prompt_start", {"prompt_id": prompt_id, "text": text,
                                           "images": refs, "limits": asdict(limits)})
            self.client.request("configure", {
                "maxRequests": before + limits.max_requests,
                "allowedTools": ["read", "write", "edit", "bash", *[t["name"] for t in nav.tool_specs]]})
            for attempt in range(2):
                active_seconds = time.monotonic() - started - (wait_seconds(nav.private) - wait_before)
                if active_seconds >= limits.seconds:
                    reason = "time_budget"
                    break
                settled = self.client.run_prompt(
                    model_text, refs, nav.execute, limits.seconds - active_seconds,
                    on_event=lambda value: self._event(value, on_event), cancel_event=nav.cancel,
                    prompt_id=prompt_id if not attempt else prompt_id + "-continue", max_turns=limits.max_turns)
                if (settled.get("reason") != "no_action_output" or nav.request.submitted or nav.broken or attempt
                        or nav.request.stop_reason or charged_requests(nav.private) - before >= limits.max_requests):
                    break
                model_text, refs = "Your last response contained no visible answer or tool call. Continue the current request using the available tools, and call submit when complete.", []
                nav.trace.emit("empty_output_continuation", {"prompt_id": prompt_id})
            if reason != "time_budget":
                reason = "submitted" if nav.request.submitted else nav.request.stop_reason or settled.get("reason", "runtime_error")
            if reason == "model_finished":
                reason = "unsubmitted"
            error_text = settled.get("error")
            if nav.broken:
                reason = nav.request.stop_reason or "runtime_error"
        except Exception as error:
            reason, error_text = "runtime_error", f"{type(error).__name__}: {error}"
            nav.broken = True
        finally:
            try:
                nav.end_request()
                elapsed = time.monotonic() - started
                if wait_before is not None:
                    elapsed -= wait_seconds(nav.private) - wait_before
                used = charged_requests(nav.private) - before if before is not None else 0
                answer = nav.request.answer if nav.request.submitted else settled.get("answer", "")
                result = PromptResult(prompt_id, nav.request.submitted, answer, reason,
                                      used, nav.environment.steps - action_start,
                                      f"private/prompts/{prompt_id}.json", max(0, elapsed), nav.trace.sanitize(error_text))
                nav.trace.emit("prompt_end", result.to_dict())
                write_json(self.directory / result.record, result.to_dict())
            except BaseException:
                nav.broken = True
                raise
            finally:
                with self._lock:
                    self._busy = False
        return result

    def abort(self):
        self.navigation.cancel.set()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            if self.client:
                self.client.close()
                self.client.drain_events(self.navigation.trace.accept)
        finally:
            self.navigation.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def open_session(repo, config, directory, environment, scene_map=None, initial_state=None, **kwargs):
    return HarnessSession(repo, config, directory, environment, scene_map, initial_state, **kwargs)

