"""Bidirectional Pi transport. Tool callbacks execute in the caller's thread."""

import json
from pathlib import Path
import queue
import subprocess
import threading
import time
import uuid


class CoreError(RuntimeError):
    pass


class CoreClient:
    def __init__(self, repo, directory, command=None):
        self.repo, self.directory = Path(repo), Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._stderr = (self.directory / "core.log").open("a")
        command = command or ["node", "src/agent-core/dist/server.js"]
        self.process = subprocess.Popen(command, cwd=self.repo, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, bufsize=1)
        self._events = queue.Queue()
        self._responses = {}
        self._condition = threading.Condition()
        self._write_lock = threading.Lock()
        self._closed = False
        self._secrets = ()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._reader.start()
        self._stderr_reader.start()

    def _read_stderr(self):
        for line in self.process.stderr:
            for secret in self._secrets:
                line = line.replace(secret, "[REDACTED]")
            self._stderr.write(line)
            self._stderr.flush()

    def _read(self):
        try:
            for line in self.process.stdout:
                value = json.loads(line)
                if "id" in value and "ok" in value:
                    with self._condition:
                        self._responses[value["id"]] = value
                        self._condition.notify_all()
                else:
                    self._events.put(value)
        except Exception as error:
            self._events.put({"type": "transport_error", "error": str(error)})
        finally:
            self._events.put({"type": "transport_eof"})
            with self._condition:
                self._condition.notify_all()

    def request(self, op, args=None, timeout=30):
        request_id = uuid.uuid4().hex
        message = {"id": request_id, "op": op, "args": args or {}}
        if op == "init":
            secret = (message["args"].get("auth") or {}).get("apiKey")
            if secret:
                self._secrets += (secret,)
        with self._write_lock:
            if self.process.poll() is not None:
                raise CoreError(f"Core exited with code {self.process.returncode}")
            try:
                self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
                self.process.stdin.flush()
            except OSError as error:
                raise CoreError(f"Core transport write failed: {error}") from error
        deadline = time.monotonic() + timeout
        with self._condition:
            while request_id not in self._responses:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CoreError(f"Core {op} acknowledgment timed out")
                if self.process.poll() is not None:
                    raise CoreError(f"Core exited while waiting for {op}")
                self._condition.wait(min(remaining, 0.2))
            response = self._responses.pop(request_id)
        if not response["ok"]:
            raise CoreError(response.get("error", f"Core {op} failed"))
        return response.get("data")

    def run_prompt(self, text, images, tool_handler, seconds, on_event=None,
                   cancel_event=None):
        """Pump callbacks while Pi runs; never wait for settled before handling tools."""
        self.request("prompt", {"text": text, "images": images, "seconds": seconds})
        deadline = time.monotonic() + seconds + 5
        abort_sent = False
        while True:
            if cancel_event is not None and cancel_event.is_set() and not abort_sent:
                self.request("abort", timeout=5)
                abort_sent = True
                deadline = min(deadline, time.monotonic() + 5)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.request("abort", timeout=5)
                raise CoreError("Core prompt deadline exceeded")
            try:
                message = self._events.get(timeout=min(remaining, 0.2))
            except queue.Empty:
                continue
            if on_event:
                on_event(message)
            if message["type"] == "tool_request":
                try:
                    reply = tool_handler(message["call_id"], message["name"], message["args"])
                except Exception as error:
                    reply = {"result": {"content": [{"type": "text", "text":
                             f"{type(error).__name__}: {error}"}], "isError": True}}
                self.request("tool_reply", {"call_id": message["call_id"], **reply})
            elif message["type"] == "settled":
                return message
            elif message["type"] in {"transport_error", "transport_eof"}:
                raise CoreError(message.get("error", "Core transport closed before settled"))

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.process.poll() is None:
                try:
                    self.request("close", timeout=5)
                except Exception:
                    self.process.terminate()
                try:
                    self.process.stdin.close()
                except OSError:
                    pass
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        finally:
            self._reader.join(timeout=1)
            self._stderr_reader.join(timeout=1)
            self.process.stdout.close()
            self.process.stderr.close()
            self._stderr.close()

    def drain_events(self, on_event):
        """Archive final snapshots emitted during close without running new tools."""
        while True:
            try:
                message = self._events.get_nowait()
            except queue.Empty:
                return
            if message["type"] != "transport_eof":
                on_event(message)
