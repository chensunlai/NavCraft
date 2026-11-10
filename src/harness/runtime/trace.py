"""One application-owned audit stream, fed by actual Core events."""

from datetime import datetime, timezone
import base64
import hashlib
import json
from pathlib import Path
import re


def resolve_trace(value, directory, *, images=False):
    """Resolve audit message references; optionally restore inline image payloads.

    Image references remain small by default so reading a trace does not
    recreate every historical image in every request.
    """
    if isinstance(value, list):
        return [resolve_trace(item, directory, images=images) for item in value]
    if not isinstance(value, dict):
        return value
    if "$trace_blob" not in value:
        return {key: resolve_trace(child, directory, images=images) for key, child in value.items()}
    digest, encoding = value["$trace_blob"], value.get("encoding")
    if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("Invalid trace blob digest")
    if encoding not in {"json", "base64", "data-url"}:
        raise ValueError("Unknown trace blob encoding")
    if encoding != "json" and not images:
        return value.copy()
    path = Path(directory) / "blobs" / digest
    if path.is_symlink():
        raise ValueError("Trace blobs cannot be symlinks")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("Trace blob hash mismatch")
    if encoding == "json":
        return resolve_trace(json.loads(content), directory, images=images)
    data = base64.b64encode(content).decode("ascii")
    if encoding == "data-url":
        prefix = value.get("prefix", "")
        if not re.fullmatch(r"data:image/[^;,]+;base64,", prefix):
            raise ValueError("Invalid trace image prefix")
        return prefix + data
    return data


def read_trace(path, *, images=False):
    """Stream audit events without retaining decoded images."""
    path = Path(path)
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield resolve_trace(json.loads(line), path.parent, images=images)


class TraceWriter:
    def __init__(self, directory, secrets=()):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self._stream = (directory / "trace.jsonl").open("a", encoding="utf-8")
        self._secrets = tuple(secret for secret in secrets if secret)
        self.types = {}
        self.calls = []
        self.closed = False

    def store(self, value):
        raw = json.dumps(self.sanitize(value), ensure_ascii=False, separators=(",", ":")).encode()
        digest = hashlib.sha256(raw).hexdigest()
        directory = self.directory / "blobs"
        directory.mkdir(exist_ok=True)
        path = directory / digest
        if not path.exists():
            path.write_bytes(raw)
        return {"$trace_blob": digest, "encoding": "json"}

    def sanitize(self, value):
        line = json.dumps(value, ensure_ascii=False)
        for secret in self._secrets:
            line = line.replace(secret, "[REDACTED]")
        return json.loads(line)

    def emit(self, kind, data=None, timestamp=None, seq=None, prompt_id=None):
        value = {"timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
                 "type": kind, "data": {} if data is None else data}
        if seq is not None:
            value["core_seq"] = seq
        if prompt_id is not None:
            value["prompt_id"] = prompt_id
        line = json.dumps(value, ensure_ascii=False)
        for secret in self._secrets:
            line = line.replace(secret, "[REDACTED]")
        self._stream.write(line + "\n")
        self._stream.flush()
        self.types[kind] = self.types.get(kind, 0) + 1
        if kind == "tool_execution_end" and isinstance(data, dict):
            self.calls.append({"name": data.get("toolName"), "isError": data.get("isError", False)})

    def accept(self, message):
        if message.get("type") == "event":
            event = message["event"]
            self.emit(event["type"], event.get("data"), event.get("timestamp"), message.get("seq"),
                      message.get("prompt_id"))
        elif message.get("type") == "tool_request":
            # Pi's tool_execution_start already records the actual call.
            return
        else:
            self.emit("core_" + message["type"], message)

    def close(self):
        if not self.closed:
            self.closed = True
            self._stream.close()
