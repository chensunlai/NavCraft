"""One application-owned audit stream, fed by actual Core events."""

from datetime import datetime, timezone
import json
from pathlib import Path


class TraceWriter:
    def __init__(self, directory, secrets=()):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._stream = (directory / "trace.jsonl").open("a", encoding="utf-8")
        self._secrets = tuple(secret for secret in secrets if secret)
        self.types = {}
        self.compactions = []
        self.calls = []
        self.closed = False

    def redact_with(self, *secrets):
        self._secrets += tuple(secret for secret in secrets if secret)

    @property
    def position(self):
        """Byte offset used to delimit one prompt in the shared audit stream."""
        return self._stream.tell()

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
        if kind == "compaction_end" and isinstance(data, dict) and data.get("result"):
            self.compactions.append(data["result"])
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
