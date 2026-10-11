"""Append-only interaction events and independent, durable observation files."""

import json
from pathlib import Path
import time
from typing import Any, Iterator

from src.common.io import redact


class TraceWriter:
    def __init__(self, directory: Path, episode_id: str, secrets: tuple[str, ...] = ()):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "trace.jsonl"
        self.episode_id = episode_id
        self.goal_id: int | None = None
        self.secrets = secrets
        self._stream = self.path.open("a", encoding="utf-8")
        self._index = 0

    def emit(self, event: str, **fields: Any) -> None:
        self._index += 1
        row = redact({"event": event, "index": self._index, "episode_id": self.episode_id, "goal_id": self.goal_id,
                      "timestamp": time.time(), **fields}, self.secrets)
        self._stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


class TraceReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def events(self) -> Iterator[dict]:
        with self.path.open(encoding="utf-8") as stream:
            for line in stream:
                yield json.loads(line)

    def decisions(self) -> Iterator[dict]:
        """Pair actual model conditions and outputs; no reconstructed extra history."""
        requests = {}
        for row in self.events():
            if row["event"] == "model_request":
                requests[row["request_id"]] = row
            elif row["event"] == "model_response":
                request = requests.pop(row["request_id"], None)
                if request is not None:
                    yield {"episode_id": row["episode_id"], "request": request, "response": row}
