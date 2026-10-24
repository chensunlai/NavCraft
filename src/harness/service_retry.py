"""Shared accounting for service outages recorded by the Agent Core."""

import json
from pathlib import Path
import time


def wait_seconds(private):
    path = Path(private) / "service-retry.json"
    if not path.exists():
        return 0.0
    state = json.loads(path.read_text())
    active = state.get("started_ms")
    return (state.get("completed_ms", 0) +
            (max(0, time.time() * 1000 - active) if active is not None else 0)) / 1000


def charged_requests(private, fallback=0):
    path = Path(private) / "requests.json"
    if not path.exists():
        return fallback
    return sum(not attempt.get("service_error", False)
               for attempt in json.loads(path.read_text())["attempts"])
