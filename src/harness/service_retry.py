"""Shared accounting for service outages recorded by the Agent Core."""

import json
import re
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


def unavailable_result(result):
    """Only settled service failures are safe to resume automatically."""
    if (result.get("finish_reason") != "model_error" or result.get("recovery_error")
            or result.get("worker_error") or result.get("cleanup_error")):
        return False
    error = (result.get("last_error")
             or (result.get("prompt_result", {}).get("last") or {}).get("errorMessage") or "")
    if re.search(r"gateway_concurrency_limit|concurrency limit exceeded for account", error, re.I):
        return True
    code = re.search(r"(?:API error\s*\(|\bHTTP\s+)(\d{3})", error, re.I)
    if code:
        return int(code[1]) in {408, 429, 500, 502, 503, 504, 529}
    return bool(re.search(r"(?:service|upstream).*?(?:temporarily unavailable|unavailable|overloaded)|(?:upstream response )?stream (?:was interrupted|interrupted|ended unexpectedly)|fetch failed|connection (?:error|reset)|request timed out|operation was aborted due to timeout", error, re.I))
