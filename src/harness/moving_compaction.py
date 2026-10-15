"""Private instrumentation for one same-session, moving compaction diagnostic."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .session import _write


NOTE = "memory/moving_compaction.md"


def snapshot(session, workspace, checkpoint):
    """Capture existing state only; never observe, move or expose private pose."""
    task = session.env.task
    workspace = Path(workspace)
    state = deepcopy(task.private_state())
    records = [json.loads(line) for line in (workspace / "observations/index.jsonl").read_text().splitlines()
               if line.strip()]
    note_path = workspace / NOTE
    note = note_path.read_text() if note_path.is_file() else ""
    image_hashes = {ref: hashlib.sha256((workspace / ref).read_bytes()).hexdigest()
                    for ref in checkpoint.get("images", []) if (workspace / ref).is_file()}
    return {"state": state, "observations": records, "checkpoint": deepcopy(checkpoint),
            "image_hashes": image_hashes, "note": note,
            "note_sha256": hashlib.sha256(note.encode()).hexdigest()}


def ready(phase):
    """A frame ID or a rotation alone cannot satisfy the pre-movement gate."""
    state = phase["state"]
    points = state.get("motion_trajectory", state.get("trajectory", []))
    travel = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
    current = phase.get("checkpoint", {}).get("metadata", {}).get("current", {})
    candidates = ([current["observation"]] if current.get("observation") else current.get("views", []))
    frame = next((record["id"] for record in candidates
                  if record.get("id") and record["id"] in phase.get("note", "")), None)
    at_new_frame = any(record.get("id") == frame and record.get("step", 0) > 0
                       for record in phase.get("observations", []))
    return {"passed": bool(travel > .05 and not state.get("terminated") and not state.get("truncated")
                            and at_new_frame and frame and frame in phase.get("note", "")),
            "actual_travel_m": travel, "local_event_not_finished": not bool(state.get("terminated")),
            "judgment_has_current_frame": bool(frame and frame in phase.get("note", "")),
            "current_frame": frame}


def save(private, diagnostic):
    _write(Path(private) / "moving_compaction.json", diagnostic)
