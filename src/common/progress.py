"""Small task-count progress bars on stderr, also readable in redirected logs."""

from collections import Counter
import sys
from threading import Lock

TERMINAL = {"finished", "ready", "partial", "failed"}


def progress_line(label, states):
    counts = Counter(states.values())
    total = len(states)
    ended = sum(counts[state] for state in TERMINAL)
    width = 24
    filled = int(width * ended / total) if total else 0
    details = " ".join(f"{key}={counts[key]}" for key in
                       ("ready", "finished", "partial", "failed", "running", "waiting", "pending") if counts[key])
    return f"{label} [{'#' * filled}{'-' * (width - filled)}] {ended}/{total} ended | {details}"


class Progress:
    def __init__(self, label, keys, initial=None):
        self.label = label
        self.states = dict.fromkeys(keys, "pending")
        self.states.update(initial or {})
        self.lock = Lock()
        self.show()

    def show(self):
        print(progress_line(self.label, self.states), file=sys.stderr, flush=True)

    def set(self, key, state):
        with self.lock:
            if key not in self.states:
                raise KeyError(key)
            # Already completed items loaded during resume must not move backwards.
            if self.states[key] in TERMINAL and state in {"running", "waiting"}:
                return
            if self.states[key] != state:
                self.states[key] = state
                self.show()
