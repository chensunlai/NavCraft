"""Thread-safe task states displayed through tqdm on stderr."""

from collections import Counter
import sys
from threading import Lock

from tqdm import tqdm

TERMINAL = {"finished", "ready", "partial", "failed"}


class Progress:
    def __init__(self, label, keys, initial=None):
        self.label = label
        self.states = dict.fromkeys(keys, "pending")
        self.states.update(initial or {})
        self.lock = Lock()
        ended, details = self._summary()
        # Keep carriage-return rendering through `2>&1 | tee` in tmux too.
        self.bar = tqdm(total=len(self.states), initial=ended, desc=label,
                        postfix=details, file=sys.stderr, disable=False,
                        dynamic_ncols=True, mininterval=.2, unit="task",
                        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} ended "
                                   "[{elapsed}<{remaining}{postfix}]")
        if ended == len(self.states):
            self.bar.close()

    def _summary(self):
        counts = Counter(self.states.values())
        ended = sum(counts[state] for state in TERMINAL)
        details = " ".join(f"{key}={counts[key]}" for key in
                           ("ready", "finished", "partial", "failed", "running", "waiting", "pending")
                           if counts[key])
        return ended, details

    def set(self, key, state):
        with self.lock:
            if key not in self.states:
                raise KeyError(key)
            # Completed items loaded during resume must not move backwards.
            if self.states[key] in TERMINAL and state in {"running", "waiting"}:
                return
            if self.states[key] == state:
                return
            self.states[key] = state
            ended, details = self._summary()
            self.bar.set_postfix_str(details, refresh=False)
            self.bar.update(ended - self.bar.n)
            if ended == len(self.states):
                self.bar.close()
            else:
                # Show dispatch/retry changes even when no task has ended yet.
                self.bar.refresh()
