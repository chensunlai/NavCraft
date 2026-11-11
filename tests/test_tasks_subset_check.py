"""Physical acceptance must check every branch used by an action-choice label."""
from copy import deepcopy

import pytest

from src.dataset.tasks_subset.bundle import snapshot_digest
from src.dataset.tasks_subset.review.check import _physical_references


def test_action_choices_require_both_independent_replays(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "state.json").write_text("{}")
    state = {"id": "cut", "snapshot": "snapshot", "frame_keys": ["before"]}
    frames = {}
    trials = []
    for name in ("near", "far"):
        end, replay_end = f"branch-{name}", f"replay-{name}"
        for key in (end, replay_end):
            frames[key] = {"stream": key, "pose": {"position": [0, 0, 0]}}
        calls = [{"name": "move", "arguments": {"mode": "steps", "actions": ["move_forward"]}}]
        trials.append({"id": name, "start_state": "cut", "start_snapshot_sha256": snapshot_digest(snapshot),
                       "calls": calls, "end_frame": end, "motions": [],
                       "replay": {"passed": True, "calls": calls, "end_frame": replay_end, "motions": []}})
    bundle = {"trials": trials, "frames": frames}
    label = {"trial": "near", "trials": ["near", "far"]}
    arguments = (tmp_path, state, {"mode": "judgment"}, {"physical_replay": True}, label)
    assert _physical_references(*arguments, bundle) is True
    failed = deepcopy(bundle)
    failed["trials"][1]["replay"]["passed"] = False
    with pytest.raises(ValueError, match="independent replay"):
        _physical_references(*arguments, failed)
    stale = deepcopy(bundle)
    stale["trials"][1]["start_snapshot_sha256"] = "wrong"
    with pytest.raises(ValueError, match="snapshot"):
        _physical_references(*arguments, stale)
    with pytest.raises(ValueError, match="reference missing"):
        _physical_references(*arguments, {**bundle, "trials": trials[:1]})
