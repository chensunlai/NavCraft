from pathlib import Path

import pytest

from src.common.io import write_json
from src.dataset.generation.records import load_records, save_records
from src.dataset.generation.teacher import solve


def setup(tmp_path, monkeypatch):
    root = tmp_path / "data/collection"
    root.mkdir(parents=True)
    write_json(root / "teacher_selection.json", ["sample"])
    save_records(root / "samples.jsonl", [{"id": "sample", "task_id": "5a"}])
    monkeypatch.setattr("src.dataset.generation.check.check", lambda *args: {"valid": True, "content_sha256": "frozen"})
    config = {"model": "gpt-6.1-sol", "reasoning_effort": "high", "concurrency": 1,
              "generation": {"attempts": 2}}
    return root, config


def test_teacher_uses_nvidia_egl_and_resumes_success_without_reexecution(tmp_path, monkeypatch):
    root, config = setup(tmp_path, monkeypatch)
    calls = []

    def worker(command, **kwargs):
        assert kwargs["env"]["__EGL_VENDOR_LIBRARY_FILENAMES"].endswith("10_nvidia.json")
        directory = Path(command[command.index("--output") + 1])
        write_json(directory / "private/requests.json", {"attempts": [{"number": 1}]})
        write_json(directory / "result.json", {"task_id": "5a", "status": "completed",
                                                "score": {"success": True, "reason": "correct", "metrics": {}}})
        calls.append(command)

    monkeypatch.setattr("src.dataset.generation.teacher.subprocess.run", worker)
    assert solve(root, config, tmp_path, ["config"]) ["successful_samples"] == 1
    assert solve(root, config, tmp_path, ["config"]) ["successful_samples"] == 1
    assert len(calls) == 1
    with pytest.raises(ValueError, match="changed"):
        solve(root, {**config, "reasoning_effort": "medium"}, tmp_path, [])


def test_pre_model_worker_crashes_are_kept_and_do_not_exhaust_teacher_attempts(tmp_path, monkeypatch):
    root, config = setup(tmp_path, monkeypatch)
    monkeypatch.setattr("src.dataset.generation.teacher.subprocess.run", lambda *args, **kwargs: None)
    assert solve(root, config, tmp_path, []) ["successful_samples"] == 0
    failed = load_records(root / "solutions.jsonl")
    assert len(failed) == 2 and all(not row["model_started"] for row in failed)

    def recovered(command, **kwargs):
        directory = Path(command[command.index("--output") + 1])
        write_json(directory / "private/requests.json", {"attempts": [{"number": 1}]})
        write_json(directory / "result.json", {"task_id": "5a", "status": "completed",
                                                "score": {"success": True, "reason": "correct", "metrics": {}}})

    monkeypatch.setattr("src.dataset.generation.teacher.subprocess.run", recovered)
    assert solve(root, config, tmp_path, []) ["successful_samples"] == 1
    rows = load_records(root / "solutions.jsonl")
    assert len(rows) == 3 and rows[-1]["attempt"] == 3 and rows[-1]["model_started"]
