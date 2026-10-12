"""Completed visual runs survive later frozen-condition failures or interrupts."""

import json
from types import SimpleNamespace

import pytest

from src.bench import visual


@pytest.fixture
def coordinator(tmp_path, monkeypatch):
    monkeypatch.setattr(visual, "ROOT", tmp_path)
    monkeypatch.setattr(visual, "harness_version", lambda: "fixture-version")
    monkeypatch.setattr(
        visual, "conditions_fingerprint", lambda config: "fixture-conditions"
    )
    manifests = {}
    for family in visual.ENVIRONMENTS:
        path = tmp_path / f"{family}.json"
        path.write_text(json.dumps({"episodes": [family]}))
        manifests[family] = path

    def config_for(
        family, stage, model, limit=None, max_requests=None, fine_actions=False
    ):
        return SimpleNamespace(
            episodes={"manifest": manifests[family]},
            model={"model": f"fixture-{model}"},
            family=family,
        )

    monkeypatch.setattr(visual, "config_for", config_for)

    def saved():
        paths = list(
            tmp_path.glob("cache/visual-harness-v1/experiments/*/experiment.json")
        )
        assert len(paths) == 1
        return json.loads(paths[0].read_text())

    return manifests, saved


def summary():
    return {
        "selected": 20,
        "finished": 20,
        "score_over_selected": 0.9,
        "subgoal_success_over_planned": 0.9,
    }


def test_completed_family_saved_before_next_family_freeze_failure(
    coordinator, tmp_path, monkeypatch
):
    manifests, saved = coordinator
    completed = []

    def run_experiment(config):
        # The first run is saved while the experiment is still running.
        assert saved()["status"] == "running"
        completed.append(config.family)
        manifests["instruction"].write_text("changed after the first family")
        return tmp_path / "runs" / config.family, summary()

    monkeypatch.setattr(visual, "run_experiment", run_experiment)
    with pytest.raises(RuntimeError, match="Frozen task manifest.*instruction"):
        visual.main(["--families", "objnav", "instruction", "--rounds", "2"])
    result = saved()
    assert completed == ["objnav"]
    assert result["status"] == "failed"
    assert result["error"]["type"] == "RuntimeError"
    assert "instruction" in result["error"]["message"]
    assert result["active_model"] == "qwen27"
    assert not result["acceptance_passed"]
    assert [(row["family"], row["round"]) for row in result["runs"]] == [("objnav", 1)]


def test_interrupt_preserves_completed_families_and_explicit_status(
    coordinator, tmp_path, monkeypatch
):
    _, saved = coordinator
    calls = 0

    def run_experiment(config):
        nonlocal calls
        calls += 1
        if calls == 2:
            # Persistence happens before the next family begins, not at batch end.
            assert [row["family"] for row in saved()["runs"]] == ["objnav"]
            raise KeyboardInterrupt
        return tmp_path / "runs" / config.family, summary()

    monkeypatch.setattr(visual, "run_experiment", run_experiment)
    with pytest.raises(KeyboardInterrupt):
        visual.main(["--families", "objnav", "instruction"])
    result = saved()
    assert result["status"] == "interrupted"
    assert result["error"] == {
        "type": "KeyboardInterrupt",
        "message": "Interrupted by user",
    }
    assert len(result["runs"]) == 1 and not result["acceptance_passed"]


def test_normal_completion_saves_each_round_once_and_legacy_batch_still_returns_rows(
    coordinator, tmp_path, monkeypatch
):
    _, saved = coordinator
    calls = 0

    def run_experiment(config):
        nonlocal calls
        assert len(saved()["runs"]) == calls
        calls += 1
        return tmp_path / "runs" / f"{config.family}-{calls}", summary()

    monkeypatch.setattr(visual, "run_experiment", run_experiment)
    result = visual.main(["--stage", "validate"])
    assert result["status"] == "completed" and result["acceptance_passed"]
    assert "active_model" not in result
    assert len(saved()["runs"]) == 6
    assert [(row["family"], row["round"]) for row in result["runs"]] == [
        (family, number) for number in (1, 2) for family in visual.ENVIRONMENTS
    ]
    monkeypatch.setattr(
        visual, "run_experiment", lambda config: (tmp_path / "legacy", summary())
    )
    batch = visual.run_batch("qwen27", "develop", ["objnav"], 1)
    assert len(batch) == 1 and batch[0]["family"] == "objnav"


def test_escalation_interrupt_keeps_prior_model_results(
    coordinator, tmp_path, monkeypatch
):
    _, saved = coordinator

    def run_experiment(config):
        if config.model["model"] == "fixture-deepseek":
            assert len(saved()["runs"]) == 3
            assert saved()["active_model"] == "deepseek"
            raise RuntimeError("fixture model service is unavailable")
        return tmp_path / "runs" / config.family, {
            **summary(),
            "score_over_selected": 0.1,
            "subgoal_success_over_planned": 0.1,
        }

    monkeypatch.setattr(visual, "run_experiment", run_experiment)
    with pytest.raises(RuntimeError, match="model service"):
        visual.main(["--auto"])
    result = saved()
    assert result["status"] == "failed" and result["active_model"] == "deepseek"
    assert len(result["runs"]) == 3
    assert {row["model"] for row in result["runs"]} == {"qwen27"}


@pytest.mark.parametrize("change", ["manifest", "harness", "conditions"])
def test_changes_during_run_are_saved_only_as_invalidated_diagnostics(
    coordinator, tmp_path, monkeypatch, change
):
    manifests, saved = coordinator
    version = {"value": "fixture-version"}
    fingerprint = {"value": "fixture-conditions"}
    monkeypatch.setattr(visual, "harness_version", lambda: version["value"])
    monkeypatch.setattr(
        visual, "conditions_fingerprint", lambda config: fingerprint["value"]
    )

    def run_experiment(config):
        if change == "manifest":
            manifests[config.family].write_text("modified while workers ran")
        elif change == "harness":
            version["value"] = "modified-version"
        else:
            fingerprint["value"] = "modified-environment"
        return tmp_path / "runs" / "completed-but-mixed", summary()

    monkeypatch.setattr(visual, "run_experiment", run_experiment)
    with pytest.raises(RuntimeError, match="Frozen task manifest.*during"):
        visual.main(["--families", "objnav"])
    result = saved()
    assert result["status"] == "failed" and not result["acceptance_passed"]
    row = result["runs"][0]
    assert row["run"] == "runs/completed-but-mixed"
    assert row["status"] == "invalidated" and row["valid"] is False
    assert row["score"] is None
    assert row["summary"]["score_over_selected"] == 0.9
    assert "during" in row["invalidation_reason"]
    assert not visual.qualifies(result["runs"])
    assert not visual.acceptance_qualifies(result["runs"])


@pytest.mark.parametrize("filename", ["scene.json", "regions.json", "floor_plan.svg"])
def test_initial_arrow_calibration_inputs_are_part_of_frozen_conditions(
    tmp_path, filename
):
    maps = tmp_path / "scene_maps"
    archive = maps / "hm3d" / "val" / "00001-fixture"
    (archive / "raw").mkdir(parents=True)
    (archive / "scene_description.md").write_text("Actor-readable scene description")
    for name in ("scene.json", "regions.json", "floor_plan.svg"):
        (archive / "raw" / name).write_text("Original private calibration")
    config = SimpleNamespace(
        harness={"scene_maps": str(maps)},
        snapshot=lambda: {"harness": {"scene_maps": str(maps)}},
    )
    before = visual.conditions_fingerprint(config)
    (archive / "raw" / filename).write_text("Changed private calibration")
    assert visual.conditions_fingerprint(config) != before
