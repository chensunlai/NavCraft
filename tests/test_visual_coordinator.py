"""Frozen acceptance, model selection and complete GOAT denominators."""

from src.bench.report import summarize
from src.bench.visual import acceptance_qualifies, qualifies, score


def test_goat_score_includes_failed_episodes_and_pending_selected_goals():
    rows = [
        {
            "benchmark": "goat",
            "status": "terminated",
            "planned_goals": 2,
            "metrics": {"partial_success": 0.5, "subtask_success": [1, 0]},
        },
        {
            "benchmark": "goat",
            "status": "worker_error",
            "planned_goals": 8,
            "metrics": {},
        },
    ]
    assert score(summarize(rows, 2), "goat") == 0.1
    assert score(summarize(rows[:1], 2, planned_goals=10), "goat") == 0.1


def test_all_families_need_the_threshold():
    assert qualifies([{"score": 0.8}, {"score": 1}])
    assert not qualifies([{"score": 1}, {"score": 0.79}])
    assert not qualifies([])


def test_acceptance_requires_two_distinct_complete_frozen_rounds():
    rows = [
        {
            "family": family,
            "round": round_id,
            "score": 0.8,
            "manifest_sha256": family,
            "conditions_sha256": family,
            "harness_version": "v1",
            "model_id": "model",
            "summary": {"selected": 20, "finished": 20},
        }
        for family in ("objnav", "instruction", "goat")
        for round_id in (1, 2)
    ]
    assert acceptance_qualifies(rows)
    assert not acceptance_qualifies([dict(row, round=1) for row in rows])
    rows[0]["summary"]["selected"] = 1
    assert not acceptance_qualifies(rows)
    rows[0]["summary"]["selected"] = 20
    rows[0]["conditions_sha256"] = "different"
    assert not acceptance_qualifies(rows)
