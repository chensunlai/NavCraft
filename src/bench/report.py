"""Summaries preserve failed episodes in the selected-sample denominator."""

import math
from statistics import mean


def summarize(
    results: list[dict], selected_count: int, planned_goals: int | None = None
) -> dict:
    statuses = {}
    for row in results:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    metric_names = sorted({key for row in results for key in row.get("metrics", {})})
    metrics = {}
    for key in metric_names:
        values = [row.get("metrics", {}).get(key) for row in results]
        values = [
            v
            for v in values
            if isinstance(v, (int, float))
            and not isinstance(v, bool)
            and math.isfinite(v)
        ]
        if values:
            metrics[key] = {"mean": mean(values), "count": len(values)}
    score_key = (
        "partial_success"
        if any(row["benchmark"] == "goat" for row in results)
        else "success"
    )
    score = sum(row.get("metrics", {}).get(score_key, 0) or 0 for row in results)
    if score_key == "partial_success" and all(
        "planned_goals" in row for row in results
    ):
        planned = (
            planned_goals
            if planned_goals is not None
            else sum(row["planned_goals"] for row in results)
        )
        successful = sum(
            sum(row.get("metrics", {}).get("subtask_success", [])) for row in results
        )
        goal_score = successful / planned if planned else 0.0
    else:
        planned, successful, goal_score = None, None, None
    return {
        "selected": selected_count,
        "finished": len(results),
        "statuses": statuses,
        "score_key": score_key,
        "score_over_selected": score / selected_count if selected_count else None,
        "metrics": metrics,
        "planned_subgoals": planned,
        "successful_subgoals": successful,
        "subgoal_success_over_planned": goal_score,
    }
