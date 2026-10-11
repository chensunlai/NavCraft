"""Summaries preserve failed episodes in the selected-sample denominator."""

import math
from statistics import mean


def summarize(results: list[dict], selected_count: int) -> dict:
    statuses = {}
    for row in results:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    metric_names = sorted({key for row in results for key in row.get("metrics", {})})
    metrics = {}
    for key in metric_names:
        values = [row.get("metrics", {}).get(key) for row in results]
        values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
        if values:
            metrics[key] = {"mean": mean(values), "count": len(values)}
    score_key = "partial_success" if any(row["benchmark"] == "goat" for row in results) else "success"
    score = sum(row.get("metrics", {}).get(score_key, 0) or 0 for row in results)
    return {"selected": selected_count, "finished": len(results), "statuses": statuses,
            "score_key": score_key, "score_over_selected": score / selected_count if selected_count else None,
            "metrics": metrics}
