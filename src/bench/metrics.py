"""Private navigation scores from physical motion and explicit submission."""

import math
from statistics import mean


def spl(success, shortest, travelled):
    if not success or not math.isfinite(shortest):
        return 0.0
    denominator = max(shortest, travelled)
    return shortest / denominator if denominator else 1.0


def score_goal(benchmark, *, submitted, distance, shortest, path_length, steps):
    threshold = 3.0 if benchmark in ("r2r", "rxr") else 1.0
    official = {"objnav": 0.1, "goat": 0.25}.get(benchmark, threshold)
    success = bool(submitted and distance < threshold)
    official_success = bool(submitted and distance < official)
    return {
        "success": float(success), "spl": spl(success, shortest, path_length),
        "navigation_error": distance, "submitted": bool(submitted),
        "success_distance": threshold,
        "official_success": float(official_success), "official_success_distance": official,
        "official_spl": spl(official_success, shortest, path_length),
        "shortest_path": shortest, "path_length": path_length, "steps": steps,
    }


def score_episode(benchmark, goal_results, planned_goals):
    if planned_goals < 1:
        raise ValueError("An episode needs at least one goal")
    if benchmark != "goat":
        return dict(goal_results[0]) if goal_results else {"success": 0.0, "spl": 0.0}
    metrics = {
        "partial_success": sum(g["success"] for g in goal_results) / planned_goals,
        "composite_success": float(len(goal_results) == planned_goals and all(g["success"] for g in goal_results)),
        "spl": sum(g["spl"] for g in goal_results) / planned_goals,
        "official_success": sum(g["official_success"] for g in goal_results) / planned_goals,
        "planned_goals": planned_goals,
    }
    for modality in ("object", "description", "image"):
        rows = [g for g in goal_results if g.get("modality") == modality]
        if rows:
            metrics[f"{modality}_success"] = mean(g["success"] for g in rows)
    return metrics
