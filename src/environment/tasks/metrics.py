"""Navigation scores from simulator state, never the agent's success claim."""

import math
from statistics import mean


def spl(success: bool, shortest: float, travelled: float) -> float:
    if not success or not math.isfinite(shortest):
        return 0.0
    denominator = max(shortest, travelled)
    return shortest / denominator if denominator else 1.0


def ndtw(trajectory: list, reference: list, threshold: float) -> float | None:
    """Exact DTW with Euclidean point cost, as supported by VLN-CE's NDTW.

    Use the dense official *_gt.json.gz locations. Sparse instruction waypoints
    are kept as metadata and are deliberately not substituted for that path.
    """
    if not reference:
        return None
    previous = [0.0] + [math.inf] * len(reference)
    last = None
    for point in trajectory:
        if point == last:
            continue
        last = point
        current = [math.inf]
        for column, target in enumerate(reference, 1):
            current.append(
                math.dist(point, target) + min(previous[column], current[-1], previous[column - 1])
            )
        previous = current
    return math.exp(-previous[-1] / (len(reference) * threshold))


def vln_metrics(
    *,
    stopped: bool,
    distance: float,
    minimum_distance: float,
    shortest: float,
    trajectory: list,
    reference: list,
    threshold: float,
    travelled: float,
) -> dict:
    success = stopped and distance < threshold
    score = ndtw(trajectory, reference, threshold)
    return {
        "success": float(success),
        "spl": spl(success, shortest, travelled),
        "navigation_error": distance,
        "oracle_success": float(minimum_distance < threshold),
        "path_length": travelled,
        "ndtw": score,
        "sdtw": score * success if score is not None else None,
    }


def goat_metrics(goals: list[dict], results: list[dict]) -> dict:
    successes = [0.0] * len(goals)
    spls = [0.0] * len(goals)
    for result in results:
        index = result["index"]
        successes[index], spls[index] = result["success"], result["spl"]
    metrics = {
        "partial_success": mean(successes),
        "composite_success": float(all(successes)),
        "composite_spl": mean(spls),
        "subtask_success": successes,
        "subtask_spl": spls,
    }
    for modality in ("object", "description", "image"):
        indices = [i for i, goal in enumerate(goals) if goal["modality"] == modality]
        metrics[f"{modality}_success"] = mean(successes[i] for i in indices) if indices else None
        metrics[f"{modality}_spl"] = mean(spls[i] for i in indices) if indices else None
    return metrics


def summarize(results: list[dict], selected_count: int, *, exploration_only=False) -> dict:
    """Infrastructure failures remain visible in denominators and coverage."""
    statuses = {}
    for result in results:
        status = result["status"]
        statuses[status] = statuses.get(status, 0) + 1
    metrics = {}
    keys = sorted({key for result in results for key in result.get("metrics", {})})
    for key in keys:
        values = [result.get("metrics", {}).get(key) for result in results]
        values = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
        if values:
            metrics[key] = {"mean": mean(values), "count": len(values)}
    score_key = (
        "partial_success" if any(r.get("benchmark") == "goat" for r in results) else "success"
    )
    score = sum(r.get("metrics", {}).get(score_key, 0.0) or 0.0 for r in results)
    if exploration_only or (results and all(r.get('protocol') == 'goat-exploration-only-v1' for r in results)):
        score_key, score = None, None
    return {
        "selected": selected_count,
        "finished": len(results),
        "statuses": statuses,
        "score_key": score_key,
        "score_over_selected": score / selected_count if selected_count and score is not None else None,
        "metrics": metrics,
    }
