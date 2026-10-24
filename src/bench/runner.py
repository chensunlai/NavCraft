"""Task prompts and private scoring around an ordinary persistent Harness."""

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

from src.common.io import redact, write_json
from src.environment import Environment
from .metrics import score_episode, score_goal

INFRASTRUCTURE_REASONS = {
    "model_error", "service_error", "core_error", "environment_error", "timeout", "aborted", "error",
    "runtime_error", "execution_uncertain", "tool_timeout", "tool_result_invalid",
}


def complete_goal_rows(rows, goals):
    """Keep unattempted GOAT targets in total and modality denominators."""
    return rows + [{"index": index, "modality": goals[index]["modality"],
                    "success": 0.0, "spl": 0.0, "official_success": 0.0,
                    "submitted": False, "reason": "not_attempted"}
                   for index in range(len(rows), len(goals))]


def validate_single_floor(environment, episode):
    """Private geometric preflight; never changes official accepted target positions."""
    start = environment.position()
    tolerance = .5
    report = {"method": "start_height_band_and_geodesic_path", "height_tolerance_m": tolerance,
              "start": start, "goals": []}
    for goal in episode.goals:
        local = [point for point in goal["positions"] if abs(point[1] - start[1]) <= tolerance]
        path = environment.sim.shortest_path(start, local)
        heights = [point[1] for point in path]
        span = max(heights) - min(heights) if heights else None
        report["goals"].append({"all_viewpoints": len(goal["positions"]), "same_height_viewpoints": len(local),
                                "reachable": bool(path), "path_height_span_m": span,
                                "passed": bool(path) and span <= tolerance})
    reference = episode.reference_path
    report["reference_height_span_m"] = max(p[1] for p in reference) - min(p[1] for p in reference) if reference else None
    report["passed"] = all(g["passed"] for g in report["goals"]) and (
        report["reference_height_span_m"] is None or report["reference_height_span_m"] <= tolerance)
    return report


def task_message(goal):
    if goal["modality"] == "image":
        return "Go next to the object shown in this picture."
    if goal["modality"] == "description":
        return f"Find this object and go next to it: {goal['instruction']}"
    return goal["instruction"]


def execute_goals(environment, session, episode, limits, directory):
    """GOAT preserves one session, environment and memory across user requests."""
    directory = Path(directory)
    rows = []
    finish_reason = "completed"
    infrastructure_error = False
    for index, goal in enumerate(episode.goals):
        shortest = environment.distance(goal["positions"])
        start_steps, start_length = environment.steps, environment.path_length
        images = ()
        if goal["modality"] == "image":
            target = directory / "private" / f"target_{index:03d}.png"
            environment.target_image(goal["image_goal"], target)
            images = (session.register_target(target),)
        result = session.prompt(task_message(goal), images=images, limits=limits)
        finish_reason = result.reason
        row = score_goal(episode.benchmark, submitted=result.submitted,
                         distance=environment.distance(goal["positions"]), shortest=shortest,
                         path_length=environment.path_length - start_length,
                         steps=environment.steps - start_steps)
        row.update(index=index, modality=goal["modality"], reason=result.reason,
                   answer=result.answer, requests=result.requests)
        rows.append(row)
        write_json(directory / "private/goals.json", rows)
        # These conditions cannot be fairly scored as navigation failures.
        if result.reason in INFRASTRUCTURE_REASONS:
            infrastructure_error = True
            break
    return complete_goal_rows(rows, episode.goals), finish_reason, infrastructure_error


def run_episode(repo, model_config, episode, environment_config, scene_map, directory, limits):
    from src.harness.session import PromptLimits, open_session

    repo, directory = Path(repo), Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result = {"episode_id": episode.id, "benchmark": episode.benchmark,
              "model": model_config["model"], "reasoning_effort": model_config["reasoning_effort"],
              "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "planned_goals": len(episode.goals), "gpu": environment_config["gpu"]}
    write_json(directory / "result.json", result)
    write_json(directory / "private/episode.json", episode.to_dict())
    environment = session = None
    rows = []
    try:
        if not Path(scene_map).is_file():
            raise FileNotFoundError(f"Prepare the wall map before running: {scene_map}")
        environment = Environment(episode, environment_config, directory / "private/environment")
        preflight = validate_single_floor(environment, episode)
        write_json(directory / "private/preflight.json", preflight)
        if not preflight["passed"]:
            raise ValueError("Single-floor/reachability preflight failed; inspect private/preflight.json")
        session = open_session(repo, model_config, directory, environment, Path(scene_map))
        rows, reason, failed = execute_goals(environment, session, episode,
                                            PromptLimits(**limits), directory)
        result.update(status="infrastructure_error" if failed else "completed", finish_reason=reason)
    except Exception as error:
        # Credentials never appear in task/config dumps; sanitize exception text too.
        result.update(status="infrastructure_error", finish_reason="worker_error",
                      error=redact(str(error)), error_type=type(error).__name__)
        (directory / "private").mkdir(exist_ok=True)
        (directory / "private/error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        if (directory / "private/goals.json").is_file():
            rows = json.loads((directory / "private/goals.json").read_text())
    finally:
        cleanup = []
        for resource in (session, environment):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    cleanup.append(f"{type(error).__name__}: {error}")
        if cleanup:
            result.update(status="infrastructure_error", cleanup_errors=cleanup)
        rows = complete_goal_rows(rows, episode.goals)
        result["environment"] = {
            "metrics": score_episode(episode.benchmark, rows, len(episode.goals)),
            "subtasks": rows,
            "trajectory": environment.trajectory if environment else [],
            "motion_trajectory": environment.motion_trajectory if environment else [],
            "steps": environment.steps if environment else 0,
            "path_length": environment.path_length if environment else 0,
        }
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(directory / "result.json", result)
    return result
