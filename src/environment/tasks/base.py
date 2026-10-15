"""Private task lifecycle and benchmark rules, independent of RGB rendering."""

from dataclasses import dataclass
import math

from src.common.register import task_registry
from src.environment.base import NavigationInputError, TRANSLATION_ACTIONS
from src.environment.tasks.metrics import goat_metrics, spl, vln_metrics


@dataclass(frozen=True)
class TaskUpdate:
    goal_changed: bool = False
    terminated: bool = False
    truncated: bool = False
    reason: str | None = None


class TaskAdapter:
    """Own goal progression, action budgets and scores; never transport annotations."""

    family: str
    default_success_distance = 3.0
    default_max_steps = 500

    def __init__(self, config: dict):
        self.config = config
        self.threshold = float(
            config.get("success_distance", self.default_success_distance)
        )
        self.max_steps = int(config.get("max_steps", self.default_max_steps))
        self.max_subtask_steps = int(config.get("max_subtask_steps", 500))
        if self.threshold <= 0 or self.max_steps < 1 or self.max_subtask_steps < 1:
            raise ValueError(
                "Task success distance and action budgets must be positive"
            )
        self.protocol = config.get("protocol", f"astrabench-{self.family}-v1")
        self.episode = None

    def reset(self, episode, point: list[float], distance: float) -> None:
        self.episode = episode
        self.goals = episode.goals
        if not self.goals or (self.family != "goat" and len(self.goals) != 1):
            raise ValueError(
                f"{self.family} needs {'one or more' if self.family == 'goat' else 'exactly one'} goals"
            )
        self.goal_index = self.steps = self.goal_steps = 0
        self.path_length = self.goal_path_length = 0.0
        self.trajectory = [list(point)]
        self.subtasks = []
        self.terminated = self.truncated = self.stopped = False
        self.reason = None
        self.set_goal_distance(distance)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    @property
    def goal(self) -> dict:
        return self.goals[self.goal_index]

    @property
    def movement_actions(self) -> tuple[str, ...]:
        actions = TRANSLATION_ACTIONS + ("turn_left", "turn_right")
        return (
            actions
            if self.episode.benchmark == "r2r"
            else actions + ("look_up", "look_down")
        )

    def set_goal_distance(self, distance: float) -> None:
        self.shortest = self.distance = self.minimum_distance = distance

    def _count(self, point: list[float], distance: float) -> None:
        if self.done:
            raise NavigationInputError("Episode is already finished")
        self.steps += 1
        self.goal_steps += 1
        travel = math.dist(point, self.trajectory[-1])
        self.path_length += travel
        self.goal_path_length += travel
        self.trajectory.append(list(point))
        self.distance = distance
        self.minimum_distance = min(self.minimum_distance, distance)

    def on_step(self, point: list[float], distance: float) -> TaskUpdate:
        self._count(point, distance)
        if self.steps >= self.max_steps:
            self._record_goal(stopped=False, reason="max_steps")
            self.truncated, self.reason = True, "max_steps"
        elif self.family == "goat" and self.goal_steps >= self.max_subtask_steps:
            return self._advance(stopped=False, reason="max_subtask_steps")
        return self.update()

    def finish(self, point: list[float], distance: float) -> TaskUpdate:
        self._count(point, distance)
        return self._advance(stopped=True, reason="stop")

    def _record_goal(self, *, stopped: bool, reason: str) -> None:
        success = stopped and self.distance < self.threshold
        self.subtasks.append(
            {
                "index": self.goal_index,
                "modality": self.goal["modality"],
                "success": float(success),
                "spl": spl(success, self.shortest, self.goal_path_length),
                "distance": self.distance,
                "shortest_path": self.shortest,
                "path_length": self.goal_path_length,
                "steps": self.goal_steps,
                "reason": reason,
            }
        )

    def _advance(self, *, stopped: bool, reason: str) -> TaskUpdate:
        self._record_goal(stopped=stopped, reason=reason)
        if self.family != "goat":
            self.stopped, self.terminated, self.reason = stopped, True, reason
            return self.update()
        self.goal_index += 1
        if self.goal_index == len(self.goals):
            self.terminated, self.reason = True, "all_subtasks_finished"
        elif self.steps >= self.max_steps:
            self.truncated, self.reason = True, "max_steps"
        else:
            self.goal_steps, self.goal_path_length = 0, 0.0
        return self.update(goal_changed=self.goal_index < len(self.goals))

    def update(self, *, goal_changed: bool = False) -> TaskUpdate:
        return TaskUpdate(goal_changed, self.terminated, self.truncated, self.reason)

    def finalize(self, reason: str) -> None:
        if not self.done:
            self._record_goal(stopped=False, reason=reason)
            self.truncated, self.reason = True, reason

    def metrics(self) -> dict:
        if self.family == "goat":
            scores = goat_metrics(self.goals, self.subtasks)
        else:
            scores = vln_metrics(
                stopped=self.stopped,
                distance=self.distance,
                minimum_distance=self.minimum_distance,
                shortest=self.shortest,
                trajectory=self.trajectory,
                reference=self.episode.gt_path,
                threshold=self.threshold,
                travelled=self.path_length,
            )
        checks = getattr(self.episode, "route_checks", [])
        if checks:
            cursor = 0
            completed = []
            for check in checks:
                hit = next(
                    (
                        index
                        for index in range(cursor, len(self.trajectory))
                        if math.dist(self.trajectory[index], check["position"])
                        <= check.get("radius", 1.5)
                    ),
                    None,
                )
                completed.append(hit is not None)
                if hit is not None:
                    cursor = hit
                else:
                    break
            completed += [False] * (len(checks) - len(completed))
            scores.update(
                route_constraint_completion=sum(completed) / len(checks),
                route_constraints_satisfied=float(all(completed)),
            )
        return {
            **scores,
            "protocol": self.protocol,
            "steps": self.steps,
            "path_length": self.path_length,
            "reason": self.reason,
            "success_distance": self.threshold,
            "ndtw_reference": (
                (
                    "official_gt"
                    if getattr(self.episode, "provenance", "official") == "official"
                    else "synthetic_path"
                )
                if self.episode.gt_path
                else "unavailable"
            ),
            "provenance": getattr(self.episode, "provenance", "official"),
        }

    def private_state(self) -> dict:
        # Include unattempted goals explicitly; aggregate GOAT scores also retain zeros.
        results = list(self.subtasks)
        completed = {result["index"] for result in results}
        for index, goal in enumerate(self.goals):
            if index not in completed:
                results.append(
                    {
                        "index": index,
                        "modality": goal["modality"],
                        "success": 0.0,
                        "spl": 0.0,
                        "reason": self.reason or "incomplete",
                    }
                )
        return {
            "metrics": self.metrics(),
            "subtasks": results,
            "trajectory": self.trajectory,
            "goal_index": self.goal_index,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }


@task_registry.register("objectnav")
class ObjectNavTask(TaskAdapter):
    family = "objectnav"
    default_success_distance = 1.0


@task_registry.register("vlnce")
class VLNCETask(TaskAdapter):
    family = "vlnce"


@task_registry.register("goat")
class GOATTask(TaskAdapter):
    family = "goat"
    default_success_distance = 1.0
    default_max_steps = 5000
