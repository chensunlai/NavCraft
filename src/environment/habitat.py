"""Navigation environment lifecycle with a lazily imported Habitat renderer."""

from pathlib import Path

from src.common.io import write_json
from src.common.register import env_registry
from src.environment.base import (
    NavigationInputError,
    Observation,
    PublicTask,
    Transition,
)
from src.environment.tasks.base import TaskAdapter, TaskUpdate


@env_registry.register("habitat")
class HabitatEnvironment:
    def __init__(
        self, config: dict, task: TaskAdapter, workspace: Path, simulator_factory=None
    ):
        self.config, self.task, self.workspace = config, task, Path(workspace).resolve()
        self.simulator_factory = simulator_factory
        self.sim = None
        self.observation = None
        self.public_task = None
        self._next_frame = 0

    def reset(self, episode) -> Transition:
        self.close()
        self.observation = self.public_task = None
        self._next_frame = 0
        self.task.episode = None
        self.workspace.mkdir(parents=True, exist_ok=True)
        factory = self.simulator_factory
        if factory is None:
            from src.environment.simulator import HabitatSimulator

            factory = HabitatSimulator
        try:
            self.sim = factory(episode=episode, config=self.config)
            distance = self.sim.distance(episode.goals[0]["positions"])
            self.task.reset(episode, self.sim.position(), distance)
            self.observation = self._capture()
            self.public_task = self._public_task()
            self._save()
            return self._transition(TaskUpdate(), {"event": "reset", "steps": 0})
        except BaseException as error:
            try:
                self.close()
            except Exception as close_error:
                raise error from close_error
            raise

    def _require_running(self) -> None:
        if self.sim is None:
            raise RuntimeError("Environment has not been reset or is already closed")
        if self.task.done:
            raise NavigationInputError("Episode is already finished")

    def _public_task(self) -> PublicTask | None:
        if self.task.done:
            return None
        goal = self.task.goal
        image = None
        if goal["modality"] == "image":
            if "image_goal" not in goal:
                raise ValueError("Image goal is missing its private target camera")
            image = (
                self.workspace
                / "observations"
                / f"target_{self.task.goal_index:03d}.png"
            )
            if not image.exists():
                self.sim.target_image(goal["image_goal"], image)
        return PublicTask(
            self.task.goal_index, goal["modality"], goal["instruction"], image
        )

    def observe(self) -> Observation:
        if self.observation is None:
            raise RuntimeError("Environment has not been reset")
        return self.observation

    def _reserve_frames(self, count: int) -> list[int]:
        frames = list(range(self._next_frame, self._next_frame + count))
        self._next_frame += count
        return frames

    def _capture(self) -> Observation:
        return self.sim.capture(self.workspace, self._reserve_frames(1)[0])

    def panorama(self) -> list[Observation]:
        """Observe six horizontal directions without progressing the task."""
        self._require_running()
        return self.sim.panorama(self.workspace, self._reserve_frames(6))

    @property
    def movement_actions(self) -> tuple[str, ...]:
        return self.task.movement_actions

    def step(self, action: str) -> Transition:
        self._require_running()
        if not isinstance(action, str) or action not in self.movement_actions:
            raise NavigationInputError(
                f"Allowed movement actions: {self.movement_actions}; use nav_finish to stop"
            )
        collision = self.sim.step(action)
        update = self.task.on_step(
            self.sim.position(), self.sim.distance(self.task.goal["positions"])
        )
        return self._after_action(action, collision, update)

    def finish_goal(self) -> Transition:
        self._require_running()
        update = self.task.finish(
            self.sim.position(), self.sim.distance(self.task.goal["positions"])
        )
        return self._after_action("stop", False, update)

    def _after_action(
        self, action: str, collision: bool, update: TaskUpdate
    ) -> Transition:
        if update.goal_changed and not self.task.done:
            self.task.set_goal_distance(self.sim.distance(self.task.goal["positions"]))
        self.observation = self._capture()
        self.public_task = self._public_task()
        self._save()
        return self._transition(
            update,
            {
                "action": action,
                "collision": bool(collision),
                "steps": self.task.steps,
                "remaining_total_actions": max(
                    0, self.task.max_steps - self.task.steps
                ),
                "remaining_goal_actions": (
                    max(0, self.task.max_subtask_steps - self.task.goal_steps)
                    if self.task.family == "goat" and not self.task.done
                    else None
                ),
            },
        )

    def _transition(self, update: TaskUpdate, feedback: dict) -> Transition:
        return Transition(
            self.observation,
            self.public_task,
            feedback,
            update.goal_changed,
            update.terminated,
            update.truncated,
            update.reason,
        )

    def _save(self) -> None:
        directory = self.workspace / "private"
        directory.mkdir(exist_ok=True)
        value = self.task.private_state()
        value["episode_id"] = self.task.episode.id
        write_json(directory / "state.json", value)

    def finalize(self, reason: str) -> None:
        if self.task.episode is not None:
            self.task.finalize(reason)
            self.public_task = None
            self._save()

    def metrics(self) -> dict:
        if self.task.episode is None:
            return {}
        return self.task.metrics()

    def close(self) -> None:
        if self.sim is not None:
            simulator, self.sim = self.sim, None
            simulator.close()
