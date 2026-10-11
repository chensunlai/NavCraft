"""Construct only selected components, using explicit builtin registration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.common.config import ExperimentConfig
from src.common.register import env_registry, model_registry, runtime_registry, task_registry
from src.dataset.episodes.base import EpisodeSpec
from src.harness.agent.base import AgentLimits
from src.harness.session import NavigationSession
from src.harness.trace import TraceWriter

if TYPE_CHECKING:
    from src.harness.agent.loop import AgentRuntime


def load_builtins() -> None:
    from src.dataset.episodes import load_builtin_sources
    from src.environment import fake, habitat  # noqa: F401
    from src.environment.tasks import base  # noqa: F401
    from src.harness.agent import loop, model  # noqa: F401
    from src.harness.tools import load_builtin_tools

    load_builtin_sources()
    load_builtin_tools()


@dataclass
class BuiltExperiment:
    session: NavigationSession
    runtime: AgentRuntime
    limits: AgentLimits

    def close(self) -> None:
        try:
            self.session.close()
        finally:
            self.runtime.close()


def build(config: ExperimentConfig, episode: EpisodeSpec, directory: Path, trace: TraceWriter) -> BuiltExperiment:
    load_builtins()
    task = task_registry.create(config.task["name"], config=config.task)
    environment = env_registry.create(config.environment["name"], config=config.environment,
                                      task=task, workspace=directory)
    model = None
    try:
        model = model_registry.create(config.model["name"], config=config.model)
        runtime = runtime_registry.create(config.harness.get("runtime", "agent"), model=model, config=config.harness)
        action_parameters = {
            "forward_step_m": config.environment.get("forward_step", 0.25),
            "turn_angle_degrees": config.environment.get("turn_angle", 30 if config.environment["name"] == "fake" else 15),
            "allow_sliding": config.environment.get("allow_sliding", config.environment["name"] == "habitat"),
        }
        session = NavigationSession(environment, episode, trace, config.harness["tools"], action_parameters=action_parameters)
        limits = AgentLimits(config.run["max_requests"], config.run["timeout_seconds"], config.run["max_context_chars"])
        return BuiltExperiment(session, runtime, limits)
    except BaseException:
        environment.close()
        if model is not None:
            model.close()
        raise
