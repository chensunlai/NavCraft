"""Five configuration roles; paths are always relative to the repository root."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .io import redact

ROOT = Path(__file__).resolve().parents[2]
ROLES = {
    "TASK_CONFIG": {"episodes", "task"},
    "ENV_CONFIG": {"environment"},
    "MODEL_CONFIG": {"model"},
    "HARNESS_CONFIG": {"harness"},
    "RUN_CONFIG": {"run"},
}


@dataclass
class ExperimentConfig:
    episodes: dict[str, Any]
    task: dict[str, Any]
    environment: dict[str, Any]
    model: dict[str, Any]
    harness: dict[str, Any]
    run: dict[str, Any]
    sources: dict[str, str]

    def snapshot(self) -> dict[str, Any]:
        return redact(asdict(self), (self.model.get("api_key", ""),))


def resolve_path(value: str) -> str:
    path = Path(value).expanduser()
    return str((ROOT / path).resolve()) if not path.is_absolute() else str(path.resolve())


def load_config(paths: Mapping[str, str]) -> ExperimentConfig:
    values: dict[str, Any] = {}
    sources = {}
    for variable, sections in ROLES.items():
        if not paths.get(variable):
            raise ValueError(f"Missing {variable}; launch through a scripts/*.sh file")
        path = Path(resolve_path(paths[variable]))
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict) or set(raw) != sections:
            raise ValueError(f"{path}: expected exactly these sections: {sorted(sections)}")
        for section in sections:
            if not isinstance(raw[section], dict):
                raise ValueError(f"{path}: {section} must be a mapping")
            values[section] = raw[section]
        sources[variable] = str(path)
    for section, names in {
        "episodes": ("root", "selection_file", "gt_path"),
        "environment": ("scenes", "egl_vendor_file"),
        "run": ("output_dir",),
    }.items():
        for name in names:
            if values[section].get(name):
                values[section][name] = resolve_path(values[section][name])
    config = ExperimentConfig(**values, sources=sources)
    validate(config)
    return config


def validate(config: ExperimentConfig) -> None:
    benchmark = config.episodes.get("name")
    expected = {"objnav": "objectnav", "r2r": "vlnce", "rxr": "vlnce", "goat": "goat"}
    if benchmark not in expected or config.task.get("name") != expected[benchmark]:
        raise ValueError("episodes.name and task.name are incompatible")
    if not config.task.get("protocol"):
        raise ValueError("task.protocol must explicitly identify the evaluation conditions")
    if config.environment.get("name") not in {"fake", "habitat"}:
        raise ValueError("environment.name must be fake or habitat")
    if config.model.get("name") not in {"responses", "scripted"}:
        raise ValueError("model.name must be responses or scripted")
    if config.harness.get("runtime", "agent") != "agent":
        raise ValueError("Only the agent runtime is implemented")
    for name in ("max_requests", "timeout_seconds", "max_context_chars"):
        value = config.run.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"run.{name} must be positive")
    if type(config.run["max_requests"]) is not int or type(config.run["max_context_chars"]) is not int:
        raise ValueError("run.max_requests and max_context_chars must be integers")
    if not config.run.get("output_dir"):
        raise ValueError("run.output_dir is required")
    for name in ("success_distance", "max_steps", "max_subtask_steps"):
        value = config.task.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"task.{name} must be positive")
    if config.model["name"] == "responses":
        for name in ("base_url", "model", "timeout_seconds", "max_output_tokens"):
            if not config.model.get(name):
                raise ValueError(f"model.{name} is required")
