"""Explicit named factories, with no import scanning or global instances."""

from typing import Any, Callable


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._factories: dict[str, Callable[..., Any]] = {}

    def register(self, name: str):
        def decorator(factory: Callable[..., Any]):
            if name in self._factories:
                raise ValueError(f"Duplicate {self.name} registration: {name}")
            self._factories[name] = factory
            return factory
        return decorator

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._factories[name]
        except KeyError as error:
            raise ValueError(f"Unknown {self.name}: {name}; available: {self.list()}") from error

    def create(self, name: str, **kwargs: Any) -> Any:
        return self.get(name)(**kwargs)

    def list(self) -> list[str]:
        return sorted(self._factories)


episode_registry = Registry("episode source")
env_registry = Registry("environment")
task_registry = Registry("task")
model_registry = Registry("model backend")
runtime_registry = Registry("agent runtime")
tool_registry = Registry("navigation tool")
