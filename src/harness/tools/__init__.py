"""Resolve flat toolsets, then bind each model schema to its executor once."""

from collections.abc import Callable
from dataclasses import dataclass, field

from jsonschema import Draft7Validator
from ..errors import ToolInputError
from . import entity_specs, navigation_specs, node_specs
from .node import NodeMemoryTools
from .runtime import NavigationTools


@dataclass(frozen=True)
class Toolsets:
    names: tuple[str, ...]
    memory_type: str
    geometry: bool

    @classmethod
    def resolve(cls, names):
        if not isinstance(names, (list, tuple)) or any(not isinstance(n, str) for n in names):
            raise ValueError("toolsets must be a list of names")
        selected = set(names)
        if len(names) != len(selected) or selected - {"navigation", "entity_memory", "node_memory", "geometry"}:
            raise ValueError("Unknown or duplicate toolset")
        if "navigation" not in selected or len(selected & {"entity_memory", "node_memory"}) != 1:
            raise ValueError("Select navigation and exactly one of entity_memory / node_memory")
        return cls(tuple(names), "node" if "node_memory" in selected else "entity", "geometry" in selected)


@dataclass
class Tool:
    schema: dict
    execute: Callable
    validator: Draft7Validator = field(init=False, repr=False)

    def __post_init__(self):
        self.validator = Draft7Validator(self.schema["parameters"])

    def call(self, args):
        error = next(self.validator.iter_errors(args), None)
        if error:
            field = ".".join(map(str, error.absolute_path)) or "arguments"
            raise ToolInputError(f"{field}: {error.message}")
        return self.execute(**args)


def bind_tools(executor, selection):
    navigation = NavigationTools(executor)
    node = selection.memory_type == "node"
    nav_schemas = navigation_specs.specifications(selection.geometry, node=node)
    tools = [Tool(schema, execute) for schema, execute in zip(nav_schemas, (
        navigation.observe_reply, navigation.move, navigation.submit, navigation.task_read))]
    if node:
        memory = NodeMemoryTools(executor, navigation)
        schemas, handlers = node_specs.memory_specifications(), (memory.read, memory.edit)
    else:
        schemas = entity_specs.specifications(selection.geometry)
        handlers = (navigation.map_reply, navigation.images, navigation.read_memory, navigation.edit_memory)
    tools.extend(Tool(schema, execute) for schema, execute in zip(schemas, handlers))
    # Stable presentation order preserves the established model interface.
    indexed = {tool.schema["name"]: tool for tool in tools}
    order = ("observe", "move", "map_view", "view_images", "memory_read", "memory_edit", "submit", "task_read")
    return {name: indexed[name] for name in order if name in indexed}
