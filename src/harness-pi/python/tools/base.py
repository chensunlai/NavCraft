"""Public Python tool contracts; no model session or simulator thread lives here."""

from copy import deepcopy
import json
from pathlib import Path

import memory


def object_schema(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required),
            "additionalProperties": False}


BOX_SCHEMA = {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}


class ToolContext:
    def __init__(self, repo, workspace, bridge, state, cancel_event=None):
        self.repo, self.workspace = Path(repo).resolve(), Path(workspace).resolve()
        self.bridge, self.state, self.cancel_event = bridge, state, cancel_event
        self.state.setdefault("done", False)
        self.state.setdefault("goal_changed", False)
        self.state.setdefault("current", {})
        self.state.setdefault("revision", 0)
        if bridge is not None and cancel_event is not None:
            bridge.cancel_event = cancel_event

    def result(self, data, images=()):
        refs = list(images)
        for ref in refs:
            path = memory._path(self.workspace, ref, {"scene", "observations", "scratch"})
            if not path.is_file():
                raise ValueError(f"Public image is missing: {ref}")
        return {"content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}],
                "images": refs, "details": {"images": refs}}

    def checkpoint(self):
        current = deepcopy(self.state["current"])
        refs = list(current.get("images", []))
        target = (current.get("task") or {}).get("image")
        if target and target not in refs:
            refs.append(target)
        if self.state.get("latest_map") and self.state["latest_map"] not in refs:
            refs.append(self.state["latest_map"])
        metadata = {"current": current, "latest_map": self.state.get("latest_map"),
                    "latest_map_metadata": deepcopy(self.state.get("latest_map_metadata"))}
        return {"content": json.dumps(metadata, ensure_ascii=False), "images": refs,
                "metadata": metadata, "revision": self.state["revision"]}


class Tool:
    def __init__(self, context):
        self.context = context

    def spec(self):
        return {"name": self.name, "description": self.description,
                "parameters": deepcopy(self.parameters)}
