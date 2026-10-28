"""Model-written place summaries indexing immutable scene panoramas."""

from copy import deepcopy
import json
import math
from pathlib import Path
import re

from .memory import atomic_json


class NodeMemory:
    def __init__(self, workspace, observations, transform, private_dir=None):
        self.workspace, self.observations, self.transform = Path(workspace), observations, transform
        self.path = self.workspace / "memory/nodes.json"
        self.used_path = Path(private_dir or self.workspace.parent / "private") / "memory_used_ids.json"
        size = self.transform.get("size_m")
        if type(size) not in (int, float) or not math.isfinite(size) or size <= 0:
            raise ValueError("Map size_m must be positive")
        if not self.path.exists():
            atomic_json(self.path, [])
        nodes = self.nodes
        for node in nodes:
            self._validate(node)
        if len({node["id"] for node in nodes}) != len(nodes):
            raise ValueError("Node snapshot contains duplicate IDs")
        used = self._used()
        identifiers = {node["id"] for node in nodes}
        if not identifiers.issubset(used):
            atomic_json(self.used_path, sorted(used | identifiers))

    @property
    def nodes(self):
        nodes = json.loads(self.path.read_text())
        if not isinstance(nodes, list):
            raise ValueError("memory/nodes.json must be an array")
        return nodes

    def _used(self):
        used = json.loads(self.used_path.read_text()) if self.used_path.exists() else []
        if not isinstance(used, list) or any(not isinstance(value, str) for value in used):
            raise ValueError("Memory ID ledger must be an array of strings")
        return set(used)

    def _detail(self, node):
        current = self.observations.panorama(node["panorama"])
        return {**deepcopy(node), "pose": self.observations.panorama(node["anchor"])["pose"],
                "observation_pose": current["pose"], "step": current["step"]}

    def get(self, identifier):
        for node in self.nodes:
            if node["id"] == identifier:
                return self._detail(node)
        raise ValueError(f"Unknown node: {identifier}")

    def summary(self, limit=10):
        return self.read(limit=min(limit, 10))

    def read(self, id=None, room=None, query=None, limit=10, offset=0):
        try:
            if type(limit) is not int or not 1 <= limit <= 300 or type(offset) is not int or offset < 0:
                raise ValueError("limit must be 1-300; offset must be a nonnegative integer")
            if id is not None:
                if room is not None or query is not None:
                    raise ValueError("id cannot be combined with room or query")
                node = self.get(id)
                total = len(node["history"])
                node["history"] = node["history"][offset:offset + limit]
                node["history_total"] = total
                return {"nodes": [node], "total": 1,
                        "next_offset": offset + limit if offset + limit < total else None}
            if room is not None and (not isinstance(room, str) or not room.strip()):
                raise ValueError("room must be a nonempty string or null")
            if query is not None and not isinstance(query, str):
                raise ValueError("query must be a string")
            nodes = self.nodes
            if room is not None:
                nodes = [node for node in nodes if node["room"] == room]
            if query is not None:
                nodes = [node for node in nodes
                         if query.casefold() in (node["description"] + " " + (node["room"] or "")).casefold()]
            summaries = [{key: detail[key] for key in ("id", "room", "description", "pose", "panorama")}
                         | {"history_count": len(detail["history"])}
                         for detail in map(self._detail, nodes[offset:offset + limit])]
            return {"nodes": summaries, "total": len(nodes),
                    "next_offset": offset + limit if offset + limit < len(nodes) else None}
        except (ValueError, TypeError, KeyError) as error:
            return {"error": str(error)}

    def edit(self, operations, current_panorama=None):
        context = ""
        try:
            if not isinstance(operations, list) or not 1 <= len(operations) <= 20:
                raise ValueError("operations must contain 1-20 items")
            state = {node["id"]: node for node in self.nodes}
            used = self._used() | set(state)
            number = max((int(identifier[1:]) for identifier in used
                          if re.fullmatch(r"N\d{3,}", identifier)), default=0)
            changes = []
            for index, operation in enumerate(operations, 1):
                if not isinstance(operation, dict):
                    raise ValueError(f"Operation {index} must be an object")
                op, identifier = operation.get("op"), operation.get("id", operation.get("keep"))
                context = f"Operation {index} {op}" + (f" {identifier}" if identifier else "") + ": "
                if op == "create":
                    self._keys(operation, {"op", "description", "room"})
                    self._text_fields(operation, require_description=True)
                    self._current(current_panorama)
                    number += 1
                    identifier = f"N{number:03d}"
                    state[identifier] = {"id": identifier, "anchor": current_panorama,
                                         "panorama": current_panorama, "room": operation.get("room"),
                                         "description": operation.get("description"), "history": [current_panorama]}
                    used.add(identifier)
                elif op == "update":
                    self._keys(operation, {"op", "id", "description", "room", "refresh"})
                    self._text_fields(operation)
                    node = self._node(state, identifier)
                    if type(operation.get("refresh", False)) is not bool:
                        raise ValueError("refresh must be a boolean")
                    for key in ("description", "room"):
                        if key in operation:
                            node[key] = operation[key]
                    if operation.get("refresh"):
                        self._current(current_panorama)
                        self._near(node["anchor"], current_panorama)
                        node["panorama"] = current_panorama
                        if current_panorama not in node["history"]:
                            node["history"].append(current_panorama)
                elif op == "delete":
                    self._keys(operation, {"op", "id"})
                    self._node(state, identifier)
                    del state[identifier]
                elif op == "merge":
                    self._keys(operation, {"op", "keep", "remove", "description", "room"})
                    self._text_fields(operation, require_description=True)
                    self._merge(state, operation)
                else:
                    raise ValueError("op must be create, update, delete or merge")
                changes.append({"op": op, "id": identifier,
                                **({"remove": operation["remove"]} if op == "merge" else {})})
            for node in state.values():
                self._validate(node)
        except (ValueError, TypeError, KeyError) as error:
            return {"error": context + str(error)}
        # Reserve issued IDs before replacing the public snapshot, as entity memory does.
        atomic_json(self.used_path, sorted(used))
        atomic_json(self.path, list(state.values()))
        changed = {change["id"] for change in changes}
        return {"changes": changes, "nodes": [self._detail(node) for node in state.values() if node["id"] in changed]}

    @staticmethod
    def _keys(value, allowed):
        if set(value) - allowed:
            raise ValueError("Unknown fields: " + ", ".join(sorted(set(value) - allowed)))

    @staticmethod
    def _node(state, identifier):
        if not isinstance(identifier, str) or identifier not in state:
            raise ValueError(f"Unknown node: {identifier}")
        return state[identifier]

    @staticmethod
    def _text_fields(value, require_description=False):
        if require_description or "description" in value:
            description = value.get("description")
            if not isinstance(description, str) or not description.strip() or len(description) > 300:
                raise ValueError("description must be a nonempty string of 1-300 characters")
        room = value.get("room")
        if room is not None and (not isinstance(room, str) or not room.strip()):
            raise ValueError("room must be a nonempty string or null")

    def _current(self, group):
        if group is None:
            raise ValueError("A current complete panorama is required; call observe(mode=panorama) first")
        return self.observations.panorama(group)

    def _near(self, anchor, panorama):
        first, second = (self.observations.panorama(group)["pose"]["pos"] for group in (anchor, panorama))
        distance = math.dist(first, second) * self.transform["size_m"]
        if distance > 1 + 1e-9:
            raise ValueError(f"Panorama is {distance:.3f}m from the fixed node anchor; maximum is 1m")

    def _merge(self, state, operation):
        keep, remove = operation.get("keep"), operation.get("remove")
        node = self._node(state, keep)
        if (not isinstance(remove, list) or not remove or any(not isinstance(item, str) for item in remove)
                or len(set(remove)) != len(remove) or keep in remove):
            raise ValueError("remove must contain distinct existing node IDs excluding keep")
        sources = [node] + [self._node(state, identifier) for identifier in remove]
        for source in sources[1:]:
            self._near(node["anchor"], source["anchor"])
        rooms = {source["room"] for source in sources if source["room"] is not None}
        if len(rooms) > 1 and "room" not in operation:
            raise ValueError("Conflicting room; specify the final room")
        node["description"] = operation.get("description")
        node["room"] = operation["room"] if "room" in operation else node["room"] or next(iter(rooms), None)
        node["history"] = list(dict.fromkeys(group for source in sources for group in source["history"]))
        for identifier in remove:
            del state[identifier]

    def _validate(self, node):
        if not isinstance(node, dict) or set(node) != {"id", "anchor", "panorama", "room", "description", "history"}:
            raise ValueError("Node must contain id, anchor, panorama, room, description and history")
        if not isinstance(node["id"], str) or not re.fullmatch(r"N\d{3,}", node["id"]):
            raise ValueError("Node ID must be N followed by at least three digits")
        self._text_fields(node, require_description=True)
        history = node["history"]
        if (not isinstance(history, list) or any(not isinstance(group, str) for group in history)
                or len(set(history)) != len(history) or node["anchor"] not in history or node["panorama"] not in history):
            raise ValueError("history must contain distinct panoramas including anchor and current panorama")
        for group in history:
            self.observations.panorama(group)
        self._near(node["anchor"], node["panorama"])
