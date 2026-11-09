"""Model-written place summaries indexing immutable scene panoramas."""

from .errors import ToolInputError

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
            atomic_json(self.path, {"nodes": [], "rooms": {}})
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
    def state(self):
        state = json.loads(self.path.read_text())
        if (not isinstance(state, dict) or set(state) != {"nodes", "rooms"}
                or not isinstance(state["nodes"], list) or not isinstance(state["rooms"], dict)):
            raise ValueError("memory/nodes.json must contain nodes array and rooms description object")
        for room, description in state["rooms"].items():
            self._room_description(room, description)
        return state

    @property
    def nodes(self):
        return self.state["nodes"]

    @property
    def rooms(self):
        return self.state["rooms"]

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
        raise ToolInputError(f"Unknown node: {identifier}")

    @staticmethod
    def _room_records(state):
        members = {}
        for node in state["nodes"]:
            if node["room"] is not None:
                members.setdefault(node["room"], []).append(node["id"])
        return [{"room": room, "description": state["rooms"].get(room), "nodes": members.get(room, [])}
                for room in dict.fromkeys([*members, *state["rooms"]])]

    def read_rooms(self, room=None, query=None, limit=10, offset=0):
        try:
            self._pagination(limit, offset)
            if query is not None and not isinstance(query, str):
                raise ToolInputError("query must be a string")
            rows = self._room_records(self.state)
            if room is not None:
                rows = [row for row in rows if row["room"] == room]
                if not rows:
                    raise ToolInputError(f"Unknown room: {room}")
            if query is not None:
                rows = [row for row in rows
                        if query.casefold() in (row["room"] + " " + (row["description"] or "")).casefold()]
            return {"rooms": rows[offset:offset + limit], "total": len(rows),
                    "next_offset": offset + limit if offset + limit < len(rows) else None}
        except ToolInputError as error:
            return {"error": str(error)}

    def read(self, id=None, room=None, query=None, limit=10, offset=0):
        try:
            self._pagination(limit, offset)
            if id is not None:
                if room is not None or query is not None:
                    raise ToolInputError("id cannot be combined with room or query")
                node = self.get(id)
                total = len(node["history"])
                node["history"] = node["history"][offset:offset + limit]
                node["history_total"] = total
                return {"nodes": [node], "total": 1,
                        "next_offset": offset + limit if offset + limit < total else None}
            if room is not None and (not isinstance(room, str) or not room.strip()):
                raise ToolInputError("room must be a nonempty string or null")
            if query is not None and not isinstance(query, str):
                raise ToolInputError("query must be a string")
            state = self.state
            nodes = state["nodes"]
            if room is not None:
                room_record = next((row for row in self._room_records(state) if row["room"] == room), None)
                if room_record is None:
                    raise ToolInputError(f"Unknown room: {room}")
                nodes = [node for node in nodes if node["room"] == room]
            if query is not None:
                nodes = [node for node in nodes
                         if query.casefold() in (node["description"] + " " + (node["room"] or "")).casefold()]
            summaries = [{key: detail[key] for key in ("id", "room", "description", "pose", "panorama")}
                         | {"history_count": len(detail["history"])}
                         for detail in map(self._detail, nodes[offset:offset + limit])]
            return {"nodes": summaries, "total": len(nodes),
                    **({"room": room_record} if room is not None else {}),
                    "next_offset": offset + limit if offset + limit < len(nodes) else None}
        except ToolInputError as error:
            return {"error": str(error)}

    def edit(self, operations, current_panorama=None):
        context = ""
        try:
            if not isinstance(operations, list) or not 1 <= len(operations) <= 20:
                raise ToolInputError("operations must contain 1-20 items")
            snapshot = self.state
            state = {node["id"]: node for node in snapshot["nodes"]}
            rooms = snapshot["rooms"]
            used = self._used() | set(state)
            number = max((int(identifier[1:]) for identifier in used
                          if re.fullmatch(r"N\d{3,}", identifier)), default=0)
            changes = []
            for index, operation in enumerate(operations, 1):
                if not isinstance(operation, dict):
                    raise ToolInputError(f"Operation {index} must be an object")
                op, identifier = operation.get("op"), operation.get("id", operation.get("keep"))
                context = f"Operation {index} {op}" + (f" {identifier}" if identifier else "") + ": "
                if op == "update" and "id" not in operation and "room" in operation:
                    self._keys(operation, {"op", "room", "description"})
                    room = operation["room"]
                    if "description" not in operation:
                        raise ToolInputError("Room update requires description; null clears it")
                    description = operation["description"]
                    self._room_description(room, description, allow_null=True)
                    if room not in rooms and not any(node["room"] == room for node in state.values()):
                        raise ToolInputError(f"Unknown room: {room}; assign a node to this room first")
                    if description is None:
                        rooms.pop(room, None)
                    else:
                        rooms[room] = description
                    changes.append({"op": op, "room": room})
                    continue
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
                        raise ToolInputError("refresh must be a boolean")
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
                    raise ToolInputError("op must be create, update, delete or merge")
                changes.append({"op": op, "id": identifier,
                                **({"remove": operation["remove"]} if op == "merge" else {})})
            for node in state.values():
                self._validate(node)
        except ToolInputError as error:
            return {"error": context + str(error)}
        # Reserve issued IDs before replacing the public snapshot, as entity memory does.
        atomic_json(self.used_path, sorted(used))
        atomic_json(self.path, {"nodes": list(state.values()), "rooms": rooms})
        changed = {change["id"] for change in changes if "id" in change}
        return {"changes": changes, "nodes": [self._detail(node) for node in state.values() if node["id"] in changed]}

    @staticmethod
    def _pagination(limit, offset):
        if type(limit) is not int or not 1 <= limit <= 300 or type(offset) is not int or offset < 0:
            raise ToolInputError("limit must be 1-300; offset must be a nonnegative integer")

    @staticmethod
    def _room_description(room, description, allow_null=False):
        if not isinstance(room, str) or not room.strip():
            raise ToolInputError("room must be a nonempty string")
        if description is None and allow_null:
            return
        if not isinstance(description, str) or not description.strip() or len(description) > 1000:
            raise ToolInputError("Room description must be a nonempty string of 1-1000 characters")

    @staticmethod
    def _keys(value, allowed):
        if set(value) - allowed:
            raise ToolInputError("Unknown fields: " + ", ".join(sorted(set(value) - allowed)))

    @staticmethod
    def _node(state, identifier):
        if not isinstance(identifier, str) or identifier not in state:
            raise ToolInputError(f"Unknown node: {identifier}")
        return state[identifier]

    @staticmethod
    def _text_fields(value, require_description=False):
        if require_description or "description" in value:
            description = value.get("description")
            if not isinstance(description, str) or not description.strip() or len(description) > 300:
                raise ToolInputError("description must be a nonempty string of 1-300 characters")
        room = value.get("room")
        if room is not None and (not isinstance(room, str) or not room.strip()):
            raise ToolInputError("room must be a nonempty string or null")

    def _current(self, group):
        if group is None:
            raise ToolInputError("A current complete panorama is required; call observe(mode=panorama) first")
        return self.observations.panorama(group)

    def _near(self, anchor, panorama):
        first, second = (self.observations.panorama(group)["pose"]["pos"] for group in (anchor, panorama))
        distance = math.dist(first, second) * self.transform["size_m"]
        if distance > 1 + 1e-9:
            raise ToolInputError(f"Panorama is {distance:.3f}m from the fixed node anchor; maximum is 1m")

    def _merge(self, state, operation):
        keep, remove = operation.get("keep"), operation.get("remove")
        node = self._node(state, keep)
        if (not isinstance(remove, list) or not remove or any(not isinstance(item, str) for item in remove)
                or len(set(remove)) != len(remove) or keep in remove):
            raise ToolInputError("remove must contain distinct existing node IDs excluding keep")
        sources = [node] + [self._node(state, identifier) for identifier in remove]
        for source in sources[1:]:
            self._near(node["anchor"], source["anchor"])
        rooms = {source["room"] for source in sources if source["room"] is not None}
        if len(rooms) > 1 and "room" not in operation:
            raise ToolInputError("Conflicting room; specify the final room")
        node["description"] = operation.get("description")
        node["room"] = operation["room"] if "room" in operation else node["room"] or next(iter(rooms), None)
        node["history"] = list(dict.fromkeys(group for source in sources for group in source["history"]))
        for identifier in remove:
            del state[identifier]

    def _validate(self, node):
        if not isinstance(node, dict) or set(node) != {"id", "anchor", "panorama", "room", "description", "history"}:
            raise ToolInputError("Node must contain id, anchor, panorama, room, description and history")
        if not isinstance(node["id"], str) or not re.fullmatch(r"N\d{3,}", node["id"]):
            raise ToolInputError("Node ID must be N followed by at least three digits")
        self._text_fields(node, require_description=True)
        history = node["history"]
        if (not isinstance(history, list) or any(not isinstance(group, str) for group in history)
                or len(set(history)) != len(history) or node["anchor"] not in history or node["panorama"] not in history):
            raise ToolInputError("history must contain distinct panoramas including anchor and current panorama")
        for group in history:
            self.observations.panorama(group)
        self._near(node["anchor"], node["panorama"])
