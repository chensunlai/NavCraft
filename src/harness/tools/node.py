"""Node operations; the executor validates schemas and enforces request boundaries."""

from ..errors import ToolInputError


class NodeMemoryTools:
    def __init__(self, executor, navigation_tools):
        self.executor = executor
        self.navigation = navigation_tools

    def read(self, mode="text", **args):
        s = self.executor
        if mode == "text":
            source = args.pop("source", "nodes")
            if source == "nodes":
                result = s.memory.read(**args)
            elif source == "rooms":
                result = s.memory.read_rooms(**args)
            else:
                result = s.observations.history(**args)
                associations = {}
                for node in s.memory.nodes:
                    for group in set([node["anchor"], node["panorama"], *node["history"]]):
                        associations.setdefault(group, []).append(node["id"])
                result = {**result, "events": [{**event, "nodes": associations.get(
                    event.get("group", event["image"]), [])} for event in result["events"]]}
            return s.reply(result, error="error" in result)
        if mode == "image":
            nodes = [s.memory.get(reference) if reference.startswith("N") else None
                     for reference in args["refs"]]
            rows = [s.observations.image(node["panorama"] if node else reference)
                    for reference, node in zip(args["refs"], nodes)]
            if "crop" in args:
                source = rows[0]
                rows[0] = {**s.observations.crop(source["image"], args["crop"]),
                           **{key: source[key] for key in ("pose", "step", "group") if key in source}}
            return self.navigation.image_reply(rows, nodes=nodes)
        room = None
        highlight = args.get("highlight")
        if "room" in args:
            rooms = s.memory.read_rooms(room=args["room"])
            if "error" in rooms:
                return s.reply(rooms, error=True)
            room = rooms["rooms"][0]
            highlight = list(dict.fromkeys([*(highlight or []), *room["nodes"]]))
        view = args.get("view", "explored" if room is not None else "local")
        radius = args.get("radius_m", 6)
        mapped = "explored" if view == "explored" else {"ego" if view == "ego" else "around_self": radius}
        value = self.navigation.map_value(view=mapped, highlight=highlight)
        if room is not None:
            value = {**value, "room": room}
        return s.reply(value, [value["map"]], [f"[MAP {value['view_id']} | model-written nodes]"])

    def edit(self, operations):
        s = self.executor
        needs_panorama = [op for op in operations
                          if op["op"] == "create" or op["op"] == "update" and op.get("refresh")]
        if len(needs_panorama) > 1:
            raise ToolInputError("A batch permits at most one create or refresh")
        panorama = s.observations.current_panorama(s.current) if needs_panorama else None
        before = s.trace.store(s.memory.state)
        result = s.memory.edit(operations, current_panorama=panorama)
        if "error" in result:
            return s.reply({**result, "committed": False}, error=True)
        s.trace.emit("memory_changed", {"before": before, "after": s.trace.store(s.memory.state)})
        changes = []
        for change, operation in zip(result["changes"], operations):
            row = dict(change)
            if operation["op"] != "delete" and "id" in row:
                node = next((n for n in s.memory.nodes if n["id"] == row["id"]), None)
                if node is not None:
                    if operation["op"] in {"create", "merge"} or "room" in operation:
                        row["room"] = node["room"]
                    if operation["op"] in {"create", "merge"} or operation.get("refresh"):
                        row["panorama"] = node["panorama"]
            changes.append(row)
        return s.reply({"changes": changes, "committed": True})
