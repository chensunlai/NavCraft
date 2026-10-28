"""Node memory tools reuse the navigation image and map readers."""

from jsonschema import Draft7Validator

from .node_specs import memory_specifications


class NodeMemoryTools:
    def __init__(self, session, navigation_tools):
        self.session = session
        self.navigation = navigation_tools
        self.validators = {spec["name"]: Draft7Validator(spec["parameters"])
                           for spec in memory_specifications()}

    def _validate(self, name, arguments):
        error = next(self.validators[name].iter_errors(arguments), None)
        if error is not None:
            field = ".".join(map(str, error.absolute_path)) or "arguments"
            raise ValueError(f"Invalid {name} {field}: {error.message}")

    def read(self, mode="text", **args):
        self._validate("memory_read", {"mode": mode, **args})
        s = self.session
        if mode == "text":
            source = args.pop("source", "nodes")
            if source == "nodes":
                result = s.memory.read(**args)
            else:
                result = s.observations.history(**args)
                associations = {}
                for node in s.memory.nodes:
                    for group in set([node["anchor"], node["panorama"], *node["history"]]):
                        associations.setdefault(group, []).append(node["id"])
                result = {**result, "events": [{**event, "nodes": associations.get(
                    event.get("group", event["image"]), [])} for event in result["events"]]}
            return s._reply(result, error="error" in result)
        if mode == "image":
            images = []
            for reference in args["refs"]:
                image = s.memory.get(reference)["panorama"] if reference.startswith("N") else reference
                images.append({"image": image, **({"crop": args["crop"]} if "crop" in args else {})})
            return self.navigation.images(images=images, layout="separate")
        view = args.get("view", "local")
        radius = args.get("radius_m", 6)
        mapped = "explored" if view == "explored" else {"ego" if view == "ego" else "around_self": radius}
        value = self.navigation.map_value(view=mapped, highlight=args.get("highlight"))
        return s._reply(value, [value["map"]], [f"[MAP {value['view_id']} | model-written nodes]"])

    def edit(self, operations):
        s = self.session
        if s._qa:
            raise ValueError("This question permits only existing-evidence queries and submission")
        self._validate("memory_edit", {"operations": operations})
        needs_panorama = [op for op in operations
                          if op["op"] == "create" or op["op"] == "update" and op.get("refresh")]
        if len(needs_panorama) > 1:
            raise ValueError("A batch permits at most one create or refresh")
        panorama = s.observations.current_panorama(s.current) if needs_panorama else None
        before = s.trace.store(s.memory.nodes)
        result = s.memory.edit(operations, current_panorama=panorama)
        if "error" in result:
            return s._reply({**result, "committed": False}, error=True)
        s._revision += 1
        s.trace.emit("memory_changed", {"before": before, "after": s.trace.store(s.memory.nodes)})
        changes = []
        for change, operation in zip(result["changes"], operations):
            row = dict(change)
            if operation["op"] != "delete":
                node = next((n for n in s.memory.nodes if n["id"] == row["id"]), None)
                if node is not None:
                    if operation["op"] in {"create", "merge"} or "room" in operation:
                        row["room"] = node["room"]
                    if operation["op"] in {"create", "merge"} or operation.get("refresh"):
                        row["panorama"] = node["panorama"]
            changes.append(row)
        return s._reply({"changes": changes, "committed": True})
