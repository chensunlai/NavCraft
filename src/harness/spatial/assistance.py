"""Geometric assistance for model-selected points; never reads task goals or scores."""

from ..errors import ToolInputError

from copy import deepcopy
import json
import math
import shutil
import time

from src.common.io import write_json
from src.common.coordinates import bearing_distance, body_forward, uv_to_world, world_to_uv
from .geometry import Geometry
from ..memory.entity import location


class SpatialTools:
    def __init__(self, executor):
        self.executor = executor
        self.geometry = Geometry(executor.workspace, executor.private, executor.transform,
                                 bounded=executor.memory_type != "node")
        self.sources_path = executor.private / "location_sources.json"
        self.enabled = executor.selection.geometry
        self.planner = executor.goto_planner
        if self.planner not in {"local", "straight"}:
            raise ValueError("goto_planner must be local or straight")

    def record_frame(self, observation, row, body_pose=None):
        if observation.depth is None:
            raise ValueError("super tools require private RGB-D captures")
        camera, pose = observation.camera, observation.pose
        directory = self.executor.private / "depth"
        directory.mkdir(exist_ok=True)
        depth = directory / (row["frame"] + ".png")
        shutil.copyfile(observation.depth, depth)
        body_pose = body_pose or pose
        record = {"frame": row["frame"], "depth": depth.name,
                  "depth_scale": camera["depth_scale"], "K": camera["K"],
                  "world_from_optical": pose["world_from_optical"],
                  "body_position": body_pose["position"],
                  "body_forward": body_forward(body_pose["rotation_xyzw"]),
                  "floor_height": self.executor.transform["floor_height"],
                  "width": camera["width"], "height": camera["height"],
                  "depth_min": camera["depth_min"], "depth_max": camera["depth_max"]}
        with (directory / "index.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    def prepare_edits(self, operations):
        prepared, sources = deepcopy(operations), {}
        entities = self.executor.memory.entities
        kinds = {e["id"]: e["type"] for e in entities}
        evidence_by_id = {e["id"]: list(e.get("evidence", [])) for e in entities}
        for op in prepared:
            identifier = op.get("id", op.get("keep"))
            if op.get("op") == "create":
                kinds[identifier] = op.get("type")
                evidence_by_id[identifier] = []
            fields = op.get("fields", {})
            reference = fields.pop("location", None)
            if reference is not None:
                if any(k in fields for k in ("point", "anchor")):
                    raise ToolInputError("location cannot be combined with point or anchor")
                kind = kinds.get(identifier)
                if kind not in {"room", "door", "object"}:
                    raise ToolInputError("location requires an existing or newly created entity type")
                value = self.target(reference)
                fields["anchor" if kind == "room" else "point"] = value["uv"]
                if value.get("frame"):
                    previous = evidence_by_id.get(identifier, []) if op.get("op") == "update" else []
                    evidence = fields.get("evidence", previous)
                    if evidence is None:
                        evidence = []
                    fields["evidence"] = list(dict.fromkeys([*evidence, value["frame"]]))
                if value["observed"]:
                    sources[identifier] = {key: value[key] for key in ("frame", "uv", "height")}
            # Follow this batch's evidence changes; omitted update fields preserve
            # prior records, while merge combines the kept and removed records.
            if op.get("op") == "merge":
                evidence_by_id[identifier] = list(dict.fromkeys([
                    *evidence_by_id.get(identifier, []),
                    *(frame for removed in op.get("remove", []) for frame in evidence_by_id.get(removed, [])),
                    *(fields.get("evidence") or []),
                ]))
                for removed in op.get("remove", []):
                    evidence_by_id.pop(removed, None)
                    kinds.pop(removed, None)
            elif op.get("op") == "delete":
                evidence_by_id.pop(identifier, None)
                kinds.pop(identifier, None)
            elif "evidence" in fields:
                evidence_by_id[identifier] = fields["evidence"] or []
        return prepared, sources

    def remember_sources(self, operations, additions):
        sources = json.loads(self.sources_path.read_text()) if self.sources_path.exists() else {}
        entities = {e["id"]: e for e in self.executor.memory.entities}
        for op in operations:
            identifier = op.get("id", op.get("keep"))
            if any(key in op.get("fields", {}) for key in ("point", "anchor")):
                sources.pop(identifier, None)
        for identifier, source in additions.items():
            if identifier in entities and source["uv"] == location(entities[identifier]):
                sources[identifier] = source
        sources = {key: value for key, value in sources.items()
                   if key in entities and value["uv"] == location(entities[key])}
        write_json(self.sources_path, sources)

    def near(self, near, ids=None, filter=None, limit=30, offset=0):
        if ids is not None or not isinstance(near, dict):
            raise ToolInputError("near must be an object and cannot be combined with ids")
        executor = self.executor
        if set(near) == {"self"}:
            centre, radius = executor.current["pose"]["pos"], near["self"]
        elif "entity" in near and not set(near) - {"entity", "radius"}:
            centre, radius = self.target({"entity": near["entity"]})["uv"], near.get("radius", 3)
        else:
            raise ToolInputError("near requires self:radius or entity:id with optional radius")
        if type(radius) not in (int, float) or not math.isfinite(radius) or radius <= 0:
            raise ToolInputError("near radius must be positive")
        # Let the base memory reader enforce filtering and pagination types.
        validation = executor.memory.read(filter=filter, limit=limit, offset=offset)
        if "error" in validation:
            return validation
        candidates = executor.memory.read(filter=filter, limit=300)["entities"]
        entities = {e["id"]: e for e in executor.memory.entities}
        found = []
        for summary in candidates:
            uv = location(entities[summary["id"]])
            if uv is None:
                continue
            distance = math.dist(centre, uv) * executor.transform["size_m"]
            if distance <= radius:
                bearing, current_distance = self.relative(uv)
                found.append((distance, {**summary, "bearing": round(bearing, 2), "distance": round(current_distance, 3)}))
        found.sort(key=lambda item: (item[0], item[1]["id"]))
        return {"entities": [item[1] for item in found[offset:offset + limit]], "total": len(found),
                "next_offset": offset + limit if offset + limit < len(found) else None}

    def target(self, target):
        if not isinstance(target, dict):
            raise ToolInputError("target must be an image/pixel, uv or entity reference")
        if set(target) == {"node"}:
            if self.executor.memory_type != "node":
                raise ToolInputError("Node references require node memory")
            node = self.executor.memory.get(target["node"])
            return {"uv": node["pose"]["pos"], "height": 0, "observed": False}
        if set(target) == {"image", "pixel"}:
            reference, pixel = self.executor.observations.resolve_pixel(target["image"], target["pixel"])
            if reference.startswith("m_"):
                return {**self.geometry.resolve_map(reference, pixel), "height": 0, "observed": False}
            if not self.enabled:
                raise ToolInputError("Basic tools cannot project RGB pixels. Use a map image (m_) and pixel, "
                                 "or global UV; omit an uncertain location rather than using the camera position.")
            value = self.geometry.resolve_frame(reference, pixel)
            if not value["valid"]:
                raise ToolInputError(value["reason"])
            return {"uv": value["uv"], "height": value["height"], "observed": True, "frame": reference}
        if set(target) == {"entity"}:
            entity = next((e for e in self.executor.memory.entities if e["id"] == target["entity"]), None)
            if entity is None or location(entity) is None:
                raise ToolInputError("Entity is missing or has no location")
            sources = json.loads(self.sources_path.read_text()) if self.sources_path.exists() else {}
            source = sources.get(entity["id"], {})
            observed = source.get("uv") == location(entity)
            return {"uv": location(entity), "height": source.get("height", .5 if entity["type"] == "object" else 0),
                    "observed": observed, **({"frame": source["frame"]} if observed else {})}
        if set(target) == {"uv"}:
            uv = target["uv"]
            if (not isinstance(uv, list) or len(uv) != 2 or
                    any(type(v) not in (int, float) or not math.isfinite(v)
                        or (self.executor.memory_type != "node" and not 0 <= v <= 1) for v in uv)):
                raise ToolInputError("uv must contain two finite numbers within the map coordinate domain")
            return {"uv": uv, "height": 0, "observed": False}
        raise ToolInputError("target must specify exactly image/pixel, uv or entity")

    def relative(self, uv):
        value = bearing_distance(self.executor.current["pose"], uv, self.executor.transform)
        return value["bearing_degrees"], value["distance_m"]

    def move(self, target, *, face=False, stop_distance=None, max_actions=40):
        value = self.target(target)
        if type(max_actions) is not int or not 1 <= max_actions <= 40:
            raise ToolInputError("max_actions must be an integer in [1,40]")
        stop_distance = (.2 if value["height"] <= .15 else .6) if stop_distance is None else stop_distance
        if type(stop_distance) not in (int, float) or not math.isfinite(stop_distance) or stop_distance <= 0:
            raise ToolInputError("stop_distance must be positive")
        executor, goal = self.executor, value["uv"]
        route, planner = None, "straight"
        bearing, remaining = self.relative(goal)
        stop = "completed"
        if not face and self.planner == "local" and value["observed"] and remaining > stop_distance:
            planner = "local"
            if remaining > 8:
                stop = "outside_local_scope"
            else:
                point = uv_to_world(goal, executor.transform)
                endpoint = executor.environment.near_point(point, stop_distance)
                route = executor.environment.shortest_path([endpoint]) if endpoint is not None else []
                start = executor.environment.position()
                length = sum(math.dist(a, b) for a, b in zip(route, route[1:]))
                if not route:
                    stop = "unreachable"
                elif length > 8 or any(abs(p[1] - start[1]) > .5 or math.dist(p, start) > 8 for p in route):
                    stop = "outside_local_scope"
        done, events, turn_degrees = [], [], 0.
        start_steps = executor.environment.steps
        deadline = time.monotonic() + 90
        while stop == "completed" and executor.environment.steps - start_steps < max_actions:
            if time.monotonic() >= deadline:
                stop = "time_limit"
                break
            bearing, remaining = self.relative(goal)
            if planner == "local" and (remaining > 8 or math.dist(executor.environment.position(), start) > 8
                                       or abs(executor.environment.position()[1] - start[1]) > .5):
                stop = "outside_local_scope"
                break
            if face and abs(bearing) <= 1 or not face and remaining <= stop_distance:
                stop = "aligned" if face else "reached"
                break
            waypoint = goal
            if route:
                current = executor.environment.position()
                while len(route) > 1 and math.dist(current, route[0]) < .26:
                    route.pop(0)
                waypoint = world_to_uv(route[0], executor.transform)
            turn, _ = self.relative(waypoint)
            degrees = max(-30., min(30., turn)) if abs(turn) > 1 else None
            action = ("turn_right" if degrees > 0 else "turn_left") if degrees is not None else "move_forward"
            receipt = executor.perform_motion(action, degrees)
            if receipt is None:
                stop = "interrupted"
                break
            motion = receipt
            if done and done[-1][0] == action:
                done[-1][1] += 1
            else:
                done.append([action, 1])
            turn_degrees += degrees or 0
            if motion.collision or motion.blocked:
                events.append({"i": executor.environment.steps - start_steps, "collision": bool(motion.collision),
                               "auto_yaw": round(-motion.auto_yaw_delta_degrees, 2)})
            if motion.blocked:
                stop = "blocked"
                break
        bearing, remaining = self.relative(goal)
        reached = abs(bearing) <= 1 if face else remaining <= stop_distance
        if stop == "completed":
            stop = ("aligned" if face else "reached") if reached else "max_actions"
        executor.publish_state()
        result = {"done": done, "stop": stop, "frame": executor.current["frame"],
                  "step": executor.environment.steps, "pose": executor.current["pose"], "turn_degrees": round(turn_degrees, 2)}
        if events:
            result["events"] = events
        if face:
            result["bearing_after"] = round(bearing, 2)
        else:
            result.update(reached=reached, remaining=round(remaining, 3), planner=planner)
        if "node" in target:
            result["target_node"] = target["node"]
            if not face and stop == "blocked":
                result["hint"] = "Straight approach blocked; choose an intermediate visible waypoint around the obstacle."
            elif not face and stop == "max_actions":
                result["hint"] = "Action limit reached before the node; inspect the endpoint before continuing."
        return result
