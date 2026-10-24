"""Geometric assistance for model-selected points; never reads task goals or scores."""

from copy import deepcopy
import json
import math
import shutil
import time

from src.common.io import write_json
from .geometry import Geometry
from .memory import location


class SuperTools:
    def __init__(self, session):
        self.session = session
        self.geometry = Geometry(session.workspace, session.private, session.transform)
        self.sources_path = session.private / "location_sources.json"
        self.planner = session.config.get("goto_planner", "local")
        if self.planner not in {"local", "straight"}:
            raise ValueError("goto_planner must be local or straight")

    def record_frame(self, observation, row, body_pose=None):
        if observation.depth is None:
            raise ValueError("super tools require private RGB-D captures")
        camera, pose = observation.camera, observation.pose
        directory = self.session.private / "depth"
        directory.mkdir(exist_ok=True)
        depth = directory / (row["frame"] + ".png")
        shutil.copyfile(observation.depth, depth)
        body_pose = body_pose or pose
        qx, qy, qz, qw = body_pose["rotation_xyzw"]
        record = {"frame": row["frame"], "depth": depth.name,
                  "depth_scale": camera["depth_scale"], "K": camera["K"],
                  "world_from_optical": pose["world_from_optical"],
                  "body_position": body_pose["position"],
                  "body_forward": [-2 * (qx * qz + qw * qy), 0, -(1 - 2 * (qx * qx + qy * qy))],
                  "floor_height": self.session.transform["floor_height"],
                  "width": camera["width"], "height": camera["height"],
                  "depth_min": camera["depth_min"], "depth_max": camera["depth_max"]}
        with (directory / "index.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    def prepare_edits(self, operations):
        if not isinstance(operations, list) or not 1 <= len(operations) <= 20:
            raise ValueError("operations must contain 1-20 items")
        prepared, sources = deepcopy(operations), {}
        kinds = {e["id"]: e["type"] for e in self.session.memory.entities}
        for op in prepared:
            if not isinstance(op, dict):
                raise ValueError("Each operation must be an object")
            identifier = op.get("id", op.get("keep"))
            if op.get("op") == "create":
                kinds[identifier] = op.get("type")
            fields = op.get("fields", {})
            if not isinstance(fields, dict):
                raise ValueError("fields must be an object")
            at, reference = fields.pop("at", None), op.pop("coordinate_ref", None)
            if at is not None:
                if reference is not None or any(k in fields for k in ("point", "anchor", "segment")):
                    raise ValueError("at cannot be combined with explicit geometry or coordinate_ref")
                if not isinstance(at, dict) or set(at) - {"frame", "pixel", "pixels"}:
                    raise ValueError("at requires an original frame and pixel or pixels")
                kind = kinds.get(identifier)
                field = {"room": "anchor", "door": "segment", "object": "point"}.get(kind)
                if field is None:
                    raise ValueError("at requires an existing or newly created entity type")
                key = "pixels" if kind == "door" else "pixel"
                if set(at) != {"frame", key}:
                    raise ValueError(f"at for {kind} requires frame and {key}")
                pixels = at[key] if kind == "door" else [at[key]]
                if not isinstance(pixels, list) or len(pixels) != (2 if kind == "door" else 1):
                    raise ValueError("A door requires two pixels")
                values = [self.geometry.resolve_frame(at["frame"], pixel) for pixel in pixels]
                if any(not value["valid"] for value in values):
                    raise ValueError("at has invalid geometry: " + ", ".join(v.get("reason", "valid") for v in values))
                fields[field] = [v["uv"] for v in values] if kind == "door" else values[0]["uv"]
                evidence = fields.get("evidence", [])
                if evidence is None:
                    evidence = []
                if not isinstance(evidence, list):
                    raise ValueError("evidence must be an array")
                fields["evidence"] = list(dict.fromkeys([*evidence, at["frame"]]))
                position = ([(values[0]["uv"][i] + values[1]["uv"][i]) / 2 for i in (0, 1)]
                            if kind == "door" else values[0]["uv"])
                sources[identifier] = {"frame": at["frame"], "uv": position,
                                       "height": sum(v["height"] for v in values) / len(values)}
            elif reference is not None:
                if not isinstance(reference, str) or not reference.startswith("m_"):
                    raise ValueError("coordinate_ref must be a saved map view ID")
                for field in ("point", "anchor", "segment"):
                    if fields.get(field) is not None:
                        value = fields[field]
                        fields[field] = ([self.geometry.resolve_map(reference, p)["uv"] for p in value]
                                         if field == "segment" else self.geometry.resolve_map(reference, value)["uv"])
        return prepared, sources

    def remember_sources(self, operations, additions):
        sources = json.loads(self.sources_path.read_text()) if self.sources_path.exists() else {}
        entities = {e["id"]: e for e in self.session.memory.entities}
        for op in operations:
            identifier = op.get("id", op.get("keep"))
            if any(key in op.get("fields", {}) for key in ("point", "anchor", "segment")):
                sources.pop(identifier, None)
        for identifier, source in additions.items():
            if identifier in entities and source["uv"] == location(entities[identifier]):
                sources[identifier] = source
        sources = {key: value for key, value in sources.items()
                   if key in entities and value["uv"] == location(entities[key])}
        write_json(self.sources_path, sources)

    def near(self, near, ids=None, filter=None, limit=30, offset=0):
        if ids is not None or not isinstance(near, dict):
            raise ValueError("near must be an object and cannot be combined with ids")
        session = self.session
        if set(near) == {"self"}:
            centre, radius = session.current["pose"]["pos"], near["self"]
        elif "entity" in near and not set(near) - {"entity", "radius"}:
            centre, radius = self.target({"entity": near["entity"]})["uv"], near.get("radius", 3)
        else:
            raise ValueError("near requires self:radius or entity:id with optional radius")
        if type(radius) not in (int, float) or not math.isfinite(radius) or radius <= 0:
            raise ValueError("near radius must be positive")
        # Let the base memory reader enforce filtering and pagination types.
        validation = session.memory.read(filter=filter, limit=limit, offset=offset)
        if "error" in validation:
            return validation
        candidates = session.memory.read(filter=filter, limit=300)["entities"]
        entities = {e["id"]: e for e in session.memory.entities}
        found = []
        for summary in candidates:
            uv = location(entities[summary["id"]])
            if uv is None:
                continue
            distance = math.dist(centre, uv) * session.transform["size_m"]
            if distance <= radius:
                bearing, current_distance = self.relative(uv)
                found.append((distance, {**summary, "bearing": round(bearing, 2), "distance": round(current_distance, 3)}))
        found.sort(key=lambda item: (item[0], item[1]["id"]))
        return {"entities": [item[1] for item in found[offset:offset + limit]], "total": len(found),
                "next_offset": offset + limit if offset + limit < len(found) else None}

    def target(self, target):
        if not isinstance(target, dict):
            raise ValueError("target must specify frame/pixel, uv or entity")
        if set(target) == {"frame", "pixel"}:
            value = self.geometry.resolve_frame(target["frame"], target["pixel"])
            if not value["valid"]:
                raise ValueError(value["reason"])
            return {"uv": value["uv"], "height": value["height"], "observed": True}
        if set(target) == {"entity"}:
            entity = next((e for e in self.session.memory.entities if e["id"] == target["entity"]), None)
            if entity is None or location(entity) is None:
                raise ValueError("Entity is missing or has no location")
            sources = json.loads(self.sources_path.read_text()) if self.sources_path.exists() else {}
            source = sources.get(entity["id"], {})
            observed = source.get("uv") == location(entity)
            return {"uv": location(entity), "height": source.get("height", .5 if entity["type"] == "object" else 0),
                    "observed": observed}
        if set(target) == {"uv"}:
            uv = target["uv"]
            if (not isinstance(uv, list) or len(uv) != 2 or
                    any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in uv)):
                raise ValueError("uv must contain two finite numbers in [0,1]")
            return {"uv": uv, "height": 0, "observed": False}
        raise ValueError("target must specify exactly frame/pixel, uv or entity")

    def relative(self, uv):
        pose = self.session.current["pose"]
        dx, dz = [(uv[i] - pose["pos"][i]) * self.session.transform["size_m"] for i in (0, 1)]
        distance = math.hypot(dx, dz)
        bearing = (math.degrees(math.atan2(dx, -dz)) - pose["heading"] + 180) % 360 - 180 if distance > .001 else 0.
        return bearing, distance

    def move(self, target, *, face=False, stop_distance=None, max_actions=40):
        value = self.target(target)
        if type(max_actions) is not int or not 1 <= max_actions <= 40:
            raise ValueError("max_actions must be an integer in [1,40]")
        stop_distance = (.2 if value["height"] <= .15 else .6) if stop_distance is None else stop_distance
        if type(stop_distance) not in (int, float) or not math.isfinite(stop_distance) or stop_distance <= 0:
            raise ValueError("stop_distance must be positive")
        session, goal = self.session, value["uv"]
        route, planner = None, "straight"
        bearing, remaining = self.relative(goal)
        stop = "completed"
        if not face and self.planner == "local" and value["observed"] and remaining > stop_distance:
            planner = "local"
            if remaining > 8:
                stop = "outside_local_scope"
            else:
                origin, size = session.transform["origin"], session.transform["size_m"]
                point = [origin[0] + goal[0] * size, session.transform["floor_height"], origin[1] + goal[1] * size]
                endpoint = session.environment.near_point(point, stop_distance)
                route = session.environment.shortest_path([endpoint]) if endpoint is not None else []
                start = session.environment.position()
                length = sum(math.dist(a, b) for a, b in zip(route, route[1:]))
                if not route:
                    stop = "unreachable"
                elif length > 8 or any(abs(p[1] - start[1]) > .5 or math.dist(p, start) > 8 for p in route):
                    stop = "outside_local_scope"
        done, events, turn_degrees = [], [], 0.
        start_steps = session.environment.steps
        deadline = time.monotonic() + 90
        while stop == "completed" and session.environment.steps - start_steps < max_actions:
            if time.monotonic() >= deadline:
                stop = "time_limit"
                break
            bearing, remaining = self.relative(goal)
            if planner == "local" and (remaining > 8 or math.dist(session.environment.position(), start) > 8
                                       or abs(session.environment.position()[1] - start[1]) > .5):
                stop = "outside_local_scope"
                break
            if face and abs(bearing) <= 1 or not face and remaining <= stop_distance:
                stop = "aligned" if face else "reached"
                break
            waypoint = goal
            if route:
                current = session.environment.position()
                while len(route) > 1 and math.dist(current, route[0]) < .26:
                    route.pop(0)
                origin, size = session.transform["origin"], session.transform["size_m"]
                waypoint = [(route[0][0] - origin[0]) / size, (route[0][2] - origin[1]) / size]
            turn, _ = self.relative(waypoint)
            degrees = max(-30., min(30., turn)) if abs(turn) > 1 else None
            action = ("turn_right" if degrees > 0 else "turn_left") if degrees is not None else "move_forward"
            receipt = session._perform_motion(action, degrees)
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
                events.append({"i": session.environment.steps - start_steps, "collision": bool(motion.collision),
                               "auto_yaw": round(-motion.auto_yaw_delta_degrees, 2)})
            if motion.blocked:
                stop = "blocked"
                break
        bearing, remaining = self.relative(goal)
        reached = abs(bearing) <= 1 if face else remaining <= stop_distance
        if stop == "completed":
            stop = ("aligned" if face else "reached") if reached else "max_actions"
        session._publish_state()
        result = {"done": done, "stop": stop, "frame": session.current["frame"],
                  "step": session.environment.steps, "pose": session.current["pose"], "turn_degrees": round(turn_degrees, 2)}
        if events:
            result["events"] = events
        if face:
            result["bearing_after"] = round(bearing, 2)
        else:
            result.update(reached=reached, remaining=round(remaining, 3), planner=planner)
        return session._reply(result, [session.current["path"]], [f"[CURRENT {session.current['frame']}]"], end=bool(session._stop_reason))
