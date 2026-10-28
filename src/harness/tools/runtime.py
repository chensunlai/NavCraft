"""Navigation operations and bounded, related material in each tool receipt."""

import uuid

from src.common.actions import ACTIONS
from src.common.coordinates import world_to_uv
from ..memory_feedback import memory_feedback


class NavigationTools:
    def __init__(self, session):
        self.session = session
        self.node_tools = None
        if session.memory_type == "node":
            from .node import NodeMemoryTools
            self.node_tools = NodeMemoryTools(session, self)

    def observe(self, mode="current", project=False):
        s = self.session
        if mode not in {"current", "panorama"} or type(project) is not bool:
            raise ValueError("Expected current/panorama and boolean project")
        if project and not s.spatial.enabled:
            raise ValueError("Observation projection requires super tools")
        if s._observation_end is not None and s._observations >= s._observation_end:
            raise ValueError("Observation limit reached; use existing evidence")
        s._observations += 1
        if mode == "current":
            s.current = s._capture(s.environment.observe())
            views = [s.current]
            value = {key: s.current[key] for key in ("frame", "step", "pose")}
            value["size"] = s.observations.image(s.current["frame"])["size"]
            refs, labels = [s.current["path"]], [f"[CURRENT {s.current['frame']}]"]
        else:
            pose = s.environment.pose()
            group = "p_" + uuid.uuid4().hex[:8]
            views = [s._capture(obs, body_pose=pose, group=group, yaw=i * 60)
                     for i, obs in enumerate(s.environment.panorama())]
            s.current = views[0]
            image = s.observations.grid([v["frame"] for v in views])
            value = {"image": image["image"], "size": image["size"], "group": group,
                     "views": [{"frame": v["frame"], "yaw": v["yaw"],
                                "size": s.observations.image(v["frame"])["size"]} for v in views],
                     "step": s.environment.steps, "pose": s.current["pose"]}
            refs, labels = [image["path"]], [f"[CURRENT {image['image']} | six views already attached]"]
        s._publish_state()
        if project:
            mapped = self.map_value(view={"ego": 6}, overlay=[v["frame"] for v in views])
            value["projection"] = mapped
            refs.append(mapped["map"])
            labels.append(f"[MAP {mapped['view_id']}]")
        return value, refs, labels

    def steps(self, actions):
        s = self.session
        if (not isinstance(actions, list) or not 1 <= len(actions) <= 6
                or any(not isinstance(a, str) or a not in ACTIONS for a in actions)):
            raise ValueError("actions must contain 1–6 valid atomic actions")
        done, events, stop = [], [], "completed"
        for i, action in enumerate(actions, 1):
            motion = s._perform_motion(action)
            if motion is None:
                stop = "interrupted"
                break
            if done and done[-1][0] == action:
                done[-1][1] += 1
            else:
                done.append([action, 1])
            if motion.collision or motion.blocked:
                events.append({"i": i, "collision": bool(motion.collision), "blocked": bool(motion.blocked),
                               "auto_yaw": round(-motion.auto_yaw_delta_degrees, 2)})
            if motion.blocked:
                stop = "blocked"
                break
        s._publish_state()
        value = {"done": done, "stop": stop, "frame": s.current["frame"],
                 "step": s.environment.steps, "pose": s.current["pose"]}
        if events:
            value["events"] = events
        return value

    def move(self, mode, actions=None, target=None, after="current", stop_distance=None, max_actions=None):
        s = self.session
        if after not in {"current", "panorama"}:
            raise ValueError("after must be current or panorama")
        if mode == "steps":
            if target is not None or stop_distance is not None or max_actions is not None:
                raise ValueError("steps accepts only actions and after")
            value = self.steps(actions)
        elif mode in {"face", "approach"}:
            if not s.spatial.enabled:
                raise ValueError("face and approach require super tools")
            if actions is not None or mode == "face" and stop_distance is not None:
                raise ValueError("Target motion does not accept actions; face has no stop_distance")
            value = s.spatial.move(target, face=mode == "face", stop_distance=stop_distance,
                                   max_actions=40 if max_actions is None else max_actions)
        else:
            raise ValueError("mode must be steps, face or approach")
        value["size"] = s.observations.image(s.current["frame"])["size"]
        refs = [s.current["path"]] if value["done"] else []
        labels = [f"[CURRENT {s.current['frame']}]"] if refs else []
        if after == "panorama":
            if s._stop_reason or s._movement_exhausted:
                value["after"] = {"executed": False, "reason": "execution_window_ended"}
            else:
                try:
                    observation, refs, labels = self.observe("panorama")
                    value["after"] = {"executed": True, **observation}
                    value["frame"] = s.current["frame"]
                except ValueError as error:
                    value["after"] = {"executed": False, "reason": str(error)}
                except Exception as error:
                    # Motion has already happened: retain the receipt and never ask to replay it.
                    value["after"] = {"executed": False, "reason": f"Observation failed: {type(error).__name__}"}
                    s._broken, s._stop_reason = True, "environment_error"
        return s._reply(value, refs, labels, end=bool(s._stop_reason))

    def images(self, images, layout="separate"):
        s = self.session
        if not isinstance(images, list) or not 1 <= len(images) <= 6 or layout not in {"separate", "grid"}:
            raise ValueError("Expected 1–6 images and separate/grid layout")
        # Resolve every reference before creating derived images.
        for item in images:
            if not isinstance(item, dict) or set(item) - {"image", "crop"}:
                raise ValueError("Expected image and optional crop")
            s.observations.image(item.get("image"))
        rows = [s.observations.crop(item["image"], item["crop"]) if item.get("crop") is not None
                else s.observations.image(item["image"]) for item in images]
        if layout == "grid":
            rows = [s.observations.grid([row["image"] for row in rows])]
        metadata = [{k: row[k] for k in ("image", "frame", "step", "pose", "group", "yaw", "size") if k in row} for row in rows]
        kinds = {"target": "TARGET", "question": "QUESTION", "map": "MAP", "derived": "VIEW", "observation": "HISTORY", "panorama": "HISTORY"}
        labels = [f"[{kinds[row['kind']]} {row['image']}]" for row in rows]
        return s._reply({"images": metadata}, [row["path"] for row in rows], labels)

    def map_value(self, overlay=None, project=None, **args):
        s = self.session
        if overlay is not None and not s.spatial.enabled:
            raise ValueError("RGB overlay requires super tools")
        points = s.spatial.geometry.overlay(overlay) if overlay else None
        node_mode = s.memory_type == "node"
        path = (getattr(s.environment, "motion_trajectory", s.environment.trajectory)
                if node_mode else s.environment.trajectory)
        trajectory = [world_to_uv(p, s.transform) for p in path]
        breaks = getattr(s.environment, "motion_trajectory_breaks" if node_mode else "trajectory_breaks", ())
        projected, markers = [], []
        if project is not None:
            if (not isinstance(project, dict) or set(project) != {"source", "points"}
                    or not isinstance(project["points"], list) or not 1 <= len(project["points"]) <= 10):
                raise ValueError("project requires source and 1–10 pixel pairs")
            for i, pixel in enumerate(project["points"], 1):
                label = f"P{i}"
                try:
                    resolved = s.spatial.target({"image": project["source"], "pixel": pixel})
                    marker = {"label": label, "uv": resolved["uv"]}
                    projected.append(marker)
                    markers.append(marker)
                except ValueError as error:
                    projected.append({"label": label, "error": str(error)})
        value = s.renderer.render(s.current["pose"], trajectory, overlay_points=points, located_points=markers,
                                  trajectory_breaks=breaks, **args)
        if node_mode:
            value.pop("visible_ids", None)
            records = value.get("records", [])
            value["records"] = records[:10]
            if len(records) > 10:
                value["omitted_records"] = len(records) - 10
            return {k: v for k, v in value.items() if k not in {"pixel_to_uv", "plot_bounds"}}
        # The renderer owns the exact displayed extent, including rotated views.
        ids = args.get("highlight") or value.pop("visible_ids")
        value.pop("visible_ids", None)
        indexed = {e["id"]: e for e in s.memory.entities}
        related = [indexed[i] for i in dict.fromkeys(ids) if i in indexed]
        value["records"] = [{k: e[k] for k in ("id", "type", "name", "room", "connects", "passage") if k in e}
                            for e in related[:12]]
        if len(related) > 12:
            value["omitted_records"] = len(related) - 12
        if project is not None:
            value["source"], value["points"] = project["source"], projected
        # Transforms remain in the image index; model-selected pixels use view_id directly.
        return {k: v for k, v in value.items() if k not in {"pixel_to_uv", "plot_bounds"}}

    def read_memory(self, **args):
        s = self.session
        result = s.spatial.near(**args) if "near" in args else s.memory.read(**args)
        if "error" in result:
            return s._reply(result, error=True)
        detailed = args.get("ids") is not None or result.get("total", 0) <= 5
        if detailed and args.get("ids") is None:
            details = s.memory.read(ids=[e["id"] for e in result["entities"]])["entities"]
            for old, new in zip(result["entities"], details):
                new.update({k: old[k] for k in ("bearing", "distance") if k in old})
            result["entities"] = details
        result["detail"] = detailed
        frames = []
        if detailed:
            # Interleave records so two different entities can each contribute evidence.
            lists = [list(reversed(e.get("evidence", []))) for e in result["entities"]]
            for i in range(max(map(len, lists), default=0)):
                for evidence in lists:
                    if i < len(evidence) and evidence[i]["frame"] not in frames:
                        frames.append(evidence[i]["frame"])
        selected = frames[:2]
        result["attached_evidence"] = selected
        images = [s.observations.image(fid) for fid in selected]
        return s._reply(result, [row["path"] for row in images],
                        [f"[HISTORY {row['image']} | {row['size'][0]}x{row['size'][1]} | recorded evidence, not verified]"
                         for row in images])

    def edit_memory(self, operations):
        s = self.session
        requested = operations
        before_entities = s.memory.entities
        before = s.trace.store(before_entities)
        try:
            operations, sources = s.spatial.prepare_edits(operations)
        except (ValueError, TypeError, KeyError) as error:
            return s._reply({"error": f"{error}. No memory changes applied."}, error=True)
        result = s.memory.edit(operations)
        if "error" in result:
            result["error"] += ". No memory changes applied."
            return s._reply(result, error=True)
        s.spatial.remember_sources(operations, sources)
        s._revision += 1
        s.trace.emit("memory_changed", {"before": before, "after": s.trace.store(s.memory.entities)})
        identifiers = {change["id"] for change in result["changes"]}
        result["entities"] = [{k: e[k] for k in ("id", "point", "anchor", "room", "connects", "passage", "evidence") if k in e}
                              for e in s.memory.entities if e["id"] in identifiers]
        try:
            preview = memory_feedback(s, requested, before_entities)
        except Exception as error:
            # The edit is committed. A failed preview must not invite a duplicate write.
            result["preview_error"] = f"Memory saved; visual feedback unavailable ({type(error).__name__}). Read the existing record; do not repeat create."
            s.trace.emit("memory_preview_error", {"error": str(error)})
            return s._reply(result)
        if preview is None:
            return s._reply(result)
        result["preview"] = {k: v for k, v in preview.items() if k != "path"}
        return s._reply(result, [preview["path"]],
                        [f"[MEMORY EDIT {preview['image']} | selected pixels and saved locations; not verified]"])

    def execute(self, name, args):
        s = self.session
        if s._qa and name not in {"view_images", "memory_read", "map_view", "submit"}:
            raise ValueError("This question permits only existing-evidence queries and submission")
        if s._movement_exhausted and name in {"move", "observe"}:
            raise ValueError("Movement window ended; finish using existing evidence")
        if name == "observe":
            return s._reply(*self.observe(**args))
        if name == "move":
            return self.move(**args)
        if name == "view_images":
            return self.images(**args)
        if name == "memory_read":
            if self.node_tools:
                return self.node_tools.read(**args)
            return self.read_memory(**args)
        if name == "memory_edit":
            if self.node_tools:
                return self.node_tools.edit(**args)
            return self.edit_memory(**args)
        if name == "map_view":
            value = self.map_value(**args)
            return s._reply(value, [value["map"]], [f"[MAP {value['view_id']} | model-written records]"])
        if name == "submit":
            s._submitted, s._answer = True, args.get("answer", "")
            return s._reply({"submitted": True}, end=True)
        raise ValueError(f"Unknown tool: {name}")
