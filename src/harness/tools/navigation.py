"""Navigation operations and bounded, related material in each tool receipt."""

from ..errors import ToolInputError

import uuid

from src.common.actions import ACTIONS
from src.common.coordinates import world_to_uv
from ..spatial.feedback import memory_feedback


class NavigationTools:
    def __init__(self, executor):
        self.executor = executor
    def observe(self, mode="current", project=False):
        s = self.executor
        if mode not in {"current", "panorama"} or type(project) is not bool:
            raise ToolInputError("Expected current/panorama and boolean project")
        if project and not s.spatial.enabled:
            raise ToolInputError("Observation projection requires super tools")
        if s.request.observation_end is not None and s.observation_count >= s.request.observation_end:
            raise ToolInputError("Observation limit reached; use existing evidence")
        s.observation_count += 1
        if mode == "current":
            s.current = s.capture(s.environment.observe())
            views = [s.current]
            value = {key: s.current[key] for key in ("frame", "step", "pose")}
            value["size"] = s.observations.image(s.current["frame"])["size"]
            refs, labels = [s.current["path"]], [f"[CURRENT {s.current['frame']}]"]
        else:
            pose = s.environment.pose()
            group = "p_" + uuid.uuid4().hex[:8]
            views = [s.capture(obs, body_pose=pose, group=group, yaw=i * 60)
                     for i, obs in enumerate(s.environment.panorama())]
            s.current = views[0]
            image = s.observations.grid([v["frame"] for v in views])
            value = {"image": image["image"], "size": image["size"], "group": group,
                     "views": [{"frame": v["frame"], "yaw": v["yaw"],
                                "size": s.observations.image(v["frame"])["size"]} for v in views],
                     "step": s.environment.steps, "pose": s.current["pose"]}
            refs, labels = [image["path"]], [f"[CURRENT {image['image']} | {image['size'][0]}x{image['size'][1]} | select pixels on this whole image; f_ IDs reopen detail]"]
        s.publish_state()
        if project:
            mapped = self.map_value(view={"ego": 6}, overlay=[v["frame"] for v in views])
            value["projection"] = mapped
            refs.append(mapped["map"])
            labels.append(f"[MAP {mapped['view_id']}]")
        return value, refs, labels

    def steps(self, actions):
        s = self.executor
        if (not isinstance(actions, list) or not 1 <= len(actions) <= 6
                or any(not isinstance(a, str) or a not in ACTIONS for a in actions)):
            raise ToolInputError("actions must contain 1–6 valid atomic actions")
        done, events, stop = [], [], "completed"
        for i, action in enumerate(actions, 1):
            motion = s.perform_motion(action)
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
        s.publish_state()
        value = {"done": done, "stop": stop, "frame": s.current["frame"],
                 "step": s.environment.steps, "pose": s.current["pose"]}
        if events:
            value["events"] = events
        return value

    def move(self, mode, actions=None, target=None, after="current", stop_distance=None, max_actions=None):
        s = self.executor
        if after not in {"current", "panorama"}:
            raise ToolInputError("after must be current or panorama")
        if mode == "steps":
            if target is not None or stop_distance is not None or max_actions is not None:
                raise ToolInputError("steps accepts only actions and after")
            value = self.steps(actions)
        elif mode in {"face", "approach"}:
            if not s.spatial.enabled:
                raise ToolInputError("face and approach require super tools")
            if actions is not None or mode == "face" and stop_distance is not None:
                raise ToolInputError("Target motion does not accept actions; face has no stop_distance")
            value = s.spatial.move(target, face=mode == "face", stop_distance=stop_distance,
                                   max_actions=40 if max_actions is None else max_actions)
        else:
            raise ToolInputError("mode must be steps, face or approach")
        value["size"] = s.observations.image(s.current["frame"])["size"]
        refs = [s.current["path"]] if value["done"] else []
        labels = [f"[CURRENT {s.current['frame']}]"] if refs else []
        if after == "panorama":
            if s.request.stop_reason:
                value["after"] = {"executed": False, "reason": "execution_window_ended"}
            else:
                try:
                    observation, refs, labels = self.observe("panorama")
                    value["after"] = {"executed": True, **observation}
                    value["frame"] = s.current["frame"]
                except ToolInputError as error:
                    value["after"] = {"executed": False, "reason": str(error)}
                except Exception as error:
                    # Motion has already happened: retain the receipt and never ask to replay it.
                    value["after"] = {"executed": False, "reason": f"Observation failed: {type(error).__name__}"}
                    s.broken, s.request.stop_reason = True, "environment_error"
        return s.reply(value, refs, labels, end=bool(s.request.stop_reason))

    def images(self, images, layout="separate"):
        s = self.executor
        # Resolve every reference before creating derived images.
        for item in images:
            s.observations.image(item["image"])
        rows = [s.observations.crop(item["image"], item["crop"]) if item.get("crop") is not None
                else s.observations.image(item["image"]) for item in images]
        if layout == "grid":
            rows = [s.observations.grid([row["image"] for row in rows])]
        return self.image_reply(rows)

    def image_reply(self, rows, nodes=None):
        """Keep the displayed image ID, capture pose and optional node anchor together."""
        nodes = nodes if nodes is not None else [None] * len(rows)
        metadata = [{k: row[k] for k in ("image", "frame", "step", "pose", "group", "yaw", "size") if k in row} for row in rows]
        kinds = {"target": "TARGET", "question": "QUESTION", "map": "MAP", "derived": "VIEW", "observation": "HISTORY", "panorama": "HISTORY"}
        labels = []
        for row, detail, node in zip(rows, metadata, nodes):
            if node is not None:
                detail.update(node=node["id"], anchor_pose=node["pose"])
            identity = f"{node['id']} | " if node else ""
            labels.append(f"[{kinds[row['kind']]} {identity}image={row['image']} | {row['size'][0]}x{row['size'][1]}]")
        return self.executor.reply({"images": metadata}, [row["path"] for row in rows], labels)

    def task_read(self):
        s = self.executor
        value = {"request": s.request.text}
        groups = [("target", s.request.target_frames)]
        images, refs, labels = [], [], []
        for kind, frames in groups:
            for frame in dict.fromkeys(frames):
                row = s.observations.image(frame)
                images.append({"kind": kind, "frame": frame, "size": row["size"]})
                refs.append(row["path"])
                labels.append(f"[{kind.upper()} {frame} | {row['size'][0]}x{row['size'][1]}]")
        if images:
            value["images"] = images
        return s.reply(value, refs, labels)

    def map_value(self, overlay=None, project=None, **args):
        s = self.executor
        if overlay is not None and not s.spatial.enabled:
            raise ToolInputError("RGB overlay requires super tools")
        points = s.spatial.geometry.overlay(overlay) if overlay else None
        node_mode = s.memory_type == "node"
        path = s.environment.motion_trajectory if node_mode else s.environment.trajectory
        trajectory = [world_to_uv(p, s.transform) for p in path]
        breaks = s.environment.motion_trajectory_breaks if node_mode else s.environment.trajectory_breaks
        projected, markers = [], []
        if project is not None:
            if (not isinstance(project, dict) or set(project) != {"source", "points"}
                    or not isinstance(project["points"], list) or not 1 <= len(project["points"]) <= 10):
                raise ToolInputError("project requires source and 1–10 pixel pairs")
            for i, pixel in enumerate(project["points"], 1):
                label = f"P{i}"
                try:
                    resolved = s.spatial.target({"image": project["source"], "pixel": pixel})
                    marker = {"label": label, "uv": resolved["uv"]}
                    projected.append(marker)
                    markers.append(marker)
                except ToolInputError as error:
                    projected.append({"label": label, "error": str(error)})
        value = s.renderer.render(s.current["pose"], trajectory, overlay_points=points, located_points=markers,
                                  trajectory_breaks=breaks, **args)
        if node_mode:
            value.pop("visible_ids", None)
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
        s = self.executor
        result = s.spatial.near(**args) if "near" in args else s.memory.read(**args)
        if "error" in result:
            return s.reply(result, error=True)
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
        return s.reply(result, [row["path"] for row in images],
                        [f"[HISTORY {row['image']} | {row['size'][0]}x{row['size'][1]} | recorded evidence, not verified]"
                         for row in images])

    def edit_memory(self, operations):
        s = self.executor
        requested = operations
        before_entities = s.memory.entities
        before = s.trace.store(before_entities)
        try:
            operations, sources = s.spatial.prepare_edits(operations)
        except ToolInputError as error:
            return s.reply({"error": f"{error}. No memory changes applied."}, error=True)
        result = s.memory.edit(operations)
        if "error" in result:
            result["error"] += ". No memory changes applied."
            return s.reply(result, error=True)
        s.spatial.remember_sources(operations, sources)
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
            return s.reply(result)
        if preview is None:
            return s.reply(result)
        result["preview"] = {k: v for k, v in preview.items() if k != "path"}
        return s.reply(result, [preview["path"]],
                        [f"[MEMORY EDIT {preview['image']} | selected pixels and saved locations; not verified]"])

    def observe_reply(self, **args):
        return self.executor.reply(*self.observe(**args))

    def map_reply(self, **args):
        value = self.map_value(**args)
        return self.executor.reply(value, [value["map"]], [f"[MAP {value['view_id']} | model-written records]"])

    def submit(self, answer=""):
        self.executor.request.submitted = True
        self.executor.request.answer = answer
        return self.executor.reply({"submitted": True}, end=True)
