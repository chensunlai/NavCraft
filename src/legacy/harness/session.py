"""One episode, one environment; tools only receive public operation callbacks."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.common.register import tool_registry
from src.dataset.episodes.base import EpisodeSpec
from src.environment.base import Environment, Observation, Transition
from src.legacy.harness.agent.base import ToolCall, ToolResult
from src.common.map import MapAsset
from src.legacy.harness.prompts import SYSTEM_PROMPT, observation_message
from src.legacy.harness.tools import ToolSet, load_builtin_tools
from src.legacy.harness.tools.navigation import ToolContext, ToolInputError
from src.legacy.harness.trace import TraceWriter


class NavigationSession:
    def __init__(
        self,
        environment: Environment,
        episode: EpisodeSpec,
        trace: TraceWriter,
        tool_names: list[str],
        map_asset: MapAsset | None = None,
        action_parameters: dict | None = None,
        assets=None,
    ):
        load_builtin_tools()
        if len(tool_names) != len(set(tool_names)) or not tool_names:
            raise ValueError("Harness tools must be a nonempty unique list")
        self.environment, self.episode, self.trace = environment, episode, trace
        self.tools = {name: tool_registry.create(name) for name in tool_names}
        self.toolset = ToolSet(self)
        self.map_asset = map_asset
        self.assets = assets
        if self.assets is not None and "nav_depth" in self.tools:
            raise ValueError("The visual map harness cannot expose nav_depth")
        if "nav_observe" in self.tools:
            self.tools["nav_observe"].visual = self.assets is not None
        self.action_parameters = dict(action_parameters or {})
        self.transition: Transition | None = None
        self.frames: dict[str, Observation] = {}
        self._calls: dict[str, tuple[str, ToolResult]] = {}
        self._call_id = ""

    def start(self) -> list[dict]:
        if self.transition is not None:
            raise RuntimeError("Session already started")
        self._accept(self.environment.reset(self.episode))
        self.trace.emit("episode_started", public=self._public())
        data = self._public()
        data["movement_actions"] = list(self.movement_actions)
        data["action_parameters"] = self.action_parameters
        images = self._images(include_goal=True)
        roles = self._image_roles(images)
        prompt = SYSTEM_PROMPT
        if self.assets is not None:
            from .prompts import visual_system_prompt

            map_data, map_images = self.assets.initial_map(
                self.transition.observation.pose
            )
            data["memory"] = {
                "description_ref": self.assets.description_ref,
                "floor_plan_ref": self.assets.map_ref,
                "room_images_ref": self.assets.rooms_ref,
                "initial_map": map_data,
                "coordinates": "Map tools use normalized [x,y] in the referenced image; origin top-left.",
            }
            images.extend(map_images)
            roles.update({str(path): "map" for path in map_images})
            prompt = visual_system_prompt()
        return [
            {"role": "system", "content": prompt},
            observation_message(data, images, roles if self.assets else None),
        ]

    def _public(self) -> dict:
        if self.assets is None:
            return self.transition.public_dict()
        transition = self.transition
        observation = transition.observation
        task = transition.task
        public_task = None
        if task is not None:
            public_task = {
                "goal_id": task.goal_id,
                "modality": task.modality,
                "instruction": task.instruction,
            }
            if task.image is not None:
                public_task["image_ref"] = self.assets.register_image(
                    task.image, ref=f"goal_{task.goal_id:03d}", kind="goal"
                )
        allowed = {
            "event",
            "action",
            "collision",
            "steps",
            "remaining_total_actions",
            "remaining_goal_actions",
        }
        return {
            "observation": {
                "frame_id": observation.frame_id,
                "image": self.assets.image_info(observation.frame_id),
            },
            "task": public_task,
            "feedback": {
                key: value
                for key, value in transition.feedback.items()
                if key in allowed
            },
            "goal_changed": transition.goal_changed,
            "terminated": transition.terminated,
            "truncated": transition.truncated,
            "reason": transition.reason,
        }

    @property
    def movement_actions(self) -> tuple[str, ...]:
        return ("move_forward", "turn_left", "turn_right") + (
            () if self.episode.benchmark == "r2r" else ("look_up", "look_down")
        )

    def _accept(self, transition: Transition) -> None:
        self.transition = transition
        self.frames[transition.observation.frame_id] = transition.observation
        if self.assets is not None:
            self.assets.register_image(
                transition.observation.rgb, ref=transition.observation.frame_id
            )
        self.trace.goal_id = (
            transition.task.goal_id if transition.task is not None else None
        )

    def _images(self, include_goal: bool = False) -> list[Path]:
        images = [Path(self.transition.observation.rgb)]
        task = self.transition.task
        if include_goal and task is not None and task.image is not None:
            images.append(Path(task.image))
        return images

    def _image_roles(self, images: list[Path]) -> dict[str, str]:
        roles = {str(path): "observation" for path in images}
        task = self.transition.task
        if task is not None and task.image is not None:
            roles[str(task.image)] = "goal"
        return roles

    def _result(self, data: dict | None = None, images: bool = True) -> ToolResult:
        transition = self.transition
        image_paths = (
            self._images(include_goal=transition.goal_changed) if images else []
        )
        return ToolResult(
            self._call_id,
            True,
            self._public() if data is None else data,
            image_paths,
            transition.terminated,
            transition.truncated,
            goal_changed=transition.goal_changed,
            image_roles=self._image_roles(image_paths),
        )

    def execute(self, call: ToolCall) -> ToolResult:
        if self.transition is None:
            raise RuntimeError("Session has not started")
        self._call_id = call.call_id
        try:
            arguments = (
                json.loads(call.arguments)
                if isinstance(call.arguments, str)
                else call.arguments
            )
        except json.JSONDecodeError:
            arguments = call.arguments
        signature = json.dumps(
            {"name": call.name, "arguments": arguments},
            sort_keys=True,
            ensure_ascii=False,
        )
        if call.call_id in self._calls:
            previous, result = self._calls[call.call_id]
            if previous != signature:
                result = ToolResult(
                    call.call_id,
                    False,
                    {"error": "call_id reused with different arguments"},
                )
            self.trace.emit("tool_replayed", call_id=call.call_id, result=result)
            return result
        self.trace.emit(
            "tool_call", call_id=call.call_id, name=call.name, arguments=arguments
        )
        try:
            if not call.call_id:
                raise ToolInputError("call_id must not be empty")
            if self.transition.terminated or self.transition.truncated:
                raise ToolInputError("Episode is already finished")
            if call.name not in self.tools:
                raise ToolInputError(f"Unavailable tool: {call.name}")
            context = ToolContext(
                self._observe,
                self._depth,
                self._step,
                self._finish,
                self.movement_actions,
                self._read_text if self.assets else None,
                self._view_images if self.assets else None,
                self._mark_pose if self.assets else None,
            )
            result = self.tools[call.name].execute(arguments, context)
            if self.assets is not None and result.goal_changed:
                self.transition = replace(self.transition, goal_changed=False)
        except ToolInputError as error:
            result = ToolResult(
                call.call_id,
                False,
                {"error": str(error)},
                [],
                self.transition.terminated,
                self.transition.truncated,
            )
        except Exception:
            # The action may already have happened before capture/transport failed.
            result = ToolResult(
                call.call_id,
                False,
                {"error": "Execution failed; this call will not be executed again"},
                [],
                False,
                True,
            )
            self._calls[call.call_id] = (signature, result)
            self.trace.emit(
                "tool_execution_failed", call_id=call.call_id, result=result
            )
            raise
        self._calls[call.call_id] = (signature, result)
        self.trace.emit("tool_result", call_id=call.call_id, result=result)
        return result

    def _observe(self, mode: str = "single") -> ToolResult:
        if mode == "panorama":
            if self.assets is None:
                raise ToolInputError("Panorama requires the visual harness")
            views = self.environment.panorama()
            refs = []
            for offset, observation in zip(range(0, 360, 60), views):
                self.frames[observation.frame_id] = observation
                ref = self.assets.register_image(
                    observation.rgb, ref=f"{observation.frame_id}_right_{offset:03d}"
                )
                refs.append(ref)
            details, paths = self.assets.view_images(refs, layout="grid")
            data = self._public()
            data.update(
                {
                    "mode": "panorama",
                    "views": [
                        {"image_ref": ref, "right_degrees": angle}
                        for ref, angle in zip(refs, range(0, 360, 60))
                    ],
                    "images": details,
                }
            )
            self.trace.emit("panorama_captured", views=data["views"])
            result = self._result(data, images=False)
            result.images = paths
            result.image_roles = {str(path): "observation" for path in paths}
            return result
        observation = self.environment.observe()
        self.frames[observation.frame_id] = observation
        self.transition = replace(
            self.transition,
            observation=observation,
            goal_changed=False,
            feedback={"event": "observe"},
        )
        return self._result()

    def _depth(self, frame_id: str, points: list[list[int]]) -> ToolResult:
        if frame_id not in self.frames:
            raise ToolInputError(
                "Unknown frame; only already observed frames can be queried"
            )
        frame = self.frames[frame_id]
        if frame.depth is None:
            raise ToolInputError("Depth is not enabled")
        depth = np.load(frame.depth, allow_pickle=False)
        height, width = depth.shape
        if any(not (0 <= u < width and 0 <= v < height) for u, v in points):
            raise ToolInputError(
                f"Pixels must be within width={width}, height={height}"
            )
        values = []
        for u, v in points:
            value = float(depth[v, u])
            valid = bool(np.isfinite(value) and value > 0)
            values.append(
                {"pixel": [u, v], "depth_m": value if valid else None, "valid": valid}
            )
        return self._result(
            {"frame_id": frame_id, "unit": "meters", "points": values}, images=False
        )

    def _step(self, actions: list[str]) -> ToolResult:
        executed = []
        for action in actions:
            transition = self.environment.step(action)
            self._accept(transition)
            feedback = (
                self._public()["feedback"]
                if self.assets is not None
                else transition.feedback
            )
            executed.append(
                {
                    "action": action,
                    "frame_id": transition.observation.frame_id,
                    **feedback,
                }
            )
            self.trace.emit(
                "action_executed",
                call_id=self._call_id,
                action=action,
                public=self._public(),
            )
            if (
                transition.terminated
                or transition.truncated
                or transition.goal_changed
                or (self.assets is not None and transition.feedback.get("collision"))
            ):
                break
        data = self._public()
        data["executed_actions"] = executed
        return self._result(data)

    def _finish(self) -> ToolResult:
        self._accept(self.environment.finish_goal())
        self.trace.emit("goal_finished", call_id=self._call_id, public=self._public())
        return self._result()

    def _memory_result(self, data: dict, paths: list[Path]) -> ToolResult:
        result = self._result(data, images=False)
        result.images = paths
        result.image_roles = {str(path): "observation" for path in paths}
        if self.assets.latest_map_ref is not None:
            latest = self.assets.path_for_image(self.assets.latest_map_ref)
            if latest in paths:
                result.image_roles[str(latest)] = "map"
        return result

    def _read_text(self, ref: str, start_line: int, max_lines: int) -> ToolResult:
        try:
            return self._memory_result(
                self.assets.read_text(ref, start_line, max_lines), []
            )
        except ValueError as error:
            raise ToolInputError(str(error)) from error

    def _view_images(self, refs: list[str], layout: str, crops) -> ToolResult:
        try:
            return self._memory_result(*self.assets.view_images(refs, layout, crops))
        except ValueError as error:
            raise ToolInputError(str(error)) from error

    def _mark_pose(self, map_ref: str, position, heading, note: str) -> ToolResult:
        try:
            return self._memory_result(
                *self.assets.mark_pose(map_ref, position, heading, note)
            )
        except ValueError as error:
            raise ToolInputError(str(error)) from error

    def close(self) -> None:
        self.environment.close()
