"""Native Habitat boundary. Imported only when a real environment is reset."""

from dataclasses import dataclass, fields
from copy import deepcopy
from pathlib import Path
import math

import habitat_sim
import magnum as mn
import numpy as np
from PIL import Image
from habitat_sim.utils.common import quat_from_coeffs, quat_to_coeffs
import quaternion

from src.environment.base import MotionResult, Observation, TRANSLATION_ACTIONS


@dataclass(frozen=True)
class SensorConfig:
    width: int = 512
    height: int = 512
    hfov: float = 90
    sensor_height: float = 1.5
    agent_height: float = 1.5
    agent_radius: float = 0.1
    navmesh_cell_height: float = 0.2
    forward_step: float = 0.25
    turn_angle: float = 15
    depth: bool = True
    seed: int = 0

    @classmethod
    def from_dict(cls, config: dict):
        value = cls(
            **{
                field.name: config[field.name]
                for field in fields(cls)
                if field.name in config
            }
        )
        if value.width < 1 or value.height < 1 or not 0 < value.hfov < 180:
            raise ValueError("Invalid camera dimensions or hfov")
        if (
            min(
                value.agent_height,
                value.agent_radius,
                value.forward_step,
                value.turn_angle,
                value.navmesh_cell_height,
            )
            <= 0
        ):
            raise ValueError("Agent dimensions and motion increments must be positive")
        if not math.isfinite(value.navmesh_cell_height):
            raise ValueError("NavMesh voxel height must be finite")
        return value


def camera_parameters(config: SensorConfig) -> dict:
    focal = config.width / (2 * math.tan(math.radians(config.hfov) / 2))
    return {
        "width": config.width,
        "height": config.height,
        "hfov_degrees": config.hfov,
        "K": [
            [focal, 0, (config.width - 1) / 2],
            [0, focal, (config.height - 1) / 2],
            [0, 0, 1],
        ],
        "depth_units": "metres",
        "depth_type": "optical_axis_z",
        "optical_frame": "X right, Y down, Z forward",
        "sensor_frame": "X right, Y up, forward -Z",
    }


class HabitatSimulator:
    def __init__(self, episode, config: dict):
        self.config = SensorConfig.from_dict(config)
        scene_id = episode.scene_id.removeprefix("data/scene_datasets/")
        scene = Path(config["scenes"]) / scene_id
        if not scene.is_file():
            raise FileNotFoundError(f"Scene does not exist: {scene}")
        native = habitat_sim.SimulatorConfiguration()
        native.scene_id = str(scene.resolve())
        native.gpu_device_id = int(config.get("gpu", 2))
        native.allow_sliding = True
        agent = habitat_sim.agent.AgentConfiguration()
        agent.height, agent.radius = self.config.agent_height, self.config.agent_radius
        sensors = []
        for name, kind in [
            ("rgb", habitat_sim.SensorType.COLOR),
            ("depth", habitat_sim.SensorType.DEPTH),
        ]:
            if name == "depth" and not self.config.depth:
                continue
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid, spec.sensor_type = name, kind
            spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            spec.resolution = [self.config.height, self.config.width]
            spec.position = mn.Vector3(0, self.config.sensor_height, 0)
            spec.hfov = self.config.hfov
            sensors.append(spec)
        agent.sensor_specifications = sensors
        agent.action_space = {
            action: habitat_sim.agent.ActionSpec(
                action,
                habitat_sim.agent.ActuationSpec(
                    amount=(
                        self.config.forward_step
                        if action in TRANSLATION_ACTIONS
                        else self.config.turn_angle
                    )
                ),
            )
            for action in TRANSLATION_ACTIONS + (
                "turn_left",
                "turn_right",
                "look_up",
                "look_down",
            )
        }
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(native, [agent]))
        try:
            self.sim.seed(self.config.seed)
            settings = habitat_sim.NavMeshSettings()
            settings.set_defaults()
            settings.agent_height, settings.agent_radius = (
                self.config.agent_height,
                self.config.agent_radius,
            )
            settings.agent_max_climb = 0.2
            settings.cell_height = self.config.navmesh_cell_height
            if (
                not self.sim.pathfinder.is_loaded
                or self.sim.pathfinder.nav_mesh_settings != settings
            ):
                if not self.sim.recompute_navmesh(self.sim.pathfinder, settings):
                    raise RuntimeError(f"Cannot build navigation mesh: {scene}")
            self.agent = self.sim.get_agent(0)
            state = habitat_sim.AgentState()
            state.position = np.asarray(episode.start_position, dtype=np.float32)
            state.rotation = quat_from_coeffs(episode.start_rotation)
            self.agent.set_state(state)
        except BaseException as error:
            try:
                self.sim.close()
            except Exception as close_error:
                raise error from close_error
            raise

    def position(self) -> list[float]:
        return [float(value) for value in self.agent.get_state().position]

    def distance(self, targets: list) -> float:
        if not targets:
            return math.inf
        query = habitat_sim.MultiGoalShortestPath()
        query.requested_start = np.asarray(self.position(), dtype=np.float32)
        query.requested_ends = np.asarray(targets, dtype=np.float32)
        self.sim.pathfinder.find_path(query)
        return float(query.geodesic_distance)

    def _motion_sample(self, *, collision=False, auto_yaw=0.0) -> dict:
        state = self.agent.get_state()
        return {"position": [float(v) for v in state.position],
                "rotation_xyzw": [float(v) for v in quat_to_coeffs(state.rotation)],
                "collision": bool(collision), "auto_yaw_delta_degrees": float(auto_yaw)}

    def _slide_projection(self, start, direction, amount, pending):
        """Subdivide a corner-cutting slide; never choose another direction."""
        pathfinder = self.sim.pathfinder
        while True:
            end = np.asarray(self.agent.controls.move_filter_fn(start, start + direction * amount))
            # The reverse check avoids false rejection of floating-point edge tangents.
            forward = np.asarray(pathfinder.try_step_no_sliding(start, end))
            reverse = np.asarray(pathfinder.try_step_no_sliding(end, start))
            error = min(np.linalg.norm((forward - end)[[0, 2]]),
                        np.linalg.norm((reverse - start)[[0, 2]]))
            if error <= .0001:
                return amount, end
            # Edge tangents can fail both casts after float rounding. Test a
            # parallel line 20 micrometres to either side; do not move the agent.
            delta = end - start
            side = np.asarray([-delta[2], 0, delta[0]])
            length = float(np.linalg.norm(side))
            if length > 1e-8:
                side *= .00002 / length
                for offset in (side, -side):
                    cast = np.asarray(pathfinder.try_step_no_sliding(start + offset, end + offset))
                    if np.linalg.norm((cast - end - offset)[[0, 2]]) <= .00001:
                        return amount, end
            if amount <= .00005:
                return amount, None
            amount /= 2
            pending.append(amount)

    def _adaptive_translation(self, action: str) -> MotionResult:
        count = math.ceil(self.config.forward_step / .05)
        micro_step = self.config.forward_step / count
        local = {"move_forward": [0, 0, -1], "move_backward": [0, 0, 1],
                 "move_left": [-1, 0, 0], "move_right": [1, 0, 0]}[action]
        samples = [self._motion_sample()]
        collision, blocked, auto_yaw = False, False, 0.0
        pending = [micro_step] * count
        while pending:
            state = self.agent.get_state()
            start = np.asarray(state.position).copy()
            direction = quaternion.as_rotation_matrix(state.rotation) @ np.asarray(local)
            amount, projected = self._slide_projection(start, direction, pending.pop(), pending)
            if projected is None:
                blocked = True
                break
            requested = start + direction * amount
            controls = self.agent.controls
            original_filter = controls.move_filter_fn
            try:
                controls.move_filter_fn = lambda _start, _end: mn.Vector3(projected)
                hit = controls.action(self.agent.scene_node, action,
                                      habitat_sim.agent.ActuationSpec(amount=amount))
            finally:
                controls.move_filter_fn = original_filter
            collision |= bool(hit)
            end = np.asarray(self.agent.get_state().position)
            delta = end - start
            travel = float(np.linalg.norm(delta[[0, 2]]))
            blocked = travel < min(.001, amount * .1)
            if not blocked and np.linalg.norm((end - requested)[[0, 2]]) > .001:
                actual_yaw = math.atan2(-float(delta[0]), -float(delta[2]))
                requested_yaw = math.atan2(-float(direction[0]), -float(direction[2]))
                error = (math.degrees(actual_yaw - requested_yaw) + 180) % 360 - 180
                turn_limit = 3 * amount / micro_step
                adjusted = float(np.clip(auto_yaw + np.clip(error, -turn_limit, turn_limit), -15, 15))
                self.agent.scene_node.rotate_y(mn.Rad(math.radians(adjusted - auto_yaw)))
                auto_yaw = adjusted
            samples.append(self._motion_sample(collision=hit, auto_yaw=auto_yaw))
            if blocked:
                break
        displacement = np.asarray(samples[-1]["position"]) - np.asarray(samples[0]["position"])
        blocked |= float(np.linalg.norm(displacement[[0, 2]])) < self.config.forward_step * .08
        # Match one native atomic action's world tick, without rendering each microstep.
        self.sim.step_physics(1.0 / 60.0)
        return MotionResult(collision, bool(blocked), auto_yaw, samples)

    def step(self, action: str) -> MotionResult:
        if action in TRANSLATION_ACTIONS:
            return self._adaptive_translation(action)
        before = self._motion_sample()
        controls = self.agent.controls
        movement_filter = controls.move_filter_fn
        try:
            if action in {"turn_left", "turn_right"}:
                controls.move_filter_fn = lambda start, end: start
            observations = self.sim.step(action)
        finally:
            controls.move_filter_fn = movement_filter
        collision = bool(observations.get("collided", False))
        return MotionResult(collision, collision, samples=[before, self._motion_sample(collision=collision)])

    def pose(self) -> dict:
        state = self.agent.get_state()
        sensor = state.sensor_states["rgb"]
        world_from_optical = np.eye(4)
        world_from_optical[:3, :3] = quaternion.as_rotation_matrix(
            sensor.rotation
        ) @ np.diag([1, -1, -1])
        world_from_optical[:3, 3] = sensor.position
        return {
            "position": [float(value) for value in state.position],
            "rotation_xyzw": [float(value) for value in quat_to_coeffs(state.rotation)],
            "camera_position": [float(value) for value in sensor.position],
            "camera_rotation_xyzw": [
                float(value) for value in quat_to_coeffs(sensor.rotation)
            ],
            "world_from_optical": world_from_optical.tolist(),
            "frame": "Habitat world; Y up; base/sensor forward -Z; quaternion XYZW",
        }

    def capture(self, workspace: Path, frame: int) -> Observation:
        observations = self.sim.get_sensor_observations()
        directory = workspace / "observations"
        directory.mkdir(parents=True, exist_ok=True)
        rgb = directory / f"rgb_{frame:06d}.png"
        Image.fromarray(np.asarray(observations["rgb"])[..., :3]).save(rgb)
        depth = None
        if self.config.depth:
            depth = directory / f"depth_{frame:06d}.npy"
            np.save(depth, np.asarray(observations["depth"], dtype=np.float32))
        return Observation(
            f"frame_{frame:06d}",
            rgb,
            depth,
            self.pose(),
            camera_parameters(self.config),
        )

    def panorama(self, workspace: Path, frame_ids: list[int]) -> list[Observation]:
        """Render six body-relative horizontal views, restoring sensor extrinsics."""
        if len(frame_ids) != 6 or len(set(frame_ids)) != 6:
            raise ValueError("A panorama requires six unique capture IDs")
        original = deepcopy(self.agent.get_state())
        forward = quaternion.as_rotation_matrix(original.rotation) @ np.array(
            [0, 0, -1]
        )
        yaw = math.atan2(-float(forward[0]), -float(forward[2]))
        views = []
        try:
            for index, frame_id in enumerate(frame_ids):
                state = deepcopy(original)
                rotation = quaternion.from_rotation_vector(
                    [0, yaw - index * math.pi / 3, 0]
                )
                for sensor in state.sensor_states.values():
                    sensor.rotation = rotation
                self.agent.set_state(
                    state, reset_sensors=False, infer_sensor_states=False
                )
                views.append(self.capture(workspace, frame_id))
        finally:
            self.agent.set_state(
                original, reset_sensors=False, infer_sensor_states=False
            )
        return views

    def target_image(self, camera: dict, destination: Path) -> None:
        # GOAT stores an absolute sensor pose, not an agent-base pose.
        uuid = "evaluation_target"
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid, spec.sensor_type = uuid, habitat_sim.SensorType.COLOR
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        spec.resolution, spec.hfov = camera["image_dimensions"], float(camera["hfov"])
        original = self.agent.get_state()
        try:
            self.sim.add_sensor(spec)
            state = habitat_sim.AgentState(
                position=original.position,
                rotation=original.rotation,
                sensor_states={
                    **original.sensor_states,
                    uuid: habitat_sim.SixDOFPose(
                        position=np.asarray(camera["position"]),
                        rotation=quat_from_coeffs(camera["rotation"]),
                    ),
                },
            )
            self.agent.set_state(state, reset_sensors=False, infer_sensor_states=False)
            sensor = self.sim._sensors[uuid]
            sensor.draw_observation()
            rgb = np.asarray(sensor.get_observation())[..., :3].copy()
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(destination)
        finally:
            try:
                self.sim._sensors.pop(uuid, None)
                if uuid in self.agent._sensors:
                    habitat_sim.SensorFactory.delete_subtree_sensor(
                        self.agent.scene_node, uuid
                    )
                    del self.agent._sensors[uuid]
                self.agent.agent_config.sensor_specifications = [
                    sensor
                    for sensor in self.agent.agent_config.sensor_specifications
                    if sensor.uuid != uuid
                ]
            finally:
                self.agent.set_state(
                    original, reset_sensors=False, infer_sensor_states=False
                )

    def close(self) -> None:
        self.sim.close()
