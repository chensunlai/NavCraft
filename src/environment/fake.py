"""Deterministic CPU environment, using the same task lifecycle as Habitat."""

import math
from pathlib import Path

import numpy as np
from PIL import Image

from src.common.register import env_registry
from src.environment.base import Observation
from src.environment.habitat import HabitatEnvironment


@env_registry.register("fake")
class FakeEnvironment(HabitatEnvironment):
    def __init__(self, config: dict, task, workspace: Path, simulator_factory=None):
        super().__init__(config, task, workspace, simulator_factory or FakeSimulator)


class FakeSimulator:
    """Open plane, known depth, and a synthetic image; not a navigation benchmark."""

    def __init__(self, episode, config: dict):
        self.config = config
        self.point = np.asarray(episode.start_position, dtype=np.float64)
        x, y, z, w = episode.start_rotation
        self.yaw = math.atan2(2 * (w * y + x * z), 1 - 2 * (y * y + z * z))
        self.pitch = 0.0
        self.closed = False

    def position(self) -> list[float]:
        return self.point.tolist()

    def distance(self, targets: list) -> float:
        return min((math.dist(self.position(), target) for target in targets), default=math.inf)

    def step(self, action: str) -> bool:
        angle = math.radians(self.config.get("turn_angle", 30))
        if action == "move_forward":
            self.point += self.config.get("forward_step", 0.25) * np.array([-math.sin(self.yaw), 0, -math.cos(self.yaw)])
        elif action == "turn_left":
            self.yaw += angle
        elif action == "turn_right":
            self.yaw -= angle
        elif action == "look_up":
            self.pitch = min(math.pi / 2, self.pitch + angle)
        elif action == "look_down":
            self.pitch = max(-math.pi / 2, self.pitch - angle)
        return False

    def capture(self, workspace: Path, frame: int) -> Observation:
        directory = workspace / "observations"
        directory.mkdir(parents=True, exist_ok=True)
        width, height = self.config.get("width", 64), self.config.get("height", 48)
        rgb = directory / f"rgb_{frame:06d}.png"
        pixels = np.zeros((height, width, 3), dtype=np.uint8)
        pixels[..., 0] = np.arange(width, dtype=np.uint8)
        pixels[..., 1] = np.arange(height, dtype=np.uint8)[:, None]
        pixels[..., 2] = frame % 256
        Image.fromarray(pixels).save(rgb)
        depth = None
        if self.config.get("depth", True):
            depth = directory / f"depth_{frame:06d}.npy"
            np.save(depth, np.full((height, width), 2.0, dtype=np.float32))
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        rotation = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        camera_position = self.point + [0, self.config.get("sensor_height", 1.5), 0]
        matrix = np.eye(4)
        matrix[:3, :3] = rotation @ np.diag([1, -1, -1])
        matrix[:3, 3] = camera_position
        focal = width / (2 * math.tan(math.radians(self.config.get("hfov", 90)) / 2))
        pose = {"position": self.position(), "rotation_xyzw": [0, math.sin(self.yaw / 2), 0, math.cos(self.yaw / 2)],
                "camera_position": camera_position.tolist(), "world_from_optical": matrix.tolist(),
                "frame": "Habitat world; Y up; base/sensor forward -Z; quaternion XYZW"}
        camera = {"width": width, "height": height, "hfov_degrees": self.config.get("hfov", 90),
                  "K": [[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]],
                  "depth_units": "metres", "depth_type": "optical_axis_z",
                  "optical_frame": "X right, Y down, Z forward",
                  "sensor_frame": "X right, Y up, forward -Z"}
        return Observation(f"frame_{frame:06d}", rgb, depth, pose, camera)

    def target_image(self, camera: dict, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        height, width = camera.get("image_dimensions", [48, 64])
        Image.new("RGB", (width, height), color="blue").save(destination)

    def close(self) -> None:
        self.closed = True
