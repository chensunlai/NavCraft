"""CPU simulator for lifecycle tests, never reported as benchmark evidence."""

import math
from pathlib import Path

import numpy as np
from PIL import Image

from .base import MotionResult, Observation, TRANSLATION_ACTIONS


class FakeSimulator:
    def __init__(self, episode, config):
        self.config = config
        self.point = np.array(episode.start_position, dtype=float)
        x, y, z, w = episode.start_rotation
        self.yaw = math.atan2(2 * (w * y + x * z), 1 - 2 * (y * y + z * z))
        self.closed = False

    def position(self):
        return self.point.tolist()

    def pose(self):
        return {"position": self.position(),
                "rotation_xyzw": [0, math.sin(self.yaw / 2), 0, math.cos(self.yaw / 2)],
                "camera_position": (self.point + [0, self.config["sensor_height"], 0]).tolist()}

    def restore_pose(self, pose):
        self.point = np.array(pose["position"], dtype=float)
        x, y, z, w = pose["rotation_xyzw"]
        self.yaw = math.atan2(2 * (w * y + x * z), 1 - 2 * (y * y + z * z))

    def distance(self, targets):
        return self.distance_from(self.position(), targets)

    def distance_from(self, point, targets):
        return min((math.dist(point, target) for target in targets), default=math.inf)

    def shortest_path(self, point, targets):
        return [list(point), list(min(targets, key=lambda target: math.dist(point, target)))] if targets else []

    def step(self, action):
        samples = [self.pose()]
        if action in TRANSLATION_ACTIONS:
            forward = np.array([-math.sin(self.yaw), 0, -math.cos(self.yaw)])
            right = np.array([math.cos(self.yaw), 0, -math.sin(self.yaw)])
            direction = {"move_forward": forward, "move_backward": -forward,
                         "move_left": -right, "move_right": right}[action]
            origin = self.point.copy()
            for i in range(1, 6):
                self.point = origin + direction * self.config["forward_step"] * i / 5
                samples.append(self.pose())
        else:
            self.yaw += math.radians(self.config["turn_angle"]) * (1 if action == "turn_left" else -1)
            samples.append(self.pose())
        return MotionResult(samples=samples)

    def capture(self, workspace, frame):
        target = Path(workspace) / "observations" / f"f_{frame:04d}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (self.config["width"], self.config["height"]),
                  (frame % 255, int(math.degrees(self.yaw)) % 255, 100)).save(target)
        return Observation(f"f_{frame:04d}", target, None, self.pose(),
                           {"width": self.config["width"], "height": self.config["height"],
                            "hfov_degrees": self.config["hfov"]})

    def panorama(self, workspace, frame_ids):
        original = self.yaw
        try:
            views = []
            for i, fid in enumerate(frame_ids):
                self.yaw = original - math.radians(60 * i)
                views.append(self.capture(workspace, fid))
            return views
        finally:
            self.yaw = original

    def target_image(self, camera, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        height, width = camera["image_dimensions"]
        Image.new("RGB", (width, height), "blue").save(destination)

    def close(self):
        self.closed = True
