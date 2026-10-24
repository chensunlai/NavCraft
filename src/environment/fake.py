"""CPU simulator for lifecycle tests, never reported as benchmark evidence."""

import math
from pathlib import Path

import numpy as np
from PIL import Image

from .base import (DEPTH_MAX, DEPTH_MIN, DEPTH_SCALE, MotionResult, Observation,
                   TRANSLATION_ACTIONS, save_depth, turn_action)


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
        rotation = [0, math.sin(self.yaw / 2), 0, math.cos(self.yaw / 2)]
        camera_position = self.point + [0, self.config["sensor_height"], 0]
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        transform = np.eye(4)
        transform[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ np.diag([1, -1, -1])
        transform[:3, 3] = camera_position
        return {"position": self.position(), "rotation_xyzw": rotation,
                "camera_position": camera_position.tolist(),
                "camera_rotation_xyzw": rotation, "world_from_optical": transform.tolist()}

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

    def snap_point(self, point):
        return [float(point[0]), float(self.point[1]), float(point[2])]

    def turn(self, degrees):
        turn_action(degrees)
        before = self.pose()
        self.yaw -= math.radians(degrees)
        return MotionResult(samples=[before, self.pose()])

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
        depth = None
        focal = self.config["width"] / (2 * math.tan(math.radians(self.config["hfov"]) / 2))
        camera = {"width": self.config["width"], "height": self.config["height"],
                  "hfov_degrees": self.config["hfov"],
                  "K": [[focal, 0, (self.config["width"] - 1) / 2],
                        [0, focal, (self.config["height"] - 1) / 2], [0, 0, 1]]}
        if self.config["depth"]:
            depth = target.with_name(f"depth_f_{frame:04d}.png")
            save_depth(depth, np.full((self.config["height"], self.config["width"]), 2.0))
            camera.update(depth_scale=DEPTH_SCALE, depth_min=DEPTH_MIN, depth_max=DEPTH_MAX,
                          depth_units="metres", depth_type="optical_axis_z",
                          depth_encoding="uint16_png", depth_invalid_value=0)
        return Observation(f"f_{frame:04d}", target, depth, self.pose(), camera)

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
