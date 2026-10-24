"""One episode's simulator and actual motion history."""

import json
import math
from pathlib import Path

from .base import ACTIONS

SENSORS = {
    "width": 640, "height": 480, "hfov": 79,
    "sensor_height": 0.88, "agent_height": 0.88, "agent_radius": 0.18,
    "navmesh_cell_height": 0.05, "forward_step": 0.25, "turn_angle": 30,
    "depth": False,
}


class Environment:
    def __init__(self, episode, config: dict, directory: Path, simulator_factory=None):
        if type(config.get("gpu")) is not int or config["gpu"] not in (0, 1):
            raise ValueError("Lite simulation requires an explicit gpu of 0 or 1")
        self.episode = episode
        self.config = {**config, **SENSORS}
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        if simulator_factory is None:
            from .simulator import HabitatSimulator
            simulator_factory = HabitatSimulator
        self.sim = simulator_factory(episode=episode, config=self.config)
        self.steps = 0
        self.path_length = 0.0
        self.trajectory = [self.position()]
        self.motion_trajectory = [self.position()]
        self.motions = []
        self.current = None
        self._frame = 0

    def _frames(self, count=1):
        start = self._frame + 1
        self._frame += count
        return list(range(start, self._frame + 1))

    def observe(self):
        self.current = self.sim.capture(self.directory, self._frames()[0])
        return self.current

    def panorama(self):
        return self.sim.panorama(self.directory, self._frames(6))

    def step(self, action: str):
        if action not in ACTIONS:
            raise ValueError(f"Unknown action: {action}")
        motion = self.sim.step(action)
        points = motion.positions or [self.position()]
        # Include the true pre-action point even for a sparse simulator adapter.
        path = [self.motion_trajectory[-1], *points]
        self.path_length += sum(math.dist(a, b) for a, b in zip(path, path[1:]))
        self.motion_trajectory.extend(points[1:] if points[0] == self.motion_trajectory[-1] else points)
        self.trajectory.append(self.position())
        self.steps += 1
        record = {"step": self.steps, "action": action,
                  "collision": motion.collision, "blocked": motion.blocked,
                  "auto_yaw_delta_degrees": motion.auto_yaw_delta_degrees,
                  "samples": motion.samples}
        self.motions.append(record)
        private = self.directory / "private"
        private.mkdir(exist_ok=True)
        with (private / "motions.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        return motion, self.observe()

    def pose(self):
        return self.sim.pose()

    def restore_pose(self, pose):
        if self.steps:
            raise ValueError("Initial pose can only be restored before movement")
        self.sim.restore_pose(pose)
        self.trajectory = [self.position()]
        self.motion_trajectory = [self.position()]
        self.current = None

    def position(self):
        return self.sim.position()

    def distance(self, targets):
        return self.sim.distance(targets)

    def target_image(self, camera, path):
        return self.sim.target_image(camera, Path(path))

    def close(self):
        if self.sim is not None:
            sim, self.sim = self.sim, None
            sim.close()
