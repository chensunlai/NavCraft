import json

import numpy as np
from PIL import Image
import pytest

from src.dataset.episodes import EpisodeSpec
from src.environment import Environment
from src.environment.fake import FakeSimulator
from src.harness.maps import MapRenderer


def test_relocation_preserves_motion_budget_and_excludes_teleport_distance(tmp_path):
    episode = EpisodeSpec("fixture", "0", "objnav", "fake", [0, 0, 0], [0, 0, 0, 1], [], "fixture")
    env = Environment(episode, {"gpu": 0}, tmp_path, FakeSimulator)
    try:
        env.step("move_forward")
        old_path = env.motion_trajectory.copy()
        env.relocate({"position": [5, 0, 5], "rotation_xyzw": [0, 0, 0, 1]})
        assert env.steps == 1 and env.path_length == pytest.approx(.25)
        assert env.motion_trajectory[:-1] == old_path
        assert env.trajectory_breaks == [2] and env.position() == [5, 0, 5]
        frame = env.observe()
        assert frame.pose["camera_position"] == [5, .88, 5]
        env.step("move_forward")
        assert env.steps == 2 and env.path_length == pytest.approx(.5)
        record = json.loads((tmp_path / "private/relocations.jsonl").read_text())
        assert record["step"] == 1 and record["trajectory_index"] == 2
        before = env.pose()
        with pytest.raises(ValueError):
            env.relocate({"position": [1, 0, 0], "rotation_xyzw": [0, 0, 0, 0]})
        assert env.pose() == before
    finally:
        env.close()


def test_map_does_not_draw_teleport_as_travel(tmp_path):
    (tmp_path / "scene").mkdir()
    (tmp_path / "scene/map.json").write_text(json.dumps({"origin": [0, 0], "size_m": 10, "walls": []}))
    renderer = MapRenderer(tmp_path)
    pose = {"pos": [.8, .8], "heading": 0}
    path = [[.2, .2], [.8, .8]]
    linked = renderer.render(pose, path, view="full")
    separated = renderer.render(pose, path, view="full", trajectory_breaks=[1])
    color = [143, 198, 237]
    assert np.any(np.all(np.asarray(Image.open(tmp_path / linked["map"])) == color, axis=2))
    assert not np.any(np.all(np.asarray(Image.open(tmp_path / separated["map"])) == color, axis=2))
