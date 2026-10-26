from copy import deepcopy
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.dataset.episodes import EpisodeSpec
from src.dataset.generation.collect import _execute_reference, facing_pose, footprint_distance, reference_pose, visible_instances
from src.dataset.generation.scenes import environment_config, semantic_instances, semantic_mesh_bounds, verify_bounds
from src.environment.base import Observation
from src.environment import Environment
from src.environment.fake import FakeSimulator
from src.harness.session import open_session


def test_semantic_identity_uses_native_id_not_list_position():
    obj = SimpleNamespace(id="region_73", semantic_id=lambda: 73,
                          category=SimpleNamespace(name=lambda: "couch"))
    rows = semantic_instances(SimpleNamespace(objects=[None, obj]), 0,
                              {73: {"center": [1, .5, 2], "size": [1, 1, 1]}})
    assert rows == [{"id": "region_73", "semantic_id": 73, "category": "sofa",
                     "center": [1., .5, 2.], "size": [1., 1., 1.]}]
    semantic = np.full((30, 40), 73, dtype=np.int32)
    assert visible_instances(semantic, rows)[0]["id"] == "region_73"
    assert not visible_instances(semantic, [{**rows[0], "semantic_id": 1}])


def test_generation_tags_and_private_depth_are_not_tool_permissions(tmp_path):
    values = environment_config({"tool_tier": "basic", "environment": {"gpu": 1}}, tmp_path)
    assert values["private_semantic"] and values["private_depth"] and values["tool_tier"] == "basic"
    with pytest.raises(ValueError, match="GPU 0 or 1"):
        environment_config({"environment": {"gpu": 3}}, tmp_path)


def test_environment_cut_restores_counters_frame_and_true_path(tmp_path):
    episode = EpisodeSpec("cut", "0", "generated", "fake.glb", [0, 0, 0], [0, 0, 0, 1], [], "fixture")
    config = {"gpu": 0, "tool_tier": "basic", "private_depth": True}
    first = Environment(episode, config, tmp_path / "a", FakeSimulator)
    first.step("move_forward")
    first.turn(15)
    state = deepcopy(first.state())
    second = Environment(episode, config, tmp_path / "b", FakeSimulator)
    second.restore_state(state)
    assert second.state() == state
    assert second.observe().frame_id == "f_0003"
    assert second.config["depth"]
    second.step("move_forward")
    assert second.steps == 3 and second.path_length == pytest.approx(.5)
    assert first.steps == 2
    assert len(state["trajectory"]) == 3
    with pytest.raises(ValueError, match="before movement"):
        second.restore_state(state)
    first.close()
    second.close()


def test_approach_distance_uses_footprint_and_facing_rotation():
    instance = {"center": [0, .5, 0], "size": [2, 1, 2]}
    assert footprint_distance([0, 0, 0], instance) == 0
    assert footprint_distance([2, 0, 0], instance) == 1
    pose = facing_pose([0, 0, 0], [1, 0, 0])
    assert pose["rotation_xyzw"] == pytest.approx([0, -2 ** -.5, 0, 2 ** -.5])


def test_semantic_vertex_bounds_do_not_include_world_origin(tmp_path):
    vertices = np.array([[2, 3, 4], [3, 5, 6], [2, 5, 4]], dtype="<f4")
    colors = np.tile(np.array([20, 40, 60, 65535], dtype="<u2"), (3, 1))
    binary = vertices.tobytes() + colors.tobytes()
    document = {"nodes": [{"mesh": 0}], "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}}]}],
                "bufferViews": [{"byteOffset": 0}, {"byteOffset": vertices.nbytes}],
                "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                              {"bufferView": 1, "componentType": 5123, "count": 3, "type": "VEC4", "normalized": True}]}
    encoded = json.dumps(document).encode()
    encoded += b" " * (-len(encoded) % 4)
    mesh = tmp_path / "semantic.glb"
    mesh.write_bytes(struct.pack("<III", 0x46546C67, 2, 28 + len(encoded) + len(binary))
                     + struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
                     + struct.pack("<II", len(binary), 0x004E4942) + binary)
    labels = tmp_path / "semantic.txt"
    labels.write_text('HM3D Semantic Annotations\n73,010203,"chair",0\n')
    assert semantic_mesh_bounds(mesh, labels) == {73: {"center": [2.5, 5., -4.], "size": [1., 2., 2.]}}


def test_depth_coordinate_check_rejects_wrong_axis_transform(tmp_path):
    mask = tmp_path / "semantic.npy"
    np.save(mask, np.full((25, 25), 73, dtype=np.int32))
    depth = tmp_path / "depth.png"
    Image.fromarray(np.full((25, 25), 2000, dtype=np.uint16)).save(depth)
    camera = {"K": [[100, 0, 12], [0, 100, 12], [0, 0, 1]], "depth_scale": .001}
    observation = Observation("frame", tmp_path / "rgb.png", depth,
                              {"world_from_optical": np.eye(4).tolist()}, camera, mask)
    row = {"id": "chair_73", "semantic_id": 73, "center": [0, 0, 2], "size": [2, 2, 2]}
    assert verify_bounds(observation, [row])[0]["inside_mesh_aabb_fraction"] == 1
    with pytest.raises(ValueError, match="coordinate check failed"):
        verify_bounds(observation, [{**row, "center": [20, 0, 2]}])


def test_reference_pitch_looks_at_target_without_changing_body_heading():
    target = [1., 3., -1.]
    pose = reference_pose([0., 0., 0.], target)
    assert pose["rotation_xyzw"] == facing_pose([0., 0., 0.], target)["rotation_xyzw"]
    x, y, z, w = pose["camera_rotation_xyzw"]
    forward = np.array([-2 * (x * z + w * y), 2 * (w * x - y * z), -(1 - 2 * (x * x + y * y))])
    expected = np.asarray(target) - pose["camera_position"]
    expected /= np.linalg.norm(expected)
    assert forward == pytest.approx(expected)


def test_basic_reference_uses_actual_atomic_tools_without_super_calls(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    episode = EpisodeSpec("reference", "0", "generated", "fake.glb", [0, 0, 0], [0, 0, 0, 1], [], "fixture")
    environment = Environment(episode, {"gpu": 0}, tmp_path / "environment", FakeSimulator)
    wall_map = tmp_path / "map.json"
    wall_map.write_text(json.dumps({"origin": [-5, -5], "size_m": 10, "floor_height": 0,
                                    "walls": [[[-5, -5], [5, -5]]]}))
    try:
        with open_session(repo, {"model": "fixture", "tool_tier": "basic"}, tmp_path / "session",
                          environment, wall_map, start_core=False) as session:
            session.begin_script("Go to the nearby point.")
            _execute_reference(session, [0, 0, -.6])
            calls = [block for message in session._history if message["role"] == "assistant" for block in message["content"]]
            assert [call["name"] for call in calls] == ["nav_step", "nav_step"]
            assert environment.position() == pytest.approx([0, 0, -.5])
            assert session.client is None
    finally:
        environment.close()
