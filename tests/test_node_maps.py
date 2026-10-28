from copy import deepcopy
import itertools
import json
import math
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageColor, ImageDraw
import pytest

from src.common.coordinates import apply_affine, invert_affine
from src.harness.geometry import Geometry
from src.harness.map_labels import overlaps
from src.harness.maps import COLORS, MapRenderer, _font


@pytest.fixture
def node_map(tmp_path):
    (tmp_path / "scene").mkdir()
    transform = {"origin": [-10, -10], "size_m": 20, "floor_height": 0,
                 "walls": [[[12, -15], [16, -15]], [[12, -15], [12, -9]]]}
    (tmp_path / "scene/map.json").write_text(json.dumps(transform))
    panoramas = {"p_1": {"pose": {"pos": [1.2, -.2], "heading": 0}},
                 "p_2": {"pose": {"pos": [1.3, -.18], "heading": 90}},
                 "p_3": {"pose": {"pos": [.4, .5], "heading": 180}},
                 "p_4": {"pose": {"pos": [1.24, -.2], "heading": 270}}}
    nodes = [{"id": "N001", "anchor": "p_1", "panorama": "p_4", "room": "living_room_1",
              "description": "Entrance beside the bookshelf", "history": ["p_1", "p_4"]},
             {"id": "N002", "anchor": "p_2", "panorama": "p_2", "room": "living_room_1",
              "description": "Window side of the room", "history": ["p_2"]},
             {"id": "N003", "anchor": "p_3", "panorama": "p_3", "room": None,
              "description": "Unassigned place", "history": ["p_3"]}]
    memory = SimpleNamespace(nodes=nodes, observations=SimpleNamespace(panorama=panoramas.__getitem__))
    renderer = MapRenderer(tmp_path, memory, memory_type="node")
    geometry = Geometry(tmp_path, tmp_path / "private", transform, bounded=False)
    return renderer, geometry, panoramas


def image_pixel(view, point):
    return apply_affine(invert_affine(view["pixel_to_uv"]), point)


def test_explored_includes_anchors_and_trajectory_outside_initial_range(node_map):
    renderer, geometry, panoramas = node_map
    before = deepcopy(renderer.memory.nodes)
    pose = {"pos": [1.25, -.1], "heading": 90}
    path = [[-.1, .8], [.4, .5], [1.25, -.1]]
    view = renderer.render(pose, path)
    assert view["bounds"] == pytest.approx([-.15, -.25, 1.35, .85])
    assert view["visible_ids"] == ["N001", "N002", "N003"]
    assert view["records"][0] == {
        "id": "N001", "room": "living_room_1", "panorama": "p_4",
        "description": "Entrance beside the bookshelf"}
    for node in renderer.memory.nodes:
        point = panoramas[node["anchor"]]["pose"]["pos"]
        assert geometry.resolve_map(view["view_id"], image_pixel(view, point))["uv"] == pytest.approx(point)
    first_label = next(row for row in renderer.last_label_layout if row["id"] == "N001")
    assert first_label["point"] == pytest.approx(image_pixel(view, [1.2, -.2]))
    assert renderer.memory.nodes == before
    full = renderer.render(pose, path, view="full")
    assert full["bounds"] == view["bounds"]
    assert full["pixel_to_uv"] == view["pixel_to_uv"]


@pytest.mark.parametrize("heading", [0, 45, 90, 180, 270])
def test_rotated_node_map_round_trips_beyond_initial_canvas(node_map, heading):
    renderer, geometry, _ = node_map
    pose = {"pos": [1.25, -.1], "heading": heading}
    view = renderer.render(pose, [], view={"ego": 6})
    assert geometry.resolve_map(view["view_id"], [384, 380])["uv"] == pytest.approx(pose["pos"])
    angle = math.radians(heading)
    ahead = geometry.resolve_map(view["view_id"], [384, 380 - 692 / 12])["uv"]
    assert ahead == pytest.approx([1.25 + math.sin(angle) / 20, -.1 - math.cos(angle) / 20], abs=5e-5)
    # A later crop must not change the source transform of a saved image.
    renderer.render({"pos": [-.2, 1.1], "heading": 0}, [], view={"around_self": 3})
    assert geometry.resolve_map(view["view_id"], [384, 380])["uv"] == pytest.approx(pose["pos"])
    with pytest.raises(ValueError, match="outside_map_plot"):
        geometry.resolve_map(view["view_id"], [384, 15])
    bounded = Geometry(renderer.workspace, geometry.private, geometry.transform)
    with pytest.raises(ValueError, match="outside_canvas"):
        bounded.resolve_map(view["view_id"], [384, 380])


def test_empty_node_map_has_local_nonzero_bounds_and_never_draws_walls(node_map):
    renderer, _, _ = node_map
    renderer.memory.nodes = []
    pose = {"pos": [1.2, -.2], "heading": 0}
    with_walls = renderer.render(pose, [])
    assert with_walls["bounds"] == pytest.approx([1.15, -.25, 1.25, -.15])
    assert min(with_walls["image_size"]) > 0
    data_path = renderer.workspace / "scene/map.json"
    data = json.loads(data_path.read_text())
    data["walls"] = []
    data_path.write_text(json.dumps(data))
    no_walls = renderer.render(pose, [])
    with Image.open(renderer.workspace / with_walls["map"]) as first, Image.open(renderer.workspace / no_walls["map"]) as second:
        assert first.tobytes() == second.tobytes()
    with pytest.raises(ValueError, match="only visited nodes"):
        renderer.render(pose, [], overlay_points=np.array([[1.2, -.2, 255, 0, 255]]))


def test_node_labels_share_room_color_keep_bold_ids_and_avoid_overlap(node_map):
    renderer, _, panoramas = node_map
    panoramas["p_3"]["pose"]["pos"] = [1.2, -.15]
    pose = {"pos": [1.25, -.1], "heading": 0}
    view = renderer.render(pose, [], view={"around_self": 6})
    labels = {row["id"]: row for row in renderer.last_label_layout}
    assert len(labels) == 3
    assert labels["N001"]["color"] == labels["N002"]["color"]
    assert labels["N003"]["color"] == COLORS["node"]
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    for label in labels.values():
        assert label["type"] == "node"
        assert label["text"] == label["id"]
        box = measure.textbbox((0, 0), label["id"], font=_font(14, bold=True))
        assert label["size"] == [box[2] - box[0] + 6, box[3] - box[1] + 4]
    for first, second in itertools.combinations(labels.values(), 2):
        assert not overlaps(first["rect"], second["rect"], 3)
    with Image.open(renderer.workspace / view["map"]) as image:
        for label in labels.values():
            assert image.getpixel(tuple(round(v) for v in label["point"])) == ImageColor.getrgb(label["color"])
    renderer.memory.nodes.reverse()
    renderer.render(pose, [], view={"around_self": 6})
    assert {row["id"]: row["color"] for row in renderer.last_label_layout} == {
        identifier: row["color"] for identifier, row in labels.items()}


def test_node_trajectory_respects_actual_samples_and_breaks(node_map):
    renderer, _, _ = node_map
    renderer.memory.nodes = []
    pose = {"pos": [1.25, -.1], "heading": 0}
    trajectory = [[1.1, -.23], [1.2, -.23], [1.2, -.13]]
    connected = renderer.render(pose, trajectory, view={"around_self": 6})
    broken = renderer.render(pose, trajectory, view={"around_self": 6}, trajectory_breaks=[1])
    mid = tuple(round(v) for v in image_pixel(connected, [1.15, -.23]))
    bend = tuple(round(v) for v in image_pixel(connected, [1.2, -.18]))
    with Image.open(renderer.workspace / connected["map"]) as first, Image.open(renderer.workspace / broken["map"]) as second:
        assert first.getpixel(mid) == ImageColor.getrgb("#8fc6ed")
        assert second.getpixel(mid) != ImageColor.getrgb("#8fc6ed")
        assert second.getpixel(bend) == ImageColor.getrgb("#8fc6ed")


def test_unbounded_rgb_projection_preserves_range_floor_and_finite_checks(node_map):
    renderer, geometry, _ = node_map
    depth_dir = geometry.private / "depth"
    depth_dir.mkdir(parents=True)
    (renderer.workspace / "observations").mkdir(exist_ok=True)
    camera = {"frame": "f_0001", "depth": "f_0001.png", "depth_scale": .001,
              "K": [[2, 0, 2], [0, 2, 2], [0, 0, 1]],
              "world_from_optical": [[1, 0, 0, 20], [0, -1, 0, 1], [0, 0, -1, -20], [0, 0, 0, 1]],
              "body_position": [20, 0, -20], "body_forward": [0, 0, -1],
              "floor_height": 0, "width": 5, "height": 5, "depth_min": .05, "depth_max": 10}
    Image.fromarray(np.full((5, 5), 2000, dtype=np.uint16)).save(depth_dir / "f_0001.png")
    Image.new("RGB", (5, 5), (50, 120, 200)).save(renderer.workspace / "observations/f_0001.png")
    (depth_dir / "index.jsonl").write_text(json.dumps(camera) + "\n")
    (renderer.workspace / "observations/index.jsonl").write_text(json.dumps({
        "frame": "f_0001", "path": "observations/f_0001.png"}) + "\n")
    result = geometry.resolve_frame("f_0001", [2, 2])
    assert result == {"uv": [1.5, -.6], "distance": 2., "bearing": 0., "height": 1., "valid": True}
    bounded = Geometry(renderer.workspace, geometry.private, geometry.transform)
    assert bounded.resolve_frame("f_0001", [2, 2]) == {"valid": False, "reason": "outside_canvas"}
    for pixel in ([math.nan, 2], [5, 2]):
        assert geometry.resolve_frame("f_0001", pixel)["valid"] is False
    camera["floor_y_range"] = [1.5, 3]
    (depth_dir / "index.jsonl").write_text(json.dumps(camera) + "\n")
    assert geometry.resolve_frame("f_0001", [2, 2])["reason"] == "outside_floor"
    camera.pop("floor_y_range")
    (depth_dir / "index.jsonl").write_text(json.dumps(camera) + "\n")
    Image.fromarray(np.full((5, 5), 0, dtype=np.uint16)).save(depth_dir / "f_0001.png")
    assert geometry.resolve_frame("f_0001", [2, 2])["reason"] == "invalid_depth_or_range"
