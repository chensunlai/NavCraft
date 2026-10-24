import json
import math

import numpy as np
import pytest
from PIL import Image

from src.harness.maps import MapRenderer


@pytest.fixture
def renderer(tmp_path):
    (tmp_path / "scene").mkdir()
    (tmp_path / "scene/map.json").write_text(json.dumps({
        "origin": [-5, -5], "size_m": 10,
        "walls": [[[-4, -4], [-4, 4]], [[-4, 4], [4, 4]]],
    }))
    return MapRenderer(tmp_path)


def uv(result, x, y):
    return [a * x + b * y + c for a, b, c in result["pixel_to_uv"]]


@pytest.mark.parametrize("heading", [0, 45, 90, 180, 270])
def test_ego_view_centres_pose_and_maps_up_to_body_forward(renderer, heading):
    pose = {"pos": [.04, .07], "heading": heading}
    result = renderer.render(pose, [], view={"ego": 3})
    left, top, right, bottom = result["plot_bounds"]
    cx, cy = (left + right) / 2, (top + bottom) / 2
    assert result["image_size"] == [768, 768]
    assert uv(result, cx, cy) == pytest.approx(pose["pos"])
    pixels_per_metre = (right - left) / 6
    angle = math.radians(heading)
    assert uv(result, cx, cy - pixels_per_metre) == pytest.approx([
        pose["pos"][0] + math.sin(angle) / 10,
        pose["pos"][1] - math.cos(angle) / 10,
    ])
    assert uv(result, cx + pixels_per_metre, cy) == pytest.approx([
        pose["pos"][0] + math.cos(angle) / 10,
        pose["pos"][1] + math.sin(angle) / 10,
    ])
    # A local view stays centred even when it extends beyond the original canvas.
    assert min(result["bounds"]) < 0


def test_map_images_persist_their_exact_pixel_affine(renderer):
    pose = {"pos": [.5, .5], "heading": 90}
    views = ["full", "explored", {"around_self": 2}, {"ego": 2}]
    results = [renderer.render(pose, [[.4, .5], [.5, .5]], view=view) for view in views]
    entries = [json.loads(line) for line in
               (renderer.workspace / "observations/maps/index.jsonl").read_text().splitlines()]
    assert len(entries) == len(views)
    assert len({entry["view_id"] for entry in entries}) == len(views)
    for result, entry in zip(results, entries):
        with Image.open(renderer.workspace / result["map"]) as image:
            assert list(image.size) == result["image_size"] == [entry["width"], entry["height"]]
        assert entry["pixel_to_uv"] == result["pixel_to_uv"]
        assert entry["plot_bounds"] == result["plot_bounds"]
        assert entry["map"] == result["map"]
        assert np.asarray(result["pixel_to_uv"]).shape == (2, 3)
    assert uv(results[0], 38, 34) == pytest.approx([0, 0])
    assert uv(results[2], 38, 34) == pytest.approx([.3, .3])


def test_rgb_overlay_uses_the_same_ego_projection_without_editing_memory(renderer):
    pose = {"pos": [.5, .5], "heading": 90}
    before = renderer.memory.entities
    result = renderer.render(pose, [], view={"ego": 3},
                             overlay_points=np.array([[.61, .54, 255, 0, 255]]))
    inverse = np.vstack((result["pixel_to_uv"], [0, 0, 1]))
    x, y, _ = np.linalg.solve(inverse, [.61, .54, 1])
    assert y < 380  # East is forward and therefore up at heading 90.
    with Image.open(renderer.workspace / result["map"]) as image:
        assert image.getpixel((round(x), round(y))) == (255, 0, 255)
        assert image.getpixel((384, 370)) == (229, 57, 53)  # Body arrow stays up.
    assert renderer.memory.entities == before


@pytest.mark.parametrize("radius", [0, -1, True, float("inf"), "3"])
def test_ego_radius_must_be_positive_finite_metres(renderer, radius):
    with pytest.raises(ValueError, match="ego must"):
        renderer.render({"pos": [.5, .5], "heading": 0}, [], view={"ego": radius})
