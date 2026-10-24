import itertools
import json

import numpy as np
from PIL import Image, ImageColor
import pytest

from src.harness.map_labels import overlaps
from src.harness.maps import COLORS, MapRenderer


@pytest.fixture
def renderer(tmp_path):
    (tmp_path / "scene").mkdir()
    (tmp_path / "scene/map.json").write_text(json.dumps({
        "origin": [-5, -5], "size_m": 20, "walls": [],
    }))
    return MapRenderer(tmp_path)


def map_pixel(result, uv):
    affine = np.vstack([result["pixel_to_uv"], [0, 0, 1]])
    return np.linalg.solve(affine, [*uv, 1])[:2]


def test_explored_crop_contains_located_points_without_persisting_entities(renderer):
    memory_path = renderer.workspace / "memory/entities.json"
    memory_before = memory_path.read_bytes()
    points = [{"label": "P1", "uv": [.02, .9]}, {"label": "P2", "uv": [.98, .1]}]
    result = renderer.render({"pos": [.3, .4], "heading": 0}, [[.3, .4], [.35, .45]], located_points=points)
    assert result["bounds"] == pytest.approx([0, 0, 1, 1])
    assert [label["id"] for label in renderer.last_label_layout] == ["P1", "P2"]
    assert memory_path.read_bytes() == memory_before
    assert renderer.memory.entities == []
    follow_up = renderer.render({"pos": [.3, .4], "heading": 0}, [])
    assert renderer.last_label_layout == []
    assert follow_up["bounds"] == pytest.approx([.2, .3, .4, .5])


@pytest.mark.parametrize("view,heading", [("full", 0), ({"ego": 4}, 90), ({"ego": 4}, 45)])
def test_temporary_marker_pixels_use_the_saved_map_affine(renderer, view, heading):
    point = {"label": "P1", "uv": [.57, .44]}
    result = renderer.render({"pos": [.5, .5], "heading": heading}, [], view=view, located_points=[point])
    assert not result["warnings"]
    expected = map_pixel(result, point["uv"])
    assert renderer.last_label_layout[0]["point"] == pytest.approx(expected)
    with Image.open(renderer.workspace / result["map"]) as image:
        assert image.getpixel(tuple(round(v) for v in expected)) == ImageColor.getrgb(COLORS["locate"])
    index = [json.loads(line) for line in (renderer.workspace / "observations/maps/index.jsonl").read_text().splitlines()]
    assert index[-1]["pixel_to_uv"] == result["pixel_to_uv"]
    assert index[-1]["plot_bounds"] == result["plot_bounds"]


def test_same_point_labels_are_distinct_and_do_not_hide_pose_marker(renderer):
    points = [{"label": f"P{i}", "uv": [.5, .5]} for i in range(1, 11)]
    result = renderer.render({"pos": [.5, .5], "heading": 0}, [], view="full", labels="none", located_points=points)
    assert not result["warnings"]
    assert len(renderer.last_label_layout) == 10
    for a, b in itertools.combinations(renderer.last_label_layout, 2):
        assert not overlaps(a["rect"], b["rect"], 3)
    for label in renderer.last_label_layout:
        assert label["point"] == pytest.approx(map_pixel(result, [.5, .5]))
    with Image.open(renderer.workspace / result["map"]) as image:
        assert image.getpixel((384, 380)) == ImageColor.getrgb(COLORS["locate"])


def test_canvas_corner_points_remain_visible_and_labels_stay_in_the_plot(renderer):
    points = [{"label": f"P{i}", "uv": uv} for i, uv in enumerate(([0, 0], [0, 1], [1, 0], [1, 1]), 1)]
    result = renderer.render({"pos": [.5, .5], "heading": 0}, [], view="full", located_points=points)
    assert not result["warnings"]
    assert len(renderer.last_label_layout) == 4
    left, top, right, bottom = result["plot_bounds"]
    with Image.open(renderer.workspace / result["map"]) as image:
        for label in renderer.last_label_layout:
            x, y = label["point"]
            assert left <= x <= right and top <= y <= bottom
            assert image.getpixel((round(x), round(y))) == ImageColor.getrgb(COLORS["locate"])
            rect = label["rect"]
            assert left <= rect[0] < rect[2] <= right
            assert top <= rect[1] < rect[3] <= bottom


@pytest.mark.parametrize("point", [{"label": "P1", "uv": [1.01, .5]}, {"label": "P1", "uv": [True, .5]},
                                   {"label": "P1", "uv": [float("nan"), .5]}, {"label": "", "uv": [.5, .5]}])
def test_invalid_temporary_geometry_is_rejected(renderer, point):
    with pytest.raises(ValueError, match="located_points"):
        renderer.render({"pos": [.5, .5], "heading": 0}, [], located_points=[point])
