import itertools
import json
from types import SimpleNamespace

import pytest
import numpy as np
from PIL import Image, ImageColor

from src.harness.map_labels import layout_labels, overlaps
from src.harness.maps import COLORS, MapRenderer, _font


def test_dense_layout_keeps_all_labels_away_from_markers_pose_and_each_other():
    items = [{"id": f"O{i:02d}", "point": [340 + i % 5 * 7, 350 + i // 5 * 7],
              "size": [92, 19], "priority": 1} for i in range(25)]
    items += [{"id": "R01", "point": [350, 350], "size": [100, 22], "priority": 0},
              {"id": "D01", "point": [370, 360], "size": [100, 22], "priority": 0}]
    obstacles = [[x - 7, y - 7, x + 7, y + 7] for item in items for x, y in [item["point"]]]
    obstacles.append([369, 365, 399, 395])
    bounds = [38, 34, 730, 726]
    placed, missing = layout_labels(items, bounds, obstacles)
    assert not missing
    assert len(placed) == 27
    assert [p["id"] for p in placed[:2]] == ["D01", "R01"]
    for a, b in itertools.combinations(placed, 2):
        assert not overlaps(a["rect"], b["rect"], 3)
    for label in placed:
        rect = label["rect"]
        assert bounds[0] <= rect[0] < rect[2] <= bounds[2]
        assert bounds[1] <= rect[1] < rect[3] <= bounds[3]
        assert not any(overlaps(rect, obstacle, 3) for obstacle in obstacles)
        assert label["leader"][0] == label["point"]
        assert next(item for item in items if item["id"] == label["id"])["point"] == label["point"]


def test_unplaceable_label_is_reported_without_moving_its_anchor():
    placed, missing = layout_labels([{"id": "O01", "point": [50, 50], "size": [200, 20]}], [0, 0, 100, 100])
    assert placed == []
    assert missing == ["O01"]


def test_structural_font_is_bold_and_supports_distinct_cjk_and_digits():
    regular, bold = _font(14), _font(14, bold=True)
    assert bytes(regular.getmask("R01 卧室")) != bytes(bold.getmask("R01 卧室"))
    masks = [bytes(bold.getmask(character)) for character in "012mNR椅门"]
    assert len(set(masks)) == len(masks)


@pytest.fixture
def dense_renderer(tmp_path):
    (tmp_path / "scene").mkdir()
    (tmp_path / "scene/map.json").write_text(json.dumps({
        "origin": [0, 0], "size_m": 10,
        "walls": [[[1, 1], [1, 9]], [[1, 9], [9, 9]]],
    }))
    entities = [{"id": "R01", "type": "room", "name": "客餐厅", "anchor": [.46, .46]},
                {"id": "D01", "type": "door", "name": "厨房入口", "point": [.55, .46]}]
    entities += [{"id": f"O{i:02d}", "type": "object", "name": f"物体{i:02d}",
                  "point": [.46 + (i % 5) * .009, .49 + (i // 5) * .009]} for i in range(1, 26)]
    return MapRenderer(tmp_path, SimpleNamespace(entities=entities))


@pytest.mark.parametrize("view,heading", [("full", 0), ({"ego": 3}, 45)])
def test_renderer_draws_all_dense_labels_and_type_markers(dense_renderer, view, heading):
    renderer = dense_renderer
    result = renderer.render({"pos": [.5, .5], "heading": heading}, [], view=view, labels="both")
    assert not result["warnings"]
    assert len(renderer.last_label_layout) == 27
    for a, b in itertools.combinations(renderer.last_label_layout, 2):
        assert not overlaps(a["rect"], b["rect"], 3)
    with Image.open(renderer.workspace / result["map"]) as image:
        inverse = np.vstack([result["pixel_to_uv"], [0, 0, 1]])
        body = np.linalg.solve(inverse, [.5, .5, 1])[:2]
        for label in renderer.last_label_layout:
            if np.linalg.norm(np.asarray(label["point"]) - body) > 15:
                assert image.getpixel(tuple(round(value) for value in label["point"])) == ImageColor.getrgb(COLORS[label["type"]])
            crop = image.crop(label["rect"])
            assert ImageColor.getrgb(COLORS[label["type"]]) in crop.getdata()


def test_around_door_uses_its_single_point(dense_renderer):
    result = dense_renderer.render({"pos": [.5, .5], "heading": 0}, [], view={"around": ["D01"]})
    assert result["bounds"] == pytest.approx([.35, .26, .75, .66])
