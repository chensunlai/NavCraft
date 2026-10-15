"""Archived map calibration and Habitat heading conventions."""

import json
import math
from pathlib import Path

from PIL import Image
import pytest

from src.common.map import MapCalibration


@pytest.fixture
def archive(tmp_path):
    root = tmp_path / "scene_maps"
    scene = root / "hm3d" / "val" / "00001-fixture"
    (scene / "raw").mkdir(parents=True)
    (scene / "evidence").mkdir()
    Image.new("RGB", (200, 400), "white").save(scene / "floor_plan.png")
    Image.new("RGB", (50, 100), "blue").save(scene / "evidence_rooms.jpg")
    Image.new("RGB", (100, 50), "green").save(scene / "evidence" / "rgb.jpg")
    Image.new("RGB", (100, 50), "black").save(scene / "evidence" / "depth_001.png")
    Image.new("RGB", (100, 50), "black").save(scene / "raw" / "secret.png")
    (scene / "scene_description.md").write_text("One\nTwo\nThree\n")
    (scene / "raw" / "scene.json").write_text(
        json.dumps({"levels": [{"id": "L1", "height_m": 0, "bounds": [0, 10, 0, 20]}]})
    )
    (scene / "raw" / "regions.json").write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "id": "R1",
                        "level": "L1",
                        "polygon": [[0, 0], [10, 0], [10, 20], [0, 20]],
                    }
                ]
            }
        )
    )
    (scene / "raw" / "floor_plan.svg").write_text(
        """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 200"><g id="axes_1"><g id="region-R1"><path d="M 10 180 L 90 180 L 90 20 L 10 20 z" clip-path="url(#clip)" style="fill: blue"/></g></g><defs><clipPath id="clip"><rect x="10" y="20" width="80" height="160"/></clipPath></defs></svg>"""
    )
    return root, scene


def test_calibration_uses_archived_polygons_and_habitat_heading(archive):
    _, scene = archive
    calibration = MapCalibration(scene)
    level, position, heading = calibration.initial_arrow(
        {"position": [5, 0, -10], "rotation_xyzw": [0, 0, 0, 1]}
    )
    assert level == "L1" and position == pytest.approx((0.5, 0.5))
    assert heading[0] == position[0] and heading[1] < position[1]
    _, _, left_heading = calibration.initial_arrow(
        {
            "position": [5, 0, -10],
            "rotation_xyzw": [0, math.sin(math.pi / 4), 0, math.cos(math.pi / 4)],
        }
    )
    assert left_heading[0] < position[0]
    with pytest.raises(ValueError, match="outside"):
        calibration.initial_arrow(
            {"position": [500, 0, -10], "rotation_xyzw": [0, 0, 0, 1]}
        )


def test_checked_in_archives_calibrate_all_levels_without_scene_constants():
    root = Path(__file__).resolve().parents[1] / "data" / "scene_maps"
    descriptions = list(root.glob("*/*/*/scene_description.md"))
    if not descriptions:
        pytest.skip("External scene-map archives are not installed")
    for description in descriptions:
        calibration = MapCalibration(description.parent)
        for panel in calibration.panels:
            x_svg = (panel.clip[0] + panel.clip[2]) / 2
            y_svg = (panel.clip[1] + panel.clip[3]) / 2
            world_x = (x_svg - panel.x_offset) / panel.x_scale
            minus_z = (y_svg - panel.y_offset) / panel.y_scale
            level, origin, _ = calibration.initial_arrow(
                {
                    "position": [world_x, panel.height, -minus_z],
                    "rotation_xyzw": [0, 0, 0, 1],
                }
            )
            assert level == panel.level_id
            assert origin == pytest.approx(
                (x_svg / calibration.viewbox[2], y_svg / calibration.viewbox[3])
            )
