"""Actor resource boundaries, archived map calibration and map-coordinate edits."""

import hashlib
import json
import math
from pathlib import Path

from PIL import Image
import pytest

from src.harness.assets import AssetError, SceneAssets
from src.harness.map import MapCalibration


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


def assets(archive, tmp_path):
    root, _ = archive
    return SceneAssets(
        "hm3d//val/00001-fixture/fixture.basis.glb", root, tmp_path / "run"
    )


@pytest.mark.parametrize(
    "ref",
    [
        "raw/scene.json",
        "raw/secret.png",
        "evidence/depth_001.png",
        "evidence/frame.npz",
        "../secret.png",
        "/etc/passwd",
        "evidence/../floor_plan.png",
        "evidence\\rgb.jpg",
    ],
)
def test_only_public_rgb_resources_are_readable(archive, tmp_path, ref):
    memory = assets(archive, tmp_path)
    with pytest.raises(AssetError):
        memory.path_for_image(ref)
    assert memory.image_info("evidence/rgb.jpg") == {
        "ref": "evidence/rgb.jpg",
        "width": 100,
        "height": 50,
        "kind": "evidence",
    }


def test_symlinks_cannot_expose_raw_or_external_files(archive, tmp_path):
    _, scene = archive
    memory = assets(archive, tmp_path)
    (scene / "evidence" / "raw_alias.jpg").symlink_to(scene / "raw" / "secret.png")
    external = tmp_path / "secret.jpg"
    Image.new("RGB", (10, 10)).save(external)
    (scene / "evidence" / "external.jpg").symlink_to(external)
    for ref in ("evidence/raw_alias.jpg", "evidence/external.jpg"):
        with pytest.raises(AssetError):
            memory.image_info(ref)


def test_text_pagination_and_generated_summary_are_public_not_paths(archive, tmp_path):
    memory = assets(archive, tmp_path)
    result = memory.read_text("scene_description.md", start_line=2, max_lines=1)
    assert result["text"] == "2: Two" and result["next_line"] == 3
    with pytest.raises(AssetError):
        memory.read_text("raw/scene.json")
    ref = memory.write_summary("Goal: chair; localization uncertain; check frame_0002")
    assert "localization uncertain" in memory.read_text(ref)["text"]
    assert str(tmp_path) not in json.dumps(result)


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


def test_crop_marker_maps_to_full_image_and_preserves_original(archive, tmp_path):
    _, scene = archive
    memory = assets(archive, tmp_path)
    checksum = hashlib.sha256((scene / "floor_plan.png").read_bytes()).hexdigest()
    initial, _ = memory.initial_map(
        {"position": [5, 0, -10], "rotation_xyzw": [0, 0, 0, 1]}
    )
    result, _ = memory.view_images(
        [initial["map_ref"]], crops=[[0.25, 0.25, 0.75, 0.75]]
    )
    crop_ref = result["images"][0]["ref"]
    marked, paths = memory.mark_pose(
        crop_ref, [0.5, 0.5], [0.5, 0.25], "matched landmark"
    )
    assert marked["map_ref"] != initial["map_ref"]
    assert memory.latest_map_ref == marked["map_ref"]
    assert marked["marker_kind"] == "Model estimate"
    with Image.open(paths[0]) as picture:
        assert picture.size == (200, 400)
        assert picture.getpixel((100, 200)) == (224, 32, 32)
        # Prior true-location arrow is not accumulated on the new base map.
        assert picture.getpixel((100, 245)) == (255, 255, 255)
    assert (
        checksum == hashlib.sha256((scene / "floor_plan.png").read_bytes()).hexdigest()
    )
    assert "position" not in initial and "rotation_xyzw" not in initial


def test_runtime_images_grids_and_crop_provenance(archive, tmp_path):
    memory = assets(archive, tmp_path)
    image = tmp_path / "run" / "rgb_001.png"
    Image.new("RGB", (80, 40), "red").save(image)
    memory.register_image(image, "frame_001")
    grid, paths = memory.view_images(["frame_001", "evidence/rgb.jpg"], layout="grid")
    assert len(paths) == 1 and len(grid["sources"]) == 2
    assert grid["sources"][0]["source_ref"] == "frame_001"
    uncropped, uncropped_paths = memory.view_images(["frame_001"], crops=[])
    assert uncropped["images"][0]["ref"] == "frame_001"
    assert uncropped_paths == [image.resolve()]
    with pytest.raises(AssetError, match="map image"):
        memory.mark_pose(grid["images"][0]["ref"], [0.2, 0.2], [0.3, 0.2], "")
    with pytest.raises(AssetError):
        memory.view_images(["frame_001"], crops=[[0.5, 0.1, 0.2, 0.9]])


def test_ten_image_view_and_contact_sheet(archive, tmp_path):
    root, _ = archive
    memory = SceneAssets("fixture", root, tmp_path / "run-ten")
    refs = ["evidence/rgb.jpg"] * 10
    result, paths = memory.view_images(refs, layout="separate")
    assert len(paths) == len(result["images"]) == 10
    result, paths = memory.view_images(refs, layout="grid")
    assert len(paths) == 1
    with Image.open(paths[0]) as grid:
        assert grid.height == 4 * (420 + 34)
    with pytest.raises(AssetError):
        memory.view_images(refs + [refs[0]])
    with pytest.raises(AssetError):
        memory.mark_pose("floor_plan.png", [float("nan"), 0.5], [0.5, 0.5], "")


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
