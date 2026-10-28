from unittest.mock import patch

import pytest
from PIL import Image, ImageDraw

from src.harness.observations import Observations


@pytest.fixture
def observations(tmp_path):
    observations = Observations(tmp_path, {"origin": [0, 0], "size_m": 10})
    Image.new("RGB", (640, 480), (90, 100, 110)).save(tmp_path / "observations/source.png")
    return observations


def test_panorama_has_frame_labels_without_pixel_grid(observations):
    source = observations.workspace / "observations/source.png"
    original = source.read_bytes()
    frames = [observations.add(source, yaw=i * 60)["frame"] for i in range(6)]
    texts = []
    original_text = ImageDraw.ImageDraw.text

    def record(draw, xy, text, *args, **kwargs):
        texts.append(text)
        return original_text(draw, xy, text, *args, **kwargs)

    with patch.object(ImageDraw.ImageDraw, "text", record):
        grid = observations.grid(frames)
    assert texts == [f"view {i + 1} | right {i * 60} deg | detail: {frame}"
                     for i, frame in enumerate(frames)]
    with Image.open(observations.workspace / grid["path"]) as image:
        assert image.size == (1920, 1012)
    assert source.read_bytes() == original
    assert observations.resolve_pixel(grid["image"], [650, 556]) == (frames[4], [10, 24])
    with pytest.raises(ValueError, match="label_or_padding"):
        observations.resolve_pixel(grid["image"], [650, 510])


def test_crop_grid_roundtrip_keeps_original_frame_coordinates(observations):
    source = observations.workspace / "observations/source.png"
    frame = observations.add(source)["frame"]
    cropped = observations.crop(frame, [.25, .25, .75, .75])
    assert cropped["size"] == [320, 240]
    assert observations.resolve_pixel(cropped["image"], [40, 60]) == (frame, [200, 180])
    grid = observations.grid([cropped["image"], frame])
    assert observations.resolve_pixel(grid["image"], [40, 86]) == (frame, [200, 180])
    assert observations.resolve_pixel(grid["image"], [740, 126]) == (frame, [100, 100])
    with pytest.raises(ValueError, match="label_or_padding"):
        observations.resolve_pixel(grid["image"], [400, 100])
    restored = Observations(observations.workspace, observations.transform)
    assert restored.resolve_pixel(grid["image"], [40, 86]) == (frame, [200, 180])


@pytest.mark.parametrize("pixel", [[-1, 0], [640, 0], [0, 480]])
def test_invalid_image_pixels_are_rejected(observations, pixel):
    frame = observations.add(observations.workspace / "observations/source.png")["frame"]
    with pytest.raises(ValueError, match="pixel_out_of_bounds"):
        observations.resolve_pixel(frame, pixel)


@pytest.mark.parametrize("pixel", [[float("nan"), 0], [True, 0], [float("inf"), 0]])
def test_non_numeric_pixels_are_rejected(observations, pixel):
    frame = observations.add(observations.workspace / "observations/source.png")["frame"]
    with pytest.raises(ValueError, match="finite"):
        observations.resolve_pixel(frame, pixel)


def test_sizes_and_errors_follow_the_selected_image_not_its_source(observations):
    frame = observations.add(observations.workspace / "observations/source.png")["frame"]
    cropped = observations.crop(frame, [.25, .25, .75, .75])
    grid = observations.grid([cropped["image"], frame])
    assert observations.image(frame)["size"] == [640, 480]
    assert observations.image(cropped["image"])["size"] == [320, 240]
    assert observations.image(grid["image"])["size"] == [1280, 506]
    with pytest.raises(ValueError) as error:
        observations.resolve_pixel(cropped["image"], [320, 10])
    assert str(error.value) == (f"pixel_out_of_bounds: image={cropped['image']}; size=320x240; "
                                "valid x:[0,320), y:[0,240)")
    with pytest.raises(ValueError, match=f"image={grid['image']}; size=1280x506"):
        observations.resolve_pixel(grid["image"], [1280, 10])
