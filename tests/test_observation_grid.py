from unittest.mock import patch

import pytest
from PIL import Image, ImageDraw

from src.harness.observations import Observations


@pytest.fixture
def observations(tmp_path):
    observations = Observations(tmp_path, {"origin": [0, 0], "size_m": 10})
    Image.new("RGB", (640, 480), (90, 100, 110)).save(tmp_path / "observations/source.png")
    return observations


def test_pixel_grid_is_a_same_size_derived_copy(observations):
    source = observations.workspace / "observations/source.png"
    original = source.read_bytes()
    destination = "observations/derived/grid.png"
    assert observations.grid_image("observations/source.png", destination) == destination
    with Image.open(observations.workspace / destination) as image:
        assert image.size == (640, 480)
        assert image.getpixel((80, 200)) != (90, 100, 110)
        assert image.getpixel((40, 40)) == (90, 100, 110)
    assert source.read_bytes() == original
    with pytest.raises(ValueError, match="new derived"):
        observations.grid_image("observations/source.png", "observations/source.png")
    assert source.read_bytes() == original


def test_cropped_grid_and_collage_labels_keep_each_frames_source_coordinates(observations):
    root = observations.workspace
    with Image.open(root / "observations/source.png") as image:
        image.crop((160, 80, 400, 320)).save(root / "observations/crop.png")
    texts = []
    original_text = ImageDraw.ImageDraw.text

    def record(draw, xy, text, *args, **kwargs):
        texts.append(text)
        return original_text(draw, xy, text, *args, **kwargs)

    with patch.object(ImageDraw.ImageDraw, "text", record):
        observations.grid_image("observations/crop.png", "observations/derived/crop-grid.png",
                                source_bounds=[160, 80, 400, 320])
    assert "x=160" in texts and "x=240" in texts and "y=80" in texts
    assert "x=0" not in texts
    with Image.open(root / "observations/derived/crop-grid.png") as image:
        assert image.size == (240, 240)
    texts.clear()
    with patch.object(ImageDraw.ImageDraw, "text", record):
        observations.collage(["observations/source.png"] * 2, "observations/panorama.png",
                             ["f_0001 | right 0 deg", "f_0002 | right 60 deg"], grid=True)
    assert texts.count("x=0") == 2
    assert "x=640" not in texts
    assert all("grid=source px" in text for text in texts if "f_000" in text)
    with Image.open(root / "observations/panorama.png") as image:
        assert image.size == (1280, 506)


def test_collage_without_grid_preserves_the_existing_layout(observations):
    observations.collage(["observations/source.png"] * 6, "observations/plain.png", ["frame"] * 6)
    with Image.open(observations.workspace / "observations/plain.png") as image:
        assert image.size == (1920, 1012)
        assert image.getpixel((80, 226)) == (90, 100, 110)
