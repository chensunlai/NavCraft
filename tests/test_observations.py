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
    captions = [f"f_{i:04d} | right {(i - 1) * 60} deg" for i in range(1, 7)]
    texts = []
    original_text = ImageDraw.ImageDraw.text

    def record(draw, xy, text, *args, **kwargs):
        texts.append(text)
        return original_text(draw, xy, text, *args, **kwargs)

    with patch.object(ImageDraw.ImageDraw, "text", record):
        observations.collage(["observations/source.png"] * 6, "observations/panorama.png", captions)
    assert texts == captions
    with Image.open(observations.workspace / "observations/panorama.png") as image:
        assert image.size == (1920, 1012)
        assert image.getpixel((80, 226)) == (90, 100, 110)
    assert source.read_bytes() == original


def test_collage_preserves_crop_pixels_and_rejects_mismatched_captions(observations):
    root = observations.workspace
    with Image.open(root / "observations/source.png") as image:
        image.crop((160, 80, 400, 320)).save(root / "observations/crop.png")
    observations.collage(["observations/crop.png"], "observations/collage.png", ["f_0001 crop"])
    with Image.open(root / "observations/collage.png") as image:
        assert image.getpixel((80, 226)) == (90, 100, 110)
    with pytest.raises(ValueError, match="matching captions"):
        observations.collage(["observations/source.png"], "observations/invalid.png", [])
