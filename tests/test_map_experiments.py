"""Model suggestions must not bypass raw-surface or traversability evidence."""
import numpy as np

from src.data.map_experiments import geometry_gate


def issue(kind, image="overlay.png", pixel=(10, 10)):
    return {"type": kind, "image": image, "pixel_hint": pixel,
            "confidence": .99, "reason": "a model hypothesis"}


def test_missing_wall_cannot_be_invented_without_measured_support():
    empty = np.zeros((25, 25), dtype=bool)
    result, decisions = geometry_gate(empty, empty, empty, [issue("missing_wall")],
        origin=[0, 0], side=1.25, allowed_images=["overlay.png"])
    assert not result.any()
    assert not decisions[0]["accepted"]


def test_supported_restoration_cannot_block_real_walkable_interior():
    walls = np.zeros((25, 25), dtype=bool)
    support, nav = walls.copy(), walls.copy()
    support[10, 4:18] = True
    nav[7:14, 7:14] = True
    result, decisions = geometry_gate(walls, support, nav, [issue("missing_wall")],
        origin=[0, 0], side=1.25, allowed_images=["overlay.png"])
    assert result[10, 5]
    assert not result[10, 10]
    assert decisions[0]["accepted"]


def test_furniture_label_alone_cannot_erase_geometry():
    walls = np.zeros((25, 25), dtype=bool)
    walls[10, 10] = True
    result, decisions = geometry_gate(walls, walls, np.zeros_like(walls), [issue("furniture")],
        origin=[0, 0], side=1.25, allowed_images=["overlay.png"])
    np.testing.assert_array_equal(result, walls)
    assert not decisions[0]["accepted"]


def test_perspective_coordinates_are_not_treated_as_map_pixels():
    walls = np.zeros((25, 25), dtype=bool)
    support = walls.copy(); support[10, 10] = True
    result, decisions = geometry_gate(walls, support, walls, [issue("missing_wall", "view.jpg")],
        origin=[0, 0], side=1.25, allowed_images=["overlay.png"])
    assert not result.any()
    assert decisions[0]["reason"].startswith("perspective_point")
