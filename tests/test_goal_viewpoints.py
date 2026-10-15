"""Local fixtures for semantic goal geometry and exact NavMesh/grid matching."""

import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from src.dataset.qa_campaign.viewpoints import horizontal_distance, semantic_geometry, validate_grid


def semantic_fixture(tmp_path, *, normalized=False, transformed=False, sparse=False):
    """Make a GLB with prefix/vertex padding and a differently labelled distractor."""
    target_srgb = np.array([255, 0, 0] if normalized else [32, 128, 224]) / 255
    target_linear = np.where(target_srgb <= .04045, target_srgb / 12.92,
                             ((target_srgb + .055) / 1.055) ** 2.4)
    binary = bytearray(8 + 3 * 32)
    for index, point in enumerate([[2, 3, 4], [4, 5, 6], [999, 999, 999]]):
        offset = 8 + index * 32
        struct.pack_into("<3f", binary, offset + 4, *point)
        color = target_linear if index < 2 else np.array([0, 1, 0])
        if normalized:
            struct.pack_into("<4B", binary, offset + 16, *np.rint(color * 255).astype(int), 255)
        else:
            struct.pack_into("<4f", binary, offset + 16, *color, 1)
    doc = {
        "asset": {"version": "2.0"}, "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 8, "byteLength": 96, "byteStride": 32}],
        "accessors": [
            {"bufferView": 0, "byteOffset": 4, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 0, "byteOffset": 16, "componentType": 5121 if normalized else 5126,
             "count": 3, "type": "VEC4", **({"normalized": True} if normalized else {})},
        ],
        "nodes": [{"mesh": 0, **({"translation": [1, 0, 0]} if transformed else {})}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}}]}],
    }
    if sparse:
        doc["accessors"][0]["sparse"] = {"count": 0}
    chunk = json.dumps(doc).encode()
    chunk += b" " * (-len(chunk) % 4)
    glb = (struct.pack("<III", 0x46546C67, 2, 28 + len(chunk) + len(binary))
           + struct.pack("<II", len(chunk), 0x4E4F534A) + chunk
           + struct.pack("<II", len(binary), 0x004E4942) + binary)
    mesh, labels = tmp_path / "semantic.glb", tmp_path / "semantic.txt"
    mesh.write_bytes(glb)
    labels.write_text('HM3D Semantic Annotations\n99,'
                      + ('ff0000' if normalized else '2080e0')
                      + ',"bathtub",3\n100,00ff00,"distractor",4\n101,ffffff,"missing",3\n')
    return mesh, labels


@pytest.mark.parametrize("normalized", [False, True])
def test_semantic_geometry_matches_instance_color_and_habitat_axes(tmp_path, normalized):
    mesh, labels = semantic_fixture(tmp_path, normalized=normalized)
    result = semantic_geometry(mesh, labels, [99])[99]
    # Far-away differently coloured vertices and the zero-filled padding must not
    # affect the physical object bounds; Habitat uses X, Z, -Y asset coordinates.
    assert result["decoded_vertex_count"] == 2
    assert result["aabb_min_xyz"] == [2, 4, -5]
    assert result["aabb_max_xyz"] == [4, 6, -3]
    assert result["center_xyz"] == result["aim_xyz"] == [3, 5, -4]
    assert result["label"] == "bathtub" and result["semantic_region_id"] == 3


@pytest.mark.parametrize("option,message", [
    ("transformed", "Transformed nodes"), ("sparse", "Sparse accessors"),
])
def test_unsupported_geometry_fails_instead_of_using_wrong_coordinates(tmp_path, option, message):
    mesh, labels = semantic_fixture(tmp_path, **{option: True})
    with pytest.raises(ValueError, match=message):
        semantic_geometry(mesh, labels, [99])


def test_missing_semantic_instance_does_not_create_origin_box(tmp_path):
    mesh, labels = semantic_fixture(tmp_path)
    with pytest.raises(ValueError, match="No decoded semantic vertices for 101"):
        semantic_geometry(mesh, labels, [101])


def pathfinder(**overrides):
    settings = dict(agent_height=.88, agent_radius=.18, cell_height=.05, agent_max_climb=.2)
    settings.update(overrides)
    return SimpleNamespace(nav_mesh_settings=SimpleNamespace(**settings),
                           snap_point=lambda point: np.asarray(point).copy())


def test_grid_accepts_matching_geometry_without_replacing_points():
    points = np.array([[1, .035739, 2], [3, .035739, 4]], dtype=np.float32)
    original = points.copy()
    report = validate_grid(pathfinder(), points)
    assert report["grid_points_checked"] == 2 and report["maximum_snap_error_m"] == 0
    np.testing.assert_array_equal(points, original)


@pytest.mark.parametrize("setting,value", [
    ("agent_height", 1.5), ("agent_radius", .3), ("cell_height", .2), ("agent_max_climb", .4),
])
def test_grid_rejects_incompatible_robot_or_navmesh(setting, value):
    with pytest.raises(ValueError, match="disagrees with declared"):
        validate_grid(pathfinder(**{setting: value}), [[1, 0, 2]])


@pytest.mark.parametrize("offset", [.001, float("nan")])
def test_grid_rejects_unsnapped_or_unreachable_points(offset):
    pf = pathfinder()
    pf.snap_point = lambda point: np.asarray(point) + [0, offset, 0]
    with pytest.raises(ValueError, match="not on this NavMesh"):
        validate_grid(pf, [[1, 0, 2]])


@pytest.mark.parametrize("points", [[], [1, 2, 3], [[1, 2]], [[1, 2, 3, 4]]])
def test_grid_requires_nonempty_xyz_samples(points):
    with pytest.raises(ValueError, match="nonempty XYZ"):
        validate_grid(pathfinder(), points)


def test_horizontal_proximity_uses_surface_extent_not_object_center():
    bounds = {"aabb_min_xyz": [2, 2, -5], "aabb_max_xyz": [4, 4, -3]}
    assert horizontal_distance([3, 0, -4], bounds) == 0
    assert horizontal_distance([5, 100, -4], bounds) == 1
    assert horizontal_distance([5, 0, -2], bounds) == pytest.approx(2 ** .5)
