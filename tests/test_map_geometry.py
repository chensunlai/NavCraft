"""Geometry-first map candidates: raw transforms, cuts and semantic separation."""

import json
import struct

import numpy as np
import pytest

from src.data.map_geometry import (
    height_projection,
    load_scene_geometry,
    multiheight_slices,
    slice_triangles,
    to_habitat,
    vertical_triangles,
)


def write_glb(path, vertices, *, nodes=None, colors=None):
    vertices = np.asarray(vertices, dtype="<f4")
    indices = np.arange(len(vertices), dtype="<u4")
    binary = vertices.tobytes() + indices.tobytes()
    views = [{"buffer": 0, "byteOffset": 0, "byteLength": vertices.nbytes},
             {"buffer": 0, "byteOffset": vertices.nbytes, "byteLength": indices.nbytes}]
    accessors = [{"bufferView": 0, "componentType": 5126, "count": len(vertices), "type": "VEC3"},
                 {"bufferView": 1, "componentType": 5125, "count": len(indices), "type": "SCALAR"}]
    attributes = {"POSITION": 0}
    if colors is not None:
        colors = np.asarray(colors, dtype="<f4")
        attributes["COLOR_0"] = len(accessors)
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": colors.nbytes})
        accessors.append({"bufferView": 2, "componentType": 5126, "count": len(colors), "type": "VEC3"})
        binary += colors.tobytes()
    nodes = nodes or [{"mesh": 0}]
    child_ids = {child for node in nodes for child in node.get("children", [])}
    document = {"asset": {"version": "2.0"}, "scene": 0,
                "scenes": [{"nodes": [index for index in range(len(nodes)) if index not in child_ids]}],
                "nodes": nodes, "meshes": [{"primitives": [{"attributes": attributes, "indices": 1}]}],
                "buffers": [{"byteLength": len(binary)}], "bufferViews": views, "accessors": accessors}
    text = json.dumps(document).encode()
    text += b" " * (-len(text) % 4)
    binary += b"\x00" * (-len(binary) % 4)
    body = struct.pack("<II", len(text), 0x4E4F534A) + text + struct.pack("<II", len(binary), 0x004E4942) + binary
    path.write_bytes(struct.pack("<III", 0x46546C67, 2, len(body) + 12) + body)


def rectangle(x0, x1, y0, y1, z=0):
    return np.array([[[x0, y0, z], [x1, y0, z], [x1, y1, z]],
                     [[x0, y0, z], [x1, y1, z], [x0, y1, z]]], dtype=float)


def test_asset_coordinate_transform_preserves_habitat_axes():
    np.testing.assert_array_equal(to_habitat([[1, 2, 3], [-4, -5, -6]]),
                                  [[1, 3, -2], [-4, -6, 5]])


@pytest.mark.parametrize("filename", ["scene.basis.glb", "scene.glb"])
def test_raw_hm3d_and_mp3d_load_without_any_semantic_labels(tmp_path, filename):
    scene = tmp_path / filename
    vertices = [[0, 0, 0], [2, 0, 0], [2, 0, 3]]
    write_glb(scene, vertices)
    result = load_scene_geometry(scene)
    np.testing.assert_allclose(result.triangles, to_habitat([vertices]))
    assert result.categories == {}
    assert result.metadata["geometry_triangles"] == 1
    assert result.metadata["semantic_sources"] == []


def test_node_hierarchy_transforms_before_world_conversion(tmp_path):
    scene = tmp_path / "scene.glb"
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    nodes = [{"translation": [3, 4, 5], "children": [1]},
             {"mesh": 0, "scale": [2, 3, 4], "translation": [1, 0, 0]}]
    write_glb(scene, vertices, nodes=nodes)
    result = load_scene_geometry(scene, include_semantics=False)
    expected = vertices * [2, 3, 4] + [4, 4, 5]
    np.testing.assert_allclose(result.triangles[0], to_habitat(expected))


def test_mesh_instances_are_not_silently_dropped(tmp_path):
    scene = tmp_path / "scene.glb"
    write_glb(scene, [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
              nodes=[{"mesh": 0}, {"mesh": 0, "translation": [10, 0, 0]}])
    result = load_scene_geometry(scene, include_semantics=False)
    assert result.triangles.shape == (2, 3, 3)
    np.testing.assert_allclose(result.triangles[1] - result.triangles[0], [[10, 0, 0]] * 3)


def test_node_matrix_uses_gltf_column_major_order(tmp_path):
    scene = tmp_path / "scene.glb"
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    matrix = np.eye(4)
    matrix[:3, 3] = [3, 4, 5]
    write_glb(scene, vertices, nodes=[{"mesh": 0, "matrix": matrix.T.reshape(-1).tolist()}])
    result = load_scene_geometry(scene, include_semantics=False)
    np.testing.assert_allclose(result.triangles[0], to_habitat(vertices + [3, 4, 5]))


def test_node_quaternion_rotation_precedes_y_up_conversion(tmp_path):
    scene = tmp_path / "scene.glb"
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    write_glb(scene, vertices, nodes=[{"mesh": 0, "rotation": [0, 0, 2 ** -.5, 2 ** -.5]}])
    result = load_scene_geometry(scene, include_semantics=False)
    np.testing.assert_allclose(result.triangles[0], to_habitat([[0, 0, 0], [0, 1, 0], [-1, 0, 0]]), atol=1e-7)


def test_hm3d_semantic_priors_are_separate_from_full_geometry(tmp_path):
    scene = tmp_path / "scene.basis.glb"
    raw = [[0, 0, 0], [2, 0, 0], [2, 0, 3], [3, 0, 0], [4, 0, 0], [4, 0, 3]]
    write_glb(scene, raw)
    write_glb(tmp_path / "scene.semantic.glb", raw[:3], colors=[[1, 0, 0]] * 3)
    (tmp_path / "scene.semantic.txt").write_text('HM3D Semantic Annotations\n1,FF0000,"door",1\n')
    result = load_scene_geometry(scene)
    assert len(result.triangles) == 2
    assert set(result.categories) == {"door"}
    np.testing.assert_allclose(result.categories["door"], result.triangles[:1])


def test_mp3d_semantic_priors_are_separate_from_full_geometry(tmp_path):
    from plyfile import PlyData, PlyElement

    scene = tmp_path / "scene.glb"
    raw = [[0, 0, 0], [2, 0, 0], [2, 0, 3], [3, 0, 0], [4, 0, 0], [4, 0, 3]]
    write_glb(scene, raw)
    vertices = np.array([tuple(point) for point in raw], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    faces = np.array([(np.array([0, 1, 2], dtype="i4"), 7), (np.array([3, 4, 5], dtype="i4"), 8)],
                     dtype=[("vertex_indices", "O"), ("object_id", "i4")])
    PlyData([PlyElement.describe(vertices, "vertex"), PlyElement.describe(faces, "face")], text=True).write(
        tmp_path / "scene_semantic.ply")
    (tmp_path / "scene.house").write_text("ASCII 1.1\nC 1 2 raw-wall 1 wall\nC 2 3 object 40 misc\nO 7 0 1\nO 8 0 2\n")
    result = load_scene_geometry(scene)
    assert len(result.triangles) == 2
    assert set(result.categories) == {"wall"}
    np.testing.assert_allclose(result.categories["wall"], result.triangles[:1])


def test_closed_door_survives_slices_while_overhead_lintel_and_low_furniture_do_not():
    closed_door = rectangle(-.5, .5, 0, 2.2)
    lintel = rectangle(-.7, .7, 2.1, 2.5)
    low_furniture = rectangle(2, 3, 0, .85)
    for segments in multiheight_slices(closed_door, 0).values():
        assert len(segments) == 2
        assert np.max(np.abs(segments[:, :, 0])) <= .5
    assert not any(len(value) for value in multiheight_slices(lintel, 0).values())
    assert not any(len(value) for value in multiheight_slices(low_furniture, 0).values())


def test_tall_furniture_is_a_geometry_candidate_not_misrepresented_as_semantic_wall():
    cabinet_side = rectangle(2, 3, 0, 2.1)
    assert all(len(value) for value in multiheight_slices(cabinet_side, 0).values())
    # Geometry alone cannot distinguish an upright cabinet from a short wall.
    # The pipeline must quantify this ambiguity, not silently filter it by name.
    assert len(vertical_triangles(cabinet_side)) == 2


def test_slice_handles_plane_through_vertices_without_zero_segments():
    triangle = np.array([[[0, 0, 0], [2, 1, 0], [0, 1, 0]]], dtype=float)
    segments = slice_triangles(triangle, 1)
    assert segments.shape == (1, 2, 2)
    assert np.linalg.norm(segments[0, 1] - segments[0, 0]) == pytest.approx(2)
    assert len(slice_triangles(triangle, 0)) == 0
    assert len(slice_triangles(np.array([[[0, 1, 0], [1, 1, 0], [0, 1, 1]]]), 1)) == 0


def test_vertical_filter_and_slab_projection_do_not_include_ceiling_or_lintel():
    wall = rectangle(0, 2, 0, 2.5)
    ceiling = np.array([[[0, 2.5, 0], [1, 2.5, 0], [0, 2.5, 1]]])
    lintel = rectangle(3, 4, 2.1, 2.5)
    full = np.concatenate([wall, ceiling, lintel])
    assert len(vertical_triangles(full)) == 4
    projected = height_projection(full, 0, .9, 2.0)
    assert projected.shape[1:] == (3, 2)
    assert len(projected) > 0
    assert projected[:, :, 0].min() >= 0 and projected[:, :, 0].max() <= 2


def test_invalid_geometry_and_heights_are_explicit():
    with pytest.raises(ValueError, match="finite"):
        slice_triangles(np.full((1, 3, 3), np.nan), 1)
    with pytest.raises(ValueError, match="finite"):
        multiheight_slices(rectangle(0, 1, 0, 2), float("nan"))
    with pytest.raises(ValueError, match="lower < upper"):
        height_projection(rectangle(0, 1, 0, 2), 0, 2, 1)
