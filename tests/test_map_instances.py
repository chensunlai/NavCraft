"""Semantic IDs and surfaces stay priors, not independent geometry labels."""

import hashlib
import json
import struct

import numpy as np
import pytest

from src.data.map_instances import _category, load_semantic_instances


def glb(path, triangles, colors):
    vertices = np.asarray(triangles, dtype="<f4").reshape(-1, 3)
    rgb = np.repeat(np.asarray(colors, dtype="<f4"), 3, axis=0)
    indices = np.arange(len(vertices), dtype="<u4")
    binary = vertices.tobytes() + rgb.tobytes() + indices.tobytes()
    views = [{"buffer": 0, "byteOffset": offset, "byteLength": length} for offset, length in
             [(0, vertices.nbytes), (vertices.nbytes, rgb.nbytes), (vertices.nbytes + rgb.nbytes, indices.nbytes)]]
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
           "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "indices": 2}]}],
           "buffers": [{"byteLength": len(binary)}], "bufferViews": views,
           "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(vertices), "type": "VEC3"},
                         {"bufferView": 1, "componentType": 5126, "count": len(vertices), "type": "VEC3"},
                         {"bufferView": 2, "componentType": 5125, "count": len(indices), "type": "SCALAR"}]}
    text = json.dumps(doc).encode()
    text += b" " * (-len(text) % 4)
    binary += b"\0" * (-len(binary) % 4)
    body = struct.pack("<II", len(text), 0x4E4F534A) + text + struct.pack("<II", len(binary), 0x004E4942) + binary
    path.write_bytes(struct.pack("<III", 0x46546C67, 2, len(body) + 12) + body)


@pytest.fixture
def hm3d(tmp_path):
    scene = tmp_path / "test.basis.glb"
    triangles = np.array([[[0, 0, 0], [1, 0, 0], [1, 0, 2]],
                          [[2, 0, 0], [3, 0, 0], [3, 0, 1]],
                          [[4, 0, 0], [5, 0, 0], [5, 0, 1]]])
    scene.write_text("independent raw render mesh is deliberately not loaded by a prior reader")
    glb(tmp_path / "test.semantic.glb", triangles, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    (tmp_path / "test.semantic.txt").write_text(
        'HM3D Semantic Annotations\n17,FF0000,"wall",3\n42,00FF00,"wall cabinet",8\n90,0000FF,"wallpaper",-1\n')
    return scene


def test_exact_furniture_taxonomy_does_not_invent_architecture():
    assert _category("wall cabinet") == _category("wallcabinet") == "furniture"
    assert _category("wall#cabinet", "wall") == "furniture"
    assert _category("floor_lamp", "floor") == "furniture"
    assert _category("ceiling lamp") == "furniture"
    assert _category("wallpaper") == "unknown"
    assert _category("window handle") == "unknown"
    assert _category("door frame") == "door"
    assert _category("floor") == "floor"
    assert _category("ceiling") == "ceiling"


def test_hm3d_instance_color_region_and_semantic_face_ids(hm3d):
    instances, meta = load_semantic_instances(hm3d)
    assert [instance.id for instance in instances] == [17, 42, 90]
    assert [instance.category for instance in instances] == ["wall", "furniture", "unknown"]
    assert [instance.region_id for instance in instances] == [3, 8, None]
    np.testing.assert_array_equal(instances[0].triangles, [[[0, 0, 0], [1, 0, 0], [1, 2, 0]]])
    np.testing.assert_array_equal(instances[1].triangle_ids, [[0, 1]])
    assert instances[1].source["color_hex"] == "00FF00"
    assert instances[1].source["annotation_instance_id"] == 42
    assert meta["geometry_acceptance"] == "not_performed"
    assert meta["role"] == "semantic_surface_priors_not_independent_raw_geometry"
    assert meta["semantic_mesh_triangle_count"] == meta["returned_triangle_count"] == 3
    assert meta["mixed_color_triangle_count"] == meta["unassigned_triangle_count"] == 0
    assert meta["sources"][0]["sha256"] == hashlib.sha256(hm3d.with_name("test.semantic.glb").read_bytes()).hexdigest()


def test_hm3d_filter_preserves_original_ids_not_reindexed(hm3d):
    instances, meta = load_semantic_instances(hm3d, wanted_categories={"furniture"})
    assert len(instances) == 1 and instances[0].id == 42
    np.testing.assert_array_equal(instances[0].triangle_ids, [[0, 1]])
    assert meta["annotated_instance_count"] == 3
    assert meta["returned_instance_count"] == meta["returned_triangle_count"] == 1
    assert meta["wanted_categories"] == ["furniture"]
    assert load_semantic_instances(hm3d, wanted_categories=[])[0] == []
    with pytest.raises(ValueError, match="Unknown wanted"):
        load_semantic_instances(hm3d, wanted_categories={"sofa"})


def test_hm3d_duplicate_annotation_ids_do_not_silently_overwrite(hm3d):
    hm3d.with_name("test.semantic.txt").write_text('1,FF0000,"wall",0\n1,00FF00,"door",0\n')
    with pytest.raises(ValueError, match="unique"):
        load_semantic_instances(hm3d)


def test_mp3d_category_object_region_and_original_global_face_ids(tmp_path):
    from plyfile import PlyData, PlyElement

    scene = tmp_path / "test.glb"
    scene.write_text("independent render mesh reference; not used as semantic evidence")
    vertices = np.array([(0, 0, 0), (1, 0, 0), (1, 0, 2),
                         (2, 0, 0), (3, 0, 0), (3, 0, 1),
                         (4, 0, 0), (5, 0, 0), (5, 0, 1)],
                        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    faces = np.array([(np.array([0, 1, 2], dtype="i4"), 123),
                      (np.array([3, 4, 5], dtype="i4"), 456),
                      (np.array([6, 7, 8], dtype="i4"), 789)],
                     dtype=[("vertex_indices", "O"), ("object_id", "i4")])
    PlyData([PlyElement.describe(vertices, "vertex"), PlyElement.describe(faces, "face")], text=True).write(
        tmp_path / "test_semantic.ply")
    (tmp_path / "test.house").write_text(
        "ASCII 1.1\nC 2 3 wall 1 wall\nC 4 5 wall#cabinet 1 wall\nC 9 10 unknown 40 misc\n"
        "O 123 11 2\nO 456 22 4\nO 789 -1 9\n")
    instances, meta = load_semantic_instances(scene)
    assert [instance.id for instance in instances] == [123, 456, 789]
    assert [instance.category for instance in instances] == ["wall", "furniture", "unknown"]
    assert [instance.region_id for instance in instances] == [11, 22, None]
    assert instances[1].label == "wall#cabinet"
    assert instances[1].source["category_index"] == 4
    assert instances[1].source["canonical_label"] == "wall"
    assert instances[1].source["object_id"] == 456
    np.testing.assert_array_equal(instances[0].triangles, [[[0, 0, 0], [1, 0, 0], [1, 2, 0]]])
    np.testing.assert_array_equal(instances[2].triangle_ids, [[0, 2]])
    assert meta["category_instance_counts"]["furniture"] == 1
    filtered, _ = load_semantic_instances(scene, wanted_categories={"wall"})
    assert len(filtered) == 1 and filtered[0].id == 123
