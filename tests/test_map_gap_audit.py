import numpy as np

from src.data.map_gap_audit import line_gap_candidates, clear_gap_interval, same_opening


def line(identifier, start, end):
    return {"id": identifier, "type": "line", "start": start, "end": end}


def test_finds_only_a_collinear_endpoint_gap():
    result = line_gap_candidates([
        line("left", [0, 0], [2, 0]),
        line("right", [3, 0], [5, 0]),
        line("parallel", [0, .3], [2, .3]),
    ])
    assert len(result) == 1
    assert result[0]["primitives"] == ["left", "right"]
    assert abs(result[0]["width_m"] - 1) < 1e-8


def test_does_not_bridge_a_large_gap_or_a_perpendicular_corner():
    result = line_gap_candidates([
        line("large", [0, 0], [1, 0]), line("far", [3, 0], [4, 0]),
        line("vertical", [2, 0], [2, 2]),
    ])
    assert result == []


def test_scattered_clear_samples_do_not_claim_one_large_opening():
    class Evidence:
        trees = {.3: None, .6: None, .9: None, 1.2: None, 1.5: None}

        def profile(self, points):
            blocked = (np.arange(len(points)) // 4) % 2 == 0
            return np.tile(blocked[:, None], (1, 5))

    result = clear_gap_interval([[0, 0], [1, 0]], Evidence())
    assert result["raw_clear_body_fraction"] > .4
    assert result["clear_width_m"] < .15


def test_duplicate_openings_require_same_plane_and_overlap():
    portal = {"segment": [[0, 0], [1, 0]]}
    assert same_opening(portal, {"segment": [[.1, 0], [.9, 0]]})
    assert not same_opening(portal, {"segment": [[1.2, 0], [2.2, 0]]})
    assert not same_opening(portal, {"segment": [[0, 1], [1, 1]]})


def test_compressed_depth_preserves_values_and_updates_reference(tmp_path):
    from src.data.map_gap_audit import _compact_view
    depth = np.arange(12, dtype=np.float32).reshape(3, 4) / 7
    path = tmp_path / 'view.npy'
    np.save(path, depth)
    result = _compact_view({"depth": str(path), "rgb": "view.jpg"})
    assert not path.exists()
    assert result["depth"].endswith('.npz')
    np.testing.assert_array_equal(np.load(result["depth"])["depth"], depth)
