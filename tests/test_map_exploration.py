import json
from pathlib import Path

import numpy as np

from src.data.map_exploration import canvas, floor_domain, flood_regions, rasterize, episode_floor_range, freeze_development


def test_canvas_has_two_metre_padding_and_ten_metre_rounding():
    origin, side = canvas(np.array([[[0.0, 0.0], [12.0, 4.0]]]))
    assert side == 20.0
    assert np.all(np.asarray(origin) <= [-2.0, -2.0])


def test_raster_uses_north_up_x_right_coordinates():
    mask = rasterize(np.array([[[0.0, 0.0], [1.0, 0.0]]]), [-1.0, -1.0], 2.0)
    assert mask.shape == (40, 40)
    assert mask[-1].sum() == 0
    assert mask[19:21].sum() > 0


def test_asymmetric_wall_is_in_north_east_not_south_west():
    mask = rasterize(np.array([[[.7, -.8], [.9, -.8]]]), [-1., -1.], 2.)
    rows, cols = np.nonzero(mask)
    assert rows.max() < 5
    assert cols.min() >= 33


def test_floor_domain_preserves_a_real_unscanned_hole_and_excludes_table():
    ground = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 0., 1.]],
                       [[2., 0., 0.], [3., 0., 0.], [2., 0., 1.]],
                       [[1., .8, 0.], [2., .8, 0.], [1., .8, 1.]]])
    domain, count = floor_domain(ground, 0., [-1., -1.], 10.)
    assert count == 2
    assert domain[22, 22]
    assert not domain[22, 48]


def test_regions_are_four_connected_and_ignore_tiny_components():
    free = np.zeros((8, 8), dtype=bool)
    free[1:4, 1:4] = True
    free[6, 6] = True
    labels, regions = flood_regions(free, minimum_pixels=4)
    assert len(regions) == 1
    assert labels[2, 2] >= 0
    assert labels[6, 6] == -1


def test_episode_height_groups_include_split_level_and_exclude_next_floor():
    seed = {"training_start_height_summary": [{"rounded_height_m": y} for y in (-2.3, 0, .4, .5, 1.6)]}
    assert episode_floor_range(seed, .4) == [-.2, .7]


def test_development_freeze_excludes_unresolved_maps_and_private_episode_pose(tmp_path):
    import hashlib
    source = tmp_path / "build"
    records = []
    for identifier, passed in (("accepted", True), ("unresolved", False)):
        scene = source / identifier
        folder = scene / "raw_vote"
        folder.mkdir(parents=True)
        for name in ("walls.png", "openings.json", "regions.png", "regions.json", "overlay.png", "leak_geometry_review.json"):
            (folder / name).write_bytes(b"fixture")
        meta = {"dataset": "hm3d", "floor_height": 0., "quality": {"pass_after_geometry_review": passed},
                "snapped_start": [1, 2, 3], "episode_start": [1, 2, 3],
                "wall_sha256": hashlib.sha256(b"fixture").hexdigest()}
        (folder / "meta.json").write_text(json.dumps(meta))
        records.append({"status": "built", "output": str(scene),
                        "result": {"seed": {"scene": identifier}, "scene": identifier}})
    (source / "summary.json").write_text(json.dumps({"scenes": records}))
    manifest = freeze_development(source, tmp_path / "frozen")
    assert len(manifest["maps"]) == 1 and len(manifest["excluded"]) == 1
    frozen = tmp_path / "frozen" / manifest["maps"][0]["path"]
    metadata = json.loads((frozen / "meta.json").read_text())
    assert "episode_start" not in metadata and "snapped_start" not in metadata
    assert (frozen / "leak_geometry_review.json").exists()
