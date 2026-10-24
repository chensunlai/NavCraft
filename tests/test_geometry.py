import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from src.harness.geometry import Geometry
from src.harness.maps import MapRenderer


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.workspace, self.private = self.root / "workspace", self.root / "private"
        (self.workspace / "observations/maps").mkdir(parents=True)
        (self.private / "depth").mkdir(parents=True)
        self.transform = {"origin": [-5, -5], "size_m": 10}
        self.geometry = Geometry(self.workspace, self.private, self.transform)
        self.camera = {"frame": "f_0001", "depth": "f_0001.png", "depth_scale": .001,
                       "K": [[2, 0, 2], [0, 2, 2], [0, 0, 1]],
                       "world_from_optical": [[1, 0, 0, 0], [0, -1, 0, 1],
                                              [0, 0, -1, 0], [0, 0, 0, 1]],
                       "body_position": [0, 0, 0], "body_forward": [0, 0, -1],
                       "floor_height": 0, "width": 5, "height": 5,
                       "depth_min": .05, "depth_max": 10}
        self._frame()

    def _frame(self, camera=None, depth=None):
        camera = camera or self.camera
        depth = depth if depth is not None else np.full((camera["height"], camera["width"]), 2000, dtype=np.uint16)
        Image.fromarray(depth).save(self.private / "depth" / camera["depth"])
        Image.new("RGB", (camera["width"], camera["height"]), (50, 120, 200)).save(self.workspace / "observations" / (camera["frame"] + ".png"))
        with (self.private / "depth/index.jsonl").open("a") as stream:
            stream.write(json.dumps(camera) + "\n")
        with (self.workspace / "observations/index.jsonl").open("a") as stream:
            stream.write(json.dumps({"frame": camera["frame"], "path": f"observations/{camera['frame']}.png"}) + "\n")

    def test_optical_z_backprojection_distance_and_height(self):
        center = self.geometry.resolve_frame("f_0001", [2, 2])
        self.assertEqual(center, {"uv": [.5, .3], "distance": 2., "bearing": 0., "height": 1., "valid": True})
        offset = self.geometry.resolve_frame("f_0001", [3, 3])
        self.assertEqual(offset["uv"], [.6, .3])
        self.assertEqual(offset["height"], 0)
        self.assertAlmostEqual(offset["distance"], math.sqrt(5), places=3)
        self.assertAlmostEqual(offset["bearing"], 26.565, places=2)
        # The selected depth is optical Z, not Euclidean ray length.
        world, _ = self.geometry.resolve_frame_point("f_0001", [3, 3])
        self.assertEqual(world, [1., 0., -2.])

    def test_panorama_bearing_uses_capture_body_not_subframe_direction(self):
        camera = {**self.camera, "frame": "f_0002", "depth": "f_0002.png",
                  "world_from_optical": [[0, 0, 1, 0], [0, -1, 0, 1], [1, 0, 0, 0], [0, 0, 0, 1]]}
        self._frame(camera)
        point = self.geometry.resolve_frame("f_0002", [2, 2])
        self.assertEqual(point["uv"], [.7, .5])
        self.assertEqual(point["bearing"], 90)
        # Updating the current public state cannot change an old frame's capture pose.
        (self.workspace / "scene").mkdir()
        (self.workspace / "scene/state.json").write_text(json.dumps({"pose": {"pos": [.9, .9], "heading": 180}}))
        self.assertEqual(self.geometry.resolve_frame("f_0002", [2, 2]), point)

    def test_rejects_bad_pixels_and_invalid_depth_without_semantic_guesses(self):
        for pixel in ([-1, 2], [5, 2], [2, 5], [math.nan, 2], [True, 2], [2]):
            self.assertFalse(self.geometry.resolve_frame("f_0001", pixel)["valid"])
        for depth in (0, 10_000, 11_000):
            self._frame(depth=np.full((5, 5), depth, dtype=np.uint16))
            result = self.geometry.resolve_frame("f_0001", [2, 2])
            self.assertEqual(result, {"valid": False, "reason": "invalid_depth_or_range"})
        self._frame()
        camera = {**self.camera, "floor_y_range": [1.5, 3]}
        self._frame(camera)
        self.assertEqual(self.geometry.resolve_frame("f_0001", [2, 2])["reason"], "outside_floor")
        camera = {**self.camera, "world_from_optical": [[1, 0, 0, 100], [0, -1, 0, 1], [0, 0, -1, 0], [0, 0, 0, 1]]}
        self._frame(camera)
        self.assertEqual(self.geometry.resolve_frame("f_0001", [2, 2])["reason"], "outside_canvas")

    def test_rotated_map_affine_round_trip_and_margin_rejection(self):
        matrix = np.array([[0, -.001, .6], [.001, 0, .4]])
        view = {"view_id": "m_0001", "width": 200, "height": 200,
                "pixel_to_uv": matrix.tolist(), "plot_bounds": [20, 20, 180, 180]}
        (self.workspace / "observations/maps/index.jsonl").write_text(json.dumps(view) + "\n")
        for pixel in ([100, 100], [100, 50], [150, 100], [20, 180]):
            uv = self.geometry.resolve_map("m_0001", pixel)["uv"]
            restored = np.linalg.solve(np.vstack([matrix, [0, 0, 1]]), [*uv, 1])
            np.testing.assert_allclose(restored[:2], pixel, atol=1e-10)
        self.assertEqual(self.geometry.resolve_map("m_0001", [100, 50])["uv"], [.55, .5])
        for pixel in ([10, 100], [100, 190], [200, 100]):
            with self.assertRaises(ValueError):
                self.geometry.resolve_map("m_0001", pixel)

    def test_batch_limit_and_per_point_failure(self):
        result = self.geometry.locate("f_0001", [[2, 2], [-1, 2]])
        self.assertTrue(result["points"][0]["valid"])
        self.assertFalse(result["points"][1]["valid"])
        self.assertEqual(result["points"][1]["pixel"], [-1, 2])
        for points in ([], [[2, 2]] * 11):
            with self.assertRaises(ValueError):
                self.geometry.locate("f_0001", points)
        with self.assertRaises(ValueError):
            self.geometry.locate("crop_0001", [[2, 2]])
        with self.assertRaises(ValueError):
            self.geometry.resolve_frame_point("f_0001", [5, 0])
        self.assertFalse(self.geometry.resolve_frame({}, [2, 2])["valid"])
        with self.assertRaises(ValueError):
            self.geometry.resolve_map([], [2, 2])

    def test_locate_reads_the_actual_ego_renderer_index(self):
        (self.workspace / "scene").mkdir()
        (self.workspace / "scene/map.json").write_text(json.dumps({**self.transform, "walls": []}))
        renderer = MapRenderer(self.workspace)
        view = renderer.render({"pos": [.5, .5], "heading": 90}, [], view={"ego": 2})
        self.assertEqual(self.geometry.resolve_map(view["view_id"], [384, 380])["uv"], [.5, .5])
        self.assertEqual(self.geometry.resolve_map(view["view_id"], [384, 34])["uv"], [.7, .5])
        with self.assertRaisesRegex(ValueError, "outside_map_plot"):
            self.geometry.resolve_map(view["view_id"], [384, 15])

    def test_private_depth_path_cannot_escape_directory(self):
        self.camera["depth"] = "../outside.png"
        with (self.private / "depth/index.jsonl").open("a") as stream:
            stream.write(json.dumps(self.camera) + "\n")
        with self.assertRaisesRegex(ValueError, "depth path"):
            self.geometry.load_frame("f_0001")

    def test_overlay_filters_height_range_and_preserves_rgb_and_uv(self):
        camera = {**self.camera, "width": 17, "height": 17, "K": [[16, 0, 8], [0, 16, 8], [0, 0, 1]]}
        self._frame(camera)
        rows = self.geometry.overlay(["f_0001"])
        self.assertEqual(rows.shape, (15, 5))
        np.testing.assert_allclose(rows[:, 2:], np.tile([50, 120, 200], (15, 1)))
        self.assertTrue(any(np.allclose(row[:2], [.5, .3]) for row in rows))
        self.assertEqual(self.geometry.overlay(["f_0001"], max_range=1).shape, (0, 5))
        with self.assertRaises(ValueError):
            self.geometry.overlay(["f_0001"] * 7)

    def test_panorama_overlay_has_a_fixed_output_size_limit(self):
        camera = {**self.camera, "width": 640, "height": 480,
                  "K": [[600, 0, 320], [0, 600, 240], [0, 0, 1]]}
        self._frame(camera)
        rows = self.geometry.overlay(["f_0001"] * 6)
        self.assertEqual(rows.shape, (50_000, 5))
        self.assertTrue(np.isfinite(rows).all())


if __name__ == "__main__":
    unittest.main()
