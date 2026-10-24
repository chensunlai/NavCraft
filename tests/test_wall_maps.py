import unittest

import numpy as np

from src.dataset.wall_maps import _world, canvas, slice_triangles


class WallMapTests(unittest.TestCase):
    def test_slice_preserves_doorway_below_lintel(self):
        triangles = np.array([
            [[0, 0, 0], [0, 2, 0], [1, 2, 0]],
            [[0, 0, 0], [1, 2, 0], [1, 0, 0]],
            [[2, 0, 0], [2, 2, 0], [3, 2, 0]],
            [[2, 0, 0], [3, 2, 0], [3, 0, 0]],
            [[1, 2, 0], [1, 3, 0], [2, 3, 0]],
            [[1, 2, 0], [2, 3, 0], [2, 2, 0]],
        ])
        segments = slice_triangles(triangles, 1.)
        self.assertEqual(len(segments), 4)
        self.assertTrue(all(s[:, 0].max() <= 1 or s[:, 0].min() >= 2 for s in segments))

    def test_coordinate_transform_and_canvas_padding(self):
        self.assertTrue(np.array_equal(_world([[1, 2, 3]]), [[1, 3, -2]]))
        segments = np.array([[[0, 0], [12, 4]], [[12, 4], [2, 6]]])
        origin, size = canvas(segments)
        self.assertEqual(size, 20)
        points = segments.reshape(-1, 2)
        self.assertTrue(np.all(points.min(0) - origin >= 2))
        self.assertTrue(np.all(np.array(origin) + size - points.max(0) >= 2))

    def test_no_slice_does_not_invent_walls(self):
        segments = slice_triangles([[[0, 2, 0], [0, 3, 0], [1, 3, 0]]], 1)
        self.assertEqual(segments.shape, (0, 2, 2))


if __name__ == "__main__":
    unittest.main()
