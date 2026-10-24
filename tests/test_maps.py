import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from PIL import Image

from src.harness.maps import MapRenderer
from src.harness.memory import MemoryStore


class MapTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "workspace"
        (self.root / "scene").mkdir(parents=True)
        (self.root / "observations").mkdir()
        self.pose = {"pos": [.5, .5], "heading": 90}
        (self.root / "scene/map.json").write_text(json.dumps({"origin": [-5, -5], "size_m": 10,
            "walls": [[[-4, -4], [-4, 4]], [[-4, 4], [4, 4]]]}))
        (self.root / "scene/state.json").write_text(json.dumps({"pose": self.pose}))
        (self.root / "observations/index.jsonl").write_text(json.dumps({"frame": "f_0001", "step": 1, "pose": self.pose}) + "\n")
        self.memory = MemoryStore(self.root)
        self.memory.edit([{"op": "create", "id": "O01", "type": "object", "fields": {"name": "椅子", "point": [.6, .5]}},
                          {"op": "create", "id": "R01", "type": "room", "fields": {"name": "未知位置"}}])

    def test_crops_do_not_rotate_and_have_new_references(self):
        renderer = MapRenderer(self.root)
        a = renderer.render(self.pose, [[.4, .5], [.5, .5]], view={"around_self": 2})
        b = renderer.render({**self.pose, "heading": 180}, [], view={"around_self": 2})
        self.assertEqual(a["bounds"], [.3, .3, .7, .7])
        self.assertEqual(a["bounds"], b["bounds"])
        self.assertNotEqual(a["map"], b["map"])
        self.assertTrue(any("R01" in warning for warning in a["warnings"]))
        with Image.open(self.root / a["map"]) as image:
            self.assertLessEqual(max(image.size), 768)

    def test_public_navlib_calibration_and_direction(self):
        source = Path(__file__).resolve().parents[1] / "src/harness/navlib.py"
        target = self.root / "scene/navlib.py"
        shutil.copyfile(source, target)
        spec = importlib.util.spec_from_file_location("public_navlib", target)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.uv_to_m([.5, .5]), [0., 0.])
        self.assertEqual(module.m_to_uv([0., 0.]), [.5, .5])
        answer = module.bearing_distance(self.pose, [.6, .5])
        self.assertAlmostEqual(answer["bearing_degrees"], 0)
        self.assertAlmostEqual(answer["distance_m"], 1)
        self.assertEqual(module.project(self.pose, 0, 1), [.6, .5])
        self.assertEqual(module.frame("f_0001")["step"], 1)
        self.assertEqual(module.pose(), self.pose)
        self.assertEqual(len(module.memory()), 2)


if __name__ == "__main__":
    unittest.main()
