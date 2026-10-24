import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from src.harness.maps import MapRenderer, _font
from src.harness.memory import MemoryStore


class MapTests(unittest.TestCase):
    def test_map_font_has_distinct_numbers_latin_and_chinese_glyphs(self):
        font = _font(16)
        masks = [bytes(font.getmask(character)) for character in "012mNR椅门"]
        self.assertEqual(len(set(masks)), len(masks))

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

    def test_renderer_reuses_memory_and_reads_later_edits(self):
        renderer = MapRenderer(self.root, self.memory)
        with patch("src.harness.maps.MemoryStore", side_effect=AssertionError("Memory was reopened")):
            initial = renderer.render(self.pose, [], view="full")
            self.assertTrue(any("R01" in warning for warning in initial["warnings"]))
            self.memory.edit([{"op": "update", "id": "R01", "fields": {"anchor": [.4, .4]}}])
            updated = renderer.render(self.pose, [], view="full", highlight=["R01"])
            self.assertFalse(any("R01" in warning for warning in updated["warnings"]))
            self.memory.edit([{"op": "delete", "id": "R01"}])
            deleted = renderer.render(self.pose, [], view="full", highlight=["R01"])
            self.assertTrue(any("R01" in warning for warning in deleted["warnings"]))
        self.assertNotEqual((self.root / initial["map"]).read_bytes(), (self.root / updated["map"]).read_bytes())

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
