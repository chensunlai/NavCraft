import json
from pathlib import Path
import tempfile
import unittest

from src.harness.memory import MemoryStore


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "workspace"
        (self.root / "observations").mkdir(parents=True)
        (self.root / "observations/index.jsonl").write_text("\n".join(json.dumps({
            "frame": f"f_{i:04d}", "step": i, "pose": {"pos": [.5, .5], "heading": 0}
        }) for i in range(3)) + "\n")
        self.store = MemoryStore(self.root)

    def create(self, identifier, kind, **fields):
        return {"op": "create", "id": identifier, "type": kind, "fields": {"name": identifier, **fields}}

    def test_batch_forward_reference_and_evidence_metadata(self):
        result = self.store.edit([
            self.create("O01", "object", room="R01", category="chair", evidence=["f_0001"]),
            self.create("R01", "room", anchor=[.4, .5]),
        ])
        self.assertNotIn("error", result)
        item = self.store.read(ids=["O01", "missing"])
        self.assertEqual(item["missing"], ["missing"])
        self.assertEqual(item["entities"][0]["evidence"][0]["step"], 1)
        self.assertEqual(self.store.read(filter={"room": "R01"})["total"], 1)
        self.assertEqual(self.store.read(filter={"room": None})["total"], 0)

    def test_evidence_preserves_panorama_camera_direction(self):
        with (self.root / "observations/index.jsonl").open("a") as stream:
            stream.write(json.dumps({"frame": "f_0003", "step": 3,
                                    "pose": {"pos": [.5, .5], "heading": 90},
                                    "group": "p_one", "yaw": 120}) + "\n")
        self.store.edit([self.create("R01", "room", evidence=["f_0003"])])
        evidence = self.store.read(ids=["R01"])["entities"][0]["evidence"][0]
        self.assertEqual(evidence["group"], "p_one")
        self.assertEqual(evidence["yaw"], 120)

    def test_failed_batch_never_partially_writes_or_reserves_ids(self):
        original = self.store.path.read_bytes()
        result = self.store.edit([self.create("R01", "room"), self.create("O01", "object", room="R09")])
        self.assertIn("error", result)
        self.assertEqual(original, self.store.path.read_bytes())
        self.assertNotIn("error", self.store.edit([self.create("R01", "room")]))

    def test_deleted_id_cannot_be_reused_after_reopen(self):
        self.store.edit([self.create("R01", "room")])
        self.store.edit([{"op": "delete", "id": "R01"}])
        result = MemoryStore(self.root).edit([self.create("R01", "room")])
        self.assertIn("error", result)
        self.assertIn("R02", result["hint"])

    def test_delete_reference_can_be_repaired_in_same_batch(self):
        self.store.edit([self.create("R01", "room"), self.create("O01", "object", room="R01")])
        self.assertIn("error", self.store.edit([{"op": "delete", "id": "R01"}]))
        self.assertNotIn("error", self.store.edit([
            {"op": "delete", "id": "R01"}, {"op": "update", "id": "O01", "fields": {"room": None}},
        ]))

    def test_merge_updates_room_references_and_unions_evidence(self):
        self.store.edit([self.create("R01", "room", evidence=["f_0000"]),
                         self.create("R02", "room", evidence=["f_0001"]),
                         self.create("D01", "door", connects=["R02", None]),
                         self.create("O01", "object", room="R02")])
        result = self.store.edit([{"op": "merge", "keep": "R01", "remove": ["R02"], "fields": {}}])
        self.assertNotIn("error", result)
        entities = {e["id"]: e for e in self.store.entities}
        self.assertEqual(entities["O01"]["room"], "R01")
        self.assertEqual(entities["D01"]["connects"], ["R01", None])
        self.assertEqual(entities["R01"]["evidence"], ["f_0000", "f_0001"])

    def test_merge_requires_resolution_of_conflicting_positions(self):
        self.store.edit([self.create("O01", "object", point=[.2, .3]),
                         self.create("O02", "object", point=[.5, .6])])
        merge = {"op": "merge", "keep": "O01", "remove": ["O02"], "fields": {}}
        self.assertIn("error", self.store.edit([merge]))
        merge["fields"] = {"point": [.4, .5]}
        self.assertNotIn("error", self.store.edit([merge]))

    def test_rejects_unknown_evidence_out_of_bounds_and_identity_change(self):
        for fields in ({"anchor": [1.1, .5]}, {"anchor": [float("nan"), .5]}, {"evidence": ["f_future"]}):
            self.assertIn("error", self.store.edit([self.create("R01", "room", **fields)]))
        self.store.edit([self.create("R01", "room")])
        self.assertIn("error", self.store.edit([{"op": "update", "id": "R01", "fields": {"type": "object"}}]))
        self.assertIn("error", self.store.read(ids=["R01"], filter={"type": "room"}))

    def test_snapshot_validation_and_id_reservation(self):
        self.store.path.write_text(json.dumps([{"id": "R07", "type": "room", "name": "room", "anchor": None,
                                               "description": "", "evidence": ["f_0000"]}]))
        store = MemoryStore(self.root)
        self.assertIn("R07", json.loads(store.used_path.read_text()))
        store.edit([{"op": "delete", "id": "R07"}])
        self.assertIn("error", store.edit([self.create("R07", "room")]))
        store.path.write_text(json.dumps([{"id": "O01", "type": "room", "name": "bad", "description": "", "evidence": []}]))
        with self.assertRaises(ValueError):
            MemoryStore(self.root)


if __name__ == "__main__":
    unittest.main()
