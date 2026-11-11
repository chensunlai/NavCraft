from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.dataset.tasks_subset.sampling.semantics import descriptor_ids, visibility


class InstanceEvidenceTests(unittest.TestCase):
    def test_descriptor_ids_are_not_row_indices(self):
        with TemporaryDirectory() as temp:
            descriptor = Path(temp) / "scene.semantic.txt"
            descriptor.write_text('HM3D Semantic Annotations\n34,6B7AA0,"tv",1\n211,AABBCC,"tv",3\n')
            self.assertEqual(descriptor_ids(descriptor), {34: "6B7AA0", 211: "AABBCC"})

    def test_zero_requires_complete_verified_mapping(self):
        def classify(pixels, valid=True, associated=True, complete=True):
            return visibility(pixels, valid=valid, associated=associated, complete=complete, threshold=100)

        self.assertEqual(classify(0), "absent")
        self.assertEqual(classify(0, complete=False), "unknown")
        self.assertEqual(classify(0, valid=False), "unknown")
        self.assertEqual(classify(100, associated=False), "unknown")
        self.assertEqual(classify(99), "unknown")
        self.assertEqual(classify(100, complete=False), "visible")

    def test_cached_apply_checks_hashes_and_marker_without_simulator(self):
        import hashlib
        from unittest.mock import patch

        import numpy as np

        from src.dataset.tasks_subset.bundle import write_json
        from src.dataset.tasks_subset.sampling.semantics import collect_instance_evidence

        def sha(p):
            return hashlib.sha256(p.read_bytes()).hexdigest()

        with TemporaryDirectory() as temp:
            root = Path(temp)
            assets = root / "hm3d"
            assets.mkdir()
            for ext in ("basis.glb", "semantic.glb", "semantic.txt", "basis.navmesh"):
                (assets / ("a." + ext)).write_text('34,C0\n')
            raw_map = root / "objects.json"
            raw_map.write_text("{}")
            rgb = root / "rgb.png"
            rgb.write_bytes(b"rgb")
            mask = root / "mask.npy"
            np.save(mask, np.array([[34, 34]]))
            empty = root / "empty.npy"
            np.save(empty, np.array([[0, 0]]))
            obj = {"id": "tv", "name": "TV", "center_xyz": [], "size_xyz": [],
                   "rectangles_xz": [], "semantic_ids": [34]}
            frame = {"rgb": "rgb.png", "pose": {"camera_position": [0,0,0]}, "camera": {}}
            row = {"valid": True, "camera": {}, "pose": {"camera_position": [0,0,0]}, "source_rgb_sha256": sha(rgb),
                   "mask": "mask.npy", "mask_sha256": sha(mask),
                   "objects": {"tv": {"mapping_valid": True, "pixels": 0}}}
            binding = {"frame": "f", "sha256": sha(rgb), "pixel": [0, 0], "world": [1, 2, 3]}
            sight = {"frame": "f", "id": "tv", "pixel": [0, 0], "world": [1, 2, 3],
                     "review_candidate": "r", "semantic_verified": True, "review_source": binding}
            paths = {"render_asset": assets/"a.basis.glb", "semantic_asset": assets/"a.semantic.glb",
                     "semantic_descriptor_filename": assets/"a.semantic.txt", "nav_asset": assets/"a.basis.navmesh"}
            capability = {"available": True, "backend": "habitat_native_hm3d",
                          "asset_sha256": {k: sha(v) for k,v in paths.items()},
                          "stage": {k: str(v) for k,v in paths.items()} | {"up": [0,0,1], "front": [0,1,0]},
                          "descriptor_ids": {"34": "C0"}, "object_map_sha256": sha(raw_map),
                          "rgb_tolerance": {"fraction_pixels_above_2": .001, "mean_channel_error": .02},
                          "min_pixels": 100}
            bundle = {"scene_id": "hm3d/a.basis.glb", "scene": {"objects": [obj]},
                      "states": [{"frame_keys": ["f", "empty"]}], "frames": {"f": frame, "empty": frame},
                      "source": {"map_directory": str(root), "sha256": {"objects.json": sha(raw_map)}},
                      "sightings": [sight], "instance_evidence": {"f": row, "empty": {**row, "mask": "empty.npy", "mask_sha256": sha(empty)}}, "instance_capability": capability}
            with patch("src.dataset.tasks_subset.sampling.semantics.load_scene", return_value={"objects": [obj]}), \
                 patch.dict("sys.modules", {"habitat_sim": None}):
                def apply(key="f"):
                    return collect_instance_evidence(bundle, root, {"assets": root}, 0, ["f"])["instance_evidence"][key]["objects"]["tv"]
                self.assertEqual(apply()["visibility"], "unknown")
                write_json(root/"private/instance_bindings.json", {"tv": {
                    "ids": [34], "frames": {"f": sha(rgb)}, "scope": "screen", "reviewer": "reviewer",
                    "method": "overlay", "objects_sha256": sha(raw_map),
                    "descriptor_sha256": sha(paths["semantic_descriptor_filename"])}})
                self.assertEqual(apply()["visibility"], "unknown")
                self.assertEqual(apply()["pixels"], 2)
                self.assertEqual(apply()["bbox"], [0, 0, 2, 1])
                self.assertEqual(apply("empty")["visibility"], "absent")
                sight["pixel"] = [1, 0]
                self.assertFalse(apply()["association_verified"])
                sight["pixel"] = [0, 0]
                rgb.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "Cached instance evidence changed"):
                    apply()
                rgb.write_bytes(b"rgb")
                mask.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "Cached instance evidence changed"):
                    apply()

    def test_shared_descriptor_id_is_ambiguous(self):
        from collections import Counter

        import numpy as np

        from src.dataset.tasks_subset.sampling.semantics import _object_pixels

        rows = _object_pixels(np.array([[34, 211]]), {"a": [34], "b": [34, 211]},
                              {34: "red", 211: "green"}, Counter({34: 2, 211: 1}))
        for row in rows.values():
            self.assertTrue(row["mapping_ambiguous"])
            self.assertFalse(row["mapping_valid"])

    def test_native_identity_handles_different_surface_points(self):
        import hashlib
        from unittest.mock import patch
        import numpy as np
        from src.dataset.tasks_subset.sampling.semantics import link_native_identity

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            mask = root/"mask.npy"
            np.save(mask, np.array([[34,211]]))
            rgb = root/"rgb"
            rgb.write_bytes(b"RGB")
            source = root/"objects.json"
            source.write_bytes(b"map")
            digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            sightings,frames,evidence=[],{},{}
            for key,pos,pixel,world in [
                ("a",[0,1,0],[0,0],[0,0,0]),
                ("b",[1,1,0],[0,0],[1,0,0]),
                ("c",[2,1,0],[1,0],[0,0,0])]:
                review="review-"+key
                frames[key]={"pose":{"camera_position":pos},"camera":{},"rgb":"rgb"}
                sightings.append({"id":"tv","frame":key,"pixel":pixel,"world":world,
                    "semantic_verified":True,"review_candidate":review,
                    "review_source":{"frame":key,"sha256":digest(rgb),"pixel":pixel,"world":world}})
                evidence[key]={"valid":True,"mask":"mask.npy","mask_sha256":digest(mask),
                    "source_rgb_sha256":digest(rgb),"pose":frames[key]["pose"],"camera":{},
                    "objects":{"tv":{"mapping_valid":True,"mapped_ids":[34,211],
                                     "association_reviews":[review]}}}
            bundle={"frames":frames,"sightings":sightings,"instance_evidence":evidence,
                "source":{"map_directory":str(root)},
                "instance_capability":{"available":True,"stage":{"render_asset":str(rgb)},
                    "asset_sha256":{"render_asset":digest(rgb)},"object_map_sha256":digest(source)}}
            with patch("src.dataset.tasks_subset.sampling.semantics.load_scene",
                       return_value={"objects":[{"id":"tv","semantic_ids":[34,211]}]}):
                link_native_identity(bundle,root)
                self.assertTrue(sightings[0]["identity_verified"])
                self.assertTrue(sightings[1]["identity_verified"])
                self.assertNotIn("native_identity",sightings[2])
                sightings[1]["review_source"]["sha256"]="wrong"
                link_native_identity(bundle,root)
                self.assertFalse(sightings[0]["identity_verified"])
                self.assertFalse(sightings[1]["identity_verified"])
                sightings[1]["review_source"]["sha256"]=digest(rgb)
                link_native_identity(bundle,root)
                evidence["b"]["camera"]={"changed":True}
                link_native_identity(bundle,root)
                self.assertFalse(sightings[0]["identity_verified"])

    def test_derived_sighting_preserves_seed_world_and_rejects_other_id(self):
        from unittest.mock import patch
        import numpy as np
        from src.dataset.tasks_subset.sampling.semantics import derive_instance_sightings

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.save(root/"mask.npy",np.array([[34,211],[34,211]]))
            (root/"depth").write_bytes(b"depth")
            world=[1.,2.,3.]
            (root/"rgb").write_bytes(b"rgb")
            import hashlib
            digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            frame={"rgb":"rgb","depth":"depth","camera":{},"world_from_optical":np.eye(4).tolist(),"pose":{}}
            bundle={"instance_capability":{"available":True,"min_pixels":1,
                "stage":{"render_asset":"/assets/hm3d/train/scene/scene.basis.glb"}},
                "frames":{"source":frame,"dest":frame},"states":[{"frame_keys":["source","dest"]}],
                "sightings":[{"kind":"object","id":"tv","frame":"source","world":world,
                    "semantic_verified":True,"review_candidate":"review","native_identity":{"descriptor_id":34}}],
                "instance_evidence":{"dest":{"valid":True,"mask":"mask.npy","mask_sha256":digest(root/"mask.npy"),"camera":{},"pose":{},
                    "source_rgb_sha256":digest(root/"rgb"),"objects":{"tv":{"mapped_ids":[34],"mapping_valid":True,"pixels":2}}}}}
            with patch("src.dataset.tasks_subset.sampling.semantics.link_native_identity", return_value=bundle), \
                 patch("src.dataset.tasks_subset.sampling.geometry.read_depth",return_value=np.ones((2,2))), \
                 patch("src.dataset.tasks_subset.sampling.geometry.visible_point", return_value=[0,0]):
                rows=derive_instance_sightings(bundle,root)
                self.assertEqual(rows[0]["world"],world)
                self.assertFalse(rows[0]["semantic_verified"])
                self.assertNotIn("review_source",rows[0])
            with patch("src.dataset.tasks_subset.sampling.semantics.link_native_identity", return_value=bundle), \
                 patch("src.dataset.tasks_subset.sampling.geometry.read_depth",return_value=np.ones((2,2))), \
                 patch("src.dataset.tasks_subset.sampling.geometry.visible_point", return_value=[1,0]):
                self.assertEqual(derive_instance_sightings(bundle,root),[])
            bundle["instance_evidence"]["dest"]["valid"] = False
            with patch.dict("sys.modules", {"habitat_sim": None}), \
                 patch("src.dataset.tasks_subset.sampling.semantics.link_native_identity", return_value=bundle), \
                 patch("src.dataset.tasks_subset.sampling.semantics.collect_instance_evidence",
                       side_effect=AssertionError("must not render")):
                self.assertEqual(derive_instance_sightings(bundle,root), [])
                self.assertGreater(bundle["instance_reprojection_report"]["skipped"]["missing_valid_cache_or_depth"], 0)
