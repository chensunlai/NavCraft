import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.dataset.tasks_subset.bundle import freeze_case
from src.dataset.tasks_subset.sampling.geometry import crossed_segment, project, unproject, visible_point
from src.dataset.tasks_subset.sampling.scene import room_at


class SceneTests(unittest.TestCase):
    def test_camera_rotation_and_occlusion(self):
        matrix = [[0, 0, 1, 3], [0, -1, 0, 2], [1, 0, 0, 0], [0, 0, 0, 1]]
        camera = {"K": [[3, 0, 2], [0, 3, 2], [0, 0, 1]], "width": 5, "height": 5}
        world = [5, 2, 0]
        self.assertEqual(project(world, camera, matrix), (2, 2, 2))
        self.assertEqual(unproject([2, 2], 2, camera, matrix), world)
        self.assertEqual(visible_point(world, np.full((5, 5), 2.), camera, matrix), [2, 2])
        self.assertIsNone(visible_point(world, np.full((5, 5), 1.), camera, matrix))
        self.assertIsNone(project([1, 2, 0], camera, matrix))

    def test_room_boundary_overlap_and_floor(self):
        square = [[0, 0], [3, 0], [3, 3], [0, 3]]
        levels = [{"id": "a", "floor_height": 0}, {"id": "b", "floor_height": 2}]
        rooms = [{"id": "R1", "level": "a", "polygon_xz": square},
                 {"id": "R2", "level": "b", "polygon_xz": square}]
        scene = {"levels": levels, "rooms": rooms}
        self.assertEqual(room_at(scene, [1, 0, 1]), "R1")
        self.assertEqual(room_at(scene, [1, 2, 1]), "R2")
        self.assertIsNone(room_at(scene, [0.06, 0, 1]))
        self.assertIsNone(room_at(scene, [1, 1, 1]))
        rooms.append({"id": "R3", "level": "a", "polygon_xz": square})
        self.assertIsNone(room_at(scene, [1, 0, 1]))

    def test_finite_opening_crossing(self):
        segment = [[0, -1], [0, 1]]
        self.assertTrue(crossed_segment([-1, 0, 0], [1, 0, 0], segment))
        self.assertFalse(crossed_segment([-1, 0, 1.2], [1, 0, 1.2], segment))
        self.assertFalse(crossed_segment([0, 0, -.8], [0, 0, .8], segment))

    def test_freeze_rejects_escaped_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "outside"):
                freeze_case(root, {"id": "case", "snapshot": "../../../outside"}, {}, {}, [])
            self.assertFalse((root / "cases/case").exists())

    def test_cut_resolves_restored_frames_from_public_index(self):
        from types import SimpleNamespace
        from src.dataset.tasks_subset.sampling.collection import _cut, _frame_index

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pose = {"position": [0, 0, 0]}
            prior = {f"walk/f_{i:04}": {"frame": f"f_{i:04}", "step": 0,
                     "group": "old", "pose": pose, "room": None} for i in range(1, 7)}
            future = {"future/f_0007": {"frame": "f_0007", "step": 1, "group": None, "pose": pose, "room": None}}
            appended = {"branch/f_0007": {"frame": "f_0007", "step": 1, "group": None, "pose": pose, "room": None}}
            bundle = {"frames": {**prior, **future, **appended}, "states": [],
                      "scene": {"rooms": [], "levels": [], "openings": []}}
            restored = _frame_index(bundle, list(prior))
            restored["f_0007"] = "branch/f_0007"
            ex = SimpleNamespace(
                bundle=bundle, output=root, snapshot=lambda directory: directory.mkdir(parents=True),
                environment=SimpleNamespace(steps=1, position=lambda: [0, 0, 0], trajectory=[[0, 0, 0]], motions=[]),
                observations=SimpleNamespace(items={fid: None for fid in restored}),
                frame_keys=restored, current={"frame": "f_0007"},
                memory=SimpleNamespace(nodes=[{"id": "N001", "panorama": "old"}]),
                transform={"origin": [-10, -10], "size_m": 20, "floor_height": 0, "walls": []})
            cut = _cut(ex, "blocked")
            self.assertEqual(cut["node_frames"]["N001"], list(prior))
            self.assertEqual(cut["current_frame"], "branch/f_0007")
            self.assertNotIn("future/f_0007", cut["frame_keys"])

    def test_snapshot_digest_binds_trial_to_memory_content(self):
        import shutil
        from src.dataset.tasks_subset.bundle import require_trial_snapshot, snapshot_digest

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "original"
            memory = Path("workspace/memory/nodes.json")
            (original / memory).parent.mkdir(parents=True)
            (original / "state.json").write_text("{}")
            (original / memory).write_text('{"description":"old"}')
            copy = root / "copy"
            shutil.copytree(original, copy)
            digest = snapshot_digest(original)
            self.assertEqual(digest, snapshot_digest(copy))
            state = {"snapshot": "copy"}
            trial = {"start_snapshot_sha256": digest}
            require_trial_snapshot(root, state, trial)
            (copy / memory).write_text('{"description":"new"}')
            with self.assertRaisesRegex(ValueError, "changed"):
                require_trial_snapshot(root, state, trial)
            with self.assertRaisesRegex(ValueError, "binding"):
                require_trial_snapshot(root, state, {})
            (copy / "link").symlink_to(original / "state.json")
            with self.assertRaisesRegex(ValueError, "Symlink"):
                snapshot_digest(copy)

    def test_hssd_carrier_requires_train_building_and_is_not_official(self):
        from src.dataset.tasks_subset.sampling.collection import _hssd_episode, _is_official_instruction

        with tempfile.TemporaryDirectory() as temporary:
            split = Path(temporary) / "hssd-hab/scene_splits.yaml"
            split.parent.mkdir()
            split.write_text("train: ['10_1', '20_1']\nval: ['20_2', '30']\n")
            config = {"assets": temporary}
            scene = lambda name: f"hssd-hab/scenes/{name}.scene_instance.json"
            episode = _hssd_episode(config, scene("10_1"))
            self.assertEqual(episode.provenance, "program_assisted")
            self.assertFalse(_is_official_instruction(episode))
            for identifier in ("20_1", "20_2", "30", "40"):
                with self.assertRaisesRegex(ValueError, "train building"):
                    _hssd_episode(config, scene(identifier))

    def test_frozen_seals_reject_mutated_inputs(self):
        from PIL import Image
        from src.dataset.tasks_subset.bundle import freeze_case, validate_frozen_inputs, write_json

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "states/s/snapshot"
            write_json(snapshot / "state.json", {})
            write_json(snapshot / "workspace/memory/nodes.json", {"nodes": []})
            image = root / "source.png"
            Image.new("RGB", (2, 2)).save(image)
            task = freeze_case(root, {"id": "x", "snapshot": "../../states/s/snapshot"},
                               {"accepted": [1]}, {}, [{"path": image, "caption": "RGB"}])
            case = root / "cases/x"
            self.assertTrue(validate_frozen_inputs(task, case))
            for path in (case/"private/label.json", case/task["images"][0]["path"],
                         snapshot/"workspace/memory/nodes.json"):
                before = path.read_bytes()
                path.write_bytes(before + b" ")
                with self.assertRaisesRegex(ValueError, "seal|digest"):
                    validate_frozen_inputs(task, case)
                path.write_bytes(before)
            del task["inputs_sha256"]
            with self.assertRaisesRegex(ValueError, "seal"):
                validate_frozen_inputs(task, case)
