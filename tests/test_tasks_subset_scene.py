import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.dataset.tasks_subset.bundle import freeze_case
from src.dataset.tasks_subset.sampling.geometry import crossed_segment, floor_patch_fraction, project, unproject, visible_point
from src.dataset.tasks_subset.sampling.scene import room_at


class SceneTests(unittest.TestCase):
    def test_floor_patch_rejects_a_hole_through_an_obstacle(self):
        camera = {"K": [[20, 0, 20], [0, 20, 0], [0, 0, 1]]}
        matrix = [[1, 0, 0, 0], [0, -1, 0, 1], [0, 0, -1, 0], [0, 0, 0, 1]]
        yy, _ = np.mgrid[0:41, 0:41]
        depth = 20. / np.maximum(yy, 1)
        self.assertEqual(floor_patch_fraction(depth, camera, matrix, [20, 20], 0), 1.)
        covered = depth.copy() * .5
        covered[19:22, 19:22] = depth[19:22, 19:22]
        self.assertIsNotNone(visible_point([0, 0, -1], covered, {**camera, "width": 41, "height": 41}, matrix))
        self.assertLess(floor_patch_fraction(covered, camera, matrix, [20, 20], 0), .1)

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

    def test_traversal_requires_same_floor_and_actual_finite_segment(self):
        from src.dataset.tasks_subset.sampling.collection import _boundary_crossings, _traversals

        scene = {"levels": [{"id": "L1", "floor_height": 0}], "rooms": [
            {"id": "a", "level": "L1", "polygon_xz": [[-2,-2],[0,-2],[0,2],[-2,2]]},
            {"id": "b", "level": "L1", "polygon_xz": [[0,-2],[2,-2],[2,2],[0,2]]}],
            "openings": [{"id": "D", "kind": "door", "level": "L1", "segment_xz": [[0,-1],[0,1]], "connects": ["a","b"]}]}
        motions = [{"step": 1, "samples": [{"position": [-.4,0,0]}, {"position": [.4,0,0]}]}]
        events = _traversals(scene,motions,[-1,0,0])
        self.assertEqual(events[0]["opening"], "D")
        scene["openings"][0]["kind"] = "open_functional_connection"
        self.assertEqual(_traversals(scene,motions,[-1,0,0]), [])
        zone, = _boundary_crossings(scene,motions,[-1,0,0])
        self.assertEqual((zone["kind"],zone["opening"],zone["from_room"],zone["to_room"]),
                         ("zone_transition","D","a","b"))
        scene["openings"][0]["kind"] = "open_passage"
        self.assertEqual(_traversals(scene,motions,[-1,0,0])[0]["opening"], "D")
        motions[0]["samples"] = [{"position": [-.4,2,0]}, {"position": [.4,2,0]}]
        self.assertEqual(_traversals(scene,motions,[-1,2,0]), [])
        motions[0]["samples"] = [{"position": [-.4,0,1.5]}, {"position": [.4,0,1.5]}]
        self.assertEqual(_traversals(scene,motions,[-1,0,1.5]), [])

    def test_measured_floor_patch_rejects_mixed_invalid_and_ambiguous_floor(self):
        import numpy as np
        from src.dataset.tasks_subset.sampling.geometry import measure_horizontal_floor_patch
        camera={"K":[[30,0,20],[0,30,20],[0,0,1]]}
        matrix=np.diag([1.,-1.,-1.,1.])
        matrix[1,3]=1.
        yy,xx=np.mgrid[0:50,0:50]
        depth=np.full((50,50),np.nan)
        depth[21:,:]=30/(yy[21:,:]-20)
        def measure(values,floors=[.073],body=0):
            return measure_horizontal_floor_patch(values,camera,matrix,[25,35],floors,body)
        self.assertAlmostEqual(measure(depth)["height_m"],0.)
        self.assertEqual(measure(depth)["support_fraction"],1.)
        bad=depth.copy()
        bad[:,25:]*=.9
        self.assertIsNone(measure(bad))
        invalid=depth.copy()
        invalid[:,25:]=np.nan
        self.assertIsNone(measure(invalid))
        self.assertIsNone(measure(depth,[0,.05]))
        self.assertIsNone(measure(depth,body=.3))

    def test_floor_height_band_does_not_accept_vertical_furniture_face(self):
        import numpy as np
        from src.dataset.tasks_subset.sampling.geometry import measure_horizontal_floor_patch
        camera={"K":[[400,0,320],[0,400,240],[0,0,1]]}
        matrix=np.diag([1.,-1.,-1.,1.]);matrix[1,3]=.5
        # At y=440 this wall patch has median world Y=0, with every pixel within
        # +/- .0225m of that height. Only its surface normal reveals it is vertical.
        depth=np.ones((480,640))
        self.assertIsNone(measure_horizontal_floor_patch(depth,camera,matrix,[320,440],[0],0))
