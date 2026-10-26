"""Local synthetic contracts only; these fixtures are never real-data evidence."""

from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.common.config import read_config
from src.common.io import read_json, write_json
from src.dataset.episodes import EpisodeSpec
from src.dataset.generation.check import check
from src.dataset.generation.derive import derive
from src.dataset.generation.records import (SceneRecord, Snapshot, TaskSample,
                                            load_records, local_file, save_records, stable_id)
from src.dataset.generation.tasks import TASK_IDS
from src.environment import Environment
from src.environment.fake import FakeSimulator
from src.harness.geometry import Geometry
from src.harness.session import open_session, read_snapshot


class _FixtureSimulator(FakeSimulator):
    """Render deliberately blank RGB plus explicitly synthetic aligned labels."""

    def capture(self, workspace, frame):
        observation = super().capture(workspace, frame)
        Image.new("RGB", (640, 480), "white").save(observation.rgb)
        values = np.zeros((480, 640), dtype=np.int32)
        values[270:330, 180:240] = self.config["fixture_semantic_ids"][0]
        values[270:330, 400:460] = self.config["fixture_semantic_ids"][1]
        semantic = Path(workspace) / "private" / f"fixture_semantic_{frame:04d}.npy"
        semantic.parent.mkdir(parents=True, exist_ok=True)
        np.save(semantic, values, allow_pickle=False)
        return replace(observation, semantic=semantic)


@pytest.fixture
def frozen_fixture(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    root = tmp_path / "collection"
    root.mkdir()
    write_json(root / "manifest.json", {"source": "local-fixture-not-real-data", "seed": 0})
    scene_records, snapshots = [], []
    config = {"model": "fixture-never-called", "tool_tier": "super", "goto_planner": "local",
              "generation": {"seed": 0, "per_task": 4, "teacher_per_task": 2}}
    for scene_index in range(3):
        scene_name = f"fixture-scene-{scene_index}"
        scene_dir = root / "scenes" / scene_name
        scene_dir.mkdir(parents=True)
        map_file = scene_dir / "map.json"
        write_json(map_file, {"origin": [0., 0.], "size_m": 10., "floor_height": 0.,
                             "walls": [[[1., 1.], [9., 1.]]]})
        instances = []
        for cut_index in range(2):
            cut_id = f"{scene_name}-cut-{cut_index}"
            semantic_ids = [100 * (scene_index + 1) + cut_index * 2 + 1,
                            100 * (scene_index + 1) + cut_index * 2 + 2]
            instance_ids = [f"fixture-object-{value}" for value in semantic_ids]
            heading = math.radians(scene_index * 90 + cut_index * 45)
            position = [5., 0., 5.]
            rotation = [0., -math.sin(heading / 2), 0., math.cos(heading / 2)]
            episode = EpisodeSpec(cut_id, cut_id, "generated", "fixture.glb", position, rotation, [], "local fixture")
            environment = Environment(episode, {"gpu": 0, "tool_tier": "super",
                                      "private_semantic": True, "fixture_semantic_ids": semantic_ids},
                                      tmp_path / "working" / cut_id / "environment", _FixtureSimulator)
            session = open_session(repo, config, tmp_path / "working" / cut_id / "session", environment,
                                   map_file, start_core=False)
            try:
                original_frame = session.current["frame"]
                reference_image = session.workspace / session.current["path"]
                session.begin_script("Find the pictured chair and go next to it.", images=[reference_image])
                target_frame = session._target_frames[0]
                np.save(session.private / "semantics" / f"{target_frame}.npy",
                        np.load(session.private / "semantics" / f"{original_frame}.npy"), allow_pickle=False)
                reply = session.script_tool("nav_observe", {"mode": "current"})
                assert not reply["result"]["isError"]
                frame = session.current["frame"]
                geometry = session.super_tools.geometry
                depth, camera = geometry.load_frame(frame)
                for semantic_id, instance_id, pixel in zip(semantic_ids, instance_ids, ([210, 300], [430, 300])):
                    point = Geometry._project(camera, np.asarray([pixel]), np.asarray([2.]))[0]
                    instances.append({"id": instance_id, "semantic_id": semantic_id, "category": "chair",
                                      "center": point.tolist(), "size": [.4, .8, .4]})
                destination = root / "snapshots" / cut_id
                session.snapshot(destination)
                masks = destination / "private/semantics"
                visible = {path.stem: instance_ids for path in masks.glob("*.npy")}
                snapshots.append(Snapshot(cut_id, scene_name, f"snapshots/{cut_id}",
                                          f"fixture-group:{scene_name}:{cut_index}", source="fixture",
                                          attributes={"target_instance": instance_ids[0], "target_frame": target_frame,
                                              "current_frame": frame, "frame_visibility": visible,
                                              "semantic_file": f"snapshots/{cut_id}/private/semantics/{frame}.npy",
                                              "target_semantic_file": f"snapshots/{cut_id}/private/semantics/{target_frame}.npy",
                                              "reference_success": True, "reference_steps": 2,
                                              "reference_actions": ["move_forward", "move_forward"],
                                              "reference_path_length": .5, "reference_endpoint": [5., 0., 4.5],
                                              "approach_points": [[5., 0., 4.5]],
                                              "fixture_only": True}))
            finally:
                session.close()
                environment.close()
        # An unseen category supplies a verifiable absence condition for 5a.
        instances.append({"id": f"{scene_name}-unseen-plant", "semantic_id": 9999,
                          "category": "plant", "center": [8., .5, 8.], "size": [.5, 1., .5]})
        instances_file = scene_dir / "instances.json"
        write_json(instances_file, {"instances": instances})
        scene_records.append(SceneRecord(scene_name, "fixture.glb", "fixture", "train", 0.,
                                        map_file.relative_to(root).as_posix(),
                                        instances_file.relative_to(root).as_posix(),
                                        {"position": [5., 0., 5.], "rotation_xyzw": [0., 0., 0., 1.]}))
    save_records(root / "scenes.jsonl", scene_records)
    save_records(root / "snapshots.jsonl", snapshots)
    return root, config, repo


def test_records_round_trip_and_reference_boundaries(tmp_path):
    record = TaskSample("sample", "5a", "qa", "ask", "cut", "Find a chair.", {}, {},
                        "samples/sample/private/label.json", "group")
    path = tmp_path / "samples.jsonl"
    save_records(path, [record])
    assert load_records(path, TaskSample) == [record]
    assert stable_id({"b": 2, "a": 1}) == stable_id({"a": 1, "b": 2})
    assert local_file(tmp_path, "samples.jsonl") == path
    external = tmp_path.parent / (tmp_path.name + "-external.json")
    external.write_text("{}")
    with pytest.raises(ValueError, match="external"):
        local_file(tmp_path, "../" + external.name)
    with pytest.raises(ValueError, match="Missing"):
        local_file(tmp_path, "missing.json")
    (tmp_path / "outside.json").symlink_to(external)
    with pytest.raises(ValueError, match="external"):
        local_file(tmp_path, "outside.json")
    with pytest.raises(ValueError, match="QA must use ask"):
        TaskSample(**{**asdict(record), "entry": "continue"})
    with pytest.raises(ValueError, match="Unsupported task"):
        TaskSample(**{**asdict(record), "task_id": "11a"})


def test_derive_and_check_all_eight_templates_with_local_fixture(frozen_fixture):
    root, config, repo = frozen_fixture
    samples = derive(root, config, repo)
    assert len(samples) == 32
    assert Counter(sample.task_id for sample in samples) == Counter(dict.fromkeys(TASK_IDS, 4))
    assert Counter(sample.form for sample in samples) == {"qa": 20, "decision": 8, "execution": 4}
    assert len(read_json(root / "teacher_selection.json")) == 16
    report = check(root)
    assert report["valid"], report["errors"]
    assert (root / "index.html").is_file()
    assert all(sample.attributes["source"] == "fixture" for sample in samples)
    assert not (root / "solutions.jsonl").exists()
    by_task = {task: [sample for sample in samples if sample.task_id == task] for task in TASK_IDS}
    assert {read_json(root / sample.label_file)["answer_kind"] for sample in by_task["5a"]} == {"pixel", "none"}
    assert {tuple(read_json(root / sample.label_file)["accepted_ids"]) for sample in by_task["6a"]} >= {("new",)}
    assert {read_json(root / sample.label_file)["same_instance"] for sample in by_task["6c"]} == {True, False}
    for sample in samples:
        public = json.dumps({"question": sample.question, "inputs": sample.inputs})
        assert "target_instance" not in public and "frame_visibility" not in public
        assert "fixture-object-" not in public
        for reference in sample.inputs.get("images", []):
            assert "/public/" in reference
        label = read_json(root / sample.label_file)
        if "mask_file" in label:
            assert np.load(root / label["mask_file"]).shape == (480, 640)
        if sample.entry == "continue":
            cut = next(cut for cut in load_records(root / "snapshots.jsonl", Snapshot) if cut.id == sample.snapshot)
            history = read_snapshot(root / cut.directory)["history"]
            assert history[-1]["role"] == "toolResult"
            assert history[-1]["toolCallId"] == history[-2]["content"][0]["id"]
            assert history[-2]["provider"] == "script"
    # Deriving again loads the frozen result and cannot silently change inputs.
    before = hashlib.sha256((root / "samples.jsonl").read_bytes()).hexdigest()
    assert derive(root, config, repo) == samples
    assert hashlib.sha256((root / "samples.jsonl").read_bytes()).hexdigest() == before


def test_checker_rejects_misaligned_masks_and_cross_split_groups(frozen_fixture):
    root, config, repo = frozen_fixture
    samples = derive(root, config, repo)
    sample = next(sample for sample in samples if sample.task_id == "5c")
    label = read_json(root / sample.label_file)
    np.save(root / label["mask_file"], np.ones((20, 20), dtype=np.int32))
    wrong = asdict(samples[1])
    wrong["group_id"] = samples[0].group_id
    wrong["split"] = "dev"
    save_records(root / "samples.jsonl", [samples[0], wrong, *samples[2:]])
    report = check(root)
    assert not report["valid"]
    assert any("crosses splits" in error["error"] for error in report["errors"])
    assert any("Mask does not align" in error["error"] for error in report["errors"])


def test_production_configuration_retains_ten_tasks_per_template():
    repo = Path(__file__).resolve().parents[1]
    config = read_config(repo / "config/generation/pilot.yaml")
    assert config["generation"]["per_task"] == 10
    assert {str(task) for task in config["generation"]["tasks"]} == set(TASK_IDS)
    assert config["model"] == "gpt-6.1-sol" and config["reasoning_effort"] == "high"
