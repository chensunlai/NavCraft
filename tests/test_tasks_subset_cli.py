import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.dataset.tasks_subset.__main__ import main
from src.dataset.tasks_subset.bundle import write_json


class SubsetCLITests(unittest.TestCase):
    def test_stage_bundles_are_explicit_and_existing_cases_survive_check(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            scene = "hm3d/train/example.glb"
            root = output / "scenes" / scene.replace("/", "__")
            config = output / "config.json"
            bundle = {"scene_id": scene, "source": {"sha256": {}}, "frames": {}, "states": [],
                      "trials": [], "scene": {"objects": []}, "sightings": []}
            write_json(config, {"scenes": [scene]})
            raw = root / "private/bundle.json"
            write_json(raw, bundle)
            original = raw.read_bytes()
            base = ["--config", str(config), "--output", str(output)]
            with patch("src.dataset.tasks_subset.review.stage.model_review") as model:
                main(base + ["--phase", "review", "--bundle", "private/bundle.json"])
                model.assert_not_called()
            reviewed = root / "private/reviewed_bundle.json"
            previous_review = reviewed.read_bytes()
            main(base + ["--phase", "enrich", "--bundle", "private/reviewed_bundle.json"])
            enriched = root / "private/enriched_bundle.json"
            with patch("src.dataset.tasks_subset.__main__.collect_trials", return_value=bundle) as trials, \
                 patch("src.dataset.tasks_subset.navigation.construction.prepare_navigation"):
                main(base + ["--phase", "trials", "--trial", "nav_02", "--bundle", "private/enriched_bundle.json"])
                self.assertEqual(trials.call_args.args[-1], enriched)
            self.assertEqual(raw.read_bytes(), original)
            self.assertEqual(reviewed.read_bytes(), previous_review)
            write_json(root / "manifest.json", {"cases": [{"kind": "nav_02"}]})
            with patch("src.dataset.tasks_subset.review.check.check_collection", return_value={}) as check, \
                 patch("src.dataset.tasks_subset.review.inspection.write_inspection"):
                main(base + ["--phase", "check", "--bundle", "private/enriched_bundle.json"])
                self.assertEqual(check.call_args.args[1], enriched)
            self.assertEqual(json.loads((root / "manifest.json").read_text())["cases"], [{"kind": "nav_02"}])

    def test_review_same_model_different_effort_is_one_attempt(self):
        from src.dataset.tasks_subset.review.stage import run_review

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "private/bundle.json"
            write_json(source, {})
            low, high = root / "low.json", root / "high.json"
            write_json(low, {"model": "same-model", "reasoning_effort": "low"})
            write_json(high, {"model": "same-model", "reasoning_effort": "high"})
            candidate = {"candidate_id": "abc", "kind": "object", "source": {}}
            with patch("src.dataset.tasks_subset.review.stage.review_candidates", return_value=[candidate]), \
                 patch("src.dataset.tasks_subset.review.stage.apply_reviews", return_value={}), \
                 patch("src.dataset.tasks_subset.review.stage.model_review", side_effect=TimeoutError) as model:
                config = {"review": {"models": [low, high]}}
                run_review({}, root, source, config)
                run_review({}, root, source, config)
                model.assert_called_once()
            entries = (root / "private/reviews.jsonl").read_text().splitlines()
            self.assertEqual(len(entries), 1)
            entry = json.loads(entries[0])
            self.assertEqual((entry["reviewer"], entry["reasoning_effort"], entry["decision"]),
                             ("same-model", "low", "uncertain"))

    def test_incremental_semantics_keeps_other_scene_index(self):
        with TemporaryDirectory() as temp:
            output = Path(temp)
            scene = "hm3d/train/test.glb"
            root = output / "scenes" / scene.replace("/", "__")
            config = output / "config.json"
            write_json(config, {"scenes": [scene]})
            bundle = {"scene_id": scene, "source": {}, "states": [], "trials": [], "frames": {}}
            source = root / "private/instance_bundle.json"
            write_json(source, bundle)
            write_json(output/"manifest.json", {"scenes": [{"scene_id": "other", "manifest": "other/manifest.json"}]})
            with patch("src.dataset.tasks_subset.sampling.semantics.collect_instance_evidence", return_value=bundle):
                main(["--config", str(config), "--output", str(output), "--phase", "semantics",
                      "--bundle", "private/instance_bundle.json"])
            top = json.loads((output/"manifest.json").read_text())
            self.assertEqual({s["scene_id"] for s in top["scenes"]}, {scene, "other"})
            stage = json.loads((root/"manifest.json").read_text())["stage"]
            self.assertEqual(stage["input_bundle"]["path"], str(source.resolve()))
            self.assertEqual(stage["output_bundle"]["path"], str(source.resolve()))

    def test_build_persists_generated_state_for_check(self):
        with TemporaryDirectory() as temp:
            output = Path(temp)
            scene = "hm3d/train/test.glb"
            root = output / "scenes" / scene.replace("/", "__")
            config = output / "config.json"
            write_json(config, {"scenes": [scene]})
            bundle = {"scene_id": scene, "source": {}, "states": [], "trials": [], "frames": {}}
            raw = root / "private/bundle.json"
            write_json(raw, bundle)
            before = raw.read_bytes()
            base = ["--config", str(config), "--output", str(output)]
            def build(value, directory, options):
                value["states"].append({"id": "generated-state"})
                return []
            with patch("src.dataset.tasks_subset.judgment.construction.build_judgments", side_effect=build), \
                 patch("src.dataset.tasks_subset.navigation.construction.build_navigation", return_value=[]):
                main(base + ["--phase", "build"])
            built = root / "private/built_bundle.json"
            self.assertEqual(json.loads(built.read_text())["states"], [{"id": "generated-state"}])
            self.assertEqual(raw.read_bytes(), before)
            stage = json.loads((root / "manifest.json").read_text())["stage"]
            self.assertEqual(stage["output_bundle"]["path"], str(built.resolve()))
            def check(directory, source, invalidations):
                self.assertEqual(json.loads(source.read_text())["states"], [{"id": "generated-state"}])
                return {}
            with patch("src.dataset.tasks_subset.review.check.check_collection", side_effect=check), \
                 patch("src.dataset.tasks_subset.review.inspection.write_inspection"):
                main(base + ["--phase", "check", "--bundle", "private/built_bundle.json"])
            with self.assertRaises(SystemExit):
                main(base + ["--phase", "build", "--bundle", "private/built_bundle.json"])
