"""Independent checks of frozen inputs; evidence quality needs explicit review."""
from hashlib import sha256
from pathlib import Path
import math

from PIL import Image

from src.common.actions import ACTIONS
from src.common.coordinates import map_pose, validate_map, world_to_uv
from src.common.io import read_json
from src.dataset.episodes import EpisodeSpec
from src.harness.spatial.observations import Observations
from src.harness.errors import ToolInputError

from ..bundle import require_trial_snapshot, snapshot_digest, validate_frozen_inputs
from ..catalog import JUDGMENT_KINDS, NAVIGATION_KINDS
from ..judgment.scoring import score_answer


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _path(root, base, value):
    value = Path(value)
    result = (base / value).resolve(strict=True)
    _need(not value.is_absolute() and result.is_relative_to(root), "reference escapes collection")
    return result


def _sha(path):
    return sha256(path.read_bytes()).hexdigest()


def _snapshot(root, directory, state, frames):
    saved = read_json(directory / "state.json")
    _need(saved["memory_type"] == "node" and saved["workspace"] == "workspace",
          "expected a node snapshot with local workspace")
    workspace = directory / "workspace"
    map_data = read_json(workspace / "scene/map.json")
    validate_map(map_data)
    _need(map_data["walls"] == [] and map_data == state["transform"], "snapshot blank map differs")
    published = read_json(workspace / "scene/state.json")
    _need(set(published) == {"memory_type", "pose", "frame", "step"}, "unexpected public state fields")
    observations = Observations(workspace, map_data)
    records = [frames[k] for k in state["frame_keys"]]
    _need(list(observations.items) == [r["frame"] for r in records], "snapshot frame index differs from prefix")
    allowed = {"scene/map.json", "scene/state.json", "scene/navlib.py", "scene/coordinates.py",
               "memory/nodes.json", "observations/index.jsonl", "observations/views.jsonl"}
    for row in records:
        observation = observations.get(row["frame"])
        _need(row["public_pose"] == observation["pose"] == map_pose(row["pose"], map_data) and row["step"] == observation["step"]
              and row["step"] <= state["step"], "non-prefix or mismatched public observation")
        path = _path(root, workspace, observation["path"])
        _need(path.parent == workspace / "observations" and _sha(path) == observation["sha256"],
              "snapshot image path or digest differs")
        allowed.add(observation["path"])
    for view in observations.views.values():
        _need(all(t["image"] in observations.items for t in view["sources"]),
              "derived view references non-prefix frames")
        _need(_path(root, workspace, view["path"]).parent == workspace / "observations/derived",
              "derived view outside its public directory")
        allowed.add(view["path"])
    for path in workspace.rglob("*"):
        _need(not path.is_symlink(), "public workspace contains a symlink")
        if path.is_file():
            _need(path.relative_to(workspace).as_posix() in allowed, "unexpected public workspace file")
    current = frames[state["current_frame"]]
    _need(state["current_frame"] in state["frame_keys"] and saved["current_frame"] == published["frame"] == current["frame"]
          and published["step"] == state["step"] == saved["environment"]["steps"]
          and published["pose"] == current["public_pose"] and published["memory_type"] == "node",
          "snapshot current observation differs")
    environment = saved["environment"]
    if "public_trajectory" in state:
        history = [(0, environment["trajectory"][0]), *(
            (m["step"], sample["position"]) for m in environment["motions"] for sample in m["samples"])]
        _need([(p["step"], p["pos"]) for p in state["public_trajectory"]] ==
              [(step, world_to_uv(pos, map_data)) for step, pos in history],
              "public trajectory does not match executed prefix")
    memory = read_json(workspace / "memory/nodes.json")
    _need(set(memory) == {"nodes", "rooms"}, "unexpected public memory fields")
    _need({n["id"] for n in memory["nodes"]} == set(state["node_frames"]), "node IDs differ from prefix")
    for node in memory["nodes"]:
        _need(set(node) == {"id", "anchor", "panorama", "room", "description", "history"},
              "unexpected public node fields")
        groups = [node["anchor"], node["panorama"], *node["history"]]
        for group in groups:
            _need(len(observations.panorama(group)["frames"]) == 6, "incomplete public node panorama")
        keys = state["node_frames"][node["id"]]
        _need(len(keys) == 6 and set(keys) <= set(state["frame_keys"])
              and all(frames[k]["group"] == node["panorama"] for k in keys),
              "node frame mapping differs from snapshot panorama")


def _replay(trial, state, frames):
    replay = trial["replay"]
    calls = trial["calls"]
    _need(trial["start_state"] == state["id"] and replay.get("passed") is True
          and calls and replay["calls"] == calls, "reference trial lacks matching independent replay")
    for call in calls:
        if call["name"] != "move":
            continue
        args = call["arguments"]
        _need(args.get("mode") == "steps" and 1 <= len(args["actions"]) <= 6
              and set(args["actions"]) <= set(ACTIONS), "non-atomic reference move")
    a, b = trial["end_frame"], replay["end_frame"]
    _need(a in frames and b in frames and a != b and frames[a]["stream"] != frames[b]["stream"]
          and a not in state["frame_keys"] and b not in state["frame_keys"], "replay frames not independent")
    _need(trial["motions"] == replay["motions"], "motion records differ in replay")
    _need(math.dist(frames[a]["pose"]["position"], frames[b]["pose"]["position"]) <= .03,
          "replay endpoint differs")


def _source_hashes(bundle):
    source = bundle["source"]
    directory = Path(source["map_directory"])
    for name, expected in source["sha256"].items():
        _need(Path(name).name == name and _sha(directory / name) == expected, f"scene source changed: {name}")
    split = source["split_evidence"]
    _need(bundle["split"] == split["split"] == "train"
          and _sha(Path(split["episode_shard"])) == split["sha256"], "training source changed")


def _physical_references(root, state, task, construction, label, bundle):
    if task["mode"] == "navigation":
        trials = [construction["trial"]] if construction.get("trial") else []
    else:
        identifiers = label.get("trials", [])
        if label.get("trial"):
            identifiers = [label["trial"], *identifiers]
        indexed = {trial["id"]: trial for trial in bundle["trials"]}
        _need(all(identifier in indexed for identifier in identifiers), "required replay reference missing")
        trials = [indexed[identifier] for identifier in dict.fromkeys(identifiers)]
    if not trials:
        _need(task["mode"] != "navigation" and not construction.get("physical_replay"),
              "required replay reference missing")
        return None
    for trial in trials:
        _need(any(candidate == trial for candidate in bundle["trials"]), "trial not bound to collection")
        require_trial_snapshot(root, state, trial)
        _replay(trial, state, bundle["frames"])
    return True


def _check_case(root, directory, task, construction, label, bundle, states, snapshot_errors):
    validate_frozen_inputs(task, directory)
    kind = task["kind"]
    judgment = kind in JUDGMENT_KINDS
    _need(task["mode"] == ("judgment" if judgment else "navigation"), "kind and mode disagree")
    _need(task["id"] == directory.name and bool(task["instruction"].strip()), "invalid case identity or empty prompt")
    state = states[task["provenance"]["state_id"]]
    _need(task["provenance"]["scene_id"] == bundle["scene_id"], "case refers to wrong scene")
    _need(_path(root, directory, task["snapshot"]) == _path(root, root, state["snapshot"]), "case snapshot differs")
    _need(state["id"] not in snapshot_errors, "case depends on invalid snapshot")
    _need(_path(root, directory, task["label"]) == directory / "private/label.json", "label outside case private directory")
    if judgment:
        _need(task["answer_schema"] == label["answer_schema"] and label["accepted"]
              and score_answer(label["accepted"][0], label)["success"], "judgment label/schema mismatch")
    else:
        EpisodeSpec(**read_json(_path(root, directory, task["episode"])))
        _need(label["kind"] == kind, "navigation label kind mismatch")
    prefix = set(state["frame_keys"])
    frames = bundle["frames"]
    targets = {f for t in bundle["targets"] for f in t.get("reference_frames", [])}
    images = task["images"]
    sources = construction.get("image_sources", [])
    _need(len(images) == len(sources), "missing per-image provenance")
    for image, source in zip(images, sources):
        path = _path(root, directory, image["path"])
        _need(path.parent == directory / "public" and image["sha256"] == _sha(path), "public image path or digest differs")
        with Image.open(path) as rgb:
            _need(rgb.mode == "RGB", "non-RGB public image")
            rgb.verify()
        keys = set(source["frames"])
        purpose = source["purpose"]
        if purpose == "target_reference":
            _need(keys and keys <= targets and all(not frames[k]["stream"].startswith(("branch-", "replay-")) for k in keys),
                  "target image is not a declared non-trial reference")
        else:
            _need(purpose in {"prefix", "prefix_map"} and keys and keys <= prefix,
                  "public image source is not in the prefix")
            if purpose == "prefix_map":
                _need(keys == prefix, "prefix map provenance omits part of the public prefix")
    if judgment:
        _need(set(construction.get("source_frames", [])) <= prefix | targets,
              "judgment construction refers to non-prefix evidence")
    return state


def check_collection(root, bundle_path=None, invalidations_path=None):
    root = Path(root).resolve()
    bundle = read_json(bundle_path or root / "private/bundle.json")
    catalog = {**JUDGMENT_KINDS, **NAVIGATION_KINDS}
    stages = ("proposed", "produced", "structural", "physical", "visual", "model_tried", "training_ready")
    coverage = {kind: {"constructor_registered": True, **dict.fromkeys(stages, 0), "rejections": {}}
                for kind in catalog}
    invalidated = {}
    invalidations_path = Path(invalidations_path) if invalidations_path else root / "invalidations.json"
    if invalidations_path.exists():
        document = read_json(invalidations_path)
        _need("old" not in document or Path(document["old"]).resolve() == root,
              "supersedes record names a different original collection")
        invalidated = {row["case"]: row["reason"] for row in document["invalidated"]}
    states = {s["id"]: s for s in bundle["states"]}
    snapshots = {}
    for state in states.values():
        try:
            _snapshot(root, _path(root, root, state["snapshot"]), state, bundle["frames"])
        except (OSError, ValueError, KeyError, TypeError, IndexError, ToolInputError) as exc:
            snapshots[state["id"]] = str(exc)
    source_error = None
    try:
        _source_hashes(bundle)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        source_error = str(exc)
    reject_path = root / "rejections.jsonl"
    if reject_path.exists():
        import json
        for line in reject_path.read_text().splitlines():
            row = json.loads(line)
            if row.get("kind") in coverage:
                data = coverage[row["kind"]]
                data["proposed"] += 1
                reason = row["reason"]
                data["rejections"][reason] = data["rejections"].get(reason, 0) + 1
    cases = []
    for path in sorted((root / "cases").glob("*/task.json")):
        report = {"path": path.relative_to(root).as_posix(), **dict.fromkeys(stages[2:])}
        cases.append(report)
        try:
            task = read_json(path)
            kind = task["kind"]
            report["kind"] = kind
            coverage[kind]["produced"] += 1
            coverage[kind]["proposed"] += 1
            construction = read_json(path.with_name("construction.json"))
            label_path = _path(root, path.parent, task["label"])
            label = read_json(label_path)
            state = _check_case(root, path.parent, task, construction, label, bundle, states, snapshots)
            _need(source_error is None, "collection source verification failed")
            report["structural"] = True
            if task["id"] in invalidated:
                report.update(invalidated=invalidated[task["id"]], training_ready=False)
                coverage[kind]["structural"] += 1
                continue
            report["physical"] = _physical_references(root, state, task, construction, label, bundle)
            review_path = path.parent / "private/review.json"
            if review_path.exists():
                review = read_json(review_path)
                _need(review["task_sha256"] == _sha(path) and review["label_sha256"] == _sha(label_path)
                      and review.get("snapshot_sha256") == snapshot_digest(root / state["snapshot"]),
                      "case review hashes differ")
                report["visual"] = review.get("visual") == "passed" and bool(review.get("reviewer"))
                report["model_tried"] = bool(review.get("model_attempt"))
                report["training_ready"] = (report["visual"] and review.get("label") == "passed"
                    and review.get("public_evidence") == "passed")
        except (OSError, ValueError, KeyError, TypeError, IndexError, ToolInputError) as exc:
            report["error"] = str(exc)
        if report.get("kind") in coverage:
            for stage in stages[2:]:
                coverage[report["kind"]][stage] += int(report[stage] is True)
    return {"scene_id": bundle["scene_id"], "source_error": source_error, "snapshot_errors": snapshots,
            "coverage": coverage, "cases": cases}
