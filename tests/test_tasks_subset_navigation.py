"""Reject plausible false passes in navigation task scoring."""
import math
from types import SimpleNamespace as NS

from src.bench.tasks_subset import _status
from src.dataset.tasks_subset.navigation.scoring import score_navigation


def _score(label, start, end, *, motions=(), nodes=(), groups=(), panorama=None, initial_nodes=(), prior_motions=(), answer="", submitted=False):
    env = NS(steps=max((m["step"] for m in motions), default=0), path_length=math.dist(start, end),
             motions=[*prior_motions, *motions], position=lambda: end,
             distance=lambda goals: min(math.dist(end, goal) for goal in goals))
    executor = NS(request=NS(submitted=submitted), memory=NS(nodes=nodes),
                  observations=NS(panorama=lambda _group: panorama))
    initial = {"steps": 0, "path_length": 0, "position": start, "nodes": [], "panoramas": groups, "memory": {"nodes": initial_nodes}}
    return score_navigation(env, executor, label, initial, {"answer": answer, "submitted": submitted})


def test_endpoint_is_not_intermediate_position():
    label = {"kind": "nav_01", "positions": [[2, 0, 0]]}
    assert not _score(label, [0, 0, 0], [0, 0, 0])["success"]
    assert not _score(label, [0, 0, 0], [1, 0, 0])["success"]
    assert _score(label, [0, 0, 0], [2, 0, 0])["success"]


def test_crossing_uses_initial_point_and_finite_correct_floor_opening():
    def room(identifier, left, right):
        return {"id": identifier, "level": "floor",
                "polygon_xz": [[left, -2], [right, -2], [right, 2], [left, 2]]}
    label = {"kind": "nav_02", "from_room": "left", "to_room": "right",
             "opening": {"level": "floor", "segment_xz": [[0, -.5], [0, .5]]},
             "scene": {"levels": [{"id": "floor", "floor_height": 0}],
                       "rooms": [room("left", -2, 0), room("right", 0, 2)]}}
    def cross(start, end):
        return _score(label, start, end, motions=[{"step": 1, "samples": [{"position": end}]}])["success"]
    assert cross([-.2, 0, 0], [.2, 0, 0])
    assert not cross([-.2, 0, 1], [.2, 0, 1])
    assert not cross([-.2, 1, 0], [.2, 0, 0])
    assert not cross([.2, 0, 0], [-.2, 0, 0])


def test_panorama_can_be_new_without_moving():
    label = {"kind": "nav_04", "room_names": ["kitchen"], "origin_uv": [0, 0], "radius_uv": .01}
    node = {"id": "N1", "room": "kitchen", "description": "Counter ahead", "panorama": "p_1"}
    panorama = {"pose": {"pos": [0, 0]}, "frames": list(range(6)), "step": 0}
    def status(nodes, groups=()):
        return _score(label, [0, 0, 0], [0, 0, 0], nodes=nodes, groups=groups, panorama=panorama)["status"]
    assert status([node]) == "needs_review"
    assert status([node], ["p_1"]) == "incorrect"
    assert status([{**node, "room": "garage"}]) == "incorrect"
    assert status([{**node, "description": " "}]) == "incorrect"


def test_wrong_instance_needs_motion_and_explicit_rejection():
    label = {"kind": "nav_05", "variant": "reference_match", "matches": False, "positions": [[1, 0, 0]]}
    point = [1, 0, 0]
    moved = [{"step": 1, "samples": [{"position": point}]}]
    assert not _score(label, point, point, answer="mismatch")["success"]
    assert not _score(label, [0, 0, 0], point, motions=moved)["success"]
    assert not _score(label, [0, 0, 0], point, motions=moved, answer="mismatch", submitted=True)["success"]
    assert _score(label, [0, 0, 0], point, motions=moved, answer="mismatch")["success"]


def test_runtime_classification_keeps_model_outcomes_separate():
    assert _status("action_budget") == "budget_exhausted"
    assert _status("runtime_error") == "infrastructure_error"
    assert _status("unsubmitted") == "completed"


def test_missing_review_is_not_verified_absence():
    from src.dataset.tasks_subset.navigation.construction import _object_candidates

    state = {"id": "cut", "room": "hall", "frame_keys": ["old", "now"], "current_frame": "now",
             "node_frames": {"N1": ["old"]}, "visited_rooms": ["hall"]}
    bundle = {"scene": {"objects": [{"id": "lamp", "room": "kitchen"}]},
              "states": [state], "frames": {"old": {}, "now": {}, "future": {}}}
    def candidates(kind, *rows):
        return list(_object_candidates({**bundle, "sightings": [], "navigation_evidence": {"visibility": [
            {"frame": frame, "visible": visible, "object": "lamp"} for frame, visible in rows]}}, kind))
    assert not candidates("nav_07", ("old", True))
    assert len(candidates("nav_07", ("old", True), ("now", False))) == 1
    assert not candidates("nav_10", ("old", True), ("now", False))
    assert len(candidates("nav_10", ("future", True), ("old", False), ("now", False))) == 1

def test_altered_snapshot_is_isolated_and_changes_one_field(tmp_path):
    import json
    from src.dataset.tasks_subset.navigation.construction import _altered_cut

    state = {"id": "cut", "snapshot": "states/cut/snapshot"}
    path = "workspace/memory/nodes.json"
    source = tmp_path / state["snapshot"] / path
    source.parent.mkdir(parents=True)
    before = {"nodes": [{"id": "N1", "room": "hall", "description": "mirror"},
                        {"id": "N2", "room": "kitchen", "description": "counter"}]}
    source.write_text(json.dumps(before))
    altered = _altered_cut({"states": [state]}, tmp_path, state,
                           {"node": "N1", "field": "room", "correct": "hall", "wrong": "kitchen"})
    assert json.loads(source.read_text()) == before
    modified = json.loads((tmp_path / altered["snapshot"] / path).read_text())
    assert modified["nodes"] == [{**before["nodes"][0], "room": "kitchen"}, before["nodes"][1]]


def test_object_approach_accepts_both_sides_and_rejects_other_room():
    def room(identifier, left, right):
        return {"id": identifier, "level": "floor",
                "polygon_xz": [[left, -2], [right, -2], [right, 2], [left, 2]]}
    scene = {"levels": [{"id": "floor", "floor_height": 0}],
             "rooms": [room("target", -2, 0), room("adjacent", 0, 2)]}
    region = {"room": "target", "floor_height": 0, "floor_tolerance": .25, "room_margin": .04,
              "rectangles": [{"center_xz": [-.25, 0], "size": [.2, .2], "angle_deg": 0}],
              "max_footprint_distance": .85}
    label = {"kind": "nav_05", "positions": [[-.12, 0, -.15], [-.12, 0, .15]], "success_distance": .32,
             "scene": scene, "approach_region": region}
    for position in label["positions"]:
        assert _score(label, [-1, 0, -1], position, submitted=True)["success"]
    assert not _score(label, [-1, 0, -1], [.12, 0, .15], submitted=True)["success"]


def test_replay_binding_rejects_changed_or_missing_snapshot(tmp_path):
    import pytest
    from src.dataset.tasks_subset.bundle import snapshot_digest
    from src.dataset.tasks_subset.navigation.construction import _verified_trial

    state = {"snapshot": "snapshot"}
    directory = tmp_path / state["snapshot"]
    directory.mkdir()
    (directory / "state.json").write_text("{}")
    memory = directory / "nodes.json"
    memory.write_text('{"room":"Room 1"}')
    calls = [{"name": "observe", "arguments": {"mode": "panorama"}}]
    trial = {"calls": calls, "replay": {"passed": True, "calls": calls}}
    with pytest.raises(ValueError, match="binding"):
        _verified_trial(trial, tmp_path, state)
    trial["start_snapshot_sha256"] = snapshot_digest(directory)
    assert _verified_trial(trial, tmp_path, state) == []
    memory.write_text('{"room":"Room 2"}')
    with pytest.raises(ValueError, match="changed"):
        _verified_trial(trial, tmp_path, state)


def test_repair_requires_both_destination_and_correction_to_matching_node():
    label = {"kind": "nav_12", "positions": [[2, 0, 0]],
             "repair": {"node": "N1", "field": "room", "wrong": "Room 2", "correct": "Room 1"}}
    old = [{"id": "N1", "room": "Room 2"}]
    fixed = [{"id": "N1", "room": "Room 1"}]
    start, goal = [0, 0, 0], [2, 0, 0]
    assert not _score(label, start, goal, initial_nodes=old, nodes=fixed)["success"]
    assert not _score(label, start, goal, initial_nodes=old, nodes=old)["success"]
    assert not _score(label, start, goal, initial_nodes=old, nodes=[{"id": "N2", "room": "Room 1"}])["success"]
    assert not _score(label, start, start, initial_nodes=old, nodes=fixed)["success"]
    assert not _score(label, start, goal, initial_nodes=fixed, nodes=fixed)["success"]


def test_bench_keeps_initial_binding_and_voids_score_on_input_change(tmp_path, monkeypatch):
    import json
    from src.bench import tasks_subset as bench

    task = tmp_path / "task.json"
    label = tmp_path / "label.json"
    label.write_text("{}")
    task.write_text(json.dumps({"id": "demo", "kind": "relative", "mode": "judgment", "label": label.name,
                                "limits": {}, "images": [], "instruction": "answer", "answer_schema": {}}))
    monkeypatch.setattr("src.dataset.tasks_subset.bundle.validate_frozen_inputs", lambda *_args: True)
    monkeypatch.setattr(bench, "credentials", lambda *_args: {"model": "fixture", "apiKey": "key"})
    def run(*_args):
        label.write_text('{"changed":true}')
        return {"reason": "window_end", "answer": "null"}
    monkeypatch.setattr(bench, "run_fixed_evidence", run)
    monkeypatch.setattr(bench, "score_answer", lambda *_args: {"success": True})
    result = bench.run_task(tmp_path, task, {}, tmp_path / "run")
    assert result["reason"] == "frozen_input_changed"
    assert result["changed_inputs"] == ["label_sha256"]
    assert len(result["task_sha256"]) == len(result["label_sha256"]) == 64
    assert result["snapshot_sha256"] is None
    assert "score" not in result


def test_current_panorama_excludes_future_frames_at_same_step():
    from src.dataset.tasks_subset.bundle import current_observation_frames

    base = {"step": 0, "pose": {"position": [0, 0, 0]},
            "public_pose": {"pos": [.5, .5], "heading": 0}}
    bundle = {"frames": {"initial": base, **{
        f"p-{yaw}": {**base, "group": "p_0", "yaw": yaw} for yaw in range(0, 360, 60)}}}
    start = {"current_frame": "initial", "frame_keys": ["initial"], "step": 0}
    assert current_observation_frames(bundle, start) == ["initial"]
    keys = [key for key in bundle["frames"] if key != "initial"]
    later = {**start, "current_frame": keys[0], "frame_keys": ["initial", *keys]}
    assert current_observation_frames(bundle, later) == keys


def test_reference_pair_changes_only_reference_and_private_answer(tmp_path):
    from PIL import Image
    from pathlib import Path
    from src.dataset.tasks_subset.navigation.construction import _instruction

    frames = {}
    for i, key in enumerate(("current", "same", "different")):
        path = tmp_path / f"{key}.png"
        Image.new("RGB", (24, 24), (0, 30 * i, 0)).save(path)
        frames[key] = {"rgb": str(path), "pose": {"position": [i, 0, 0]}}
    state = {"current_frame": "current", "frame_keys": list(frames), "room": "room"}
    sightings = [{"kind": "object", "id": obj, "frame": key, "pixel": [12, 12],
                  "semantic_verified": True, "identity_verified": True}
                 for obj, key in (("A", "current"), ("A", "same"), ("B", "different"))]
    pair = {"observed_object": "A", "requested_object": "B", "current_frame": "current",
            "reference_frames": ["different"], "same_category": True, "review_candidate": "pair",
            "public_context": [{"frame": "same", "caption": "Historical panorama, public pose"},
                               {"frame": "different", "caption": "Other historical panorama, public pose"}]}
    bundle = {"frames": frames, "sightings": sightings,
              "scene": {"objects": [{"id": "A", "room": "room", "name": "chair"}]},
              "targets": [{"id": "target", "object_id": "A", "positions": [[2, 0, 0]]}],
              "navigation_evidence": {"distinct_instances": [pair]}}
    def build(variant, calls):
        trial = {"id": variant, "variant": variant, "target": "target", "comparison_id": "pair",
                 "visible_frame": "current", "goal_measurement": {"positions": [[2, 0, 0]], "distance": 0., "initial_distance": 2.},
                 "replay": {"calls": calls, "goal_measurement": {"positions": [[2, 0, 0]], "distance": 0., "initial_distance": 2.}}}
        return _instruction("nav_05", trial, state, frames["current"], bundle, [[2, 0, 0]], tmp_path)

    pos, pos_label, pos_images = build("reference_match", [{"name": "submit"}])
    neg, neg_label, neg_images = build("wrong_instance", [])
    assert pos == neg
    assert [im["caption"] for im in pos_images] == [im["caption"] for im in neg_images]
    assert pos_images[0]["path"].read_bytes() == neg_images[0]["path"].read_bytes()
    assert pos_images[1]["source"] != neg_images[1]["source"]
    assert [Path(im["path"]).read_bytes() for im in pos_images[2:]] == [
        Path(im["path"]).read_bytes() for im in neg_images[2:]]
    assert pos_label["matches"] and not neg_label["matches"]
    assert not _score(pos_label, [0, 0, 0], [2, 0, 0], answer="mismatch")["success"]


def test_reference_budget_rejects_motion_and_observation_overruns():
    import pytest
    from src.dataset.tasks_subset.navigation.construction import _reference_budget

    trial = {"replay": {"motions": [{}, {}], "calls": [{"name": "observe"}] * 2}}
    points = [[1, 0, 0], [0, 0, 0]]
    start = [0, 0, 0]
    for limit, message in [({"max_actions": 2}, "action"), ({"max_path_length": 1.5}, "path"),
                           ({"max_observations": 1}, "observation")]:
        with pytest.raises(ValueError, match=message):
            _reference_budget(trial, points, start, limit)
    assert _reference_budget(trial, points, start, {"max_requests": 1})["script_tool_calls"] == 2


def test_known_room_route_uses_prefix_events_only():
    from src.dataset.tasks_subset.navigation.construction import _intervening_room
    state = {"visited_rooms": ["A", "B", "C"], "traversals": [
        {"from_room": "A", "to_room": "B"}, {"from_room": "B", "to_room": "C"}]}
    assert _intervening_room(state, "A", "C")
    assert not _intervening_room(state, "A", "B")
    assert not _intervening_room({**state, "traversals": []}, "A", "C")


def test_close_across_wall_or_exact_threshold_does_not_count_as_arrival():
    import pytest
    from src.dataset.tasks_subset.navigation.construction import _require_goal_measurements

    end = [0.05, 0, 0]
    label = {"kind": "nav_01", "positions": [[0, 0, 0]], "success_distance": .35}
    branch = {"positions": label["positions"], "distance": .05}
    for distance in (2.0, .35, None):
        replay = {**branch, "distance": distance}
        with pytest.raises(ValueError, match="geodesic arrival"):
            _require_goal_measurements({"goal_measurement": branch, "replay": {"goal_measurement": replay}}, label, end)
        env = NS(steps=0, path_length=0, motions=[], position=lambda: end,
                 distance=lambda _: math.inf if distance is None else distance)
        assert not score_navigation(env, NS(request=NS(submitted=False)), label,
                                    {"position": end, "steps": 0, "path_length": 0})["success"]
    trial = {"goal_measurement": branch, "replay": {"goal_measurement": branch}}
    _require_goal_measurements(trial, label, end)


def test_visible_point_producer_passes_explicit_goals_to_branch(tmp_path, monkeypatch):
    from src.dataset.tasks_subset.bundle import write_json
    from src.dataset.tasks_subset.sampling import collection

    world = [0, 0, 1]
    bundle = {"seed": 0, "scene": {}, "episode": {"id": "x", "source_id": "x", "benchmark": "r2r",
              "scene_id": "x", "start_position": [0, 0, 0], "start_rotation": [0, 0, 0, 1],
              "goals": [], "source_file": "fixture"},
              "states": [{"id": "start"}, {"id": "stop", "current_frame": "f"}],
              "frames": {"f": {"pose": {"position": [0, 0, 0]}}},
              "sightings": [{"id": "point", "kind": "point", "frame": "f", "world": world}],
              "trials": [], "targets": [], "official_tasks": []}
    write_json(tmp_path / "private/bundle.json", bundle)
    def branch(bundle, root, config, episode, state, kind, operation, **fields):
        assert kind == "nav_01"
        assert fields["goal_positions"] == [world]
        return fields
    monkeypatch.setattr(collection, "_branch", branch)
    monkeypatch.setattr(collection, "read_depth", lambda *_args: None)
    monkeypatch.setattr(collection, "floor_patch_fraction", lambda *_args: 1.)
    bundle["frames"]["f"].update(depth="depth.png", camera={}, world_from_optical=[])
    bundle["states"][1]["transform"] = {"floor_height": 0}
    bundle["sightings"][0]["pixel"] = [20, 20]
    write_json(tmp_path / "private/bundle.json", bundle)
    output = collection.collect_trials({"assets": "."}, tmp_path, 1, ["nav_01"])
    assert output["targets"][0]["positions"] == [world]


def test_safe_waypoint_recovery_can_go_back_but_must_reach_waypoint():
    label = {"kind": "nav_03", "variant": "safe_waypoint", "positions": [[-1, 0, 0]],
             "success_distance": .30}
    start = [0, 0, 0]
    blocked = [{"step": 0, "blocked": True}]
    for end, expected in [([-1, 0, 0], True), ([-.2, 0, 0], False), (start, False)]:
        assert _score(label, start, end, prior_motions=blocked)["success"] is expected
    assert not _score(label, start, [-1, 0, 0])["success"]


def test_other_unknown_nodes_and_prefix_views_cannot_be_treated_as_absent():
    from src.dataset.tasks_subset.navigation.construction import _known_once, _reviewed_absent
    state = {"node_frames": {"seen": ["seen"], "unknown": [f"f{i}" for i in range(6)]}}
    bundle = {"navigation_evidence": {"visibility": []}}
    assert not _known_once(bundle, state, "lamp", {"seen"})
    for key in state["node_frames"]["unknown"]:
        bundle["navigation_evidence"]["visibility"].append({"object": "lamp", "frame": key, "visible": False})
    assert _known_once(bundle, state, "lamp", {"seen"})
    assert not _reviewed_absent(bundle, "lamp", ["unknown-history"])
    assert not _reviewed_absent(bundle, "lamp", [])


def test_bench_rejects_tampered_seal_before_credentials_or_model(tmp_path, monkeypatch):
    import json
    from src.bench.tasks_subset import run_task

    (tmp_path / "label.json").write_text("{}")
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"id": "x", "kind": "nav_01", "mode": "navigation", "label": "label.json",
                                "images": [], "inputs_sha256": {"label": "incorrect"}}))
    def forbidden(*_args):
        raise AssertionError("model credentials must not be accessed")
    monkeypatch.setattr("src.bench.tasks_subset.credentials", forbidden)
    result = run_task(tmp_path, task, {}, tmp_path / "run")
    assert result["reason"] == "frozen_input_mismatch"
    assert "score" not in result


def test_correction_requires_independent_complete_alias_references():
    from src.dataset.tasks_subset.navigation.construction import _correction_comparison
    nodes = [{"id": "subject", "room": "Room 1"}, {"id": "same", "room": "Room 1"},
             {"id": "other", "room": "Room 2"}]
    frames, indexed = {}, {}
    for node in nodes:
        indexed[node["id"]] = []
        for yaw in range(0, 360, 60):
            key = f"{node['id']}-{yaw}"
            indexed[node["id"]].append(key)
            frames[key] = {"yaw": yaw, "group": "p_" + node["id"],
                           "public_pose": {"pos": [.4, .6], "heading": 10}}
    state = {"node_frames": indexed, "frame_keys": list(frames)}
    bundle = {"frames": frames}
    assert _correction_comparison(bundle, state, nodes[:1], nodes[0], "Room 2") is None
    keys, prompt = _correction_comparison(bundle, state, nodes, nodes[0], "Room 2")
    assert len(keys) == 18
    assert "Images 7–12" in prompt and "Room 1" in prompt and "capture pose" in prompt
    state["frame_keys"].remove("same-60")
    assert _correction_comparison(bundle, state, nodes, nodes[0], "Room 2") is None


def test_absence_review_producer_includes_intermediate_history(tmp_path):
    from src.dataset.tasks_subset.navigation.construction import navigation_review_proposals
    frames = {key: {} for key in ("initial", "middle", "current", "target")}
    bundle = {"frames": frames, "scene": {"objects": [{"id": "chair", "name": "chair", "room": "R"}]},
              "sightings": [{"kind": "object", "id": "chair", "frame": "target", "pixel": [4, 4]}],
              "states": [{"id": "cut", "room": "other", "snapshot": "missing",
                          "current_frame": "current", "frame_keys": ["initial", "middle", "current"],
                          "visited_rooms": ["other"], "node_frames": {}}]}
    proposals = list(navigation_review_proposals(bundle, tmp_path))
    assert any(row["fact"]["value"] == {"object": "chair", "frame": "middle", "visible": False}
               and row["frames"] == ["middle", "target"] for row in proposals)


def test_execution_semantics_cannot_pass_with_one_review_or_stale_binding(tmp_path):
    import pytest
    from src.dataset.tasks_subset.bundle import write_json
    from src.dataset.tasks_subset.review.execution import finalize_execution, _hash
    path = tmp_path / "private/execution-review/manifest.json"
    write_json(path, {"structural_score": {"details": {"event": True}}, "room_assignment_valid": True, "nodes": [{"role": "created", "node": {"room": "Room 1"}, "images": []},
                {"role": "room_reference", "node": {"room": "Room 1"}, "images": []}]})
    row = {"reviewer": "first", "model": "model-a", "manifest_sha256": _hash(path),
           "answer": {"decision": "accept", "description_supported": True, "room_supported": True}}
    assert not finalize_execution(tmp_path, [row])["success"]
    assert finalize_execution(tmp_path, [row, {**row, "reviewer": "second", "model": "model-b"}])["success"]
    write_json(path, {"changed": True})
    with pytest.raises(ValueError, match="different execution"):
        finalize_execution(tmp_path, [row])


def test_execution_reviewer_identity_and_typed_new_room(tmp_path):
    import pytest
    from src.dataset.tasks_subset.bundle import write_json
    from src.dataset.tasks_subset.review.execution import finalize_execution, _hash
    path = tmp_path / "private/execution-review/manifest.json"
    manifest = {"structural_score": {"details": {"event": True}}, "room_assignment_valid": True,
                "room_assignment_mode": "first_room",
                "nodes": [{"role": "created", "node": {"room": "Room 1"}, "images": []}]}
    write_json(path, manifest)
    def rows():
        return [{"reviewer": name, "model": model, "manifest_sha256": _hash(path),
                 "answer": {"decision": "accept", "description_supported": True, "room_supported": False}}
                for name, model in [("a", "model-a"), ("b", "model-b")]]
    assert finalize_execution(tmp_path, rows())["success"]
    duplicate = rows()
    duplicate[1]["model"] = duplicate[0]["model"]
    with pytest.raises(ValueError, match="distinct models"):
        finalize_execution(tmp_path, duplicate)
    manifest["room_assignment_mode"] = "new_room"
    manifest["nodes"][0]["node"]["room"] = "bedroom"
    write_json(path, manifest)
    assert not finalize_execution(tmp_path, rows())["success"]


def test_correction_reuse_requires_exact_comparison_prefix_and_node_assignments(tmp_path):
    from copy import deepcopy
    from src.dataset.tasks_subset.bundle import write_json
    from src.dataset.tasks_subset.navigation.construction import correction_applicable
    frames, node_frames, nodes = {}, {}, []
    for index in range(3):
        identifier = f"N{index}"
        keys = [f"{identifier}-{yaw}" for yaw in range(6)]
        node_frames[identifier] = keys
        frames.update({key: {"group": f"p_{index}"} for key in keys})
        nodes.append({"id": identifier, "anchor": f"p_{index}", "panorama": f"p_{index}",
                      "room": "Room 1" if index < 2 else "Room 2"})
    source = {"id": "later", "snapshot": "later", "frame_keys": list(frames), "node_frames": node_frames,
              "room_aliases": {"A": "Room 1", "B": "Room 2", "C": "Room 3"}}
    earlier = {**source, "id": "earlier", "snapshot": "earlier", "room_aliases": {"A": "Room 1", "B": "Room 2"}}
    for s in (source, earlier):
        write_json(tmp_path / s["snapshot"] / "workspace/memory/nodes.json", {"nodes": nodes})
    bundle = {"states": [source, earlier], "frames": frames}
    fact = {"state": "later", "review_candidate": "review", "review_source": {"sha": "original"},
            "node": "N0", "field": "room", "correct": "Room 1", "wrong": "Room 2",
            "evidence_frame": "N0-0", "comparison_frames": list(frames)}
    assert correction_applicable(bundle, earlier, fact, tmp_path)
    assert not correction_applicable(bundle, {**earlier, "frame_keys": ["N0-0"]}, fact, tmp_path)
    changed = deepcopy(nodes); changed[2]["room"] = "Room 1"
    write_json(tmp_path / "earlier/workspace/memory/nodes.json", {"nodes": changed})
    assert not correction_applicable(bundle, earlier, fact, tmp_path)


def test_repair_order_requires_new_local_panorama_before_committed_edit():
    import json
    from src.dataset.tasks_subset.navigation.scoring import observed_before_repair
    repair = {"node": "N1", "field": "room", "correct": "Room 1",
              "anchor_uv": [.5, .5], "map_size_m": 20, "radius_m": 1., "floor_height": 0}
    def receipt(name, args, value):
        return json.dumps([name, args]), {"result": {"isError": False, "content": [{"text": json.dumps(value)}]}}
    observe = receipt("observe", {"mode": "panorama"}, {"group": "p_new"})
    edit = receipt("memory_edit", {"operations": [{"op": "update", "id": "N1", "room": "Room 1"}]},
                   {"committed": True})
    pano = {"step": 4, "frames": [1, 2, 3, 4, 5, 6], "pose": {"pos": [.51, .5]}}
    env = NS(motions=[{"step": 4, "samples": [{"position": [0, 0, 0]}]}])
    initial = {"position": [0, 0, 0], "receipt_ids": [], "panoramas": []}
    executor = NS(_calls={"observe": observe, "edit": edit}, observations=NS(panorama=lambda _: pano))
    assert observed_before_repair(env, executor, repair, initial)
    executor._calls = {"edit": edit, "observe": observe}
    assert not observed_before_repair(env, executor, repair, initial)
    executor._calls = {"observe": observe, "edit": edit}
    pano["pose"]["pos"] = [.8, .5]
    assert not observed_before_repair(env, executor, repair, initial)


def test_derived_observation_requires_current_bound_asset_bytes(tmp_path):
    import hashlib
    from src.dataset.tasks_subset.navigation.construction import _target_sightings
    for name in ("rgb", "depth", "mask"):
        (tmp_path / name).write_bytes(name.encode())
    frame = {"rgb": "rgb", "depth": "depth", "pose": {"position": [0, 0, 0]}, "camera": {"K": []}}
    binding = {name + "_sha256": hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()
               for name in ("rgb", "depth", "mask")}
    binding.update(pose=frame["pose"], camera=frame["camera"])
    seed = {"id": "A", "frame": "old", "semantic_verified": True, "review_candidate": "review"}
    derived = {"id": "A", "frame": "now", "source_frame": "old", "method": "native_instance_reprojection",
               "identity_verified": True, "derived_from_review": "review", "evidence_binding": binding}
    bundle = {"sightings": [seed], "derived_sightings": [derived], "frames": {"now": frame},
              "instance_evidence": {"now": {"mask": "mask"}}}
    assert list(_target_sightings(bundle, tmp_path)) == [seed, derived]
    (tmp_path/"depth").write_bytes(b"changed")
    assert list(_target_sightings(bundle, tmp_path)) == [seed]


def test_execution_uncertain_is_not_promoted_by_positive_components(tmp_path):
    from src.dataset.tasks_subset.bundle import write_json
    from src.dataset.tasks_subset.review.execution import finalize_execution, _hash
    path = tmp_path / "private/execution-review/manifest.json"
    write_json(path, {"structural_score": {"details": {"event": True}},
                     "room_assignment_valid": True, "room_assignment_mode": "first_room",
                     "nodes": [{"role": "created", "node": {"room": "Room 1"}, "images": []}]})
    rows = [{"reviewer": model, "model": model, "manifest_sha256": _hash(path),
             "answer": {"decision": decision, "description_supported": True, "room_supported": True}}
            for model, decision in [("model-a", "accept"), ("model-b", "uncertain")]]
    result = finalize_execution(tmp_path, rows)
    assert result["success"] is None and result["status"] == "needs_review"
