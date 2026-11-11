from collections import Counter
import json

import pytest
from PIL import Image

from src.bench import tasks_subset as bench
from src.dataset.tasks_subset.catalog import JUDGMENT_KINDS
from src.dataset.tasks_subset.judgment.scoring import score_answer
from src.dataset.tasks_subset.judgment.construction import (
    Context, _rooms, _views, arr, build_judgments, ego, enum, num, sector, shortest_routes,
)
from src.dataset.tasks_subset.review.evidence import apply_reviews, review_candidates


def _bundle(tmp_path):
    snapshot = tmp_path / "states/cut/snapshot/state.json"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("{}")
    Image.new("RGB", (160, 100), "gray").save(tmp_path / "rgb.jpg")
    frames = {}
    for key, step, x in [("before", 0, 0), ("current", 1, .25), ("future", 2, .5)]:
        frames[key] = {"rgb": "rgb.jpg", "step": step, "stream": "walk",
                       "pose": {"position": [x, 0, 0]}, "public_pose": {"pos": [x, .5], "heading": 0}}
    def sight(frame, identifier, pixel, world, verified=True, **extra):
        return {"frame": frame, "kind": "point", "id": identifier, "pixel": pixel, "world": world,
                "verified": verified, "method": "fixture", **extra}

    state = {"id": "cut", "snapshot": "states/cut/snapshot", "current_frame": "current",
             "step": 1, "frame_keys": ["before", "current"], "node_frames": {},
             "transform": {"origin": [-1, -1], "size_m": 2}}
    return {"scene_id": "fixture", "scene": {"objects": []}, "states": [state], "frames": frames,
            "sightings": [sight("before", "shared", [70, 50], [0, 0, -1]),
                          sight("current", "shared", [55, 50], [0, 0, -1]),
                          sight("current", "other", [110, 55], [1, 0, -2]),
                          sight("current", "occluded", [130, 55], [1, 0, -3], False),
                          sight("future", "shared", [1, 1], [0, 0, -1])]}


def test_projection_orientation_and_border():
    assert ego((0, -2), (0, 0), 0) == (0, 2)
    assert sector(*ego((0, -2), (0, 0), 90)) == "left"
    assert sector(*ego((0, 2), (0, 0), 90)) == "right"
    assert sector(1, 1) is None
    assert sector(.05, .06) is None


def test_routes_accept_equivalent_paths():
    edges = [("A", "B", "1"), ("A", "C", "2"), ("B", "D", "3"), ("C", "D", "4")]
    expected = [["A", "1", "B", "3", "D"], ["A", "2", "C", "4", "D"]]
    assert shortest_routes(edges, "A", "D") == expected
    assert not shortest_routes(edges, "A", "E")
    label = {"answer_schema": arr(enum(*"ABCD1234")), "accepted": expected}
    assert score_answer(expected[1], label)["success"]
    assert not score_answer(["A", "D"], label)["success"]


@pytest.mark.parametrize("answer", [True, "1", float("nan"), float("inf"), 10**500])
def test_strict_number_answers(answer):
    label = {"answer_schema": num("m", .2), "accepted": [1]}
    assert score_answer(answer, label)["status"] == "invalid_answer"


def test_identity_needs_nonoccluded_shared_and_distinguishable_negative(tmp_path):
    bundle = _bundle(tmp_path)
    before, after, distinct, blocked, _ = bundle["sightings"]
    for item in (before, after, distinct, blocked):
        item.update(kind="object", identity_verified=True)
    bundle["navigation_evidence"] = {"distinct_instances":[{"review_candidate":"pair","same_category":True,
        "observed_object":"other","requested_object":"shared","current_frame":"current","reference_frames":["before"]}]}
    c = Context(bundle, bundle["states"][0], tmp_path, {})
    _views(c)
    answers = Counter(item[2][0] for item in c.options["cross_view_identity"])
    assert answers == {True: 1, False: 1}
    assert set(c.sights) <= {"before", "current"}
    after["verified"] = False
    c = Context(bundle, bundle["states"][0], tmp_path, {})
    _views(c)
    assert not c.options["cross_view_identity"]


def test_empty_history_does_not_invent_crossings(tmp_path):
    b = _bundle(tmp_path)
    c = Context(b, b["states"][0], tmp_path, {})
    _rooms(c)
    assert not c.options


def test_frozen_judgment_runner(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path)
    from src.dataset.tasks_subset.sampling.geometry import project
    import numpy as np
    monkeypatch.setattr("src.dataset.tasks_subset.sampling.geometry.read_depth",lambda *args: np.ones((100,160)))
    monkeypatch.setattr("src.dataset.tasks_subset.sampling.geometry.floor_patch_fraction",lambda *args: 1.)
    bundle["scene"]["levels"]=[{"floor_height":0}]
    for key,frame in bundle["frames"].items():
        x=frame["pose"]["position"][0]
        frame["depth"]="test-depth.png"
        frame.update(camera={"K":[[70,0,80],[0,70,50],[0,0,1]],"width":160,"height":100},
                     world_from_optical=[[1,0,0,x],[0,-1,0,1],[0,0,-1,0],[0,0,0,1]])
    for sight in bundle["sightings"]:
        sight["world"][2] -= 1
        sight["pixel"]=list(project(sight["world"],bundle["frames"][sight["frame"]]["camera"],
                                    bundle["frames"][sight["frame"]]["world_from_optical"])[:2])
    tasks = build_judgments(bundle, tmp_path, {})
    assert len(JUDGMENT_KINDS) == 37
    task = next(t for t in tasks if t["kind"] == "distance_estimation")
    directory = (tmp_path / task["path"]).parent
    label = json.loads((directory / "private/label.json").read_text())
    monkeypatch.setattr(bench, "credentials", lambda *_: {"apiKey": "dummy", "model": "fixture"})
    def fake_call(repo, config, output, text, images, limits):
        assert limits.max_turns == 1
        assert all("public" in x["path"].parts for x in images)
        assert "future" not in text and "accepted" not in text
        return {"reason": "completed", "answer": json.dumps(label["accepted"][0])}

    monkeypatch.setattr(bench, "run_fixed_evidence", fake_call)
    result = bench.run_task(tmp_path, directory, {}, tmp_path / "result")
    assert result["score"]["success"]


def test_rgb_review_requires_two_bound_agreeing_reviews(tmp_path):
    b = _bundle(tmp_path)
    b["scene"]["objects"] = [{"id": "shared", "category": "chair"}]
    s = b["sightings"][0]
    s.update(kind="object", verified=False, method="foreground_in_schematic_bounds")
    c, = review_candidates(b, tmp_path)
    base = {"candidate_id": c["candidate_id"], "kind": c["kind"], "source": c["source"],
            "reviewer": "A", "decision": "accept", "reason": "Chair back at marker",
            "checks": {"visible": True, "name_type_match": True}}
    second = dict(base, reviewer="B")
    assert not apply_reviews(b, tmp_path, [base])["sightings"][0]["verified"]
    assert not apply_reviews(b, tmp_path, [base, dict(second, decision="uncertain")])["sightings"][0]["verified"]
    assert apply_reviews(b, tmp_path, [base, second])["sightings"][0]["verified"]
    assert not b["sightings"][0]["verified"]
    Image.new("RGB", (160, 100), "red").save(tmp_path / "rgb.jpg")
    with pytest.raises(ValueError, match="bind"):
        apply_reviews(b, tmp_path, [base, second])


def test_panorama_uses_sensor_heading_not_constant_public_body_heading(tmp_path):
    from math import cos, radians, sin

    b = _bundle(tmp_path)
    state = b["states"][0]
    for yaw in range(0, 360, 60):
        key = f"view-{yaw}"
        a = radians(yaw)
        frame = b["frames"][key] = {
            **b["frames"]["current"], "group": "p_1", "yaw": yaw,
            "world_from_optical": [[cos(a), 0, sin(a), 0], [0, -1, 0, 0],
                                   [sin(a), 0, -cos(a), 0], [0, 0, 0, 1]],
        }
        state["frame_keys"].append(key)
    c = Context(b, state, tmp_path, {})
    assert len(c.current_panorama) == 6
    assert c.heading("view-120") == pytest.approx(120)
    b["frames"]["view-120"]["world_from_optical"] = b["frames"]["view-0"]["world_from_optical"]
    c = Context(b, state, tmp_path, {})
    assert not c.current_panorama


def test_rejected_portal_cannot_reenter_as_geometry_verified(tmp_path):
    b = _bundle(tmp_path)
    b["scene"]["openings"] = [{"id": "shared", "kind": "passage"}]
    b["sightings"][0].update(kind="opening", method="depth_clear_at_annotated_portal")
    c, = review_candidates(b, tmp_path)
    record = {"candidate_id": c["candidate_id"], "kind": c["kind"], "source": c["source"],
              "reviewer": "A", "decision": "reject", "reason": "Reflection only",
              "checks": {"visible": False, "name_type_match": False}}
    s = apply_reviews(b, tmp_path, [record])["sightings"][0]
    assert s["geometry_verified"]
    assert not s["verified"] and not s["semantic_verified"] and not s["type_verified"]


def test_trial_binding_changes_on_description_only(tmp_path):
    from src.dataset.tasks_subset.bundle import snapshot_digest
    from src.dataset.tasks_subset.judgment.construction import _bound_trial
    b = _bundle(tmp_path)
    s = b["states"][0]
    root = tmp_path / s["snapshot"]
    memory = root / "workspace/memory/nodes.json"
    memory.parent.mkdir(parents=True)
    memory.write_text('{"nodes":[{"id":"N1","description":"old"}]}')
    c = Context(b, s, tmp_path, {})
    trial = {"start_snapshot_sha256": snapshot_digest(root)}
    assert _bound_trial(c, trial)
    memory.write_text('{"nodes":[{"id":"N1","description":"new"}]}')
    assert not _bound_trial(c, trial)


def test_target_set_requires_every_node_and_cut_reference(tmp_path, monkeypatch):
    from src.dataset.tasks_subset.judgment.construction import _targets
    b = _bundle(tmp_path)
    s = b["states"][0]
    s["node_frames"] = {"N1": [], "N2": []}
    b["targets"] = [{"id": "t", "object_id": "o", "identity_verified": True,
                     "reference_frames": ["future"], "reference_pixel": [20, 20]}]
    b["judgment_evidence"] = {"target_nodes": [
        {"node": "N1", "state": "cut", "target": "t", "review_candidate": "1", "review_result": "present"},
        {"node": "N2", "state": "cut", "target": "t", "review_candidate": "2", "review_result": "absent"}]}
    monkeypatch.setattr(Context, "prefix", lambda self: [("before", "test", [])])
    c = Context(b, s, tmp_path, {})
    _targets(c)
    assert not c.options["target_seen_nodes"]
    b["targets"][0]["reference_frames"] = ["before"]
    b["instance_evidence"]={"before":{"objects":{"o":{"completeness_binding":{"scope":"screen"}}}}}
    monkeypatch.setattr("src.dataset.tasks_subset.review.instances.node_visibility",
                        lambda bundle,keys,*a,**k: "visible" if keys==["before"] else "absent")
    s["node_frames"]={"N1":["before"],"N2":["current"]}
    c = Context(b,s,tmp_path,{})
    _targets(c)
    assert c.options["target_seen_nodes"][0][2] == [["N1"]]
    monkeypatch.setattr("src.dataset.tasks_subset.review.instances.node_visibility",
                        lambda bundle,keys,*a,**k: "unknown")
    c = Context(b, s, tmp_path, {})
    _targets(c)
    assert not c.options["target_seen_nodes"]


def test_native_scope_low_pixels_and_existential_node():
    from src.dataset.tasks_subset.review.instances import frame_visibility, node_visibility
    b={"frames":{},"instance_evidence":{},"instance_capability":{"min_pixels":100}}
    for key in map(str,range(6)):
        b["frames"][key]={"pose":{},"camera":{}}
        b["instance_evidence"][key]={"valid":True,"pose":{},"camera":{},"objects":{"o":{
            "mapping_valid":True,"association_verified":True,"mapping_complete":True,
            "completeness_binding":{"scope":"screen"},"pixels":10}}}
    assert frame_visibility(b,"0","o")=="unknown"
    b["instance_evidence"]["0"]["objects"]["o"]["pixels"]=101
    assert node_visibility(b,list(b["frames"]),"o")=="visible"
    assert node_visibility(b,list(b["frames"]),"o",scope="cabinet")=="unknown"


def test_nonzero_floor_motion_rejects_zero(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from src.dataset.tasks_subset.judgment.calibration import measured_floor_motion
    from src.dataset.tasks_subset.judgment.construction import MOTION
    k=np.array([[100.,0,100],[0,100.,100],[0,0,1]])
    angle=np.radians(30)
    rotation=np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
    shift=np.array([.3,.7])
    pairs=[]
    for point in (np.array([-.5,3.]),np.array([.8,4.])):
        other=rotation.T@(point-shift)
        pixels=[]
        for p in (point,other):
            optical=np.array([p[0],1.,p[1]])
            pixel=k@optical
            pixels.append((pixel[:2]/pixel[2]).tolist())
        pairs.append(({"pixel":pixels[0]},{"pixel":pixels[1]}))
    public={"K":k.tolist(),"height_m":1.,"optical_to_level":[[1,0,0],[0,-1,0],[0,0,1]]}
    monkeypatch.setattr("src.dataset.tasks_subset.judgment.calibration.floor_support",lambda *_:public)
    recovered=measured_floor_motion(SimpleNamespace(),pairs)
    assert recovered["right"]==pytest.approx(.3)
    assert recovered["forward"]==pytest.approx(.7)
    assert recovered["yaw"]==pytest.approx(30)
    label={"answer_schema":MOTION,"accepted":[recovered]}
    assert score_answer(recovered,label)["success"]
    assert not score_answer({**recovered,"right":0,"forward":0},label)["success"]


def test_review_reapplication_clears_unbound_derived_facts(tmp_path):
    b=_bundle(tmp_path)
    b["judgment_evidence"]={"trials":[{"review_candidate":"stale","trial":"invented"}]}
    b["targets"]=[{"id":"judgment-instance-stale"}]
    a=apply_reviews(b,tmp_path,[])
    assert not a.get("judgment_evidence")
    assert not a["targets"]
    assert a==apply_reviews(a,tmp_path,[])


def test_arrival_branch_is_reachable(tmp_path, monkeypatch):
    from src.dataset.tasks_subset.judgment.construction import _targets
    b=_bundle(tmp_path)
    t={"id":"t","reference_frames":["before"],"reference_pixel":[10,10]}
    b["judgment_evidence"]={"arrivals":[{"state":"cut","target":"t","review_candidate":"bound"}]}
    sight=b["sightings"][1]
    monkeypatch.setattr("src.dataset.tasks_subset.judgment.targets.arrival_candidates",
                        lambda bundle: iter([(t,b["states"][0],sight,.85,True)]))
    c=Context(b,b["states"][0],tmp_path,{})
    _targets(c)
    assert c.options["target_arrival"][0][2]==[True]
