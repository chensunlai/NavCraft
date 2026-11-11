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
    assert [case[2] for case in c.options["cross_view_identity"]] == [[False]]


def test_empty_history_does_not_invent_crossings(tmp_path):
    b = _bundle(tmp_path)
    c = Context(b, b["states"][0], tmp_path, {})
    _rooms(c)
    assert not c.options


def test_room_graph_does_not_publish_unindexed_private_room_alias(tmp_path, monkeypatch):
    import src.dataset.tasks_subset.judgment.construction as construction
    b = _bundle(tmp_path)
    state = b["states"][0]
    state.update(room="b", visited_rooms=["a", "b", "private-c"],
                 room_aliases={"a": "Room A", "b": "Room B", "private-c": "Hidden Room"},
                 node_frames={"N1": ["before"], "N2": ["current"]},
                 traversals=[{"opening": "D", "from_room": "a", "to_room": "b",
                              "before_frame": "before", "after_frame": "current",
                              "from_node": "N1", "to_node": "N2"}])
    b["scene"].update(rooms=[{"id": key} for key in ("a", "b", "private-c")],
                      openings=[{"id": "D", "kind": "door", "connects": ["a", "b"]}])
    for key, room in (("before", "a"), ("current", "b")):
        b["frames"][key]["room"] = room
        b["frames"][key]["yaw"] = 0
    b["sightings"] = [{"frame": key, "kind": "opening", "id": "D", "pixel": [80, 50],
                       "world": [0, 0, -1], "verified": True, "method": "fixture"}
                      for key in ("before", "current")]
    monkeypatch.setattr(construction, "_opening_supported", lambda *_: True)
    monkeypatch.setattr(construction, "_node_snapshot", lambda _: {"nodes": [
        {"id": "N1", "room": "Room A"}, {"id": "N2", "room": "Room B"}]})
    c = Context(b, state, tmp_path, {})
    c.prefix = lambda: [construction.row("before"), construction.row("current")]
    _rooms(c)
    assert c.options["room_adjacency"]
    assert "Hidden Room" not in json.dumps(c.options)


def test_frozen_judgment_runner(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path)
    from src.dataset.tasks_subset.sampling.geometry import project
    import numpy as np
    monkeypatch.setattr("src.dataset.tasks_subset.sampling.geometry.read_depth",lambda *args: np.ones((100,160)))
    monkeypatch.setattr("src.dataset.tasks_subset.sampling.geometry.measure_horizontal_floor_patch",
                        lambda *args: {"height_m":0.,"normal_world":[0,1,0],"tilt_degrees":0.})
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


def test_rejected_portal_cannot_reenter_as_geometry_verified(tmp_path, monkeypatch):
    b = _bundle(tmp_path)
    b["scene"]["openings"] = [{"id": "shared", "kind": "passage", "level": "L"}]
    b["scene"]["levels"] = [{"id":"L","floor_height":0}]
    b["frames"]["before"].update(camera={"fx":100},world_from_optical=[[1]])
    monkeypatch.setattr("src.dataset.tasks_subset.sampling.geometry.project_opening_band",
                        lambda *args: [[10,10],[30,10],[30,70],[10,70]])
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
    t={"id":"t","object_id":"O","reference_frames":["before"],"reference_pixel":[10,10]}
    b["targets"]=[{**t,"approach_region":{"room":"R"}}]
    monkeypatch.setattr("src.dataset.tasks_subset.judgment.targets.arrival_pictures",
                        lambda *args: [("before","Target reference",[("T",[10,10])])])
    b["judgment_evidence"]={"arrivals":[{"state":"cut","target":"t","review_candidate":"bound","frames":["before"]}]}
    sight=b["sightings"][1]
    monkeypatch.setattr("src.dataset.tasks_subset.judgment.targets.arrival_candidates",
                        lambda bundle: iter([(t,b["states"][0],sight,.85,True)]))
    c=Context(b,b["states"][0],tmp_path,{})
    _targets(c)
    assert c.options["target_arrival"][0][2]==[True]


def test_camera_distance_variant_has_public_floor_calibration(tmp_path,monkeypatch):
    import numpy as np
    from src.dataset.tasks_subset.judgment.construction import _visual
    b=_bundle(tmp_path);b["scene"]["levels"]=[{"floor_height":0}]
    f=b["frames"]["current"]
    f.update(depth="depth.png",camera={"K":[[70,0,80],[0,70,50],[0,0,1]],"width":160,"height":100},
             world_from_optical=[[1,0,0,.25],[0,-1,0,1],[0,0,-1,0],[0,0,0,1]])
    b["sightings"]=[{"id":"p","kind":"point","frame":"current","pixel":[80,85],
                     "world":[.25,0,-2],"verified":True,"method":"fixture"}]
    monkeypatch.setattr("src.dataset.tasks_subset.sampling.geometry.read_depth",lambda *_:np.ones((100,160)))
    monkeypatch.setattr("src.dataset.tasks_subset.sampling.geometry.measure_horizontal_floor_patch",
                        lambda *args: {"height_m":0.,"normal_world":[0,1,0],"tilt_degrees":0.})
    c=Context(b,b["states"][0],tmp_path,{})
    _visual(c)
    row=c.options["distance_estimation"][0]
    assert row[-1]["variant"]=="camera_distance_calibrated_floor"
    assert "height_m" in row[0] and row[2]==[2.]
    assert not score_answer(0,{"answer_schema":row[1],"accepted":row[2]})["success"]


def test_receipt_has_three_distinct_outcomes(tmp_path,monkeypatch):
    from src.dataset.tasks_subset.judgment import construction as module
    b=_bundle(tmp_path)
    c=Context(b,b["states"][0],tmp_path,{})
    event={"before_frame":"before","after_frame":"current","receipt":{"action":"move_forward","collision":True,"blocked":True}}
    monkeypatch.setattr(module,"_motion_events",lambda _:iter([event]))
    module._outcomes(c)
    case=c.options["blocked_receipt"][0]
    assert case[2]==["collision_with_motion"]
    assert "camera moved" not in case[0].lower()
    assert not score_answer("collision_no_motion",{"answer_schema":case[1],"accepted":case[2]})["success"]
    b["frames"]["current"]["pose"]["position"]=[0,0,0]
    b["frames"]["current"]["public_pose"]["pos"]=[0,.5]
    c=Context(b,b["states"][0],tmp_path,{})
    module._outcomes(c)
    assert c.options["blocked_receipt"][0][2]==["collision_no_motion"]
    event["receipt"]["collision"]=False
    c=Context(b,b["states"][0],tmp_path,{})
    module._outcomes(c)
    assert c.options["blocked_receipt"][0][2]==["no_collision"]


def test_size_endpoint_geometry_binds_real_rgb_and_both_views(tmp_path):
    import numpy as np
    from src.dataset.tasks_subset.judgment.size import MEASUREMENT,validate_endpoint_selection
    Image.new("RGB",(140,140),"gray").save(tmp_path/"frame.jpg")
    Image.fromarray(np.full((140,140),2000,dtype=np.uint16)).save(tmp_path/"depth.png")
    camera={"K":[[100,0,60],[0,100,50],[0,0,1]],"width":140,"height":140,
            "depth_scale":.001,"depth_invalid_value":0,"depth_min":.01,"depth_max":10}
    frames={}
    for key,x in (("a",0),("b",.6)):
        frames[key]={"rgb":"frame.jpg","depth":"depth.png","camera":camera,"public_pose":{"pos":[x/2,0],"heading":0},
                     "world_from_optical":[[1,0,0,x],[0,-1,0,1],[0,0,-1,0],[0,0,0,1]]}
    b={"frames":frames,"scene":{"levels":[{"floor_height":0}],"objects":[{"id":"chair","semantic_ids":[7]}]},
       "states":[{"frame_keys":["a","b"],"transform":{"origin":[0,0],"size_m":2}}]}
    from hashlib import sha256
    np.save(tmp_path/"mask.npy",np.full((140,140),7,dtype=np.int32))
    b["instance_evidence"]={}
    for key,frame in frames.items():
        b["instance_evidence"][key]={"valid":True,"pose":frame.get("pose"),"camera":camera,
            "source_rgb_sha256":sha256((tmp_path/"frame.jpg").read_bytes()).hexdigest(),
            "mask":"mask.npy","mask_sha256":sha256((tmp_path/"mask.npy").read_bytes()).hexdigest(),
            "objects":{"chair":{"mapping_valid":True,"mapped_ids":[7]}}}
    proposed={"object":"chair","name":"Chair","frames":["a","b"],"measurement":MEASUREMENT}
    selected={"measurement":MEASUREMENT,"pixels":[[40,100],[80,100]],"axis":"width","axis_description":"outside-to-outside front feet"}
    span=validate_endpoint_selection(b,tmp_path,proposed,selected)
    assert span["length_m"]==pytest.approx(.8)
    assert len(span["source"])==2 and span["source"][1]["pixels"]==[[10,100],[50,100]]
    with pytest.raises(ValueError,match="horizontal plane"):
        validate_endpoint_selection(b,tmp_path,proposed,{**selected,"pixels":[[40,80],[80,100]]})


def test_reviewed_nonpassable_type_is_not_passability(tmp_path):
    from src.dataset.tasks_subset.judgment.construction import _rooms
    b=_bundle(tmp_path)
    b["scene"]["objects"]=[{"id":"shared","name":"Wall mirror"}]
    b["sightings"][1].update(kind="object",method="foreground_in_schematic_bounds",verified=False)
    candidates=list(review_candidates(b,tmp_path))
    c=next(c for c in candidates if c["kind"]=="judgment_opening_surface_type")
    record={"candidate_id":c["candidate_id"],"kind":c["kind"],"source":c["source"],
            "reviewer":"first","decision":"accept","reason":"Mirror frame and reflected room are clear",
            "checks":{"surface_bound":True,"type_supported":True},
            "values":{"surface_type":"mirror","door_state":"not_applicable"}}
    reviewed=apply_reviews(b,tmp_path,[record,{**record,"reviewer":"second"}])
    c=Context(reviewed,reviewed["states"][0],tmp_path,{})
    _rooms(c)
    generated=c.options["opening_type"]
    assert generated and generated[0][2]==[{"type":"mirror"}]
    assert not c.options["opening_passability"]
    assert not score_answer({"type":"window"},{"answer_schema":generated[0][1],"accepted":generated[0][2]})["success"]
    different={**record,"reviewer":"second","values":{"surface_type":"glass","door_state":"not_applicable"}}
    reviewed=apply_reviews(b,tmp_path,[record,different])
    assert not reviewed.get("judgment_evidence",{}).get("opening_types")


def test_action_choices_require_same_real_pair_and_bound_start(tmp_path,monkeypatch):
    from src.dataset.tasks_subset.bundle import snapshot_digest
    from src.dataset.tasks_subset.judgment.construction import _point_choices
    monkeypatch.setattr("src.dataset.tasks_subset.judgment.construction.floor_support",lambda *_:{"height_m":1})
    b=_bundle(tmp_path);s=b["states"][0];point=b["sightings"][1]
    for name,position in (("end-near",point["world"]),("end-far",[0,0,0])):
        b["frames"][name]={**b["frames"]["future"],"pose":{"position":position}}
    binding=snapshot_digest(tmp_path/s["snapshot"])
    b["trials"]=[]
    for name in ("near","far"):
        b["trials"].append({"id":name,"start_state":"cut","target":point["id"],"pair_id":"pair",
                           "start_snapshot_sha256":binding,"replay":{"passed":True},
                           "motions":[{"collision":False,"blocked":False}],
                           "calls":[{"name":"move","arguments":{"actions":["move_forward"]}}],
                           "end_frame":"end-"+name,"goal_positions":[point["world"]]})
    c=Context(b,s,tmp_path,{})
    assert len(list(_point_choices(c)))==1
    b["trials"][1]["pair_id"]="unrelated"
    assert not list(_point_choices(c))
    b["trials"][1]["pair_id"]="pair"
    b["trials"][0]["start_snapshot_sha256"]="stale"
    assert not list(_point_choices(c))


def test_model_review_accepts_explicit_none_markers(tmp_path,monkeypatch):
    from src.dataset.tasks_subset.review.evidence import model_review
    b=_bundle(tmp_path)
    candidate={"candidate_id":"nav","kind":"navigation_correction","source":{"frames":[{"frame":"before"}]},
               "frames":["before"],"markers":None,"prompt":"Review the unmarked room panorama.",
               "check_keys":["answerable"]}
    captured=[]
    def fake_model(*args):
        captured.extend(args[4])
        return {"answer":'{"decision":"uncertain","answerable":false,"reason":"Ambiguous room."}'}
    monkeypatch.setattr("src.dataset.tasks_subset.review.model.run_fixed_evidence",fake_model)
    record=model_review(tmp_path,{},tmp_path/"review",tmp_path,b,candidate,"fixture")
    assert record["decision"]=="uncertain" and len(captured)==1
    outline=[[10,10],[40,10],[40,70],[10,70]]
    model_review(tmp_path,{},tmp_path/"outline",tmp_path,b,{**candidate,"markers":[outline]},"fixture")
    with Image.open(tmp_path/"outline/candidate-1.png") as marked:
        assert marked.getpixel((10,20))==(0,255,255)
    with pytest.raises(ValueError):
        model_review(tmp_path,{},tmp_path/"invalid",tmp_path,b,{**candidate,"markers":[]}, "fixture")


def test_endpoint_stage_exports_bound_candidates_without_model_calls(tmp_path):
    from src.dataset.tasks_subset.judgment.size import run_endpoints
    b=_bundle(tmp_path)
    b["scene"]["objects"]=[{"id":"shared","name":"Chair"}]
    b["frames"]["current"]["pose"]["position"]=[.75,0,0]
    for sight in b["sightings"][:2]:
        sight.update(kind="object",method="foreground_in_schematic_bounds")
    source=tmp_path/"private/raw.json";source.parent.mkdir(exist_ok=True);source.write_text(json.dumps(b))
    result=run_endpoints(b,tmp_path,source,{})
    export=json.loads((tmp_path/"private/size_candidates.json").read_text())
    assert len(export["candidates"])==1
    candidate=export["candidates"][0]
    assert candidate["binding"]["annotation_sha256"]
    assert all(row["sha256"] for row in candidate["binding"]["frames"])
    assert not result.get("size_endpoint_selections")
    assert (tmp_path/"private/size_bundle.json").is_file()
    assert not (tmp_path/"private/size_attempts").exists()


def test_review_derivations_rebuild_through_three_stages(tmp_path,monkeypatch):
    from src.dataset.tasks_subset.review import evidence
    b=_bundle(tmp_path)
    candidates=[
        {"candidate_id":"seed","kind":"stage","source":{"hash":"seed"},"check_keys":["supported"],
         "fact":{"section":"navigation_evidence","key":"seed","value":{"value":1}}},
        {"candidate_id":"middle","kind":"stage","source":{"hash":"middle"},"check_keys":["supported"],
         "fact":{"section":"judgment_evidence","key":"middle","value":{"value":2}}},
        {"candidate_id":"current","kind":"stage","source":{"hash":"current"},"check_keys":["supported"],
         "fact":{"section":"judgment_evidence","key":"current","value":{"value":3}}}]
    def produce(bundle,root):
        yield candidates[0]
        if bundle.get("navigation_evidence",{}).get("seed"):
            yield candidates[1]
        if bundle.get("judgment_evidence",{}).get("middle"):
            yield candidates[2]
    monkeypatch.setattr(evidence,"review_candidates",produce)
    records=[{"candidate_id":c["candidate_id"],"kind":"stage","source":c["source"],"reviewer":reviewer,
              "decision":"accept","checks":{"supported":True},"reason":"Bound evidence checked"}
             for c in candidates for reviewer in ("first","second")]
    result=evidence.apply_reviews(b,tmp_path,records)
    assert len(result["judgment_evidence"]["current"])==1
    assert result==evidence.apply_reviews(result,tmp_path,records)
    assert not evidence.apply_reviews(result,tmp_path,[]).get("judgment_evidence")


def test_budget_attempt_is_not_semantic_uncertainty_and_models_are_independent(tmp_path):
    b=_bundle(tmp_path)
    b["scene"]["objects"]=[{"id":"shared","category":"chair"}]
    b["sightings"][0].update(kind="object",verified=False,method="foreground_in_schematic_bounds")
    c=next(c for c in review_candidates(b,tmp_path) if c["kind"]=="object")
    first={"candidate_id":c["candidate_id"],"kind":"object","source":c["source"],
           "reviewer":"sol/high","model":"Sol","execution_status":"completed",
           "decision":"accept","checks":{"visible":True,"name_type_match":True},"reason":"Visible chair"}
    second={**first,"reviewer":"root-visual","model":None}
    budget={**first,"reviewer":"qwen","model":"Qwen","execution_status":"budget_exhausted",
            "decision":"uncertain","checks":{},"response_reason":"output_limit"}
    result=apply_reviews(b,tmp_path,[first,second,budget])
    assert result["sightings"][0]["semantic_verified"]
    assert result["review_attempt_counts"]=={"completed":2,"budget_exhausted":1}
    with pytest.raises(ValueError,match="distinct actual models"):
        apply_reviews(b,tmp_path,[first,{**second,"model":"Sol"}])


def test_model_review_records_execution_failure_separately(tmp_path,monkeypatch):
    from src.dataset.tasks_subset.review.evidence import model_review
    b=_bundle(tmp_path)
    candidate={"candidate_id":"budget","kind":"audit","source":{"frames":[{"frame":"before"}]},
               "frames":["before"],"markers":None,"prompt":"Review this picture.","check_keys":["answerable"]}
    monkeypatch.setattr("src.dataset.tasks_subset.review.model.run_fixed_evidence",
                        lambda *args:{"reason":"output_limit","answer":"","last":{"model":"Actual-Qwen"}})
    record=model_review(tmp_path,{"model":"alias","reasoning_effort":"medium"},
                        tmp_path/"budget",tmp_path,b,candidate,"tag")
    assert record["execution_status"]=="budget_exhausted"
    assert record["response_reason"]=="output_limit" and record["model"]=="Actual-Qwen"


def test_build_rebinds_native_measurements_after_rgb_depth_or_pose_change(tmp_path,monkeypatch):
    from copy import deepcopy
    from hashlib import sha256
    import numpy as np
    # Base identity is the separately tested input boundary. Target projection/depth,
    # pixels and cache hashes below are real, not mocked.
    monkeypatch.setattr("src.dataset.tasks_subset.sampling.semantics.link_native_identity",lambda b,r:b)
    monkeypatch.setattr("src.dataset.tasks_subset.sampling.semantics.collect_instance_evidence",
                        lambda *_a,**_k:pytest.fail("build must never render"))
    for name in ("source.png","dest.png"):
        Image.new("RGB",(5,5),"gray").save(tmp_path/name)
    Image.fromarray(np.full((5,5),2000,dtype=np.uint16)).save(tmp_path/"depth.png")
    np.save(tmp_path/"mask.npy",np.full((5,5),34,dtype=np.int32))
    camera={"K":[[1,0,2],[0,1,2],[0,0,1]],"width":5,"height":5,"depth_scale":.001,
            "depth_invalid_value":0,"depth_min":.01,"depth_max":10}
    matrix=np.eye(4).tolist();pose={"position":[0,0,0],"camera_position":[0,0,0]}
    frames={k:{"rgb":k+".png","depth":"depth.png","camera":camera,"world_from_optical":matrix,
               "pose":deepcopy(pose),"public_pose":{"pos":[.5,.5],"heading":180},"step":i,"stream":"walk"}
            for i,k in enumerate(("source","dest"))}
    row={"valid":True,"pose":deepcopy(pose),"camera":camera,"source_rgb_sha256":sha256((tmp_path/"dest.png").read_bytes()).hexdigest(),
         "mask":"mask.npy","mask_sha256":sha256((tmp_path/"mask.npy").read_bytes()).hexdigest(),
         "objects":{"tv":{"mapping_valid":True,"mapped_ids":[34]}}}
    b={"scene_id":"fixture","scene":{"objects":[]},"frames":frames,"targets":[],
       "states":[{"id":"cut","snapshot":"unused","current_frame":"dest","frame_keys":["source","dest"],
                  "node_frames":{},"step":1,"transform":{"origin":[-1,-1],"size_m":2}}],
       "sightings":[{"frame":"source","kind":"object","id":"tv","pixel":[2,2],"world":[0,0,2],
                     "method":"fixture","verified":True,"semantic_verified":True,"identity_verified":True,
                     "review_candidate":"seed","native_identity":{"descriptor_id":34}}],
       "instance_capability":{"available":True,"min_pixels":1},
       "instance_evidence":{"dest":row}}
    build_judgments(b,tmp_path,{"judgment_kinds":[]})
    assert len(b["derived_sightings"])==1
    baseline=deepcopy(b);depth=(tmp_path/"depth.png").read_bytes();rgb=(tmp_path/"dest.png").read_bytes()
    for changed in ("depth","rgb","pose"):
        b=deepcopy(baseline)
        if changed=="depth":
            Image.fromarray(np.full((5,5),1000,dtype=np.uint16)).save(tmp_path/"depth.png")
        elif changed=="rgb":
            Image.new("RGB",(5,5),"red").save(tmp_path/"dest.png")
        else:
            b["frames"]["dest"]["pose"]["camera_position"]=[1,0,0]
        build_judgments(b,tmp_path,{"judgment_kinds":[]})
        assert b["derived_sightings"]==[],changed
        (tmp_path/"depth.png").write_bytes(depth);(tmp_path/"dest.png").write_bytes(rgb)


def test_distinct_instance_preserves_reviewed_public_context(tmp_path):
    from src.dataset.tasks_subset.judgment.targets import build_instance_correspondence
    from types import SimpleNamespace
    fact = {"review_candidate":"pair", "same_category":True,
            "reference_frames":["before"],"current_frame":"current",
            "requested_object":"one","observed_object":"two",
            "public_context":[{"frame":"context","caption":"Node N001; public capture pose"}]}
    sights={"before":[{"kind":"object","id":"one","frame":"before","pixel":[10,20],"identity_verified":True}],
            "current":[{"kind":"object","id":"two","frame":"current","pixel":[30,20],"identity_verified":True}],
            "context":[]}
    results=[]
    c=SimpleNamespace(state={"frame_keys":list(sights)},sights=sights,keys=set(sights),
                      bundle={"navigation_evidence":{"distinct_instances":[fact]}},
                      put=lambda *a,**kw:results.append((a,kw)))
    build_instance_correspondence(c)
    args,private=results[0]
    assert args[2:4] == ({"type":"boolean"},[False])
    assert args[4][-1] == ("context","Node N001; public capture pose",[])
    assert "static scene" in args[1]
    assert private["private"]["pair_fact"] == fact
    c.keys.remove("context")
    results.clear()
    build_instance_correspondence(c)
    assert not results


def test_rotation_is_recoverable_from_public_bearings_and_rejects_other_surface():
    import numpy as np
    from types import SimpleNamespace
    from src.dataset.tasks_subset.judgment.calibration import calibrated_turn
    K=[[100,0,100],[0,100,100],[0,0,1]]
    frame={"camera":{"K":K},"world_from_optical":np.eye(4).tolist()}
    c=SimpleNamespace(frames={"a":frame,"b":frame})
    a={"frame":"a","kind":"point","id":"P","pixel":[100,100],"world":[0,0,-2]}
    b={**a,"frame":"b","pixel":[100-100/3**.5,100]}
    public,recovered=calibrated_turn(c,a,b,30)
    assert recovered == pytest.approx(30)
    assert set(public["first"])=={"K"}
    assert calibrated_turn(c,a,{**b,"world":[.1,0,-2]},30) is None
    assert calibrated_turn(c,a,b,-30) is None
    assert calibrated_turn(c,a,{**b,"pixel":[100,100]},30) is None


def test_opening_outline_is_reviewed_and_bound_not_a_background_dot(tmp_path,monkeypatch):
    b=_bundle(tmp_path)
    b["scene"].update(openings=[{"id":"shared","kind":"passage","level":"L"}],levels=[{"id":"L","floor_height":0}])
    b["sightings"]=b["sightings"][:1]
    b["sightings"][0].update(kind="opening",method="depth_clear_at_annotated_portal")
    b["frames"]["before"].update(camera={"K":[[100,0,70],[0,100,50],[0,0,1]]},world_from_optical=[[1]])
    outline=[[10,10],[40,10],[40,80],[10,80]]
    monkeypatch.setattr("src.dataset.tasks_subset.sampling.geometry.project_opening_band",lambda *args:outline)
    candidate,=review_candidates(b,tmp_path)
    assert candidate["source"]["outline"]==outline
    checks={"visible":True,"name_type_match":True,"frame_alignment":True}
    reviews=[{"candidate_id":candidate["candidate_id"],"source":candidate["source"],"kind":"opening",
              "reviewer":r,"checks":checks,"decision":"accept","reason":"Actual nearby physical entrance"} for r in ("a","b")]
    accepted=apply_reviews(b,tmp_path,reviews)["sightings"][0]
    assert accepted["semantic_verified"] and accepted["outline"]==outline
    assert accepted["review_source"]["outline"]==outline
    checks["frame_alignment"]=False
    assert not apply_reviews(b,tmp_path,reviews)["sightings"][0]["semantic_verified"]


def test_arrival_producer_consumes_current_panorama_bound_derived_identity(tmp_path,monkeypatch):
    from src.dataset.tasks_subset.judgment.targets import arrival_candidates
    b=_bundle(tmp_path)
    target={"id":"target","object_id":"O","reference_frames":["before"],"reference_pixel":[10,10]}
    region={"room":"R","max_footprint_distance":.85,"rectangles":[{"center_xz":[.5,0],"size":[.2,.2],"angle_deg":0}]}
    b["targets"]=[{**target,"positions":[[1,0,0]],"approach_region":region}]
    b["states"][0]["room"]="R"
    b["frames"]["current"]["group"]="pano"
    b["frames"]["current-other"]={**b["frames"]["current"],"yaw":60}
    b["states"][0]["frame_keys"].append("current-other")
    derived={"kind":"object","id":"O","frame":"current-other","pixel":[25,25],"world":[1,0,0],
             "identity_verified":True,"semantic_verified":False,"method":"native_instance_reprojection",
             "evidence_binding":{"rgb_sha256":"bound"}}
    b["derived_sightings"]=[derived]
    monkeypatch.setattr("src.dataset.tasks_subset.judgment.targets.instance_targets",lambda *args:iter([target]))
    monkeypatch.setattr("src.dataset.tasks_subset.navigation.scoring.in_approach_region",lambda *args:True)
    rows=list(arrival_candidates(b))
    assert len(rows)==1 and rows[0][2] is derived and rows[0][4] is True
    derived.pop("evidence_binding")
    assert not list(arrival_candidates(b))
