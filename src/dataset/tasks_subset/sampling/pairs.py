"""Matched physical counterfactuals from one immutable observation cut."""
from pathlib import Path
import math

from src.common.io import read_json
from src.dataset.episodes import EpisodeSpec

from ..bundle import reject, snapshot_digest, write_json
from .geometry import floor_patch_fraction, project, read_depth, unproject
from .collection import _branch, _environment_config, _move, _yaw_delta


def collect_point_pairs(config, root, gpu, bundle_path=None):
    root = Path(root).resolve()
    path = Path(bundle_path) if bundle_path else root / "private/bundle.json"
    bundle = read_json(path)
    episode = EpisodeSpec(**bundle["episode"])
    env_config = _environment_config(config, episode, gpu)
    for state in bundle["states"]:
        key = state["current_frame"]
        frame = bundle["frames"][key]
        depth = read_depth(root / frame["depth"], frame["camera"])
        h, w = depth.shape
        choices = []
        for y in range(int(h*.7), int(h*.96), 14):
            for x in range(int(w*.3),int(w*.7), 22):
                point = unproject((x,y),depth[y,x], frame["camera"],frame["world_from_optical"])
                floor = state["transform"]["floor_height"]
                if (point is None or abs(point[1]-floor) > .12 or
                        floor_patch_fraction(depth, frame["camera"], frame["world_from_optical"],
                                             [x,y], floor) < .9):
                    continue
                distance = math.dist(point,frame["pose"]["position"])
                turn = _yaw_delta(frame["pose"],point)
                steps = round(distance/.25)
                nturn = round(abs(turn)/30)
                if 5 <= steps <= 8 and abs(turn-nturn*30*(1 if turn>=0 else -1)) < 7 and steps+nturn<=12:
                    choices.append((abs(turn), point, [x,y], steps,nturn,turn))
        if not choices:
            reject(root,"point",state["id"],"no_visible_floor_patch_within_action_budget",stage="pair_proposal")
            continue
        _, point,pixel,steps,nturn,turn = min(choices)
        identifier = f"{state['id']}-floor-pair"
        if any(t.get("pair_id")==identifier and t.get("start_snapshot_sha256")==snapshot_digest(root/state["snapshot"])
               for t in bundle["trials"]):
            continue
        turns = ["turn_right" if turn>0 else "turn_left"]*nturn
        positive = turns+["move_forward"]*steps
        negative = turns+["move_forward"]*max(1,steps-4)
        target = identifier+"-target"
        bundle["sightings"].append({"frame":key,"kind":"point","id":target,"pixel":pixel,
                                   "world":point,"verified":True,"geometry_verified":True,
                                   "method":"optical_depth_surface","quality":"runtime_depth"})
        bundle["targets"].append({"id":target,"positions":[point],"success_distance":.35,
                                   "reference_frames":[key],"object_id":None})
        trials=[]
        for variant,actions in (("arrive",positive),("short",negative)):
            def operation(ex,calls):
                for i in range(0,len(actions),6):
                    _move(ex,actions[i:i+6],calls)
                return {"positions":[point]}
            trial=_branch(bundle,root,env_config,episode,state,"point",operation,
                          target=target,goal_positions=[point],pair_id=identifier,variant=variant,
                          visible_frame=key, requested_actions=actions, max_actions=12)
            trial["reference"]["reached"] = trial["goal_measurement"]["distance"] is not None and trial["goal_measurement"]["distance"] <= .25
            trials.append(trial)
            write_json(path,bundle)
        a,b=trials
        da=a["goal_measurement"]["distance"]
        db=b["goal_measurement"]["distance"]
        bundle.setdefault("pair_checks",[]).append({"pair_id":identifier,
             "produced": bool(da is not None and da<=.25 and db is not None and db>=.55),
             "distances":[da,db],"trial_ids":[a["id"],b["id"]]})
        if not bundle["pair_checks"][-1]["produced"]:
            reject(root,"point",identifier,"physical_pair_failed_separation",stage="pair_execution",distances=[da,db])
        write_json(path,bundle)
        if da is not None and da<=.25 and db is not None and db>=.55:
            return bundle
    return bundle


def collect_motion_probes(config, root, gpu, bundle_path=None, limit=6, kinds=()):
    """Produce short motion and straight-node candidates, retaining blocked attempts."""
    from .collection import _straight
    root = Path(root).resolve()
    path = Path(bundle_path) if bundle_path else root / "private/bundle.json"
    bundle = read_json(path)
    episode = EpisodeSpec(**bundle["episode"])
    env_config = _environment_config(config, episode, gpu)
    seen = {(t["start_state"], t["kind"], t.get("candidate"), t.get("start_snapshot_sha256")) for t in bundle["trials"]}
    candidates = []
    support = {}
    for state in bundle["states"]:
        if not kinds or "forward" in kinds:
            if "pre-" in state["id"] or "post-" in state["id"]:
                for n in (1, 6):
                    candidates.append((state, "forward", f"forward-{n}", ["move_forward"]*n, None))
        if not kinds or "direct" in kinds:
            current = bundle["frames"][state["current_frame"]]
            for node, keys in state["node_frames"].items():
                if not keys:
                    continue
                anchor = bundle["frames"][keys[0]]
                point = anchor["pose"]["position"]
                distance = math.dist(point,current["pose"]["position"])
                delta = _yaw_delta(current["pose"],point)
                if not .8 <= distance <= 2.:
                    continue
                steps = round(distance/.25)
                turns = round(abs(delta)/30)
                if steps+turns > 12:
                    continue
                cameras = [bundle["frames"][k] for k in state["frame_keys"]
                           if bundle["frames"][k]["step"] == state["step"]]
                supported = None
                for frame in cameras:
                    world = [point[0], point[1]+.1, point[2]]
                    pixel = project(world,frame["camera"],frame["world_from_optical"])
                    if pixel is None:
                        continue
                    depth = read_depth(root/frame["depth"],frame["camera"])
                    x,y,z = pixel
                    measured = depth[int(y),int(x)]
                    if not math.isfinite(measured):
                        continue
                    if anchor["room"] == state["room"] and measured >= z-.15:
                        supported = "clear_same_room"
                    elif measured < z-.35:
                        supported = "visible_depth_obstruction"
                    if supported:
                        break
                if supported:
                    candidates.append((state,"direct",node,None,point))
                    support[(state["id"], node)] = {"method":supported,"frame":next(k for k,v in bundle["frames"].items() if v is frame),"projected_pixel":[x,y],"surface_depth":float(measured),"anchor_depth":float(z)}
        if kinds == ("forward",) and len(candidates)>=limit:
            break
    candidates = [c for c in candidates if not kinds or c[1] in kinds]
    if kinds == ("direct",):
        candidates.sort(key=lambda c: (
            (c[0]["id"],"direct",c[2],snapshot_digest(root/c[0]["snapshot"])) in seen,
            support[(c[0]["id"],c[2])]["method"] != "clear_same_room",
            math.dist(c[4],bundle["frames"][c[0]["current_frame"]]["pose"]["position"])))
    for state,kind,candidate,actions,point in candidates[:limit]:
        if kinds and kind not in kinds:
            continue
        if (state["id"],kind,candidate,snapshot_digest(root/state["snapshot"])) in seen:
            continue
        def operation(ex,calls):
            if point is not None:
                delta=_yaw_delta(ex.environment.pose(),point)
                turns=round(abs(delta)/30)
                direct_actions=(["turn_right" if delta>0 else "turn_left"]*turns+
                         ["move_forward"]*round(math.dist(ex.environment.position(),point)/.25))
                for i in range(0,len(direct_actions),6):
                    _move(ex,direct_actions[i:i+6],calls)
                return {"reached":ex.environment.distance([point])<=.35}
            value=_move(ex,actions,calls)
            return {"stop":value["stop"]}
        fields={"candidate":candidate}
        if point is not None:
            fields.update(goal_positions=[point],target_node=candidate, max_actions=12,
                          public_support=support[(state["id"],candidate)])
        trial=_branch(bundle,root,env_config,episode,state,kind,operation,**fields)
        trial["observed_outcome"]={"blocked":any(m["blocked"] for m in trial["motions"]),
                                  "collision":any(m["collision"] for m in trial["motions"])}
        write_json(path,bundle)
    return bundle
