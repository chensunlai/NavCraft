"""Independent RGB request materials, never part of an agent observation prefix."""
from copy import deepcopy
from hashlib import sha256
import math
from pathlib import Path

from src.dataset.episodes import EpisodeSpec
from src.environment.habitat import Environment
from ..bundle import write_json
from .collection import _environment_config


def collect_target_reference(bundle, root, config, gpu, object_id):
    root = Path(root).resolve()
    result = deepcopy(bundle)
    obj = next(o for o in bundle["scene"]["objects"] if o["id"] == object_id)
    identifier = f"independent-target-{object_id}"
    key = f"target-reference/{identifier}"
    if key in bundle["frames"]:
        raise FileExistsError(key)
    episode = EpisodeSpec(**bundle["episode"])
    options = _environment_config(config, episode, gpu)
    env = Environment(episode, options, root/"private/target_references"/identifier/"environment")
    try:
        center = obj["center_xyz"]
        # Reachable station proposals are private; rendering a request image is not navigation.
        candidates = []
        for radius in (1.5, 2., 1., 2.5):
            for i in range(16):
                angle = i*math.tau/16
                point = env.snap_point([center[0]+radius*math.sin(angle),episode.start_position[1],
                                        center[2]+radius*math.cos(angle)])
                if point is None or abs(point[1]-episode.start_position[1])>.35:
                    continue
                if .8 <= math.hypot(point[0]-center[0],point[2]-center[2]) <= 2.5:
                    candidates.append(point)
        if not candidates:
            raise ValueError("No reachable target-reference candidate")
        # Prefer the same side as a verified observation, then render at a new station.
        reviewed = [s for s in bundle["sightings"] if s["id"]==object_id and s.get("semantic_verified")]
        if not reviewed:
            raise ValueError("Reference requires a reviewed positive instance")
        known = bundle["frames"][reviewed[0]["frame"]]["pose"]["camera_position"]
        point = min(candidates,key=lambda p:math.dist(p,[known[0],p[1],known[2]]))
        position = [point[0],center[1],point[2]]
        yaw = math.atan2(-(center[0]-position[0]),-(center[2]-position[2]))
        rotation = [0.,math.sin(yaw/2),0.,math.cos(yaw/2)]
        camera = {"position":position,"rotation":rotation,"image_dimensions":[480,640],"hfov":79}
        relative = Path("private/target_references")/identifier/"reference.png"
        env.target_image(camera,root/relative)
        provenance = {"source_kind":"independently_rendered_target_reference",
            "capture_method":"Environment.target_image absolute sensor pose; no agent motion claim",
            "camera":camera,"candidate_ground_position":point,"object_id":object_id,
            "object_binding":obj,"scene_source":bundle["source"],"assets":bundle.get("environment_spec"),
            "sha256":sha256((root/relative).read_bytes()).hexdigest(),"semantic_review":"needs_review"}
        result["frames"][key] = {"key":key,"stream":"target-reference","rgb":str(relative),
                                  **provenance}
        result["targets"].append({"id":identifier,"object_id":object_id,"reference_frames":[key],
                                  "source_kind":provenance["source_kind"],"reference_review":"needs_review"})
        write_json(root/"private/target_references"/identifier/"provenance.json",provenance)
    finally:
        env.close()
    write_json(root/"private/reference_bundle.json",result)
    return result
