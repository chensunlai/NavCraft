"""Private, instance-specific goal information for the Qa continuous campaign.

Evidence photographs identify objects; their camera positions are not goal sets.
Official instance viewpoints are retained exactly. Other objects require the
separately rendered and verified custom viewpoint artifact.
"""

from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path


SCENE = "QaLdnwvtxbs"
GENERATED = "data/datasets/navcraft/qa_campaign_goals_v2.json"
OFFICIAL = "data/datasets/objectnav/hm3d/v2/val/content/QaLdnwvtxbs.json.gz"
TARGETS = {
    "O020": ("卧室的床", "R01", [20]),
    "O042": ("卧室床对面的电视", "R01", [42]),
    "O099": ("主浴室浴缸", "R03", [99]),
    "O108": ("主浴室双台盆洗手台", "R03", [108, 109, 110]),
    "O175": ("客餐厅长餐桌", "R05", [175]),
    "O188": ("客厅沙发与电视之间的方形木茶几", "R05", [188]),
    "O196": ("客餐厅电视", "R05", [196]),
    "O204": ("备餐区水槽", "R06", [204]),
    "O243": ("独立卫生间洗手盆", "R07", [243]),
}


def validate_positions(positions, identifier):
    if not positions or any(
        not isinstance(p, (list, tuple)) or len(p) != 3
        or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in p)
        for p in positions
    ):
        raise ValueError(f"Invalid or empty accepted viewpoints: {identifier}")


def load_goal_catalog(repo, generated_path=None):
    repo = Path(repo)
    generated_path = Path(generated_path) if generated_path else repo / GENERATED
    if not generated_path.exists():
        raise FileNotFoundError(
            f"Verified custom goal viewpoints are required: {generated_path}. "
            f"Generate them with python -m src.bench.goal_viewpoints --output {generated_path}; "
            "evidence-camera positions are not an acceptable fallback."
        )
    generated = json.loads(generated_path.read_text())
    if generated["scene_id"] != SCENE:
        raise ValueError("Custom goal artifact belongs to another scene")
    raw = repo / f"data/scene_maps/hm3d/val/00829-{SCENE}/raw"
    room_hash = generated.get("provenance", {}).get("rooms_sha256")
    if room_hash and hashlib.sha256((raw / "rooms.json").read_bytes()).hexdigest() != room_hash:
        raise ValueError("Custom viewpoint room geometry has changed; regenerate the artifact")
    objects = {o["id"]: o for o in json.loads((raw / "objects.json").read_text())["objects"]}
    official_path = repo / OFFICIAL
    with gzip.open(official_path, "rt") as handle:
        official = json.load(handle)
    by_instance = {
        int(goal["object_id"]): goal
        for goals in official["goals_by_category"].values() for goal in goals
    }
    catalog = {}
    for identifier, (label, room, instances) in TARGETS.items():
        object_ids = [f"O{i:03d}" for i in instances]
        evidence = [frame for oid in object_ids
                    for frame in objects[oid].get("evidence_frames", [])]
        common = dict(id=identifier, label=label, room_id=room,
                      object_ids=object_ids, semantic_instance_ids=instances,
                      evidence_ids=evidence)
        if all(i in by_instance for i in instances):
            goals = [by_instance[i] for i in instances]
            positions = [v["agent_state"]["position"] for goal in goals for v in goal["view_points"]]
            value = dict(common, positions=positions,
                         source_type="official_instance_viewpoints", source=OFFICIAL,
                         source_sha256=hashlib.sha256(official_path.read_bytes()).hexdigest(),
                         instance_centers_xyz=[g["position"] for g in goals],
                         source_instance_names=[g["object_name"] for g in goals],
                         limitations=["Official seed positions are preserved exactly; the campaign's 1 m tolerance and robot configuration remain custom."])
        else:
            item = generated["targets"][identifier]
            if sorted(item["semantic_instance_ids"]) != sorted(instances) or item["room_id"] != room:
                raise ValueError(f"Custom goal identity mismatch: {identifier}")
            validate_positions(item["positions"], identifier)
            views = item["viewpoints"]
            if (item["accepted_count"] != len(item["positions"])
                    or [v["position"] for v in views] != item["positions"]
                    or any(not v["accepted"] or v["target_pixels"] < generated["method"]["minimum_visible_pixels"]
                           for v in views)):
                raise ValueError(f"Custom viewpoint verification records disagree: {identifier}")
            value = dict(common, positions=item["positions"],
                         source_type="rendered_custom_viewpoints", source=str(generated_path),
                         source_sha256=hashlib.sha256(generated_path.read_bytes()).hexdigest(),
                         geometry=item.get("geometry"), method=generated["method"],
                         candidate_count=item.get("candidate_count"),
                         evidence_refs=item.get("evidence_refs", []),
                         limitations=item.get("limitations", []))
        validate_positions(value["positions"], identifier)
        value["viewpoint_count"] = len(value["positions"])
        catalog[identifier] = value
    return {"schema_version": 2, "scene_id": SCENE, "targets": catalog,
            "environment": generated["environment"],
            "generation_provenance": generated.get("provenance", {})}


def complete_scenario(scenario, catalog):
    """Replace sparse proxies and attach private identity/provenance to each round."""
    result = deepcopy(scenario)
    if catalog.get("scene_id") and Path(result["scene_id"]).name != catalog["scene_id"] + ".basis.glb":
        raise ValueError("Goal catalog belongs to another scene")
    # Geometry derived with one robot/camera must not silently follow a new config.
    for key in ("agent_height", "agent_radius", "navmesh_cell_height", "sensor_height", "hfov", "width", "height"):
        if key in catalog.get("environment", {}) and result["environment"].get(key) != catalog["environment"][key]:
            raise ValueError(f"Goal artifact environment mismatch: {key}")
    for spec in result["rounds"]:
        if spec["kind"] != "navigation":
            continue
        private = spec["private_verification"]
        target = catalog["targets"][private["target_object_id"]]
        spec["positions"] = deepcopy(target["positions"])
        spec["snap_viewpoints"] = False
        private["target"] = {k: deepcopy(v) for k, v in target.items() if k != "positions"}
        private["target_object_ids"] = target["object_ids"]
        for event in spec.get("route_events", []):
            if event["kind"] == "waypoint" and event["id"] == "dining_table":
                event["positions"] = deepcopy(catalog["targets"]["O175"]["positions"])
                event["target_object_id"] = "O175"
                event["viewpoint_source"] = catalog["targets"]["O175"]["source_type"]
    result.update(schema_version=3, goal_catalog_version=2,
                  protocol="navcraft-map-free-12-round-goal-catalog-v2",
                  limitations="Custom continuous chain, not benchmark SR. STOP and geodesic distance <1m to complete official instance seeds or documented rendered custom seeds. Ordered directed portal crossings and dining waypoint are scored separately. Seed visibility is verified during construction, not at final STOP. The 1m tolerance does not enforce final room membership or visibility. Exploration has no arrival goal.")
    return result
