"""Publicly solvable planar camera variants, independently checked against private points."""
import json
from math import atan2, degrees, hypot
import numpy as np


def floor_support(c,sight):
    key=(sight["frame"],tuple(sight["pixel"]),tuple(sight["world"]))
    cache=c.__dict__.setdefault("_floor_support_cache",{})
    if key not in cache:
        cache[key]=_measure_floor_support(c,sight)
    return cache[key]


def _measure_floor_support(c, sight):
    f = c.frames[sight["frame"]]
    levels = c.bundle["scene"].get("levels", [])
    floor_heights=[level["floor_height"] for level in levels]
    if not floor_heights or not f.get("camera") or not f.get("world_from_optical") or not f.get("depth"):
        return None
    from ..sampling.geometry import measure_horizontal_floor_patch, read_depth
    cache = c.__dict__.setdefault("_depth_cache",{})
    try:
        if sight["frame"] not in cache:
            cache[sight["frame"]]=read_depth(c.root/f["depth"],f["camera"])
        measured=measure_horizontal_floor_patch(cache[sight["frame"]],f["camera"],f["world_from_optical"],
                                   sight["pixel"],floor_heights,f["pose"]["position"][1])
    except (OSError,KeyError):
        return None
    if measured is None:
        return None
    c.__dict__.setdefault("_floor_measurements",{})[(sight["frame"],tuple(sight["pixel"]))]={
        "frame":sight["frame"],"pixel":sight["pixel"],"measurement":measured}
    matrix = np.asarray(f["world_from_optical"], dtype=float)
    if np.linalg.norm(matrix[[0,2],3]-np.asarray(c.position(sight["frame"])))>.01:
        return None
    ray = np.linalg.solve(f["camera"]["K"], [*sight["pixel"], 1])
    direction = matrix[:3, :3] @ ray
    height = matrix[1, 3]-measured["height_m"]
    if not .3 < height < 2.5 or direction[1] >= -.18:
        return None
    origin = matrix[:3, 3]
    point = origin-direction*height/direction[1]
    if np.linalg.norm(point-np.asarray(sight["world"])) > .12:
        return None
    # Local horizontal coordinates; publish pitch/roll via optical-to-level basis.
    heading = np.radians(c.heading(sight["frame"]))
    basis = np.array([[np.cos(heading), 0, np.sin(heading)], [0,1,0],
                      [np.sin(heading),0,-np.cos(heading)]])
    public = {"K": f["camera"]["K"], "height_m": round(float(height),4),
              "optical_to_level": (basis @ matrix[:3,:3]).round(7).tolist()}
    return public


def describe(public):
    return " Calibration: K (pixels), optical-to-level rotation (right/up/forward), camera height above the marked common floor: " + json.dumps(public)


def shared_pair(c, sight):
    """Public bearing intersection with a known baseline; no private depth in the prompt."""
    f = c.frames[sight["frame"]]
    matrix = np.asarray(f.get("world_from_optical", []))
    if matrix.shape != (4,4) or abs(matrix[1,2])>.001 or abs(matrix[1,0])>.001 or np.linalg.norm(
            matrix[[0,2],3]-np.asarray(c.position(sight["frame"])))>.01:
        return None
    for key in c.state["frame_keys"]:
        g = c.frames[key]
        if not f.get("camera") or not g.get("camera") or not g.get("world_from_optical"):
            continue
        other_matrix=np.asarray(g["world_from_optical"])
        if abs(other_matrix[1,2])>.001 or abs(other_matrix[1,0])>.001 or np.linalg.norm(
                other_matrix[[0,2],3]-np.asarray(c.position(key)))>.01:
            continue
        if not .55 <= np.linalg.norm(np.asarray(c.position(key))-c.position(sight["frame"])) <= 6:
            continue
        for other in c.sights[key]:
            if other["id"] != sight["id"] or np.linalg.norm(np.asarray(other["world"])-sight["world"]) > .06:
                continue
            origins, rays = [], []
            for s, frame in ((sight,f),(other,g)):
                matrix = np.asarray(frame["world_from_optical"])
                ray = matrix[:3,:3] @ np.linalg.solve(frame["camera"]["K"], [*s["pixel"],1])
                direction = ray[[0,2]]
                origins.append(matrix[[0,2],3])
                rays.append(direction/np.linalg.norm(direction))
            det = np.linalg.det(np.column_stack((rays[0],-rays[1])))
            if abs(det) < np.sin(np.radians(5)):
                continue
            lengths = np.linalg.solve(np.column_stack((rays[0],-rays[1])),origins[1]-origins[0])
            point = origins[0]+lengths[0]*rays[0]
            if min(lengths) <= 0 or np.linalg.norm(point-np.asarray(sight["world"])[[0,2]]) > .15:
                continue
            return other
    return None


def public_camera(c, key):
    f = c.frames[key]
    return json.dumps({"pose": f["public_pose"], "camera_heading": round(c.heading(key),2),
                       "K": f["camera"]["K"], "map_scale_m": c.state["transform"]["size_m"]})


def measured_floor_motion(c, pairs):
    """Recover a 2-D rigid transform using only public calibration and marked pixels."""
    ground=[]
    for index in (0,1):
        points=[]
        for pair in pairs:
            sight=pair[index]
            public=floor_support(c,sight)
            if public is None:
                return None
            ray=np.asarray(public["optical_to_level"]) @ np.linalg.solve(public["K"],[*sight["pixel"],1])
            points.append((ray*(-public["height_m"]/ray[1]))[[0,2]])
        ground.append(np.asarray(points))
    a,b=ground
    delta_a=a[1]-a[0]; delta_b=b[1]-b[0]
    yaw=atan2(delta_a[0],delta_a[1])-atan2(delta_b[0],delta_b[1])
    # B coordinates transformed into A coordinates, right/forward convention.
    rotation=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])
    offset=np.mean(a-b@rotation.T,axis=0)
    residual=np.max(np.linalg.norm((b@rotation.T+offset)-a,axis=1))
    if residual>.06:
        return None
    return {"right":float(offset[0]),"forward":float(offset[1]),
            "yaw":float((degrees(yaw)+180)%360-180)}


def public_intersection(first,second,first_pixel,second_pixel,transform):
    """Recover XZ from public UV/body+yaw/K only; validate this horizontal-camera variant."""
    origins=[];rays=[]
    for frame,pixel in ((first,first_pixel),(second,second_pixel)):
        matrix=np.asarray(frame.get("world_from_optical",[]))
        if matrix.shape!=(4,4) or abs(matrix[1,0])>.001 or abs(matrix[1,2])>.001:
            return None
        pose=frame["public_pose"]
        origin=np.asarray(transform["origin"])+np.asarray(pose["pos"])*transform["size_m"]
        if np.linalg.norm(origin-matrix[[0,2],3])>.015:
            return None
        heading=np.radians(pose["heading"]+(frame.get("yaw") or 0))
        optical=np.linalg.solve(frame["camera"]["K"],[*pixel,1])
        bearing=np.arctan2(optical[0],optical[2])
        direction=np.array([np.sin(heading+bearing),-np.cos(heading+bearing)])
        actual=matrix[:3,:3]@optical;actual=actual[[0,2]]/np.linalg.norm(actual[[0,2]])
        if np.linalg.norm(direction-actual)>.002:
            return None
        origins.append(origin);rays.append(direction)
    if np.linalg.norm(origins[1]-origins[0])<.55:
        return None
    matrix=np.column_stack((rays[0],-rays[1]))
    if abs(np.linalg.det(matrix))<np.sin(np.radians(5)):
        return None
    lengths=np.linalg.solve(matrix,origins[1]-origins[0])
    if min(lengths)<=0:
        return None
    return (origins[0]+lengths[0]*rays[0]).tolist()


def calibrated_turn(c, first, second, expected):
    """In-place yaw recoverable from the same marked surface and public intrinsics."""
    if first["kind"] != second["kind"] or first["id"] != second["id"] or np.linalg.norm(
            np.asarray(first["world"])-second["world"]) > .06:
        return None
    bearings=[]; public={}
    matrices=[]
    for name,sight in (("first",first),("second",second)):
        frame=c.frames[sight["frame"]]
        matrix=np.asarray(frame.get("world_from_optical",[]))
        if matrix.shape!=(4,4) or not frame.get("camera") or max(abs(matrix[1,0]),abs(matrix[1,2]))>.001:
            return None
        K=np.asarray(frame["camera"]["K"])
        ray=np.linalg.solve(K,[*sight["pixel"],1])
        bearings.append(degrees(atan2(ray[0],ray[2])))
        public[name]={"K":K.tolist()}
        matrices.append(matrix)
    if np.linalg.norm(matrices[0][:3,3]-matrices[1][:3,3])>.03:
        return None
    recovered=(bearings[0]-bearings[1]+180)%360-180
    if abs((recovered-expected+180)%360-180)>2:
        return None
    return public,recovered
