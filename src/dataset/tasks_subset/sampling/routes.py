"""Private route proposals; every accepted station is reached by atomic execution."""
import math

from .scene import room_at
from .geometry import crossed_segment


def interior_points(executor, room, count=2):
    scene, env = executor.bundle["scene"], executor.environment
    row = next(r for r in scene["rooms"] if r["id"] == room)
    xs, zs = zip(*row["polygon_xz"])
    height = next(l["floor_height"] for l in scene["levels"] if l["id"] == row["level"])
    points = []
    for i in range(1, 9):
        for j in range(1, 9):
            point = [min(xs)+(max(xs)-min(xs))*i/9, height, min(zs)+(max(zs)-min(zs))*j/9]
            if room_at(scene, point, margin=.45) != room:
                continue
            snapped = env.snap_point(point)
            if snapped is None or math.dist(snapped, point) > .3 or room_at(scene, snapped, margin=.4) != room:
                continue
            path = env.shortest_path([snapped])
            if path:
                points.append(snapped)
    chosen = []
    while points and len(chosen) < count:
        origin = env.position() if not chosen else chosen[-1]
        point = min(points, key=lambda p: math.dist(p, origin)) if not chosen else max(
            points, key=lambda p: min(math.dist(p, q) for q in chosen))
        if chosen and min(math.dist(point, q) for q in chosen) < .8:
            break
        chosen.append(point)
        points = [p for p in points if math.dist(p, point) > .8]
    return chosen


def passage_points(executor, opening, destination):
    env, scene = executor.environment, executor.bundle["scene"]
    a, b = opening["segment_xz"]
    dx, dz = b[0]-a[0], b[1]-a[1]
    length = math.hypot(dx, dz)
    middle = [(a[0]+b[0])/2, env.position()[1], (a[1]+b[1])/2]
    for offset in (.55, .8):
        for sign in (-1, 1):
            before = [middle[0]+sign*dz/length*offset, middle[1], middle[2]-sign*dx/length*offset]
            after = [2*middle[0]-before[0], middle[1], 2*middle[2]-before[2]]
            points = [env.snap_point(p) for p in (before, middle, after)]
            if any(p is None for p in points):
                continue
            if any(math.dist(p, q) > .15 for p,q in zip(points,(before,middle,after))):
                continue
            if room_at(scene, points[0]) != room_at(scene, env.position()) or room_at(scene, points[2]) != destination:
                continue
            if crossed_segment(points[0], points[2], opening["segment_xz"]):
                return points
    return None


def walk_room_route(executor, route, settings):
    from .collection import _call, _cut, _follow, _move, _panorama, _straight
    env, scene = executor.environment, executor.bundle["scene"]
    index = 0
    executor.bundle["route_provenance"] = {"room_route": route, "stops_per_room": settings.get("stops_per_room",2),
        "interior_points": "private room polygons grid, snapped and reachable; real atomic execution",
        "passage_points": "source-map finite segment midpoint and .55/.8m normals; real crossing required",
        "map_hashes": executor.bundle["source"]["sha256"]}
    def cut(label, node=False):
        nonlocal index
        _panorama(executor, node)
        value = _cut(executor, f"route-{index:02d}-{label}")
        index += 1
        return value
    for room in route:
        current = room_at(scene, env.position())
        if current != room:
            choices = [o for o in scene["openings"] if o["status"] == "confirmed_open"
                       and set(o["connects"] or []) == {current, room}]
            crossed = False
            for opening in choices:
                points = passage_points(executor, opening, room)
                if not points or not _follow(executor, points[0], budget=settings.get("route_actions", 64), radius=.15):
                    continue
                start = env.position()
                before_state = cut("pre-"+opening["id"])
                from ..bundle import current_observation_frames
                visible = next((s["frame"] for s in executor.bundle["sightings"]
                                if s["kind"] == "opening" and s["id"] == opening["id"]
                                and s["frame"] in current_observation_frames(executor.bundle,before_state)), None)
                if visible:
                    executor.bundle.setdefault("passage_proposals",[]).append({
                        "state_id":before_state["id"],"opening":opening["id"],
                        "point":points[2],"visible_frame":visible})
                # Turning and forward movement alone preserve the selected door route.
                _straight(executor, [], points[2])
                crossed = (room_at(scene, env.position()) == room and
                           any(crossed_segment(a,b,opening["segment_xz"])
                               for a,b in zip(env.trajectory[-10:], env.trajectory[-9:])))
                cut("post-"+opening["id"])
                if crossed:
                    break
            if not crossed:
                executor.bundle["route_failures"] = executor.bundle.get("route_failures", []) + [
                    {"destination": room, "from_room": current, "reason": "selected_passage_not_executed"}]
                break
        for point in interior_points(executor, room, settings.get("stops_per_room", 2)):
            if _follow(executor, point, budget=settings.get("route_actions", 64), radius=.25):
                cut(room, True)
    return executor.bundle
