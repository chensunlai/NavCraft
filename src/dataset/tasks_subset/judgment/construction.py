"""Build static judgments from verified, cut-scoped observation evidence."""
from collections import defaultdict, deque
import hashlib
import random
from .calibration import describe, floor_support, public_camera, shared_pair, measured_floor_motion
from itertools import combinations
import json
from math import atan2, cos, degrees, dist, hypot, radians, sin
from pathlib import Path

from PIL import Image, ImageDraw

from ..bundle import freeze_case, reject, require_trial_snapshot
from ..catalog import JUDGMENT_KINDS

BOOL = {"type": "boolean"}
SIDES = ("forward", "right", "behind", "left")


def enum(*values):
    return {"type": "string", "enum": list(dict.fromkeys(values))}


def num(unit, tolerance):
    return {"type": "number", "unit": unit, "tolerance": tolerance}


def arr(items, *, unordered=False):
    return {"type": "array", "items": items, "unordered": unordered}


def obj(**properties):
    return {"type": "object", "properties": properties}


def xyz(point):
    return point[0], point[2]


def ego(point, origin, heading):
    dx, dz = point[0] - origin[0], point[1] - origin[1]
    angle = radians(heading)
    return dx * cos(angle) + dz * sin(angle), dx * sin(angle) - dz * cos(angle)


def signed(angle):
    return (angle + 180) % 360 - 180


def sector(right, forward, margin=8):
    angle = degrees(atan2(right, forward))
    if hypot(right, forward) < .18 or abs((angle % 90) - 45) < margin:
        return None
    return SIDES[int((angle + 45) % 360 // 90)]


def shortest_routes(edges, source, goal):
    """All fewest-crossing routes on an undirected, observed multigraph."""
    graph = defaultdict(list)
    for a, b, opening in edges:
        graph[a].append((b, opening))
        graph[b].append((a, opening))
    todo, distance, routes = deque([source]), {source: 0}, {source: [[source]]}
    while todo:
        here = todo.popleft()
        if here == goal:
            return routes[here]
        for other, opening in sorted(set(graph[here])):
            step = distance[here] + 1
            if other not in distance:
                distance[other], routes[other] = step, []
                todo.append(other)
            if distance[other] == step:
                routes[other].extend([*route, opening, other] for route in routes[here])
    return []


def row(key, caption="", markers=()):
    return key, caption, list(markers)


def mark(sighting, name):
    return name, sighting["pixel"]


class Context:
    def __init__(self, bundle, state, root, config):
        self.bundle, self.state, self.root, self.config = bundle, state, root, config
        self.frames = bundle["frames"]
        self.current = state["current_frame"]
        self.keys = set(state["frame_keys"])
        self.nodes = state["node_frames"]
        self.rooms = {x["id"]: x for x in bundle["scene"].get("rooms", [])}
        self.openings = {x["id"]: x for x in bundle["scene"].get("openings", [])}
        self.objects = {x["id"]: x for x in bundle["scene"].get("objects", [])}
        self.sights = defaultdict(list)
        for item in bundle.get("sightings", []):
            if (item["frame"] in self.keys and item.get("verified") is True and item.get("method")
                    and item.get("pixel") is not None and item.get("world") is not None):
                self.sights[item["frame"]].append(item)
        self.current_panorama = self.panorama(self.current)
        self.options = defaultdict(list)

    def panorama(self, key):
        f = self.frames[key]
        group = f.get("group")
        candidates = [k for k in self.state["frame_keys"] if group and self.frames[k].get("group") == group
                      and self.frames[k].get("stream") == f.get("stream")
                      and self.frames[k]["step"] == f["step"]]
        if not candidates and f.get("public_pose"):
            candidates = [k for k in self.state["frame_keys"] if self.frames[k].get("group")
                          and self.frames[k].get("stream") == f.get("stream")
                          and self.frames[k]["step"] == f["step"]
                          and self.frames[k]["public_pose"]["pos"] == f["public_pose"]["pos"]]
        candidates.sort(key=lambda k: self.frames[k]["yaw"])
        if [self.frames[k]["yaw"] for k in candidates] != list(range(0, 360, 60)):
            return []
        if len({self.frames[k]["group"] for k in candidates}) != 1:
            return []
        base = self.frames[candidates[0]]["public_pose"]
        if any(self.frames[k]["public_pose"]["pos"] != base["pos"] or abs(signed(
               self.heading(k) - self.heading(candidates[0]) - self.frames[k]["yaw"])) > .2
               for k in candidates):
            return []
        return candidates

    def heading(self, key):
        frame = self.frames[key]
        matrix = frame.get("world_from_optical")
        if matrix:
            return degrees(atan2(matrix[0][2], -matrix[2][2]))
        return frame["public_pose"]["heading"]

    def position(self, key):
        f = self.frames[key]
        pose = f["pose"]
        if "position" in pose:
            return xyz(pose["position"])
        uv = f["public_pose"]["pos"]
        scale = self.state["transform"]
        return tuple(o + v * scale["size_m"] for o, v in zip(scale["origin"], uv))

    def prefix(self):
        if not self.nodes:
            return []
        if any(len(keys) != 6 or set(keys) - self.keys or set(keys) != set(self.panorama(keys[0]))
               for keys in self.nodes.values()):
            return []
        return [row(key, f"Node {node}, view {self.frames[key]['yaw']}°")
                for node, keys in self.nodes.items() for key in keys]

    def put(self, kind, prompt, schema, accepted, pictures, *, memory=False, private=None):
        targets = {k for t in self.bundle.get("targets", []) for k in t.get("reference_frames", [])}
        if not accepted or not pictures or any(k not in self.keys and
                not (caption == "Target reference" and k in targets) for k, caption, _ in pictures):
            return
        if memory and not self.prefix():
            return
        self.options[kind].append((prompt, schema, accepted, pictures, memory, private or {}))


def _panorama_rows(c, keys):
    return [row(k, f"View {c.frames[k]['yaw']}° relative to the panorama's forward view") for k in keys]


def _distinct(c, keys, kinds):
    best = {}
    for k in keys:
        for s in c.sights[k]:
            if s["kind"] in kinds:
                best.setdefault((s["kind"], s["id"]), s)
    return list(best.values())


def _layout(c, items, base, pictures, kind, memory=False):
    if len(items) < 2:
        return
    supported = [(s, shared_pair(c,s)) for s in items]
    supported = [(s,other) for s,other in supported if other is not None]
    if len(supported) < 2:
        return
    shown = [s for s,_ in supported[:4]]
    pictures = pictures + [row(other["frame"], "Second measurement view " + public_camera(c,other["frame"]),
                              [mark(other,name)]) for name,(_,other) in zip("ABCD",supported[:4])]
    marks = [row(s["frame"], "Observed landmark " + public_camera(c,s["frame"]), [mark(s, name)])
             for name, s in zip("ABCD", shown)]
    coordinates = {name: ego(xyz(s["world"]), c.position(base), c.heading(base))
                   for name, s in zip("ABCD", shown)}
    frame = "current panorama forward direction" if not memory else "current body heading"
    fields = {name: obj(right=num("m", .45), forward=num("m", .45)) for name in coordinates}
    answer = {name: dict(zip(("right", "forward"), p)) for name, p in coordinates.items()}
    c.put(kind, f"Sketch the marked visible landmarks only, with the current camera as origin and "
          f"{frame} as up. Report each label as right/forward in metres; signs indicate side.",
          obj(**fields), [answer], pictures + marks, memory=memory, private={"measurements":shown,"variant":"calibrated_multiview"})


def _visual(c):
    for key in [c.current, *c.current_panorama]:
        for a, b in combinations(c.sights[key][:12], 2):
            if a["id"] == b["id"] or dist(a["pixel"], b["pixel"]) < 24:
                continue
            seed = hashlib.sha256(f"{c.bundle.get('seed',0)}:{c.state['id']}:{key}:{a['id']}:{b['id']}".encode()).digest()
            if seed[0] % 2:
                a,b = b,a
            pictures = [row(key, "Visible landmarks", [mark(a, "A"), mark(b, "B")])]
            calibration = floor_support(c,a)
            calibrated = calibration is not None and floor_support(c,b) == calibration
            pa, pb, start = xyz(a["world"]), xyz(b["world"]), c.position(key)
            heading = c.heading(key)
            direction = sector(*ego(pa, pb, heading))
            if direction and calibrated:
                c.put("object_relative_position", "From this heading, in which horizontal direction is A relative to B?" + describe(calibration),
                      enum(*SIDES), [direction], pictures, private={"measurements":[a,b],"variant":"calibrated_floor"})
            other = sector(*ego(pa, pb, heading + 90))
            if other and calibrated:
                c.put("change_reference_frame", "Imagine standing at B facing 90° clockwise from the shown camera. "
                      "In which horizontal direction is A?" + describe(calibration), enum(*SIDES), [other], pictures, private={"measurements":[a,b],"variant":"calibrated_floor"})
            tol = max(.35, .12 * dist(pa, pb))
            if calibrated and dist(pa, pb) > .35:
                c.put("distance_estimation", "Estimate the horizontal ground distance between A and B in metres." + describe(calibration),
                      num("m", tol), [dist(pa, pb)], pictures, private={"measurements":[a,b],"variant":"calibrated_floor"})
            da, db = dist(pa, start), dist(pb, start)
            if calibrated and abs(da - db) > .4:
                c.put("compare_distance_size", "Which mark is nearer horizontally to this camera, A or B?" + describe(calibration),
                      enum("A", "B"), ["A" if da < db else "B"], pictures, private={"measurements":[a,b],"variant":"calibrated_floor"})
    panorama = c.current_panorama
    if not panorama:
        return
    landmarks = _distinct(c, panorama, {"object", "opening"})
    base = panorama[0]
    pictures = _panorama_rows(c, panorama)
    _layout(c, landmarks, base, pictures, "visible_topdown_layout")
    directions = [(s, sector(*ego(xyz(s["world"]), c.position(base), c.heading(base)))) for s in landmarks]
    directions = [(s, d) for s, d in directions if d][:4]
    if len(directions) >= 2:
        fields = {name: enum(*SIDES) for name in "ABCD"[:len(directions)]}
        answer = dict(zip(fields, (d for _, d in directions)))
        annotated = [row(s["frame"], "Marked observed landmark", [mark(s, name)])
                     for name, (s, _) in zip(fields, directions)]
        c.put("egocentric_layout", "For each marked observed landmark report its sector around the "
              "panorama origin: forward, right, behind or left.", obj(**fields), [answer], pictures + annotated)
    for s, d in directions:
        if _identity_verified(s) or s["kind"] == "opening":
            # Match the crop to the full views. Do not attach the crop's original yaw.
            c.put("panorama_view_relation", "Match the cropped landmark to the six views. In which "
                  "body-relative sector is it: forward, right, behind or left?",
                  enum(*SIDES), [d], pictures + [row(s["frame"], "Cropped landmark", [mark(s, "A")])],
                  private={"landmark_crop": s["frame"]})


def _motion(c, a, b):
    right, forward = ego(c.position(b), c.position(a), c.heading(a))
    return {"right": right, "forward": forward, "yaw": signed(c.heading(b) - c.heading(a))}


MOTION = obj(right=num("m", .16), forward=num("m", .16), yaw=num("signed_degrees", 12))


def _views(c):
    historical = [k for k in c.state["frame_keys"] if c.frames[k].get("stream") ==
                  c.frames[c.current].get("stream")][-48:]
    for a, b in combinations(historical, 2):
        motion = _motion(c, a, b)
        if hypot(motion["right"], motion["forward"]) > 3:
            continue
        shared = [(p, q) for p in c.sights[a] for q in c.sights[b]
                  if (p["kind"], p["id"]) == (q["kind"], q["id"]) and
                  dist(p["world"], q["world"]) < .25]
        if not shared or all(abs(v) < .15 for v in motion.values()):
            continue
        p, q = shared[0]
        pictures = [row(a, "First view", [mark(p, "A")]), row(b, "Second view", [mark(q, "A")])]
        floors = [(p,q) for p,q in shared if floor_support(c,p) and floor_support(c,q)]
        displacement = hypot(motion["right"],motion["forward"])
        if 3 <= c.frames[b]["step"]-c.frames[a]["step"] <= 5 and displacement >= .55 and len(floors)>=2:
            floors = next(((x,y) for x,y in combinations(floors,2)
                           if dist(x[0]["world"],y[0]["world"])>=.5
                           and floor_support(c,x[0])==floor_support(c,y[0])
                           and floor_support(c,x[1])==floor_support(c,y[1])
                           and dist(x[0]["pixel"],y[0]["pixel"])>=24
                           and dist(x[1]["pixel"],y[1]["pixel"])>=24),None)
            recovered=measured_floor_motion(c,floors) if floors else None
            if floors and recovered and abs(signed(recovered["yaw"]-motion["yaw"]))<=2 and all(
                    abs(recovered[k]-motion[k])<=.08 for k in ("right","forward")):
                marks_a=[mark(x[0],name) for name,x in zip("AB",floors)]
                marks_b=[mark(x[1],name) for name,x in zip("AB",floors)]
                evidence=[row(a,"First view",marks_a),row(b,"Second view",marks_b)]
                prompt="The same two marked points lie on the floor in both views. Estimate camera motion in the first camera frame: right/forward metres and clockwise yaw." + describe(
                    {"first":floor_support(c,floors[0][0]),"second":floor_support(c,floors[0][1])})
                private={"measurements":[s for pair in floors for s in pair],
                         "baseline_m":displacement,"public_reconstruction":recovered,"variant":"calibrated_floor_motion"}
                c.put("relative_camera_pose",prompt,MOTION,[motion],evidence,private=private)
                c.put("camera_motion",prompt,MOTION,[motion],evidence,private=private)
        elif displacement <= .03 and abs(motion["yaw"]) >= 20:
            c.put("relative_camera_pose","The camera only rotated in place. Estimate its clockwise signed turn in degrees.",
                  num("signed_degrees",8),[motion["yaw"]],pictures,
                  private={"measurements":[p,q],"baseline_m":displacement,"variant":"rotation_only"})
        for p, q in shared:
            if not (_identity_verified(p) and _identity_verified(q)):
                continue
            def add(other, answer):
                c.put("cross_view_identity", "Do the marked instances A and B denote the same physical entity?",
                      BOOL, [answer], [row(a, "First view", [mark(p, "A")]),
                                      row(b, "Second view", [mark(other, "B")])])
            add(q, True)
            for other in c.sights[b]:
                if (_identity_verified(other) and other["kind"] == p["kind"] and other["id"] != p["id"]
                        and dist(q["pixel"], other["pixel"]) >= 36
                        and any(pair.get("review_candidate") and pair.get("same_category") is True
                                and pair["observed_object"] == other["id"] and pair["requested_object"] == p["id"]
                                and pair["current_frame"] == b and pair["reference_frames"] == [a]
                                for pair in c.bundle.get("navigation_evidence", {}).get("distinct_instances", []))):
                    add(other, False)


def _identity_verified(s):
    return s["kind"] in {"object", "opening"} and s.get("identity_verified") is True


def _memory(c):
    if not c.prefix():
        return
    current = [row(c.current, "Current observation")]
    sightings = _distinct(c, [k for keys in c.nodes.values() for k in keys], {"object", "opening"})
    if len(c.nodes) >= 2 and len({s["frame"] for s in sightings}) >= 2:
        _layout(c, sightings, c.current, current, "memory_topdown_layout", memory=True)
    anchors = _anchors(c, _node_snapshot(c)["nodes"])
    here_uv = c.frames[c.current]["public_pose"]["pos"]
    distances = sorted((dist(here_uv, uv) * c.state["transform"]["size_m"], node)
                       for node, uv in anchors.items())
    if len(distances) >= 2 and distances[1][0] - distances[0][0] > .3:
        c.put("nearest_node", "Which node is closest horizontally to the current camera on the prefix map? "
              "Use the fixed node anchors.", enum(*c.nodes), [distances[0][1]], current, memory=True)
    levels = c.state.get("trajectory_level")
    history = _trajectory(c)
    if len(history) >= 2 and levels:
        here = c.frames[c.current]["public_pose"]["pos"]
        scale = c.state["transform"]["size_m"]
        # Skip the recent samples so the final approach cannot count as an earlier visit.
        prior = [x for x in history if x["step"] < c.state["step"] - 8 and x.get("level") == levels]
        if prior:
            nearest = min(dist(here, x["pos"]) * scale for x in prior)
            if abs(nearest - .75) > .2:
                c.put("visited_place", "Before the last eight steps, did the plotted trajectory contain a "
                      "atomic-step endpoint within 0.75 m of the current camera, on this floor?", BOOL, [nearest < .75],
                      current, memory=True)
    transform = c.state["transform"]
    for s in c.sights[c.current]:
        second = shared_pair(c,s)
        if second is None:
            continue
        u, v = ((x - o) / transform["size_m"] for x, o in zip(xyz(s["world"]), transform["origin"]))
        c.put("image_to_node_map", "Estimate the map coordinates [u,v] of mark A. Use the public map "
                  "scale and origin; u increases right and v increases down.",
                  arr(num("map_uv", .55 / transform["size_m"])), [[u, v]],
                  [row(c.current, "First observation " + public_camera(c,c.current), [mark(s, "A")]),
                   row(second["frame"], "Second observation " + public_camera(c,second["frame"]), [mark(second,"A")])], memory=True,
                  private={"measurements":[s,second],"variant":"calibrated_multiview"})
    objects = [s for s in sightings if s["kind"] == "object" and shared_pair(c,s)]
    for a, b in combinations(objects[:8], 2):
        if a["frame"] == b["frame"]:
            continue
        side = sector(*ego(xyz(a["world"]), xyz(b["world"]), c.heading(c.current)))
        if side:
            c.put("memory_object_relation", "Using the current heading as forward, which direction is "
                  "the marked object A from the marked object B?", enum(*SIDES), [side],
                  current + [row(s["frame"], "Historical object "+public_camera(c,s["frame"]),[mark(s,name)])
                             for name,item in (("A",a),("B",b)) for s in (item,shared_pair(c,item))], memory=True,
                  private={"measurements":[a,b],"variant":"calibrated_multiview"})
    for goal in objects[:8]:
        others = sorted((dist(xyz(goal["world"]), xyz(s["world"])), s["id"], s)
                        for s in objects if s is not goal)[:3]
        if len(others) >= 2 and others[1][0] - others[0][0] > .35:
            # Assign labels independently of distance.
            others.sort(key=lambda x: x[1])
            names = list("ABC"[:len(others)])
            answer = names[min(range(len(others)), key=lambda i: others[i][0])]
            c.put("nearest_landmark", "Which marked landmark is closest horizontally to T?",
                  enum(*names), [answer], current + [row(s["frame"],"Landmark "+public_camera(c,s["frame"]),[mark(s,name)])
                             for name,item in [("T",goal)]+[(name,item[2]) for name,item in zip(names,others)]
                             for s in (item,shared_pair(c,item))], memory=True,
                  private={"measurements":[goal]+[x[2] for x in others],"variant":"calibrated_multiview"})


def _rooms(c):
    from ..review.memory import public_room

    aliases = lambda room: public_room(c.state, room)
    panorama = c.current_panorama or [c.current]
    current = _panorama_rows(c, panorama) if c.current_panorama else [row(c.current, "Current observation")]
    here = c.state.get("room")
    prefix_openings = _distinct(c, c.state["frame_keys"], {"opening"})
    seen = {s["id"]: s for s in prefix_openings if s["id"] in c.openings}
    by_room = defaultdict(dict)
    for k in c.state["frame_keys"]:
        for s in c.sights[k]:
            room = c.frames[k].get("room")
            if s["kind"] == "opening" and s["id"] in seen and room in (c.openings[s["id"]]["connects"] or []):
                by_room[room].setdefault(s["id"], s)
    traversals = c.state.get("traversals", [])
    actual_crossed = {event["opening"] for event in traversals}
    crossings = []
    for event in traversals:
        a, b, opening = (event.get(k) for k in ("from_room", "to_room", "opening"))
        before, after = event.get("before_frame"), event.get("after_frame")
        if (a in c.rooms and b in c.rooms and a != b and opening in seen and
                {a, b} == set(c.openings[opening]["connects"]) and before in c.keys and after in c.keys and
                c.frames[before].get("room") == a and c.frames[after].get("room") == b):
            crossings.append((a, b, opening, event))
    crossed = {opening for _, _, opening, _ in crossings}
    room_ids = sorted(r for r in c.state.get("visited_rooms", []) if aliases(r))
    review = next((x for x in c.bundle.get("judgment_evidence", {}).get("rooms", [])
                   if x["state"] == c.state["id"] and x.get("review_candidate")), {})
    choices = review.get("rooms", [])
    if len(choices) >= 2 and aliases(here) in choices:
        c.put("current_room", "Which previously labelled room does the current observation show?",
              enum(*choices), [aliases(here)], current, memory=True)
    exits = [s for s in _distinct(c, panorama, {"opening"}) if s["id"] in by_room[here]]
    directions, marked = {}, []
    for s in exits:
        base = panorama[0]
        d = sector(*ego(xyz(s["world"]), c.position(base), c.heading(base)))
        if d:
            directions[s["id"]] = d
            marked.append(row(s["frame"], "Observed opening", [mark(s, s["id"])]))
        kind = c.openings[s["id"]]["kind"]
        kind = "opening" if kind == "passage" else kind
        if s.get("type_verified") is True and kind in {"door", "opening", "window", "mirror", "glass"}:
            c.put("opening_type", "What is marked at A: door, opening, window, mirror, or glass?",
                  enum("door", "opening", "window", "mirror", "glass"), [kind],
                  [row(s["frame"], "Opening candidate", [mark(s, "A")])])
    if marked and c.current_panorama:
        c.put("room_exit_directions", "Report each marked visible exit's direction relative to the body.",
              obj(**{k: enum(*SIDES) for k in directions}), [directions], current + marked)
    if not c.prefix():
        return
    # Include observed endpoints for each crossing, preserving public order and direction.
    log = []
    for a, b, opening, event in crossings:
        if not aliases(a) or not aliases(b):
            continue
        s = seen[opening]
        log.extend([row(s["frame"], "Recognizable opening", [mark(s, opening)]),
                    row(event["before_frame"], "Earlier observed view"),
                    row(event["after_frame"], "Later observed view")])
    missing_crossings = actual_crossed - crossed
    for s in exits:
        opening = s["id"]
        if opening in missing_crossings or not aliases(here):
            continue
        opposite = next((b if a == here else a for a, b, key, _ in crossings if key == opening), "unknown")
        c.put("room_exit_destination", f"Which room has been confirmed on the other side of {opening}? "
              'Use "unknown" if this opening has not been crossed in the shown history.',
              enum(*(aliases(r) for r in room_ids), "unknown"), [aliases(opposite) or "unknown"],
              current + log + [row(s["frame"], "Opening under consideration", [mark(s, opening)])], memory=True)
    frontiers = {room: set(items) - actual_crossed for room, items in by_room.items() if room in room_ids}
    if frontiers and not missing_crossings & seen.keys():
        rooms = sorted(room for room, values in frontiers.items() if values)
        pictures = current + log + [row(s["frame"], f"Opening observed from {aliases(room)}", [mark(s, key)])
                                    for room, items in by_room.items() if room in room_ids for key, s in items.items()]
        c.put("rooms_with_untraversed_exits", "Which rooms have at least one marked exit that was seen "
              "but never crossed in the shown history? Return a list of room IDs.",
              arr(enum(*(aliases(r) for r in room_ids)), unordered=True), [[aliases(r) for r in rooms]], pictures, memory=True)
        choices = sorted(frontiers.get(here, ()))
        if choices:
            c.put("exploration_frontier", "To explore new space from the current room, choose any marked "
                  "opening seen from here that has not been crossed in the shown history.",
                  enum(*by_room[here]), choices, pictures, memory=True)
    edges = [(a, b, opening) for a, b, opening, _ in crossings if aliases(a) and aliases(b)]
    node_edges = [(event["from_node"], event["to_node"], opening) for _, _, opening, event in crossings
                  if event.get("from_node") in c.nodes and event.get("to_node") in c.nodes]
    if edges and here in room_ids:
        for target in room_ids:
            if target == here:
                continue
            routes = shortest_routes(edges, here, target)
            c.put("room_adjacency", f"Which confirmed opening(s) directly connect rooms {aliases(here)} and {aliases(target)}? "
                  "Return an empty list if none were crossed.",
                  arr(enum(*sorted(crossed)), unordered=True),
                  [sorted({opening for a, b, opening in edges if {a, b} == {here, target}})],
                  current + log, memory=True)
            if not routes:
                continue
            c.put("room_route", f"Give a route from {aliases(here)} to {aliases(target)} using the fewest observed crossings. "
                  "Return the room IDs in order, including both endpoints.",
                  arr(enum(*(aliases(r) for r in room_ids))), list({tuple(route[::2]): [aliases(r) for r in route[::2]] for route in routes}.values()),
                  current + log, memory=True)
            visible_ids = {s["id"] for s in exits}
            first = sorted({r[1] for r in routes})
            if set(first) <= visible_ids:
                c.put("destination_exit", f"Which currently visible opening starts a route to {aliases(target)} "
                      "with the fewest observed crossings?", enum(*sorted(visible_ids)), first,
                      current + log + marked, memory=True)
    for source, target in combinations(c.nodes, 2):
        routes = shortest_routes(node_edges, source, target)
        if routes:
            extra = []
            for a, b, opening, event in crossings:
                if event.get("from_node") in c.nodes and event.get("to_node") in c.nodes:
                    extra.append(row(event["before_frame"], "Earlier observed view"))
                    extra.append(row(event["after_frame"], "Later observed view"))
            c.put("node_opening_route", f"From {source} to {target}, give a shortest observed route as an "
                  "array alternating node IDs and opening IDs, including both endpoints.",
                  arr(enum(*c.nodes, *sorted(crossed))), routes, current + log + extra, memory=True)


def _actions(trial):
    return [a for call in trial["calls"] if call["name"] == "move"
            for a in call.get("arguments", call.get("args", {})).get("actions", [])]


def _bound_trial(c, trial):
    try:
        require_trial_snapshot(c.root, c.state, trial)
    except ValueError:
        return False
    return True


def _point_choices(c):
    trials = [t for t in c.bundle.get("trials", []) if t["start_state"] == c.state["id"]
              and _bound_trial(c, t) and t.get("replay", {}).get("passed") is True and t.get("motions")
              and t["end_frame"] not in c.keys and 0 < len(_actions(t)) <= 12]
    for point in c.sights[c.current]:
        if point["kind"] != "point" or not any(t.get("target") == point["id"] for t in trials):
            continue
        near, far = [], []
        for t in trials:
            distance = dist(c.position(t["end_frame"]), xyz(point["world"]))
            if distance <= .25:
                near.append(t)
            elif distance >= .55:
                far.append(t)
        if near and far:
            pair = sorted([near[0], far[0]], key=lambda t: t["id"])
            yield point, pair, "AB"[pair.index(near[0])]


def judgment_review_proposals(bundle, root):
    """Propose answerability checks from the observed cut, without revealing trial outcomes."""
    for state in bundle["states"]:
        c = Context(bundle, state, root, {})
        prefix = c.prefix()
        node_rooms = {node["id"]: node["room"] for node in _node_snapshot(c)["nodes"]
                      if node.get("room") and node["id"] in c.nodes}
        rooms = sorted(set(node_rooms.values()))
        if (prefix and len(rooms) >= 2 and state.get("room_aliases", {}).get(state.get("room")) in rooms and
                c.current not in {key for key, _, _ in prefix} and not c.current_panorama):
            references = [key for key, _, _ in prefix]
            grouping = [node_rooms.get(node, "unlabelled") for node in c.nodes for _ in c.nodes[node]]
            yield {"kind": "judgment_room_identity", "name": "Current room",
                   "frames": [c.current, *references],
                   "prompt": "Does the first image contain distinctive features to match it to exactly one "
                             f"room in the following node views? Their room labels in order: {grouping}.",
                   "check_keys": ["unambiguous_room"],
                   "fact": {"section": "judgment_evidence", "key": "rooms", "value": {
                       "state": state["id"], "rooms": rooms}}}
        for point, pair, _ in _point_choices(c):
            yield {"kind": "judgment_point_choices", "name": "Atomic approach",
                   "frames": [c.current], "markers": [point["pixel"]],
                   "prompt": "Can the marked point and its approach be judged from the starting view well "
                             f"enough to distinguish these two action sequences: {[ _actions(t) for t in pair ]}? "
                             "Reject if either route's outcome depends on an occluded obstacle.",
                   "check_keys": ["answerable"],
                   "fact": {"section": "judgment_evidence", "key": "point_choices", "value": {
                       "state": state["id"], "point": point["id"], "trials": [t["id"] for t in pair]}}}
        for trial in bundle.get("trials", []):
            if (trial["start_state"] != state["id"] or not _bound_trial(c, trial) or trial.get("replay", {}).get("passed") is not True
                    or trial.get("end_frame") in c.keys or not trial.get("motions")):
                continue
            actions = _actions(trial)
            if not actions or len(actions) > 12:
                continue
            frames = c.current_panorama or [c.current]
            purposes = [
                ("collision", "whether any commanded motion would contact an obstacle"),
                ("forward_range", "where the first forward block occurs, to within about 0.30 m"),
            ]
            opening = trial.get("opening")
            sight = next((s for k in frames for s in c.sights[k]
                          if s["kind"] == "opening" and s["id"] == opening), None)
            if sight:
                purposes.append(("opening", "whether the actions would cross the cyan-marked opening"))
            target_node = trial.get("target_node")
            if target_node in c.nodes:
                purposes.append(("direct", "whether a direct route to the node shown in the last view is traversable"))
            for purpose, question in purposes:
                if purpose == "forward_range" and (set(actions) != {"move_forward"} or
                        not trial["motions"][-1]["blocked"]):
                    continue
                if purpose == "collision" and not (len(trial["motions"]) == len(actions) or any(
                        m["collision"] for m in trial["motions"])):
                    continue
                view_keys = list(frames)
                markers = [None] * len(frames)
                if purpose == "opening":
                    markers[frames.index(sight["frame"])] = sight["pixel"]
                if purpose == "direct":
                    view_keys.append(c.nodes[target_node][0])
                    markers.append(None)
                yield {"kind": f"judgment_trial_{purpose}", "name": purpose, "frames": view_keys,
                       "markers": markers,
                       "prompt": f"From the starting images, is there visible evidence to determine {question} "
                                 f"when attempting {json.dumps(actions)}? The first view is forward. "
                                 "Consider occlusions, hidden floor obstacles and scale. Reject if uncertain.",
                       "check_keys": ["answerable"],
                       "fact": {"section": "judgment_evidence", "key": "trials", "value": {
                           "trial": trial["id"], "frames": view_keys, "purpose": purpose}}}


def _trials(c):
    choices = c.bundle.get("judgment_evidence", {}).get("point_choices", [])
    for point, pair, answer in _point_choices(c):
        trial_ids = [t["id"] for t in pair]
        if any(r.get("review_candidate") and r["state"] == c.state["id"] and r["point"] == point["id"]
               and r["trials"] == trial_ids for r in choices):
            options = "; ".join(f"{name}: {json.dumps(_actions(t))}" for name, t in zip("AB", pair))
            c.put("motion_to_point", "Which attempted atomic sequence ends within 0.35 m horizontally "
                  f"of the marked point? {options}", enum("A", "B"), [answer],
                  [row(c.current, "Visible waypoint", [mark(point, "P")])],
                  private={"trial": trial_ids[0], "trials": trial_ids,
                           "future_frames": [t["end_frame"] for t in pair]})

    for trial in c.bundle.get("trials", []):
        if (trial["start_state"] != c.state["id"] or not _bound_trial(c, trial) or trial.get("replay", {}).get("passed") is not True
                or not trial.get("motions") or trial["end_frame"] in c.keys):
            continue
        reviews = {v["purpose"]: v for v in c.bundle.get("judgment_evidence", {}).get("trials", [])
                   if v["trial"] == trial["id"] and v.get("review_candidate")}
        review = next(iter(reviews.values()), {})
        if not review.get("frames") or set(review["frames"]) - c.keys:
            continue
        actions = [a for call in trial["calls"] if call["name"] == "move"
                   for a in call.get("arguments", call.get("args", {})).get("actions", [])]
        if not actions:
            continue
        evidence = [row(k, "Before the proposed movement") for k in review["frames"]]
        private = {"trial": trial["id"], "future_frame": trial["end_frame"]}
        ref = trial["reference"]
        motions = trial["motions"]
        if "collision" in reviews and (any(m["collision"] for m in motions) or len(motions) == len(actions)):
            c.put("action_collision", f"Would any action collide if you attempted {json.dumps(actions)} from here?",
                  BOOL, [any(m["collision"] for m in motions)], evidence, private=private)
        start = c.position(c.current)
        end = c.position(trial["end_frame"])
        if "forward_range" in reviews and all(a == "move_forward" for a in actions) and motions[-1]["blocked"]:
            c.put("traversable_space", "Repeatedly attempt forward steps until blocked. Estimate horizontal "
                  "displacement from the starting position in metres.", num("m", .3), [dist(start, end)],
                  evidence, private=private)
        opening = trial.get("opening")
        seen = next((s for k in reviews.get("opening", {}).get("frames", []) for s in c.sights[k]
                     if s["kind"] == "opening" and s["id"] == opening), None)
        if seen and opening in c.openings:
            from ..navigation.scoring import doorway_crossed
            a = c.frames[c.current]["pose"]["position"]
            positions = [a] + [sample["position"] for m in motions for sample in m["samples"]]
            door = c.openings[opening]
            end_room = c.frames[trial["end_frame"]].get("room")
            start_room = c.state.get("room")
            if not start_room or not end_room or start_room not in (door.get("connects") or []):
                continue
            crossed = end_room != start_room and end_room in door["connects"] and doorway_crossed(
                positions, c.bundle["scene"], door, start_room, end_room)
            c.put("opening_passability", f"Would the proposed actions {json.dumps(actions)} take you through "
                  "the marked opening?", BOOL, [crossed], evidence +
                  [row(seen["frame"], "Opening", [mark(seen, "A")])], private=private)
        target_node = trial.get("target_node")
        if ("direct" in reviews and target_node in c.nodes and type(ref.get("reached")) is bool):
            c.put("direct_node_passability",
                  f"Can you travel directly from the current place to node {target_node} with the shown approach?",
                  BOOL, [ref["reached"]], [row(k, "Route evidence") for k in reviews["direct"]["frames"]],
                  private=private, memory=True)


def _motion_events(c):
    snapshot = c.root / c.state["snapshot"] / "state.json"
    if not snapshot.is_file():
        return
    motions = json.loads(snapshot.read_text()).get("environment", {}).get("motions", [])
    frames_by_step = defaultdict(list)
    for key in c.state["frame_keys"]:
        f = c.frames[key]
        if not f.get("group"):
            frames_by_step[f["step"]].append(key)
    for motion in motions:
        step = motion["step"]
        for before, after in zip(frames_by_step[step-1][-1:], frames_by_step[step][:1]):
            yield {"before_frame": before, "after_frame": after, "receipt": motion}


def _outcomes(c):
    for event in _motion_events(c):
        a, b = event["before_frame"], event["after_frame"]
        if {a, b} - c.keys or c.frames[b]["step"] != c.frames[a]["step"] + 1:
            continue
        motion = _motion(c, a, b)
        pictures = [row(a, "Before"), row(b, "After")]
        common = {s["id"] for s in c.sights[a]} & {s["id"] for s in c.sights[b]}
        if common and abs(motion["yaw"]) >= 20:
            c.put("observed_action_outcome", "Estimate the camera's clockwise signed turn in degrees.",
                  num("signed_degrees", 10), [motion["yaw"]], pictures)
        receipt = event.get("receipt", {})
        if receipt.get("blocked") is True and type(receipt.get("collision")) is bool:
            displacement = hypot(motion["right"], motion["forward"])
            if abs(displacement - .05) >= .02:
                c.put("blocked_receipt",
                      f"The movement reported blocked=true, collision={str(receipt['collision']).lower()}. "
                      f"The camera moved {displacement:.3f} m horizontally. Did it collide? Did it move at least 0.05 m?",
                      obj(collided=BOOL, moved=BOOL),
                      [{"collided": receipt["collision"], "moved": displacement >= .05}], pictures)


def _targets(c):
    from .targets import arrival_candidates, different_view

    current = c.current_panorama or [c.current]
    for target in c.bundle.get("targets", []):
        if not target.get("identity_verified") or not target.get("reference_pixel"):
            continue
        ref = target["reference_frames"][0]
        if ref not in c.keys:
            continue
        refs = [row(ref, "Target reference", [("T", target["reference_pixel"])])]
        for sight in _distinct(c, current, {"object"}):
            if (sight["id"] == target["object_id"] and _identity_verified(sight)
                    and different_view(c.frames[ref], c.frames[sight["frame"]])):
                c.put("target_identity", "Do marks T and A identify the same physical instance?",
                      BOOL, [True], refs + [row(sight["frame"], "Candidate", [mark(sight, "A")])])
        for pair in c.bundle.get("navigation_evidence", {}).get("distinct_instances", []):
            if (pair.get("review_candidate") and pair["requested_object"] == target["object_id"]
                    and pair["reference_frames"] == [ref] and pair["current_frame"] in current
                    and different_view(c.frames[ref], c.frames[pair["current_frame"]])):
                sight = next((s for s in c.sights[pair["current_frame"]] if
                              s["id"] == pair["observed_object"] and _identity_verified(s)), None)
                if sight:
                    c.put("target_identity", "Do marks T and A identify the same physical instance?",
                          BOOL, [False], refs + [row(sight["frame"], "Candidate", [mark(sight, "A")])])
        from ..review.instances import node_visibility
        native = [record.get("completeness_binding") for row in c.bundle.get("instance_evidence",{}).values()
                  for identifier,record in row.get("objects",{}).items() if identifier==target["object_id"]
                  and record.get("completeness_binding")]
        scope = native[0]["scope"] if native else None
        coverage = {node:node_visibility(c.bundle,keys,target["object_id"],scope=scope,root=c.root)
                    for node,keys in c.nodes.items()}
        if c.prefix() and scope and coverage and "unknown" not in coverage.values():
            c.put("target_seen_nodes", "Which node panoramas show the target instance T? "
                  f"Instance scope: {scope}. Support/stands/cabinets outside that scope do not count. "
                  "Return a list of IDs.", arr(enum(*c.nodes), unordered=True),
                  [sorted(node for node,result in coverage.items() if result=="visible")],refs,memory=True,
                  private={"instance":target["object_id"],"scope":scope,"visibility":coverage,
                           "rule":"complete scoped native instance masks; all six frames per node",
                           "instance_frames":[key for keys in c.nodes.values() for key in keys]})


    arrivals = c.bundle.get("judgment_evidence", {}).get("arrivals", [])
    if any(r["state"] == c.state["id"] and r.get("review_candidate") for r in arrivals):
        for target, state, sight, threshold, arrived in arrival_candidates(c.bundle):
            if state["id"] != c.state["id"] or not any(
                    r.get("review_candidate") and r["state"] == state["id"] and r["target"] == target["id"]
                    for r in arrivals):
                continue
            c.put("target_arrival", f"Is the camera within {threshold} m horizontally of the marked "
                  "target object's footprint, on the same floor and in the same room?",
                  BOOL, [arrived], [row(target["reference_frames"][0], "Target reference",
                                        [("T", target["reference_pixel"])]),
                                    row(sight["frame"], "Current view", [mark(sight, "A")])])


def _node_snapshot(c):
    path = c.root / c.state["snapshot"] / "workspace/memory/nodes.json"
    return json.loads(path.read_text()) if path.is_file() else {"nodes": [], "rooms": {}}


def _contradictions(c):
    from ..navigation.construction import _altered_cut

    records = {x["id"]: x for x in _node_snapshot(c)["nodes"]}
    current = c.current_panorama or [c.current]
    for fact in c.bundle.get("navigation_evidence", {}).get("corrections", []):
        node = fact.get("node")
        if (fact.get("state") != c.state["id"] or not fact.get("review_candidate")
                or fact.get("evidence_frame") not in current or fact["field"] != "room"
                or records.get(node, {}).get("room") != fact["correct"]):
            continue
        # The erroneous record is saved in a separate snapshot; the original remains immutable.
        altered = _altered_cut(c.bundle, c.root, c.state, fact)
        c.put("memory_contradiction", "Which node's recorded room conflicts with the current observation?",
              enum(*records), [node], [row(fact["evidence_frame"], "Current observation")], memory=True,
              private={"altered_state": altered, "correction": fact})
        break


def _anchors(c, nodes):
    anchors = {}
    for node in nodes:
        key = next((k for k in c.state["frame_keys"] if
                    c.frames[k].get("group") == node["anchor"] and c.frames[k].get("yaw") == 0), None)
        if key is not None:
            anchors[node["id"]] = c.frames[key]["public_pose"]["pos"]
    return anchors


def _trajectory(c):
    endpoints = {}
    for p in c.state.get("public_trajectory", []):
        if p["step"] <= c.state["step"]:
            endpoints[p["step"]] = {**p, "pos": [round(v, 4) for v in p["pos"]]}
    return [endpoints[k] for k in sorted(endpoints)]


def _map_image(c, path, nodes):
    """Draw only public anchors, trajectory and current pose, in map orientation."""
    transform = c.state["transform"]
    here = c.frames[c.current]["public_pose"]
    history = _trajectory(c)
    anchors = _anchors(c, nodes)
    points = [here["pos"], *anchors.values(), *(p["pos"] for p in history)]
    left, top = map(min, zip(*points))
    right, bottom = map(max, zip(*points))
    width = max(right-left, bottom-top, .1)
    pad = 34
    scale = (600-2*pad)/width

    def px(p):
        return pad + (p[0]-left)*scale, pad + (p[1]-top)*scale

    image = Image.new("RGB", (600, 640), "white")
    draw = ImageDraw.Draw(image)
    if len(history) > 1:
        draw.line([px(p["pos"]) for p in history], fill="#76767f", width=2)
    for node, uv in anchors.items():
        x, y = px(uv)
        draw.ellipse((x-4, y-4, x+4, y+4), fill="blue")
        draw.text((x+5, y-12), node, fill="navy")
    x, y = px(here["pos"])
    angle = radians(here["heading"])
    draw.line((x, y, x+20*sin(angle), y-20*cos(angle)), fill="red", width=3)
    draw.text((x+6, y+4), "Now", fill="red")
    draw.text((12, 606), f"Top-left reference ({left:.4f}, {top:.4f}) at ({pad},{pad}) px; "
              f"{scale:.1f} px per UV unit", fill="black")
    draw.text((12, 621), f"One UV unit = {transform['size_m']} m; U right; V down.", fill="black")
    image.save(path)


def _render(c, pictures, memory, directory, crop_key=None, include_trajectory=False):
    nodes = _node_snapshot(c)
    complete = c.prefix() if memory else []
    if memory:
        used = {key for key,_,_ in complete}
        for node in nodes["nodes"]:
            for group in node.get("history",[]):
                for key in c.state["frame_keys"]:
                    if c.frames[key].get("group")==group and key not in used:
                        complete.append(row(key,f"Historical panorama for node {node['id']}, view yaw {c.frames[key]['yaw']}"))
                        used.add(key)
    texts = []
    if memory:
        by_id = {n["id"]: n for n in nodes["nodes"]}
        if set(c.nodes) != set(by_id):
            raise ValueError("Snapshot node IDs do not match frozen prefix")
        texts.append("Current public pose: " + json.dumps(c.frames[c.current]["public_pose"]))
        anchors = _anchors(c,nodes["nodes"])
        for node, keys in c.nodes.items():
            texts.append(json.dumps({"node":node,"anchor_pos":anchors.get(node),
                                     "panorama_pose":c.frames[keys[0]]["public_pose"],
                                     "view_yaws":[c.frames[k]["yaw"] for k in keys]}))
        texts += [f"{node}: {n['description']}" for node, n in by_id.items() if n.get("description")]
        texts += [f"Room {name}: {desc}" for name, desc in nodes["rooms"].items() if desc]
        for node, n in by_id.items():
            if n.get("room"):
                texts.append(f"{node} is recorded in room {n['room']}.")
    combined = {}
    crop = None
    for key, caption, markers in complete + pictures:
        if key == crop_key and caption == "Cropped landmark":
            crop = (key, markers[0][1])
            continue
        old_caption, old_markers = combined.setdefault(key, ([], []))
        if caption and caption not in old_caption:
            old_caption.append(caption)
        for marker in markers:
            if marker not in old_markers:
                old_markers.append(marker)
    budget = c.config.get("judgment_image_budget", 24)
    if len(combined) + bool(memory) + bool(crop) > budget:
        raise ValueError(f"Image budget {budget} cannot show complete evidence ({len(combined) + bool(memory) + bool(crop)})")
    directory.mkdir(parents=True, exist_ok=True)
    images, image_sources = [], []
    for i, (key, (captions, markers)) in enumerate(combined.items()):
        source = c.root / c.frames[key]["rgb"]
        if markers:
            with Image.open(source) as raw:
                image = raw.convert("RGB")
            draw = ImageDraw.Draw(image)
            for text, (x, y) in markers:
                if not (0 <= x < image.width and 0 <= y < image.height):
                    raise ValueError(f"Verified marker outside source RGB: {key}")
                draw.ellipse((x-7, y-7, x+7, y+7), outline="#00ffff", width=3)
                draw.text((max(0, x-8), max(0, y-19)), text, fill="white", stroke_width=2, stroke_fill="black")
            source = directory / f"marked-{i}.jpg"
            image.save(source, quality=92)
        images.append({"path": source.resolve(), "caption": "; ".join(captions)})
        image_sources.append({"purpose": "target_reference" if "Target reference" in captions else "prefix", "frames": [key],
                              "markers": [{"text":name,"pixel":pixel} for name,pixel in markers]})
    if crop:
        key, (x, y) = crop
        path = directory / "landmark.jpg"
        with Image.open(c.root / c.frames[key]["rgb"]) as raw:
            box = (max(0, x-48), max(0, y-48), min(raw.width, x+48), min(raw.height, y+48))
            raw.convert("RGB").crop(tuple(map(int, box))).save(path)
        images.append({"path": path.resolve(), "caption": "Cropped landmark to match"})
        image_sources.append({"purpose": "prefix", "frames": [key], "crop": list(box)})
    if memory:
        path = directory / "map.png"
        _map_image(c, path, nodes["nodes"])
        images.append({"path": path.resolve(), "caption": "Blank prefix map, public pose, anchors and walked history"})
        image_sources.append({"purpose": "prefix_map", "frames": list(c.state["frame_keys"])})
        if include_trajectory:
            texts.append("Public trajectory samples (step, [u,v], floor): " + json.dumps([
                [p["step"], p["pos"], p.get("level")] for p in _trajectory(c)]))
    return images, texts, list(combined), image_sources


def build_judgments(bundle, root, config):
    from .targets import register_instance_targets

    root = Path(root)
    register_instance_targets(bundle)
    result = []
    limit = config.get("judgments_per_kind_per_state", 2)
    selected = config.get("judgment_kinds", list(JUDGMENT_KINDS))
    if set(selected) - JUDGMENT_KINDS.keys():
        raise ValueError("Unknown judgment kind in judgment_kinds")
    for state in list(bundle["states"]):
        if state.get("altered_memory"):
            continue
        c = Context(bundle, state, root, config)
        for builder in (_visual, _views, _memory, _rooms, _trials, _outcomes, _targets, _contradictions):
            builder(c)
        for kind in selected:
            description = JUDGMENT_KINDS[kind]
            choices = c.options[kind]
            if not choices:
                reject(root, kind, state["id"], f"Missing qualifying evidence: {description['requirements']}")
                continue
            # Preserve both polarities when they are available within the per-kind budget.
            if choices[0][1] == BOOL and len(choices) > 1:
                choices.sort(key=lambda x: x[2][0])
                opposite = next((i for i, x in enumerate(choices) if x[2] == [True]), None)
                if opposite:
                    choices.insert(1, choices.pop(opposite))
            rng = random.Random(hashlib.sha256(f"{bundle.get('seed',0)}:{state['id']}:{kind}".encode()).hexdigest())
            buckets = defaultdict(list)
            for candidate in choices:
                buckets[json.dumps(candidate[2],sort_keys=True)].append(candidate)
            groups = list(buckets.values())
            rng.shuffle(groups)
            for group in groups:
                rng.shuffle(group)
            choices = [group[i] for i in range(max(map(len,groups))) for group in groups if i<len(group)]
            produced = 0
            for i, (prompt, schema, accepted, pictures, memory, private) in enumerate(choices):
                if produced >= limit:
                    break
                identifier = f"{state['id']}-{kind}-{i+1}"
                render_context = (Context(bundle, private["altered_state"], root, config)
                                  if "altered_state" in private else c)
                actual_state = render_context.state
                try:
                    images, texts, source_frames, image_sources = _render(render_context, pictures, memory, root / "private" / "judgments" / identifier,
                                                             private.get("landmark_crop"), kind == "visited_place")
                except (OSError, ValueError) as exc:
                    reject(root, kind, identifier, str(exc))
                    continue
                if texts:
                    prompt += "\n\nPublic prefix memory:\n" + "\n".join(texts)
                task = {"id": identifier, "kind": kind, "mode": "judgment", "instruction": prompt,
                        "snapshot": f"../../{actual_state['snapshot']}", "answer_schema": schema,
                        "limits": {"max_requests": 1, "max_turns": 1, "seconds": 180},
                        "provenance": {"scene_id": bundle["scene_id"], "state_id": actual_state["id"], "source_kind": "observations"}}
                label = {"answer_schema": schema, "accepted": accepted, **private}
                construction = {"image_sources": image_sources, "source_frames": source_frames, "state_id": actual_state["id"],
                                "scene_source": bundle.get("source", {}), "physical_replay": bool(private.get("trial"))}
                result.append(freeze_case(root, task, label, construction, images))
                produced += 1
    from collections import Counter
    stats = {}
    for task in result:
        label = json.loads(((root/task["path"]).parent/"private/label.json").read_text())
        item = stats.setdefault(task["kind"],{"answers":Counter(),"baseline_m":[],"variants":Counter()})
        item["answers"][json.dumps(label["accepted"],sort_keys=True)] += 1
        item["variants"][label.get("variant","default")] += 1
        if "baseline_m" in label:
            item["baseline_m"].append(label["baseline_m"])
    from ..bundle import write_json
    write_json(root/"private/judgment_statistics.json",stats)
    return result
