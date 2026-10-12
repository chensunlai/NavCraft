"""Private geometric predicates; no simulator ground truth in tool results."""

import math


def horizontal(position):
    return [float(position[0]), -float(position[2])]


def distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def map_point(calibration, point):
    panel = calibration.panels[0]
    vx, vy, vw, vh = calibration.viewbox
    return [(panel.x_scale * point[0] + panel.x_offset - vx) / vw,
            (panel.y_scale * point[1] + panel.y_offset - vy) / vh]


def crossed_opening(before, after, segment, *, margin=0.03):
    """A nonzero actual movement must intersect the interior of this opening."""
    a, b = segment
    u = [after[i] - before[i] for i in range(2)]
    v = [b[i] - a[i] for i in range(2)]
    cross = lambda x, y: x[0] * y[1] - x[1] * y[0]
    denom = cross(u, v)
    if abs(denom) < 1e-8:
        return False
    delta = [a[i] - before[i] for i in range(2)]
    t, s = cross(delta, v) / denom, cross(delta, u) / denom
    inset = margin / max(distance(a, b), 1e-8)
    return 0 < t <= 1 and inset < s < 1 - inset


def local_event(labels, history):
    """IA success includes the designated crossing and arrival, not just room ID."""
    if not history:
        return {"passed": False, "crossed": False, "arrival": False}
    points = [horizontal(item["pose"]["position"]) for item in history]
    crossed = any(crossed_opening(a, b, labels["door_segment"])
                  for a, b in zip(points, points[1:]))
    arrival = distance(points[-1], labels["destination"]) <= labels.get("radius", 0.35)
    length = sum(distance(a, b) for a, b in zip(points, points[1:]))
    actions = len(history) - 1
    passed = crossed and arrival and actions <= 24 and length <= 4.0001
    return {"passed": passed, "crossed": crossed, "arrival": arrival,
            "actions": actions, "travel_metres": length,
            "method": "replayed actual base positions intersect designated opening and reach local target"}
