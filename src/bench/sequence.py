"""Private ordered route and exploration geometry for multi-round tasks."""

import math


def inside(point, polygon):
    x, y = point
    hit = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if (a[1] > y) != (b[1] > y) and x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]:
            hit = not hit
    return hit


def _local_room_side(point, normal, polygon):
    """Find the room's side immediately beside the portal, in world XZ.

    Room polygons use map X,-Z. Whole-room centers do not identify a local
    doorway's side in a concave room. Small normal probes also tolerate minor
    rounding gaps between the archived door and room boundaries. An overlapping
    or unresolved boundary is not sufficient evidence for a directed crossing.
    """
    for distance in (0.0001, 0.001, 0.01, 0.05, 0.1):
        membership = [inside([point[0] + sign * distance * normal[0],
                              -(point[1] + sign * distance * normal[1])], polygon)
                      for sign in (-1, 1)]
        if all(membership):
            return 0
        if any(membership):
            return 1 if membership[1] else -1
    return 0


def crosses(before, after, event, rooms):
    """Intersect designated segment in the requested direction, not nearby radius."""
    a, b = event['segment_xz']
    p, q = [before[0], before[2]], [after[0], after[2]]
    vx, vy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(vx, vy)
    if length < 1e-8:
        return False
    side = lambda t: vx * (t[1] - a[1]) - vy * (t[0] - a[0])
    ps, qs = side(p), side(q)
    if abs(ps - qs) < 1e-8 or ps * qs > 1e-8:
        return False
    t = ps / (ps - qs)
    if not 0 <= t <= 1:
        return False
    intersection = [p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]
    projection = ((intersection[0] - a[0]) * vx + (intersection[1] - a[1]) * vy) / (vx * vx + vy * vy)
    if not -1e-6 <= projection <= 1 + 1e-6:
        return False
    # Sample just inside the segment ends to avoid polygon-vertex ambiguity.
    local_t = min(1 - 1e-4, max(1e-4, projection))
    local_point = [a[0] + local_t * vx, a[1] + local_t * vy]
    normal = [-vy / length, vx / length]
    destination_side = _local_room_side(local_point, normal, rooms[event['to_room']]['polygon'])
    if not destination_side:
        return False
    if event.get('from_room'):
        origin_side = _local_room_side(local_point, normal, rooms[event['from_room']]['polygon'])
        if origin_side != -destination_side:
            return False
    return (qs - ps) * destination_side > 1e-8


def score_route(round_spec, trajectory, rooms):
    cursor, results = 1, []
    for event in round_spec.get('route_events', []):
        found = None
        for i in range(cursor, len(trajectory)):
            if event['kind'] == 'portal':
                matched = crosses(trajectory[i - 1], trajectory[i], event, rooms)
            else:
                matched = any(math.dist(trajectory[i], point) < event['radius'] for point in event['positions'])
            if matched:
                found = i
                break
        results.append({'id': event['id'], 'kind': event['kind'], 'passed': found is not None,
                        'trajectory_index': found})
        if found is None:
            results.extend({'id': e['id'], 'kind': e['kind'], 'passed': False, 'trajectory_index': None}
                           for e in round_spec['route_events'][len(results):])
            break
        cursor = found + 1
    return {'passed': all(r['passed'] for r in results), 'events': results}
