"""Private ordered route and exploration geometry for multi-round tasks."""

import math

def inside(point, polygon):
    x, y = point
    hit = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if (a[1] > y) != (b[1] > y) and x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]:
            hit = not hit
    return hit


def crosses(before, after, event, rooms):
    """Intersect designated segment in the requested direction, not nearby radius."""
    a, b = event['segment_xz']
    p, q = [before[0], before[2]], [after[0], after[2]]
    vx, vy = b[0] - a[0], b[1] - a[1]
    side = lambda t: vx * (t[1] - a[1]) - vy * (t[0] - a[0])
    ps, qs = side(p), side(q)
    if abs(ps - qs) < 1e-8 or ps * qs > 1e-8:
        return False
    t = ps / (ps - qs)
    intersection = [p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]
    projection = ((intersection[0] - a[0]) * vx + (intersection[1] - a[1]) * vy) / (vx * vx + vy * vy)
    polygon = rooms[event['to_room']]['polygon']
    centroid = [sum(c[0] for c in polygon) / len(polygon), sum(c[1] for c in polygon) / len(polygon)]
    return -1e-6 <= projection <= 1 + 1e-6 and (qs - ps) * side([centroid[0], -centroid[1]]) > 1e-8


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
