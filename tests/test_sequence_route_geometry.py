"""Directed route checks use room geometry local to the actual opening."""

from copy import deepcopy

import pytest

from src.bench.sequence import crosses, score_route


# Local, self-contained fixture copied from the Qa scene's reviewed room/door
# geometry. In particular R01 is concave: its vertex average lies on the wrong
# side of D01 despite the room being north of that opening.
ROOMS = {key: {'polygon': polygon} for key, polygon in {
    'R01': [[.9, 1.2], [2.13, 1.2], [2.13, -3.8], [6.05, -3.8],
            [6.05, -2.1], [6.65, -2.1], [6.65, 1.65], [6.05, 1.65],
            [6.05, 2.55], [.9, 2.55]],
    'R02': [[2.55, 2.55], [6.4, 2.55], [6.4, 5.08], [2.55, 5.08]],
    'R03': [[-1.1, 2.55], [2.55, 2.55], [2.55, 5.08], [.65, 5.08],
            [.65, 4.45], [-1.1, 4.45]],
    'R04': [[-1.1, 1.2], [.9, 1.2], [.9, 2.55], [-1.1, 2.55]],
    'R05': [[-6.55, 1.2], [2.13, 1.2], [2.13, -3.7], [-1.42, -3.7],
            [-1.42, -2.84], [-2.56, -2.84], [-2.56, -3.48], [-6.2, -3.48],
            [-6.2, -2.32], [-6.55, -2.32]],
    'R06': [[-4.35, 1.2], [-2.4, 1.2], [-2.4, 2.22], [-4.35, 2.22]],
    'R07': [[-4.35, 2.58], [-2.4, 2.58], [-2.4, 5.55], [-4.35, 5.55]],
    'R08': [[-6.55, 1.2], [-4.35, 1.2], [-4.35, 5.22], [-6.55, 5.22]],
}.items()}

PORTALS = [
    ('D01', [[1.07, -1.2], [2., -1.2]], 'R05', 'R01', [0, -1]),
    ('D02', [[3.8, -2.55], [5.12, -2.55]], 'R01', 'R02', [0, -1]),
    ('D03', [[1.4, -2.55], [2.28, -2.55]], 'R01', 'R03', [0, -1]),
    ('D04', [[-.28, -2.55], [.78, -2.55]], 'R03', 'R04', [0, 1]),
    ('D05', [[-4.35, -2.76], [-4.35, -3.65]], 'R08', 'R07', [1, 0]),
    ('P01', [[-6.5, -1.2], [-4.4, -1.2]], 'R05', 'R08', [0, -1]),
    ('P02', [[-4.28, -1.2], [-2.47, -1.2]], 'R05', 'R06', [0, -1]),
]


def _event(row):
    identifier, segment, origin, destination, _ = row
    return {'kind': 'portal', 'id': identifier, 'segment_xz': segment,
            'from_room': origin, 'to_room': destination}


@pytest.mark.parametrize('row', PORTALS, ids=[row[0] for row in PORTALS])
def test_all_reviewed_portals_accept_both_declared_directions(row):
    _, segment, _, _, direction = row
    midpoint = [(segment[0][axis] + segment[1][axis]) / 2 for axis in (0, 1)]
    before = [midpoint[0] - .125 * direction[0], .0357,
              midpoint[1] - .125 * direction[1]]
    after = [midpoint[0] + .125 * direction[0], .0357,
             midpoint[1] + .125 * direction[1]]
    event = _event(row)
    assert crosses(before, after, event, ROOMS)
    assert not crosses(after, before, event, ROOMS)
    reverse = dict(event, from_room=event['to_room'], to_room=event['from_room'])
    assert crosses(after, before, reverse, ROOMS)
    assert not crosses(before, after, reverse, ROOMS)
    assert crosses(before, after, dict(event, segment_xz=segment[::-1]), ROOMS)


def test_actual_round_11_d01_crossing_is_not_reversed_by_concave_room():
    before = [1.416506290435791, .035739123821258545, -1.1361546516418457]
    after = [1.4165064096450806, .035739123821258545, -1.3861546516418457]
    event = _event(PORTALS[0])
    assert crosses(before, after, event, ROOMS)
    assert not crosses(after, before, event, ROOMS)
    assert not crosses(before, after, dict(event, from_room='R04'), ROOMS)


def test_crossing_d01_line_outside_designated_opening_does_not_count():
    event = _event(PORTALS[0])
    assert not crosses([.95, 0, -1.1], [.95, 0, -1.35], event, ROOMS)
    assert not crosses([1.5, 0, -.9], [1.5, 0, -1.1], event, ROOMS)
    assert not crosses([1.3, 0, -1.2], [1.6, 0, -1.2], event, ROOMS)


def test_directed_portals_must_occur_in_requested_order():
    bedroom = _event(PORTALS[0])
    bathroom = _event(PORTALS[2])
    spec = {'route_events': [bedroom, bathroom]}
    ordered = [[1.7, 0, -1.1], [1.7, 0, -1.35], [1.7, 0, -2.4], [1.7, 0, -2.7]]
    assert score_route(spec, ordered, ROOMS) == {
        'passed': True,
        'events': [{'id': 'D01', 'kind': 'portal', 'passed': True, 'trajectory_index': 1},
                   {'id': 'D03', 'kind': 'portal', 'passed': True, 'trajectory_index': 3}],
    }
    reversed_order = [[1.7, 0, -2.4], [1.7, 0, -2.7], [1.7, 0, -1.1], [1.7, 0, -1.35]]
    result = score_route(spec, reversed_order, ROOMS)
    assert not result['passed']
    assert result['events'][0]['passed']
    assert not result['events'][1]['passed']


def test_optional_origin_remains_compatible_but_ambiguous_geometry_is_rejected():
    event = {'segment_xz': [[0, 0], [1, 0]], 'to_room': 'B'}
    rooms = {'B': {'polygon': [[0, 0], [1, 0], [1, -1], [0, -1]]}}
    assert crosses([.5, 0, -.2], [.5, 0, .2], event, rooms)
    overlapping = deepcopy(rooms)
    overlapping['A'] = deepcopy(rooms['B'])
    assert not crosses([.5, 0, -.2], [.5, 0, .2], dict(event, from_room='A'), overlapping)
    interior = {'B': {'polygon': [[0, 1], [1, 1], [1, -1], [0, -1]]}}
    assert not crosses([.5, 0, -.2], [.5, 0, .2], event, interior)
    assert not crosses([.5, 0, -.2], [.5, 0, .2],
                       dict(event, segment_xz=[[0, 0], [0, 0]]), rooms)
