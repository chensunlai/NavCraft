"""Curved atomic motion keeps private geometry dense and action budgets discrete."""

import math

import pytest

from src.bench.sequence import score_route
from src.dataset.atomic.geometry import local_event
from src.dataset.episodes.base import EpisodeSpec
from src.environment.tasks.base import GOATTask, ObjectNavTask, VLNCETask
from src.environment.tasks.metrics import ndtw


def episode(*, benchmark="objnav", goals=None, **kwargs):
    return EpisodeSpec(
        id="motion:fixture", source_id="1", benchmark=benchmark,
        scene_id="fake.glb", start_position=[0, 0, 0],
        start_rotation=[0, 0, 0, 1], source_file="fixture",
        goals=goals or [{"modality": "object", "instruction": "chair",
                         "positions": [[0, 0, -1]]}], **kwargs,
    )


def history(points):
    return [{"pose": {"position": [x, 0, -y]}} for x, y in points]


def test_curve_counts_actual_length_but_one_action_and_one_endpoint():
    task = ObjectNavTask({})
    start, bend, end = [0, 0, 0], [.15, 0, 0], [.15, 0, -.2]
    task.reset(episode(), start, 1)
    task.on_step(end, .8, motion_path=[start, bend, end])

    assert task.path_length == pytest.approx(.35)
    assert task.path_length > math.dist(start, end)
    assert task.goal_path_length == task.path_length
    assert task.steps == task.goal_steps == 1
    assert task.trajectory == [start, end]
    assert task.private_state()["motion_trajectory"] == [start, bend, end]
    assert task.private_state()["motion_spans"] == [[0, 2]]

    task.finish(end, .8)
    assert task.steps == task.goal_steps == 2  # STOP remains one action.
    assert task.path_length == pytest.approx(.35)
    assert task.subtasks[0]["steps"] == 2
    assert task.motion_spans == [[0, 2], [2, 3]]
    assert task.metrics()["success"] == 1
    task.reset(episode(), start, 1)
    assert task.motion_trajectory == task.trajectory == [start]
    assert task.motion_spans == []


def test_dense_motion_does_not_spend_microsteps_from_action_budget():
    task = ObjectNavTask({"max_steps": 24})
    point = [0, 0, 0]
    task.reset(episode(), point, 1)
    for step in range(24):
        path = [[0, 0, -(step + fraction / 5) * .1] for fraction in range(6)]
        update = task.on_step(path[-1], 1, motion_path=path)
        assert update.truncated is (step == 23)

    assert task.steps == 24
    assert len(task.trajectory) == 25
    assert len(task.motion_trajectory) == 121
    assert task.motion_spans == [[index * 5, (index + 1) * 5] for index in range(24)]
    assert task.path_length == pytest.approx(2.4)


def test_legacy_steps_preserve_endpoint_length_and_sampling():
    task = ObjectNavTask({})
    task.reset(episode(), [0, 0, 0], 1)
    task.on_step([0, 0, -.25], .75)
    task.finish([0, 0, -.25], .75)
    assert task.motion_trajectory == task.trajectory
    assert task.path_length == .25
    assert task.steps == 2


def test_waypoint_uses_curve_while_ndtw_keeps_atomic_endpoints():
    start, bend, end = [0, 0, 0], [.15, 0, .2], [.3, 0, 0]
    spec = episode(
        benchmark="r2r", gt_path=[start, end],
        route_checks=[{"position": bend, "radius": .01}],
    )
    task = VLNCETask({})
    task.reset(spec, start, .3)
    task.on_step(end, 0, motion_path=[start, bend, end])
    task.finish(end, 0)
    scores = task.metrics()
    assert scores["route_constraints_satisfied"] == 1
    assert scores["ndtw"] == ndtw(task.trajectory, spec.gt_path, task.threshold)
    assert scores["ndtw"] != ndtw(task.motion_trajectory, spec.gt_path, task.threshold)


def test_goat_stop_resets_only_current_goal_motion_accounting():
    goals = [{"modality": "object", "instruction": "chair",
              "positions": [[0, 0, 0]]}] * 2
    task = GOATTask({})
    task.reset(episode(benchmark="goat", goals=goals), [0, 0, 0], 1)
    first = [[0, 0, 0], [.2, 0, 0], [.2, 0, -.2]]
    task.on_step(first[-1], .5, motion_path=first)
    update = task.finish(first[-1], .5)
    assert update.goal_changed and not update.terminated
    assert task.subtasks[0]["path_length"] == pytest.approx(.4)
    assert task.subtasks[0]["steps"] == 2
    assert task.goal_path_length == task.goal_steps == 0
    task.set_goal_distance(1)

    second = [first[-1], [.2, 0, -.3], [.4, 0, -.3]]
    task.on_step(second[-1], .5, motion_path=second)
    update = task.finish(second[-1], .5)
    assert update.terminated and task.steps == 4
    assert task.subtasks[1]["path_length"] == pytest.approx(.3)
    assert task.subtasks[1]["steps"] == 2
    assert task.path_length == pytest.approx(.7)
    assert task.metrics()["composite_success"] == 1
    assert task.motion_trajectory == first + [first[-1]] + second[1:] + [second[-1]]


@pytest.mark.parametrize("points,expected", [
    ([(.5, -.5), (1.3, -.5), (1.3, .5), (.5, .5)], False),
    ([(-.3, -.5), (.5, -.5), (.5, .5), (-.3, .5)], True),
])
def test_door_crossing_uses_real_segments_instead_of_endpoint_chord(points, expected):
    records = history(points)
    labels = {"door_segment": [[0, 0], [1, 0]],
              "destination": list(points[-1]), "radius": .01}
    dense = local_event(labels, records, actions=1)
    chord = local_event(labels, [records[0], records[-1]], actions=1)
    assert dense["crossed"] is expected
    assert dense["passed"] is expected
    assert chord["crossed"] is not expected

    rooms = {
        "origin": {"polygon": [[-1, -1], [2, -1], [2, 0], [-1, 0]]},
        "destination": {"polygon": [[-1, 0], [2, 0], [2, 1], [-1, 1]]},
    }
    route = {"route_events": [{"id": "door", "kind": "portal",
                               "segment_xz": [[0, 0], [1, 0]],
                               "from_room": "origin", "to_room": "destination"}]}
    trajectory = [item["pose"]["position"] for item in records]
    assert score_route(route, trajectory, rooms)["passed"] is expected
    assert score_route(route, [trajectory[0], trajectory[-1]], rooms)["passed"] is not expected


def test_local_event_uses_explicit_actions_with_dense_motion():
    labels = {"door_segment": [[0, 0], [1, 0]],
              "destination": [.5, .5], "radius": .01}
    records = history([(.5, -.5 + index / 30) for index in range(31)])
    assert local_event(labels, records, actions=24)["passed"]
    assert local_event(labels, records, actions=24)["actions"] == 24
    assert not local_event(labels, records, actions=25)["passed"]
    assert not local_event(labels, records)["passed"]  # Old endpoint-count fallback.

    detour = history([(.5, -.5), (.5, .5), (2.6, .5), (.5, .5)])
    result = local_event(labels, detour, actions=1)
    assert result["crossed"] and result["arrival"]
    assert result["travel_metres"] == pytest.approx(5.2)
    assert not result["passed"]
