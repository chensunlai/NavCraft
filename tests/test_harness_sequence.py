"""Multi-round contracts: no resets, target leakage or budget recounting."""

from copy import deepcopy
from pathlib import Path

import pytest

from src.bench.astra_campaign import build_scenario
from src.bench.sequence import crosses, score_route
from src.harness.sequence import validate_scenario

REPO = Path(__file__).resolve().parents[1]


def test_latest_twelve_rounds_have_fixed_sequence_and_private_targets():
    scenario = build_scenario(REPO)
    validate_scenario(scenario)
    rounds = scenario['rounds']
    assert len(rounds) == 12
    assert [r['id'] for r in rounds] == ['explore', 'find_bed', 'find_bathtub', 'find_dining_table',
        'bedroom_tv', 'preparation_area_sink', 'hallway_bathroom_sink', 'double_basin',
        'living_room_tv', 'relocate_closet', 'relocate_hallway', 'relocate_bathroom']
    assert all('teleport' not in r for r in rounds[:9])
    assert all('teleport' in r for r in rounds[9:])
    assert rounds[11]['private_verification']['target_object_id'] == 'O188'
    assert rounds[11]['private_verification']['source_evidence_ids'] == ['render_O188_1']
    assert rounds[7]['private_verification']['target_object_id'] == 'O108'
    assert scenario['start_position'][0] == -.7
    assert rounds[9]['teleport']['position'][0] == 3.55


@pytest.mark.parametrize('corruption', ['duplicate', 'empty', 'budget', 'target'])
def test_invalid_sequences_fail_before_any_environment_or_model_call(corruption):
    s = build_scenario(REPO)
    if corruption == 'duplicate': s['rounds'][1]['id'] = s['rounds'][0]['id']
    if corruption == 'empty': s['rounds'] = []
    if corruption == 'budget': s['rounds'][2]['max_requests'] = 0
    if corruption == 'target': s['rounds'][3]['positions'] = []
    with pytest.raises(ValueError): validate_scenario(s)


def test_designated_portal_requires_crossing_in_correct_direction_and_on_segment():
    event = {'segment_xz': [[0,0], [1,0]], 'to_room':'B'}
    rooms = {'B': {'polygon': [[0,0], [1,0], [1,-1], [0,-1]]}}
    assert crosses([.5,0,-.2], [.5,0,.2], event, rooms)
    assert not crosses([.5,0,.2], [.5,0,-.2], event, rooms)
    assert not crosses([1.5,0,-.2], [1.5,0,.2], event, rooms)
    assert not crosses([.5,0,-.3], [.5,0,-.2], event, rooms)


def test_waypoint_cannot_be_completed_after_the_later_portal():
    spec = {'route_events':[{'kind':'waypoint','id':'table','positions':[[.5,0,-.5]],'radius':.1},
            {'kind':'portal','id':'door','segment_xz':[[0,0],[1,0]],'to_room':'B'}]}
    rooms = {'B':{'polygon':[[0,0],[1,0],[1,-1],[0,-1]]}}
    ordered = [[.5,0,-1],[.5,0,-.5],[.5,0,-.2],[.5,0,.2]]
    assert score_route(spec,ordered,rooms)['passed']
    reversed_order = [[.5,0,-.2],[.5,0,.2],[.5,0,-.5]]
    assert not score_route(spec,reversed_order,rooms)['passed']


def test_round_transition_preserves_environment_pose_frames_and_memory(tmp_path, monkeypatch):
    import json
    import src.harness.sequence as module
    from src.environment.fake import FakeSimulator
    from src.environment.habitat import HabitatEnvironment

    monkeypatch.setattr(module, 'HabitatEnvironment', lambda config, task, workspace:
        HabitatEnvironment(config, task, workspace, simulator_factory=FakeSimulator))
    s = build_scenario(REPO)
    s['environment']['depth'] = False
    bridge = module.SequenceSession(REPO, tmp_path / 'workspace', tmp_path / 'private', s)
    try:
        initial = bridge.begin(0)
        first_env, first_sim = bridge.env, bridge.env.sim
        bridge.step(['move_forward'])
        end_position = first_sim.position()
        bridge.finish(0)
        (tmp_path / 'workspace/memory').mkdir()
        note = tmp_path / 'workspace/memory/note.md'
        note.write_text('Only self-authored current evidence')
        second = bridge.begin(1)
        assert bridge.env is first_env and bridge.env.sim is first_sim
        assert bridge.env.sim.position() == end_position
        assert second['task']['goal_id'] == 1
        assert second['task']['instruction'] == s['rounds'][1]['instruction']
        assert second['observation']['id'] != initial['observation']['id']
        assert note.read_text() == 'Only self-authored current evidence'
        for forbidden in ('positions', 'rotation_xyzw', 'depth', 'distance', 'target_object_id'):
            assert forbidden not in json.dumps(second)
        with pytest.raises(ValueError): bridge.finish(0)
        assert not bridge.env.task.done
    finally:
        bridge.close()
