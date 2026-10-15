"""Multi-round contracts: no resets, target leakage or budget recounting."""

from copy import deepcopy
import json
from pathlib import Path
import threading
from types import SimpleNamespace

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


def _small_fake_scenario(monkeypatch):
    import src.harness.sequence as module
    from src.environment.fake import FakeSimulator
    from src.environment.habitat import HabitatEnvironment

    monkeypatch.setattr(module, 'HabitatEnvironment', lambda config, task, workspace:
        HabitatEnvironment(config, task, workspace, simulator_factory=FakeSimulator))
    scenario = build_scenario(REPO)
    scenario['rounds'] = deepcopy(scenario['rounds'][:3])
    scenario['environment'].update(width=32, height=24, depth=False)
    for spec in scenario['rounds']:
        spec.pop('snap_viewpoints', None)
    return module, scenario


def _fake_runtime(monkeypatch, module, *, failing_prompt=None):
    opened = []

    class FakeRuntime:
        session_id = 'persistent-test-session'
        close_status = {'acknowledged': True}
        cleanup_error = None

        def __init__(self, workspace, tools, trace):
            self.workspace, self.tools, self.trace = workspace, tools, trace
            self.requests, self.prompts, self.checkpoints, self.closed = 0, [], [], False
            self.environment = tools.context.bridge.env
            self.simulator = self.environment.sim

        def prompt(self, text, images=(), limits=None, on_event=None):
            self.requests += 1
            self.prompts.append({'text': text, 'images': list(images), 'limits': limits})
            index = len(self.prompts)
            assert self.tools.context.bridge.env is self.environment
            assert self.environment.sim is self.simulator
            if failing_prompt == index:
                raise RuntimeError('Current prompt transport failed')
            self.tools.turn_start()
            self.tools.call(f'move-{index}', 'nav_step', {'actions': ['move_forward']})
            if index == 1:
                (self.workspace / 'memory/note.md').write_text('Evidence from the first prompt')
                reason = 'completed'
            else:
                self.tools.call(f'finish-{index}', 'nav_finish',
                                {'goal_id': self.tools.context.bridge.round_index})
                reason = 'host_end'
            value = {'reason': reason, 'answer': f'Answer for prompt {index}',
                     'prompt_id': f'prompt-{index}', 'requests': 1, 'total_requests': self.requests}
            return SimpleNamespace(to_dict=lambda: value)

        def update_checkpoint(self):
            self.checkpoints.append(self.tools.checkpoint())

        def close(self):
            if not self.closed:
                self.closed = True
                self.tools.context.bridge.close()

    def fake_open(repo, config, directory, workspace, system_prompt, tools, *,
                  max_requests, trace=None, on_event=None, cancel_event=None):
        expected = '\n\n'.join((repo / 'src/harness' / name).read_text().strip()
                               for name in ('navigation_prompt.md', 'navigation_experience.md')) + '\n'
        assert system_prompt == expected
        instance = FakeRuntime(workspace, tools, trace)
        opened.append(instance)
        return instance

    monkeypatch.setattr(module, 'open_session', fake_open)
    return opened


def test_sequence_uses_one_persistent_runtime_and_explicit_task_changes(tmp_path, monkeypatch):
    module, scenario = _small_fake_scenario(monkeypatch)
    opened = _fake_runtime(monkeypatch, module)
    result = module.run_sequence({'model': 'fixture'}, scenario, tmp_path, threading.Event())

    assert result['error'] is None
    assert result['finish_reason'] == 'sequence_complete'
    assert len(opened) == 1
    session = opened[0]
    assert session.closed and len(session.prompts) == 3
    assert len(session.checkpoints) == 2
    assert result['session_id'] == session.session_id
    assert result['requests'] == 3
    assert [entry['requests'] for entry in result['rounds']] == [1, 1, 1]
    assert [entry['prompt_id'] for entry in result['rounds']] == ['prompt-1', 'prompt-2', 'prompt-3']
    assert [call['limits'].max_requests for call in session.prompts] == [r['max_requests'] for r in scenario['rounds']]
    assert [call['limits'].seconds for call in session.prompts] == [r['seconds'] for r in scenario['rounds']]
    assert scenario['rounds'][1]['instruction'] not in session.prompts[0]['text']
    assert result['rounds'][0]['memory_after'] == result['rounds'][1]['memory_before']
    assert result['rounds'][0]['result']['terminated'] is False
    assert result['rounds'][0]['result']['metrics']['reason'] == 'completed'
    assert result['rounds'][1]['result']['terminated'] is True
    assert result['rounds'][1]['result']['metrics']['reason'] == 'stop'
    assert (tmp_path / 'workspace/memory/note.md').read_text() == 'Evidence from the first prompt'
    records = [json.loads(line) for line in (tmp_path / 'private/trace.jsonl').read_text().splitlines()]
    assert sum(record['type'] == 'sequence_round_end' for record in records) == 3


def test_failed_prompt_has_its_own_round_record_and_no_previous_answer(tmp_path, monkeypatch):
    module, scenario = _small_fake_scenario(monkeypatch)
    opened = _fake_runtime(monkeypatch, module, failing_prompt=2)
    result = module.run_sequence({'model': 'fixture'}, scenario, tmp_path, threading.Event())

    assert result['finish_reason'] == 'sequence_incomplete'
    assert result['error'] == 'RuntimeError: Current prompt transport failed'
    assert len(result['rounds']) == 2
    first, failed = result['rounds']
    assert first['answer'] == 'Answer for prompt 1'
    assert failed['answer'] is None and failed['last_error'] is None
    assert failed['reason'] == 'runtime_error'
    assert failed['requests'] == 1 and result['requests'] == 2
    assert failed['memory_before'] == first['memory_after']
    assert json.loads((tmp_path / 'private/round_02.json').read_text()) == failed
    assert not (tmp_path / 'private/round_03.json').exists()
    assert opened[0].closed


def test_last_round_runtime_error_does_not_report_sequence_complete(tmp_path, monkeypatch):
    module, scenario = _small_fake_scenario(monkeypatch)
    _fake_runtime(monkeypatch, module, failing_prompt=3)
    result = module.run_sequence({'model': 'fixture'}, scenario, tmp_path, threading.Event())
    assert len(result['rounds']) == len(scenario['rounds'])
    assert result['finish_reason'] == 'sequence_incomplete'
    assert result['rounds'][-1]['answer'] is None
    assert result['rounds'][-1]['error'] == 'RuntimeError: Current prompt transport failed'


def test_persistent_runtime_allows_followups_without_resetting_current_navigation(tmp_path, monkeypatch):
    module, scenario = _small_fake_scenario(monkeypatch)
    from src.harness import runtime
    from src.harness.tools import ToolSet

    workspace, private = tmp_path / 'workspace', tmp_path / 'private'
    for name in ('scene', 'observations', 'memory', 'scratch'):
        (workspace / name).mkdir(parents=True)
    bridge = module.SequenceSession(REPO, workspace, private / 'environment', scenario)
    initial = bridge.begin(0)
    state = {'current': initial, 'done': False, 'goal_changed': False}
    tools = ToolSet(REPO, workspace, bridge, state)
    clients = []

    class FakeCore:
        close_status = {'acknowledged': True}

        def __init__(self, repo, directory):
            self.directory, self.prompts, self.operations = directory, [], []
            clients.append(self)

        def request(self, op, args=None):
            self.operations.append((op, args))
            return {'session_id': 'same-core'} if op == 'init' else {}

        def run_prompt(self, text, images, tool_handler, seconds, on_event=None,
                       cancel_event=None, prompt_id=None):
            self.prompts.append(text)
            index = len(self.prompts)
            (self.directory / 'requests.json').write_text(json.dumps({'attempts': list(range(index))}))
            on_event({'type': 'event', 'event': {'type': 'turn_start', 'data': {}}})
            if index == 1:
                reply = tool_handler(f'{prompt_id}-observe', 'nav_observe', {'mode': 'single'})
                assert not reply['result'].get('isError')
            elif index in (3, 5):
                reply = tool_handler(f'{prompt_id}-move', 'nav_step', {'actions': ['move_forward']})
                assert not reply['result'].get('isError')
                if index == 3:
                    reply = tool_handler(f'{prompt_id}-finish', 'nav_finish', {'goal_id': 0})
                    assert reply['control']['end_run']
            elif index == 4:
                reply = tool_handler(f'{prompt_id}-view', 'view_images', {'refs': initial['images']})
                assert not reply['result'].get('isError')
            return {'type': 'settled', 'prompt_id': prompt_id, 'requests': index,
                    'reason': 'host_end' if index == 3 else 'completed', 'answer': f'answer-{index}'}

        def close(self):
            self.closed = True

        def drain_events(self, callback):
            pass

    monkeypatch.setattr(runtime, 'CoreClient', FakeCore)
    monkeypatch.setattr(runtime, 'core_options', lambda *args, **kwargs: {'auth': {'apiKey': 'fixture'}})
    original_task, original_env, original_sim = bridge.env.task, bridge.env, bridge.env.sim
    note = workspace / 'memory/note.md'
    note.write_text('Keep across followups and task changes')
    session = runtime.open_session(REPO, {}, tmp_path, workspace,
                                   module.load_navigation_prompt(REPO), tools, max_requests=4)
    try:
        first = session.prompt('Observe this task', initial['images'])
        frame = state['current']['observation']['id']
        second = session.prompt('Explain the observation')
        assert first.answer == 'answer-1' and second.answer == 'answer-2'
        assert bridge.round_index == 0 and bridge.env.task is original_task
        assert state['current']['observation']['id'] == frame
        assert not original_task.done and original_task.steps == 0
        assert bridge.env is original_env and bridge.env.sim is original_sim

        finished = session.prompt('Move then finish the same target')
        endpoint = bridge.env.sim.position()
        assert finished.reason == 'host_end' and original_task.done and state['done']
        followup = session.prompt('Review an existing image after finishing')
        assert followup.answer == 'answer-4'
        assert state['done'] and bridge.env.task is original_task
        assert bridge.env.sim.position() == endpoint and bridge.round_index == 0

        next_state = bridge.begin(1)
        state.update(current=next_state, done=False, goal_changed=False)
        state['revision'] += 1
        session.update_checkpoint()
        next_result = session.prompt('Start the next prepared task', next_state['images'])
        assert next_result.answer == 'answer-5'
        assert bridge.env.task is not original_task and bridge.round_index == 1
        assert bridge.env is original_env and bridge.env.sim is original_sim
        assert bridge.env.task.steps == 1 and not bridge.env.task.done
        assert note.read_text() == 'Keep across followups and task changes'
        assert len(clients) == 1 and len(clients[0].prompts) == 5
        assert sum(op == 'init' for op, _ in clients[0].operations) == 1
        assert session.requests == 5
        assert [args['maxRequests'] for op, args in clients[0].operations if op == 'configure'] == [4, 5, 6, 7, 8]
        assert len({first.prompt_id, second.prompt_id, finished.prompt_id, followup.prompt_id, next_result.prompt_id}) == 5
    finally:
        session.close()
        session.close()
    assert clients[0].closed and original_sim.closed
