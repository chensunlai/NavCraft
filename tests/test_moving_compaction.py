"""Offline moving-compaction acceptance must distinguish movement from new frame IDs."""

import base64
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image
import pytest

from src.bench.harness_verify import verify_moving_compaction
from src.harness.moving_compaction import ready, snapshot
from src.harness import runner


def fixture(tmp_path):
    workspace, trace, env, sample = [tmp_path / name for name in ('workspace', 'trace', 'env', 'sample')]
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    def lines(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    trajectory = [[0, 0, -i * .25] for i in range(5)]
    records, frames, actions = [], [], []
    for i, point in enumerate(trajectory):
        identifier = f'frame_{i:06d}'
        ref = f'observations/{identifier}.png'
        path = workspace / ref
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (8, 6), (i * 20, 30, 40)).save(path)
        original = env / f'environment/observations/rgb_{i:06d}.png'
        original.parent.mkdir(parents=True, exist_ok=True)
        original.write_bytes(path.read_bytes())
        records.append({'id': identifier, 'path': ref, 'step': i, 'source': 'observation'})
        frames.append({'id': identifier, 'step': i, 'pose': {'position': point}})
        if i:
            actions.append({'step': i, 'action': 'move_forward', 'before_frame': f'frame_{i-1:06d}',
                            'after_frame': identifier})
    labels = {'door_segment': [[-1, .6], [1, .6]], 'destination': [0, 1], 'radius': .2}
    write(sample / 'private/labels.json', labels)
    lines(workspace / 'observations/index.jsonl', records)
    lines(workspace / 'observations/actions.jsonl', actions)
    lines(env / 'frames.jsonl', frames)
    write(env / 'result.json', {'trajectory': trajectory, 'terminated': True,
                              'metrics': {'steps': 4, 'reason': 'local_event_complete'}})
    checkpoint = {'metadata': {'current': {'task': {'goal_id': 0}, 'observation': records[1]}},
                  'images': [records[1]['path']]}
    state = {'trajectory': trajectory[:2], 'terminated': False, 'truncated': False,
             'metrics': {'steps': 1}}
    note = 'Tentative at frame_000001 step 1; goal D01/T pending, uncertain heading.'
    phase = {'state': state, 'observations': records[:2], 'checkpoint': checkpoint,
             'note': note, 'note_sha256': hashlib.sha256(note.encode()).hexdigest(),
             'image_hashes': {records[1]['path']: hashlib.sha256((workspace / records[1]['path']).read_bytes()).hexdigest()}}
    normal = {'contextWindow': 128000, 'compaction': {'reserveTokens': 28000, 'keepRecentTokens': 20000}}
    diagnostic = {'normal_config': normal, 'phase1': phase, 'phase1_gate': ready(phase),
                  'forced_config': {'contextWindow': 128000, 'compaction': {'enabled': True, 'reserveTokens': 127488, 'keepRecentTokens': 1200}},
                  'restored_config': normal, 'at_compaction': deepcopy(phase)}
    diagnostic['forced_receipt'] = {'configured': True, 'compaction': diagnostic['forced_config']['compaction']}
    diagnostic['restored_receipt'] = {'configured': True, 'compaction': {'enabled': True, **normal['compaction']}}
    def action_events(action):
        identifier = 'call' + str(action['step'])
        assistant = {'role': 'assistant', 'content': [{'type': 'toolCall', 'id': identifier,
                     'name': 'nav_step', 'arguments': {'actions': [action['action']]}}]}
        reply = {'role': 'toolResult', 'toolCallId': identifier, 'toolName': 'nav_step',
                 'content': [{'type': 'text', 'text': json.dumps({'executed_actions': [action]})}]}
        return [{'type': 'message_end', 'data': {'message': assistant}},
                {'type': 'tool_execution_start', 'data': {'toolCallId': identifier, 'toolName': 'nav_step'}},
                {'type': 'tool_execution_end', 'data': {'toolCallId': identifier, 'toolName': 'nav_step',
                 'isError': False, 'result': {'content': reply['content']}}},
                {'type': 'message_end', 'data': {'message': reply}}]
    events = [{'type': 'runtime_started', 'data': {**normal, 'compaction': {'enabled': True, **normal['compaction']}}}]
    events += action_events(actions[0])
    events += [{'type': 'workspace_changes', 'data': {'changes': [{'path': 'memory/moving_compaction.md', 'after': phase['note_sha256']}]}},
               {'type': 'moving_compaction_boundary', 'data': {}},
               {'type': 'runtime_configured', 'data': diagnostic['forced_receipt']},
               {'type': 'compaction_end', 'data': {'result': {'summary': note}}},
               {'type': 'runtime_configured', 'data': diagnostic['restored_receipt']},
               {'type': 'moving_compaction_config_restored', 'data': normal},
               {'type': 'model_input', 'data': {'context': {'messages': [
                   *[row['data']['message'] for row in action_events(actions[0]) if row['type'] == 'message_end'],
                   {'content': [
                   {'type': 'text', 'text': json.dumps(checkpoint)},
                   {'type': 'image', 'data': base64.b64encode((workspace / records[1]['path']).read_bytes()).decode()}]}]}}}]
    for action in actions[1:]:
        events += action_events(action)
    def save():
        lines(trace / 'trace.jsonl', events)
    save()
    args = {'workspace': workspace, 'sample_dir': sample, 'trace_dir': trace,
            'environment_private_dir': env, 'diagnostic': diagnostic,
            'finish_reason': 'local_event_complete',
            'core_close': {'acknowledged': True, 'forced': False, 'exit_code': 0, 'readers_stopped': True}}
    return args, events, save, write


def test_real_translation_both_sides_and_independent_crossing_pass(tmp_path):
    args, _, _, _ = fixture(tmp_path)
    result = verify_moving_compaction(**args)
    assert result['passed'], result['checks']
    assert result['actual_travel_before_m'] == .25
    assert result['actual_travel_after_m'] == .75
    assert result['normal_config']['contextWindow'] == 128000


@pytest.mark.parametrize('change, check', [
    ('remove_compaction', 'compression_after_pause'),
    ('wrong_note_hash', 'judgment_grounded_at_pause'),
    ('no_restore', 'normal_config_restored'),
    ('stale_resumed_image', 'latest_public_images_in_resumed_input'),
    ('unpaired_tool', 'tool_calls_paired'),
    ('event_already_finished', 'goal_pending_before_compression'),
])
def test_partial_or_false_continuation_is_rejected(tmp_path, change, check):
    args, events, save, _ = fixture(tmp_path)
    if change == 'remove_compaction':
        events[:] = [row for row in events if row['type'] != 'compaction_end']
    elif change == 'wrong_note_hash':
        args['diagnostic']['phase1']['note'] += 'Unrecorded claim'
    elif change == 'no_restore':
        args['diagnostic'].pop('restored_config')
    elif change == 'stale_resumed_image':
        row = next(row for row in events if row['type'] == 'model_input')
        row['data']['context']['messages'][-1]['content'].pop()
    elif change == 'unpaired_tool':
        events[:] = [row for row in events if not (row['type'] == 'tool_execution_end'
                     and row['data']['toolCallId'] == 'call4')]
    else:
        args['diagnostic']['phase1']['state']['terminated'] = True
    save()
    result = verify_moving_compaction(**args)
    assert not result['passed']
    assert not result['checks'][check]


def test_changed_public_rgb_cannot_be_verified_from_ids_only(tmp_path):
    args, _, _, _ = fixture(tmp_path)
    Image.new('RGB', (8, 6), 'red').save(args['workspace'] / 'observations/frame_000003.png')
    result = verify_moving_compaction(**args)
    assert not result['passed'] and not result['checks']['motion_records_authentic']


@pytest.mark.parametrize('change', ['old_before_frame', 'duplicate_action', 'reordered_action'])
def test_motion_replay_must_have_correct_predecessor_and_complete_step_sequence(tmp_path, change):
    args, _, _, _ = fixture(tmp_path)
    path = args['workspace'] / 'observations/actions.jsonl'
    actions = [json.loads(line) for line in path.read_text().splitlines()]
    if change == 'old_before_frame':
        actions[2]['before_frame'] = 'frame_000000'
    elif change == 'duplicate_action':
        actions[2] = deepcopy(actions[1])
    else:
        actions[1:3] = reversed(actions[1:3])
    path.write_text(''.join(json.dumps(row) + '\n' for row in actions))
    result = verify_moving_compaction(**args)
    assert not result['passed'] and not result['checks']['motion_records_authentic']


@pytest.mark.parametrize('change, check', [
    ('actual_reply_missing', 'actual_model_tool_messages_paired'),
    ('orphan_model_result', 'model_inputs_tool_messages_paired'),
    ('fake_execution_id', 'execution_ids_from_actual_calls'),
    ('duplicate_start', 'tool_calls_paired'),
])
def test_execution_events_do_not_substitute_for_actual_model_message_pairs(tmp_path, change, check):
    args, events, save, _ = fixture(tmp_path)
    if change == 'actual_reply_missing':
        events[:] = [row for row in events if not (row['type'] == 'message_end'
                     and row['data']['message'].get('toolCallId') == 'call4')]
    elif change == 'orphan_model_result':
        row = next(row for row in events if row['type'] == 'model_input')
        row['data']['context']['messages'][1]['toolCallId'] = 'nonexistent'
    elif change == 'fake_execution_id':
        for row in events:
            if row['type'] in {'tool_execution_start', 'tool_execution_end'} and row['data']['toolCallId'] == 'call4':
                row['data']['toolCallId'] = 'fake'
    else:
        row = next(row for row in events if row['type'] == 'tool_execution_start')
        events.insert(2, deepcopy(row))
    save()
    result = verify_moving_compaction(**args)
    assert not result['passed'] and not result['checks'][check]


@pytest.mark.parametrize('change, check', [
    ('forced_ack_missing', 'temporary_trigger_applied'),
    ('restored_ack_wrong', 'normal_config_restored'),
    ('actual_restore_missing', 'normal_config_restored'),
    ('forced_shutdown', 'core_closed_cleanly'),
])
def test_configuration_claims_and_completion_reason_are_not_runtime_proof(tmp_path, change, check):
    args, events, save, _ = fixture(tmp_path)
    if change == 'forced_ack_missing':
        args['diagnostic'].pop('forced_receipt')
    elif change == 'restored_ack_wrong':
        args['diagnostic']['restored_receipt'] = {'configured': True, 'compaction': {}}
    elif change == 'actual_restore_missing':
        events[:] = [row for row in events if not (row['type'] == 'runtime_configured'
                     and row['data']['compaction']['reserveTokens'] == 28000)]
    else:
        args['core_close']['forced'] = True
    save()
    result = verify_moving_compaction(**args)
    assert not result['passed'] and not result['checks'][check]


def test_rotation_and_fresh_id_are_not_phase1_translation(tmp_path):
    args, _, _, _ = fixture(tmp_path)
    phase = args['diagnostic']['phase1']
    phase['state']['trajectory'][1] = phase['state']['trajectory'][0]
    assert not ready(phase)['passed']
    assert ready(phase)['actual_travel_m'] == 0


def test_snapshot_does_not_observe_or_move_and_tracks_existing_images(tmp_path):
    args, _, _, _ = fixture(tmp_path)
    phase = args['diagnostic']['phase1']
    task = SimpleNamespace(private_state=lambda: deepcopy(phase['state']))
    session = SimpleNamespace(env=SimpleNamespace(task=task))
    path = args['workspace'] / 'memory/moving_compaction.md'
    path.parent.mkdir(parents=True)
    path.write_text(phase['note'])
    captured = snapshot(session, args['workspace'], phase['checkpoint'])
    assert captured['note_sha256'] == phase['note_sha256']
    assert captured['image_hashes'] == phase['image_hashes']
    assert ready(captured)['passed']


@pytest.mark.parametrize('translation', [True, False])
def test_runner_uses_one_core_and_forces_compression_only_after_real_movement_gate(tmp_path, monkeypatch, translation):
    args, _, _, _ = fixture(tmp_path)
    phase = args['diagnostic']['phase1']
    if not translation:
        phase['state']['trajectory'][1] = phase['state']['trajectory'][0]
    repo = tmp_path / 'repo'
    (repo / 'sample').mkdir(parents=True)
    (repo / 'sample/task_map.png').write_bytes(b'fixture map')
    (repo / 'src/harness').mkdir(parents=True)
    (repo / 'src/harness/navigation_prompt.md').write_text('Fixture navigation')
    config = {'output_dir': 'runs', 'scene_archive': 'unused', 'model': 'fixture',
              'api': 'openai-responses', 'python': '/fixture/python'}
    monkeypatch.setattr(runner, 'credentials', lambda *_: {'apiKey': 'fixture key',
                        'baseUrl': 'https://service.invalid', 'model': 'fixture'})
    monkeypatch.setattr('src.harness.config.credentials', runner.credentials)
    monkeypatch.setattr(runner, 'case_config', lambda *_: {'sample': 'sample', 'requests': 24, 'seconds': 30})
    monkeypatch.setattr(runner.memory, 'prepare', lambda *_args, **_kwargs:
                        (args['workspace'] / 'memory/state.json').write_text('{}'))
    (args['workspace'] / 'memory').mkdir()
    monkeypatch.setattr(runner.memory, 'render', lambda *_: {'images': ['observations/frame_000000.png'], 'data': {}})
    monkeypatch.setattr(runner.moving_compaction, 'snapshot', lambda *_: deepcopy(phase))
    monkeypatch.setattr(runner.verify, 'verify_moving_compaction', lambda **_: {'fixture_only': True})
    bridge = Mock()
    bridge.init.return_value = {'task': {'goal_id': 0}, 'initial_hint': {'map_id': 'map_L1'}, 'images': []}
    bridge.finalize.return_value = {'truncated': True}
    monkeypatch.setattr(runner, 'NavigationSession', Mock(return_value=bridge))
    core = Mock()
    core.close_status = {'acknowledged': True, 'forced': False, 'exit_code': 0, 'readers_stopped': True}
    constructor = Mock(return_value=core)
    monkeypatch.setattr(runner, 'CoreClient', constructor)
    core.request.side_effect = lambda op, options: ({'configured': True, 'compaction': options['compaction']}
                                                   if op == 'configure' else {'initialized': True})
    phases = []

    def prompt(text, images, execute, seconds, on_event, cancel_event):
        phases.append(text)
        initial = core.request.call_args_list[0].args[1]
        assert initial['contextWindow'] - initial['compaction']['reserveTokens'] == 100000
        if len(phases) == 2:
            forced = core.request.call_args.args[1]['compaction']
            assert initial['contextWindow'] - forced['reserveTokens'] == 512
            on_event({'type': 'event', 'event': {'type': 'compaction_end',
                      'data': {'result': {'summary': 'fixture summary'}}}})
            restored = core.request.call_args.args[1]['compaction']
            assert restored == {'enabled': True, 'reserveTokens': 28000, 'keepRecentTokens': 20000}
        return {'reason': 'model_finished', 'requests': len(phases), 'last': {'stopReason': 'stop'}}

    core.run_prompt.side_effect = prompt
    runner.run_case(repo, config, 'moving_compaction', directory=tmp_path)
    assert constructor.call_count == 1
    assert sum(call.args[0] == 'init' for call in core.request.call_args_list) == 1
    assert len(phases) == (2 if translation else 1)
    saved = json.loads((tmp_path / 'private/moving_compaction.json').read_text())
    assert saved['phase1_gate']['passed'] == translation
    assert ('forced_receipt' in saved) == translation
    assert ('restored_receipt' in saved) == translation
