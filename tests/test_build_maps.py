"""Offline workflow tests; no model service, renderer or GPU required."""

import asyncio
import json
from pathlib import Path
import sys

import pytest

from src.dataset.build_maps.appserver import AppServer
from src.dataset.build_maps import runner
from src.dataset.build_maps.settings import Settings, read_env


def settings(tmp_path):
    return Settings(tmp_path, tmp_path / '.env', tmp_path / 'codex',
                    'chosen-model', 'xhigh', 'http://example.test/custom', 'private-value', gpu=1)


def test_env_and_provider_parameters(tmp_path):
    env = tmp_path / '.env'
    env.write_text('export URL="http://example.test/v1"\nKEY="literal$VALUE#suffix" # comment\n')
    assert read_env(env) == {'URL': 'http://example.test/v1', 'KEY': 'literal$VALUE#suffix'}
    config = settings(tmp_path)
    command = config.command()
    assert 'model="chosen-model"' in command
    assert 'model_reasoning_effort="xhigh"' in command
    assert 'model_providers.scene_maps.request_max_retries=5' in command
    assert 'model_providers.scene_maps.stream_max_retries=5' in command
    assert 'model_providers.scene_maps.base_url="http://example.test/custom"' in command
    assert config.api_key not in json.dumps(command) + json.dumps(config.public()) + repr(config)
    assert config.environment()['CODEX_HOME'] == str(tmp_path / 'codex')
    assert config.environment()['CUDA_VISIBLE_DEVICES'] == '1'
    assert config.environment()['SCENE_MAP_API_KEY'] == config.api_key
    with pytest.raises(ValueError, match='GPUs'):
        Settings(tmp_path, env, tmp_path, 'm', 'high', 'http://test', 'key', gpu=2)


def fixture_plan(tmp_path):
    skill = tmp_path / 'skill'
    (skill / 'scripts').mkdir(parents=True)
    (skill / 'SKILL.md').write_text('---\nname: build-scene-map\n---\nBuild and review.')
    (skill / 'scripts/validate_archive.py').write_text('print("{}")')
    asset = tmp_path / 'asset.glb'
    asset.touch()
    scenes = [{'scene_id': 'a', 'dataset': 'hm3d', 'scene_path': 'asset.glb',
               'output_dir': 'maps/hm3d/a', 'order': 1},
              {'scene_id': 'b', 'dataset': 'hssd-hab', 'scene_path': 'asset.glb',
               'output_dir': 'maps/hssd/b', 'order': 2}]
    queue = tmp_path / 'queue.json'
    queue.write_text(json.dumps({'output_root': 'maps', 'scenes': scenes}))
    config = settings(tmp_path)
    plan = runner.make_plan(queue, tmp_path, skill, config, 'full', Path(sys.executable))
    return plan, config


def test_plan_preserves_queue_order_and_does_not_modify_archives(tmp_path):
    plan, config = fixture_plan(tmp_path)
    output = tmp_path / 'batch'
    runner.save_plan(plan, output)
    assert [r['scene_id'] for r in plan['scenes']] == ['a', 'b']
    assert not (tmp_path / 'maps').exists()
    assert not config.codex_home.exists()
    assert config.api_key not in (output / 'plan.json').read_text()
    assert 'GPU 1' in next(output.glob('*/prompt.md')).read_text()
    runner.save_plan(plan, output, resume=True)
    changed = dict(plan, stage='walls')
    with pytest.raises(ValueError, match='changed'):
        runner.save_plan(changed, output, resume=True)
    with pytest.raises(ValueError, match='Unknown scene'):
        runner.make_plan(Path(plan['queue']), tmp_path, Path(plan['skill']), config,
                         'full', Path(sys.executable), ['missing'])


def test_queue_rejects_duplicate_or_external_output(tmp_path):
    plan, config = fixture_plan(tmp_path)
    queue = Path(plan['queue'])
    data = json.loads(queue.read_text())
    data['scenes'][1]['output_dir'] = data['scenes'][0]['output_dir']
    queue.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Duplicate'):
        runner.make_plan(queue, tmp_path, Path(plan['skill']), config, 'full', Path(sys.executable))
    data['scenes'][1]['output_dir'] = '../outside'
    queue.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Invalid scene output'):
        runner.make_plan(queue, tmp_path, Path(plan['skill']), config, 'full', Path(sys.executable))


def test_review_requires_local_evidence(tmp_path):
    report = {'status': 'reviewed', 'summary': 'done', 'limitations': [], 'review_evidence': []}
    with pytest.raises(ValueError, match='evidence'):
        runner.parse_report(json.dumps(report), tmp_path)
    report['review_evidence'] = ['../outside.txt']
    with pytest.raises(ValueError, match='external'):
        runner.parse_report(json.dumps(report), tmp_path)
    (tmp_path / 'review.json').write_text('{}')
    report['review_evidence'] = ['review.json']
    assert runner.parse_report(json.dumps(report), tmp_path) == report


def test_home_lock_prevents_overlapping_batch(tmp_path):
    with runner.home_lock(tmp_path):
        with pytest.raises(RuntimeError, match='Another map builder'):
            with runner.home_lock(tmp_path):
                pass
    with runner.home_lock(tmp_path):
        pass


FAKE_SERVER = '''
import json,sys

def send(x):
    print(json.dumps(x),flush=True)

for line in sys.stdin:
    req=json.loads(line)
    method=req.get('method')
    if method=='initialize':
        send({'id':req['id'],'result':{}})
    elif method=='turn/start':
        tid=req['params']['threadId']
        # A stale event and the real completion both arrive before the RPC response.
        send({'method':'turn/completed','params':{'threadId':'other','turn':{'id':'old','status':'completed'}}})
        send({'method':'item/completed','params':{'threadId':tid,'turnId':'turn',
             'item':{'type':'agentMessage','text':'{"ready":true}'}}})
        send({'method':'turn/completed','params':{'threadId':tid,'turn':{'id':'turn','status':'completed'}}})
        send({'id':req['id'],'result':{'turn':{'id':'turn'}}})
    elif method=='crash':
        sys.exit(2)
    elif method=='interactive':
        send({'id':99,'method':'item/tool/requestUserInput','params':{}})
'''


def test_stdio_handles_early_events_redaction_and_eof(tmp_path):
    script = tmp_path / 'server.py'
    script.write_text(FAKE_SERVER)
    async def exercise():
        async with AppServer([sys.executable, str(script)], {}, tmp_path, tmp_path / 'log', ('secret',)) as server:
            reply = await server.turn('thread', [{'type': 'text', 'text': 'secret'}], 'model', 'high', {}, 2)
            assert json.loads(reply['answer']) == {'ready': True}
            with pytest.raises(RuntimeError, match='closed stdout'):
                await server.request('crash', {}, timeout=2)
        assert server.process.returncode is not None
    asyncio.run(exercise())
    assert 'secret' not in (tmp_path / 'log/events.jsonl').read_text()
    assert '[REDACTED]' in (tmp_path / 'log/events.jsonl').read_text()


def test_stdio_interactive_request_fails_instead_of_hanging(tmp_path):
    script = tmp_path / 'server.py'
    script.write_text(FAKE_SERVER)
    async def exercise():
        async with AppServer([sys.executable, str(script)], {}, tmp_path, tmp_path / 'log') as server:
            with pytest.raises(RuntimeError, match='Unexpected interactive'):
                await server.request('interactive', {}, timeout=2)
    asyncio.run(exercise())


def test_turn_timeout_interrupts_and_reaps_process(tmp_path):
    script = tmp_path / 'server.py'
    script.write_text('''
import json,sys
for line in sys.stdin:
 r=json.loads(line)
 if r.get('method')=='initialize': print(json.dumps({'id':r['id'],'result':{}}),flush=True)
 if r.get('method')=='turn/start': print(json.dumps({'id':r['id'],'result':{'turn':{'id':'t'}}}),flush=True)
 if r.get('method')=='turn/interrupt': print(json.dumps({'id':r['id'],'result':{}}),flush=True)
''')
    async def exercise():
        async with AppServer([sys.executable, str(script)], {}, tmp_path, tmp_path / 'log') as server:
            with pytest.raises(asyncio.TimeoutError):
                await server.turn('thread', [], 'm', 'high', {}, .05)
        assert server.process.returncode is not None
    asyncio.run(exercise())
    assert 'turn/interrupt' in (tmp_path / 'log/events.jsonl').read_text()


class FakeServer:
    methods = []
    count = 0
    def __init__(self, *args):
        pass
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        pass
    async def request(self, method, params):
        self.methods.append(method)
        return {'thread': {'id': 'persistent-thread'}, 'model': params['model'],
                'modelProvider': params['modelProvider'], 'reasoningEffort': params['config']['model_reasoning_effort']}
    async def turn(self, thread_id, inputs, model, effort, schema, timeout, on_started):
        assert thread_id == 'persistent-thread'
        assert inputs[1]['type'] == 'skill'
        type(self).count += 1
        on_started(str(self.count))
        return {'turn_id': str(self.count), 'answer': json.dumps({
            'status': 'reviewed', 'summary': 'fixture only', 'limitations': [], 'review_evidence': ['review.json']})}


def test_scene_resume_keeps_thread_and_structural_failure_blocks_success(tmp_path, monkeypatch):
    plan, config = fixture_plan(tmp_path)
    output = tmp_path / 'batch'
    runner.save_plan(plan, output)
    scene = plan['scenes'][0]
    destination = tmp_path / scene['output_dir']
    destination.mkdir(parents=True)
    (destination / 'review.json').write_text('{}')
    directory = runner.scene_directory(output, scene)
    passed = False
    async def check(*args):
        return {'structural_pass': passed, 'errors': [] if passed else ['bad geometry']}
    monkeypatch.setattr(runner, 'validate_archive', check)
    FakeServer.methods = []
    first = asyncio.run(runner.build_scene(plan, scene, directory, config, 1, 1, FakeServer))
    assert first['status'] == 'partial'  # Teacher's "reviewed" is insufficient.
    passed = True
    second = asyncio.run(runner.build_scene(plan, scene, directory, config, 1, 1, FakeServer))
    assert second['status'] == 'ready_for_review'
    assert FakeServer.methods == ['thread/start', 'thread/resume']
    assert len(second['turns']) == 2
    third = asyncio.run(runner.build_scene(plan, scene, directory, config, 1, 1, FakeServer))
    assert len(third['turns']) == 2  # Already verified result is not rerun.


def test_batch_stops_before_next_scene_on_partial(tmp_path, monkeypatch):
    plan, config = fixture_plan(tmp_path)
    output = tmp_path / 'batch'
    runner.save_plan(plan, output)
    calls = []
    async def build(plan, scene, *args):
        calls.append(scene['scene_id'])
        return {'status': 'partial'}
    monkeypatch.setattr(runner, 'build_scene', build)
    assert not asyncio.run(runner.run_batch(plan, output, config, 1, 1))
    assert calls == ['a']


def test_validator_exit_code_cannot_be_overridden_by_json(tmp_path):
    plan, _ = fixture_plan(tmp_path)
    script = Path(plan['skill']) / 'scripts/validate_archive.py'
    script.write_text('import sys\nprint(\'{"structural_pass":true}\')\nsys.exit(1)\n')
    report = asyncio.run(runner.validate_archive(plan, plan['scenes'][0]))
    assert report['exit_code'] == 1
    assert not report['structural_pass']


def test_resolved_effort_mismatch_is_not_silent(tmp_path):
    class WrongEffort(FakeServer):
        async def request(self, method, params):
            response = await super().request(method, params)
            return dict(response, reasoningEffort='low')
    with pytest.raises(ValueError, match='effort'):
        asyncio.run(runner.open_thread(WrongEffort(), settings(tmp_path), {}))


def test_parallel_slots_continue_other_scenes_and_keep_assignments(tmp_path, monkeypatch):
    plan, config = fixture_plan(tmp_path)
    plan['scenes'] = [dict(plan['scenes'][0], scene_id=f's{i}', order=i+1,
                          output_dir=f'maps/s{i}') for i in range(6)]
    plan['parallel'] = {'concurrency': 2, 'gpus': [0, 1], 'min_free_gb': 0}
    output = tmp_path / 'batch'
    runner.save_plan(plan, output)
    active = peak = 0
    calls = []
    async def build(plan, scene, directory, settings, *args):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        calls.append((scene['scene_id'], settings.gpu, settings.codex_home.name))
        assert f'GPU {settings.gpu}' in (directory / 'prompt.md').read_text()
        await asyncio.sleep(.01)
        active -= 1
        return {'status': 'error' if scene['scene_id'] == 's0' else 'ready_for_review'}
    monkeypatch.setattr(runner, 'build_scene', build)
    assert not asyncio.run(runner.run_batch(plan, output, config, 1, 1))
    assert peak == 2 and len(calls) == 6
    assert set(calls) == {(f's{i}', i%2, f'worker-{i%2:02d}') for i in range(6)}
    first = calls.copy()
    calls.clear()
    asyncio.run(runner.run_batch(plan, output, config, 1, 1))
    assert set(first) == set(calls)
    plan['parallel']['concurrency'] = 1
    runner.save_plan(plan, output, resume=True)
    assert plan['parallel']['slots'] == 2
    calls.clear()
    peak = 0
    asyncio.run(runner.run_batch(plan, output, config, 1, 1))
    assert peak == 1
    assert set(first) == set(calls)


def test_parallel_low_space_stops_new_dispatch(tmp_path, monkeypatch):
    plan, config = fixture_plan(tmp_path)
    plan['parallel'] = {'concurrency': 2, 'gpus': [0, 1], 'min_free_gb': 20}
    output = tmp_path / 'batch'
    runner.save_plan(plan, output)
    from types import SimpleNamespace
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda _: SimpleNamespace(free=0))
    async def unexpected(*args):
        pytest.fail('No model should run when space is insufficient')
    monkeypatch.setattr(runner, 'build_scene', unexpected)
    assert not asyncio.run(runner.run_batch(plan, output, config, 1, 1))
    assert json.loads((output / 'dispatch_stop.json').read_text())['reason'] == 'low_disk_space'


def test_scene_workspace_and_native_thread_cwd(tmp_path):
    plan, config = fixture_plan(tmp_path)
    scene = plan['scenes'][0]
    scene['workspace_dir'] = str(tmp_path / scene['output_dir'] / 'raw')
    output = tmp_path / 'batch'
    runner.save_plan(plan, output)
    directory = runner.scene_directory(output, scene)
    assert directory == tmp_path / 'maps/hm3d/a/raw'
    assert '不另建 build/' in (directory / 'prompt.md').read_text()
    class CheckCwd(FakeServer):
        async def request(self, method, params):
            assert params['cwd'] == str(directory)
            return await super().request(method, params)
    asyncio.run(runner.open_thread(CheckCwd(), config, {}, directory))
    summary = runner.summarize(plan, output)
    assert summary['scenes'][0]['result'] == str(directory / 'result.json')


def test_manifest_excludes_live_workspace_logs(tmp_path):
    import importlib.util
    script = Path(runner.__file__).parent / 'build_scenes_map_skill_v2/scripts/refresh_manifest.py'
    spec = importlib.util.spec_from_file_location('refresh', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = tmp_path / 'raw'
    raw.mkdir()
    (raw / 'manifest.json').write_text('{"checksum_path_base":".."}')
    (raw / 'walls.json').write_text('{}')
    for name in ['events.jsonl','result.json','stderr.log','prompt.md']:
        (raw / name).write_text('live')
    module.refresh(tmp_path, write=True)
    checks = json.loads((raw / 'manifest.json').read_text())['checksums']
    assert set(checks) == {'raw/walls.json'}


@pytest.mark.parametrize('message,retry', [
    ('unexpected status 503 Service Unavailable', True),
    ('unexpected status 429 Too Many Requests', True),
    ('stream disconnected before completion', True),
    ('responseStreamDisconnected: upstream failed', True),
    ('unexpected status 401 Unauthorized', False),
    ('unexpected status 400 invalid parameter', False),
    ('missing scene asset', False),
])
def test_service_retry_classification(message, retry):
    assert runner.retryable_service_error(RuntimeError(message)) is retry


def test_outer_retry_waits_and_resumes_same_thread(tmp_path, monkeypatch):
    plan, config = fixture_plan(tmp_path)
    output = tmp_path / 'batch'
    runner.save_plan(plan, output)
    scene = plan['scenes'][0]
    directory = runner.scene_directory(output, scene)
    destination = tmp_path / scene['output_dir']
    destination.mkdir(parents=True)
    (destination / 'review.json').write_text('{}')
    class Flaky(FakeServer):
        methods = []
        attempts = 0
        closed = 0
        async def __aexit__(self, *args):
            type(self).closed += 1
        async def turn(self, *args):
            type(self).attempts += 1
            if self.attempts <= 2:
                args[-1](f'failed-{self.attempts}')
                raise RuntimeError('unexpected status 503 Service Unavailable')
            return await super().turn(*args)
    build = runner.build_scene
    async def run(*args):
        return await build(*args, server_factory=Flaky)
    async def check(*args):
        return {'structural_pass': True}
    waits = []
    async def sleep(delay):
        assert Flaky.closed == len(waits) + 1
        state = json.loads((directory / 'result.json').read_text())
        assert state['status'] == 'retry_wait'
        assert state['thread_id'] == 'persistent-thread'
        assert 299 <= delay <= 300
        waits.append(delay)
    monkeypatch.setattr(runner, 'build_scene', run)
    monkeypatch.setattr(runner, 'validate_archive', check)
    monkeypatch.setattr(runner.asyncio, 'sleep', sleep)
    result = asyncio.run(runner.build_with_retries(plan, scene, directory, config, 1, 1, output))
    assert result['status'] == 'ready_for_review'
    assert len(waits) == 2 and result['retry_count'] == 2
    assert Flaky.methods == ['thread/start', 'thread/resume', 'thread/resume']
    assert len(result['turns']) == 1  # Service failures do not consume repair turns.
    assert 'next_retry_at' not in result


def test_cancel_retry_wait_does_not_start_another_attempt(tmp_path, monkeypatch):
    plan, config = fixture_plan(tmp_path)
    output = tmp_path / 'batch'
    runner.save_plan(plan, output)
    scene = plan['scenes'][0]
    directory = runner.scene_directory(output, scene)
    async def build(*args):
        return {'status': 'error', 'retryable': True, 'error': '503', 'thread_id': 'saved'}
    async def cancel(delay):
        raise asyncio.CancelledError()
    monkeypatch.setattr(runner, 'build_scene', build)
    monkeypatch.setattr(runner.asyncio, 'sleep', cancel)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runner.build_with_retries(plan, scene, directory, config, 1, 1, output))
    state = json.loads((directory / 'result.json').read_text())
    assert state['status'] == 'interrupted' and state['thread_id'] == 'saved'
