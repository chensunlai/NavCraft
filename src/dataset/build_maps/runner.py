"""Serial scene construction with durable Codex threads and independent archive checks."""

import asyncio
from contextlib import contextmanager
from dataclasses import replace
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

from jsonschema import ValidationError

from src.common.io import redact, write_json
from .appserver import AppServer

REPORT_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'status': {'type': 'string', 'enum': ['reviewed', 'partial', 'blocked']},
        'summary': {'type': 'string'},
        'limitations': {'type': 'array', 'items': {'type': 'string'}},
        'review_evidence': {'type': 'array', 'items': {'type': 'string'}},
    },
    'required': ['status', 'summary', 'limitations', 'review_evidence'],
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def skill_files(skill):
    return {str(p.relative_to(skill)): digest(p) for p in sorted(skill.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}


def make_plan(queue_file, repo, skill, settings, stage, python, scene_ids=(), start=1, limit=None):
    queue = json.loads(queue_file.read_text())
    scenes = queue['scenes']
    if scene_ids:
        missing = set(scene_ids) - {r['scene_id'] for r in scenes}
        if missing:
            raise ValueError(f'Unknown scene IDs: {sorted(missing)}')
        scenes = [r for r in scenes if r['scene_id'] in scene_ids]
    scenes = scenes[start - 1:]
    if limit is not None:
        scenes = scenes[:limit]
    if not scenes:
        raise ValueError('No scenes selected')
    outputs = set()
    output_root = (repo / queue['output_root']).resolve()
    for scene in scenes:
        destination = (repo / scene['output_dir']).resolve()
        if destination == output_root or not destination.is_relative_to(output_root):
            raise ValueError(f'Invalid scene output directory: {destination}')
        if destination in outputs:
            raise ValueError(f'Duplicate scene output: {destination}')
        outputs.add(destination)
        if not (repo / scene['scene_path']).is_file():
            raise FileNotFoundError(scene['scene_path'])
        if scene.get('scene_dataset_config') and not (repo / scene['scene_dataset_config']).is_file():
            raise FileNotFoundError(scene['scene_dataset_config'])
    for required in ('SKILL.md', f'scripts/{validator_name(stage)}'):
        if not (skill / required).is_file():
            raise FileNotFoundError(skill / required)
    return {'queue': str(queue_file), 'queue_sha256': digest(queue_file),
            'settings': settings.public(), 'stage': stage, 'python': str(python),
            'skill': str(skill), 'skill_files': skill_files(skill),
            'scene_count': len(scenes), 'scenes': scenes}


def validator_name(stage):
    return 'validate_archive.py' if stage == 'full' else 'validate_wall_stage.py'


def scene_directory(output, scene):
    if scene.get('workspace_dir'):
        return Path(scene['workspace_dir'])
    # Queue order stays stable when a subset is selected or a batch resumes.
    name = re.sub(r'[^a-zA-Z0-9_.-]', '_', scene['scene_id'])
    return output / f"{scene['order']:04d}-{scene['dataset']}-{name}"


def prompt_for(plan, scene, workdir):
    repo = Path(plan['settings']['repo'])
    return Path(__file__).with_name('prompt.md').read_text().format(
        stage=('完整场景地图' if plan['stage'] == 'full' else
               '仅墙体与楼梯（wall / wall_real）。按 skill 的正常流程从场景检查、楼层识别、真实渲染取证开始，'
               '完成 wall、wall_real 与楼梯的构建、纠错、最终复查和墙体阶段归档后立即停止。'
               '不进入房间/功能区分区、物体 box、逐物体描述或证据构建；不运行完整地图阶段校验'),
        repo=repo, skill=plan['skill'], scene=json.dumps(scene, ensure_ascii=False, indent=2), workdir=workdir,
        destination=(repo / scene['output_dir']).resolve(), python=plan['python'],
        gpu=scene_gpu(plan, plan['scenes'].index(scene)),
        workspace_note=('本次直接在归档 raw/ 中工作，按最终位置原地保存文件，不另建 build/ 或嵌套 archive/，不要复制目录到自身。'
                        'raw/result.json、raw/events.jsonl、raw/stderr.log、raw/prompt.md 是调度器文件，不要改动或计入产物校验和。'
                        '需要备份只复制将修改的静态文件到 raw/backups/，不要递归备份整个场景。'
                        if scene.get('workspace_dir') else ''))


def scene_gpu(plan, index):
    parallel = plan.get('parallel')
    return parallel['gpus'][index % parallel.get('slots', parallel['concurrency']) % len(parallel['gpus'])] if parallel else plan['settings']['gpu']


def save_plan(plan, output, resume=False):
    path = output / 'plan.json'
    if path.exists():
        if not resume:
            raise ValueError('Output already has a plan; use --resume or choose a new directory')
        previous = json.loads(path.read_text())
        if resume and previous.get('parallel') and plan.get('parallel'):
            # Preserve Home/thread ownership when changing the number of active workers.
            old_parallel, new_parallel = previous['parallel'], plan['parallel']
            slots = old_parallel.get('slots', old_parallel['concurrency'])
            new_parallel['slots'] = slots
            previous['parallel'] = dict(old_parallel, slots=slots,
                                        concurrency=new_parallel['concurrency'])
        if previous != plan:
            raise ValueError('Resume configuration, skill or scene queue changed; use the original parameters')
        write_json(path, plan)
    else:
        if resume:
            raise ValueError('Cannot resume without plan.json')
        if output.exists() and any(output.iterdir()):
            raise ValueError('Output directory must be empty for a new batch')
        write_json(path, plan)
    for scene in plan['scenes']:
        directory = scene_directory(output, scene)
        directory.mkdir(parents=True, exist_ok=True)
        prompt_file = directory / 'prompt.md'
        if not prompt_file.exists():
            prompt_file.write_text(prompt_for(plan, scene, directory))


@contextmanager
def home_lock(home):
    home.mkdir(parents=True, exist_ok=True)
    with (home / 'scene-build.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another map builder is using this Codex Home') from None
        yield


async def validate_archive(plan, scene):
    destination = Path(plan['settings']['repo']) / scene['output_dir']
    command = [plan['python'], str(Path(plan['skill']) / 'scripts' / validator_name(plan['stage'])),
               str(destination)]
    proc = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), 120)
    except BaseException:
        if proc.returncode is None:
            proc.kill()
        await proc.wait()
        raise
    try:
        report = json.loads(stdout)
    except (ValueError, UnicodeError):
        report = {'structural_pass': False, 'errors': [{'kind': 'validator_output',
                  'detail': stdout.decode(errors='replace')[-4000:]}]}
    report['exit_code'] = proc.returncode
    report['structural_pass'] = proc.returncode == 0 and report.get('structural_pass') is True
    if stderr:
        report['stderr'] = stderr.decode(errors='replace')[-4000:]
    return report


def parse_report(answer, destination):
    from jsonschema import validate
    report = json.loads(answer)
    validate(report, REPORT_SCHEMA)
    if report['status'] == 'reviewed':
        if not report['review_evidence']:
            raise ValueError('Reviewed report must name actual archive review evidence')
        for ref in report['review_evidence']:
            path = (destination / ref).resolve()
            if not path.is_relative_to(destination.resolve()) or not path.is_file():
                raise ValueError(f'Missing or external review evidence: {ref}')
    return report


async def open_thread(server, settings, state, cwd=None):
    params = {'model': settings.model, 'modelProvider': 'scene_maps', 'cwd': str(cwd or settings.repo),
              'approvalPolicy': 'never', 'sandbox': 'danger-full-access',
              'config': {'model_reasoning_effort': settings.effort}}
    if state.get('thread_id'):
        method = 'thread/resume'
        params.update(threadId=state['thread_id'], excludeTurns=True)
    else:
        method = 'thread/start'
    response = await server.request(method, params)
    state['thread_id'] = response['thread']['id']
    state['resolved_model'] = response['model']
    state['resolved_effort'] = response.get('reasoningEffort')
    if response['model'] != settings.model or response['modelProvider'] != 'scene_maps':
        raise ValueError('Codex resolved an unexpected model/provider')
    if response.get('reasoningEffort') != settings.effort:
        raise ValueError('Codex did not confirm the requested reasoning effort')
    return response


async def build_scene(plan, scene, directory, settings, max_turns, timeout, server_factory=AppServer):
    state_file = directory / 'result.json'
    state = json.loads(state_file.read_text()) if state_file.exists() else {
        'scene_id': scene['scene_id'], 'output_dir': scene['output_dir'], 'turns': []}
    if state.get('status') == 'ready_for_review':
        check = await validate_archive(plan, scene)
        state['validation'] = check
        state['updated_at'] = time.time()
        write_json(state_file, state)
        if check['structural_pass']:
            return state
    state.update(status='running', updated_at=time.time(), retryable=False)
    state.pop('next_retry_at', None)
    write_json(state_file, state)
    try:
        async with server_factory(settings.command(), settings.environment(), directory,
                                  directory, (settings.api_key,)) as server:
            await open_thread(server, settings, state, directory)
            write_json(state_file, state)
            skill = Path(plan['skill']) / 'SKILL.md'
            match = re.search(r'^name:\s*(.+)$', skill.read_text(), flags=re.MULTILINE)
            if not match:
                raise ValueError('Skill frontmatter has no name')
            for _ in range(max_turns):
                prompt = (directory / 'prompt.md').read_text()
                if state['turns'] or state.get('active_turn_id'):
                    prompt += '\n继续当前场景，先检查已保存文件和历史，不从头盲目重建。上一轮主程序检查：\n'
                    prompt += json.dumps({key: state.get(key) for key in ('validation', 'agent_report', 'error')},
                                         ensure_ascii=False)[:8000]
                def started(turn_id):
                    state['active_turn_id'] = turn_id
                    write_json(state_file, state)
                reply = await server.turn(state['thread_id'], [
                    {'type': 'text', 'text': prompt},
                    {'type': 'skill', 'name': match.group(1).strip(), 'path': str(skill)}],
                    settings.model, settings.effort, REPORT_SCHEMA, timeout, started)
                state.pop('active_turn_id', None)
                state['turns'].append(reply)
                try:
                    report = parse_report(reply['answer'], settings.repo / scene['output_dir'])
                except (ValueError, ValidationError) as error:
                    # Invalid final formatting is repairable in the same thread.
                    report = {'status': 'partial', 'summary': f'Invalid final report: {error}'}
                state['agent_report'] = report
                state['validation'] = await validate_archive(plan, scene)
                state['updated_at'] = time.time()
                if report['status'] == 'reviewed' and state['validation']['structural_pass']:
                    state['status'] = 'ready_for_review'
                    state.pop('error', None)
                    write_json(state_file, state)
                    return state
                state['status'] = 'blocked' if report['status'] == 'blocked' else 'partial'
                write_json(state_file, state)
                if state['status'] == 'blocked':
                    break
    except (Exception, asyncio.CancelledError) as error:
        state.update(status='interrupted' if isinstance(error, asyncio.CancelledError) else 'error',
                     error=redact(f'{type(error).__name__}: {error}', (settings.api_key,)),
                     retryable=retryable_service_error(error), updated_at=time.time())
        write_json(state_file, state)
        if isinstance(error, asyncio.CancelledError):
            raise
    return state


def retryable_service_error(error):
    message = str(error).lower()
    # Configuration/authentication errors cannot recover by repeating a request.
    if re.search(r'\b(?:400|401|403|404|422)\b', message):
        return False
    if isinstance(error, (ValueError, asyncio.CancelledError)):
        return False
    return bool(re.search(r'\b(?:408|429|500|502|503|504|520|521|522|524)\b', message)
                or any(text in message for text in (
                    'stream disconnected', 'responsestreamdisconnected',
                    'service temporarily unavailable', 'connection reset',
                    'connection refused', 'error sending request',
                    'connection timed out', 'rate limit exceeded')))


async def build_with_retries(plan, scene, directory, settings, max_turns, timeout, output):
    """Retry transient service failures in the saved thread, after releasing its process."""
    path = directory / 'result.json'
    state = json.loads(path.read_text()) if path.exists() else {}
    initial_turns = len(state.get('turns', []))
    while True:
        if state.get('status') == 'retry_wait':
            try:
                await asyncio.sleep(max(0, state['next_retry_at'] - time.time()))
            except asyncio.CancelledError:
                state.update(status='interrupted', updated_at=time.time())
                write_json(path, state)
                raise
        remaining = max_turns - (len(state.get('turns', [])) - initial_turns)
        if remaining <= 0:
            return state
        state = await build_scene(plan, scene, directory, settings, remaining, timeout)
        if state.get('status') != 'error' or not state.get('retryable'):
            return state
        now = time.time()
        state.update(status='retry_wait', retry_count=state.get('retry_count', 0) + 1,
                     next_retry_at=now + 300, updated_at=now)
        write_json(path, state)
        event = {'scene': scene['scene_id'], 'event': 'retry_wait',
                 'retry_count': state['retry_count'], 'delay_seconds': 300,
                 'next_retry_at': state['next_retry_at'], 'error': state['error']}
        with (directory / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps({'time': now, 'direction': 'host', 'message': event}, ensure_ascii=False) + '\n')
        summarize(plan, output)
        print(json.dumps(event, ensure_ascii=False), flush=True)


def summarize(plan, output):
    rows = []
    for scene in plan['scenes']:
        result = scene_directory(output, scene) / 'result.json'
        state = json.loads(result.read_text()) if result.exists() else {'status': 'pending'}
        rows.append({'scene_id': scene['scene_id'], 'status': state['status'], 'result': str(result)})
    summary = {'selected': len(rows), 'ready_for_review': sum(r['status'] == 'ready_for_review' for r in rows),
               'scenes': rows, 'note': 'Agent-reported visual review plus host structural checks; not independent visual acceptance.'}
    write_json(output / 'summary.json', summary)
    return summary


async def run_batch(plan, output, settings, max_turns, timeout):
    if plan.get('parallel'):
        return await run_parallel(plan, output, settings, max_turns, timeout)
    with home_lock(settings.codex_home):
        try:
            for scene in plan['scenes']:
                print(json.dumps({'scene': scene['scene_id'], 'event': 'starting'}, ensure_ascii=False), flush=True)
                result = await build_with_retries(plan, scene, scene_directory(output, scene), settings, max_turns, timeout, output)
                summarize(plan, output)
                print(json.dumps({'scene': scene['scene_id'], 'status': result['status']}, ensure_ascii=False), flush=True)
                if result['status'] != 'ready_for_review':
                    return False
            return True
        finally:
            summarize(plan, output)


async def run_parallel(plan, output, settings, max_turns, timeout):
    """Fixed slots preserve GPU and Home assignments across resume; scenes keep separate threads."""
    count = plan['parallel'].get('slots', plan['parallel']['concurrency'])
    active = asyncio.Semaphore(plan['parallel']['concurrency'])
    reserve = plan['parallel']['min_free_gb'] * 1024 ** 3
    settings.codex_home.mkdir(parents=True, exist_ok=True)
    checks = [output, settings.codex_home]
    for scene in plan['scenes']:
        path = settings.repo / scene['output_dir']
        while not path.exists():
            path = path.parent
        checks.append(path)
    disks = {path.stat().st_dev: path for path in checks}
    stop_dispatch = asyncio.Event()

    async def worker(slot):
        config = replace(settings, codex_home=settings.codex_home / f'worker-{slot:02d}',
                         gpu=scene_gpu(plan, slot))
        passed = True
        with home_lock(config.codex_home):
            for scene in plan['scenes'][slot::count]:
                if stop_dispatch.is_set():
                    return False
                low = [str(path) for path in disks.values() if shutil.disk_usage(path).free < reserve]
                if low:
                    stop_dispatch.set()
                    write_json(output / 'dispatch_stop.json', {'reason': 'low_disk_space',
                               'paths': low, 'min_free_gb': plan['parallel']['min_free_gb']})
                    print(json.dumps({'event': 'dispatch_stopped', 'reason': 'low_disk_space', 'paths': low}), flush=True)
                    return False
                async with active:
                    if stop_dispatch.is_set():
                        return False
                    print(json.dumps({'scene': scene['scene_id'], 'event': 'starting',
                                      'worker': slot, 'gpu': config.gpu}), flush=True)
                    result = await build_with_retries(plan, scene, scene_directory(output, scene),
                                                      config, max_turns, timeout, output)
                passed = passed and result['status'] == 'ready_for_review'
                summarize(plan, output)
                print(json.dumps({'scene': scene['scene_id'], 'status': result['status'],
                                  'worker': slot}), flush=True)
        return passed

    with home_lock(settings.codex_home):
        try:
            summarize(plan, output)
            results = await asyncio.gather(*(worker(i) for i in range(min(count, len(plan['scenes'])))))
            return all(results)
        finally:
            summarize(plan, output)
