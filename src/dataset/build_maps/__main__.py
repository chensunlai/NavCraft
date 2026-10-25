"""Parameter-driven batch scene mapping through Codex app-server."""

import argparse
import asyncio
import json
from pathlib import Path
import sys

from src.common.io import redact, write_json
from .appserver import AppServer
from .runner import home_lock, make_plan, open_thread, run_batch, save_plan
from .settings import Settings, read_env

REPO = Path(__file__).resolve().parents[3]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['plan', 'probe', 'run'])
    p.add_argument('--env', type=Path, required=True, help='dotenv file; never executed as shell code')
    p.add_argument('--base-url-env', default='NAVCRAFT_TEST_BASE_URL')
    p.add_argument('--api-key-env', default='NAVCRAFT_TEST_API_KEY')
    p.add_argument('--model-env', default='NAVCRAFT_TEST_MODEL')
    p.add_argument('--base-url', help='Exact Responses API base override; no automatic /v1 suffix')
    p.add_argument('--model', help='Override model from the env file')
    p.add_argument('--effort', required=True, help='Exact provider-supported effort; never substituted')
    p.add_argument('--codex-home', type=Path, required=True)
    p.add_argument('--codex', default='codex', help='Codex executable')
    p.add_argument('--repo', type=Path, default=REPO)
    p.add_argument('--queue', type=Path, default=Path('data/samples/scene_build_queue.json'))
    p.add_argument('--skill', type=Path, default=Path(__file__).parent / 'build_scenes_map_skill_v2')
    p.add_argument('--stage', choices=['full', 'walls'], default='full')
    p.add_argument('--scene', action='append', default=[], help='Select scene ID; repeatable, queue order preserved')
    p.add_argument('--start', type=int, default=1, help='1-based position after scene filtering')
    p.add_argument('--limit', type=int)
    p.add_argument('--gpu', type=int, choices=[0, 1], default=0)
    p.add_argument('--concurrency', type=int, default=1, help='Independent scene workers')
    p.add_argument('--gpus', type=int, choices=[0, 1], nargs='+', default=[0, 1], help='GPU assignment for parallel workers')
    p.add_argument('--min-free-gb', type=float, default=20, help='Parallel dispatch pauses below this free disk space')
    p.add_argument('--output', type=Path, required=True, help='Batch logs/work directory, separate from final scene archives')
    p.add_argument('--scene-workspaces', action='store_true', help='Use each scene archive raw/ as its workspace')
    p.add_argument('--resume', action='store_true', help='Continue saved threads with the same plan and settings')
    p.add_argument('--max-turns', type=int, default=3, help='At most this many additional turns per scene per invocation')
    p.add_argument('--turn-timeout', type=float, default=3600)
    return p


async def probe(settings, output, timeout):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Probe output directory must be empty')
    output.mkdir(parents=True, exist_ok=True)
    state = {'settings': settings.public(), 'status': 'running'}
    write_json(output / 'result.json', state)
    try:
        with home_lock(settings.codex_home):
            async with AppServer(settings.command(), settings.environment(), settings.repo, output,
                                 (settings.api_key,)) as server:
                await open_thread(server, settings, state)
                write_json(output / 'result.json', state)
                result = await server.turn(state['thread_id'], [{'type': 'text', 'text':
                    'Protocol probe only. Do not call tools or modify files. Reply with JSON {"ready":true}.'}],
                    settings.model, settings.effort, {'type': 'object', 'additionalProperties': False,
                        'properties': {'ready': {'type': 'boolean'}}, 'required': ['ready']}, timeout)
                state.update(result)
                state['status'] = 'completed' if json.loads(result['answer']) == {'ready': True} else 'invalid_output'
    except (Exception, asyncio.CancelledError) as error:
        state.update(status='error', error=redact(f'{type(error).__name__}: {error}', (settings.api_key,)))
        if isinstance(error, asyncio.CancelledError):
            raise
    finally:
        write_json(output / 'result.json', state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return state['status'] == 'completed'


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.start < 1 or (args.limit is not None and args.limit < 1) or args.max_turns < 1 or args.turn_timeout <= 0 or args.concurrency < 1 or args.min_free_gb < 0:
        p.error('start, limit, max-turns and turn-timeout must be positive')
    repo = args.repo.resolve()
    def path(value):
        # Keep logical data/... paths, including symlink roots, in the frozen plan.
        return Path(value) if Path(value).is_absolute() else repo / value
    values = read_env(path(args.env))
    settings = Settings(repo, path(args.env), path(args.codex_home),
                        args.model or values.get(args.model_env, ''), args.effort,
                        (args.base_url or values.get(args.base_url_env, '')).rstrip('/'),
                        values.get(args.api_key_env, ''), args.codex, args.gpu)
    output = path(args.output)
    try:
        if args.command == 'probe':
            return 0 if asyncio.run(probe(settings, output, args.turn_timeout)) else 1
        plan = make_plan(path(args.queue), repo, path(args.skill), settings, args.stage,
                         Path(sys.executable), args.scene, args.start, args.limit)
        if args.concurrency > 1:
            plan['parallel'] = {'concurrency': args.concurrency, 'gpus': args.gpus,
                                'min_free_gb': args.min_free_gb}
        if args.scene_workspaces:
            for scene in plan['scenes']:
                scene['workspace_dir'] = str((repo / scene['output_dir'] / 'raw').absolute())
        if any(output.resolve().is_relative_to((repo / s['output_dir']).resolve()) for s in plan['scenes']):
            raise ValueError('Batch output must not be inside a final scene archive')
        save_plan(plan, output, args.resume)
        if args.command == 'plan':
            print(json.dumps({'plan': str(output / 'plan.json'), 'scenes': plan['scene_count'],
                              'settings': settings.public()}, ensure_ascii=False, indent=2))
            return 0
        return 0 if asyncio.run(run_batch(plan, output, settings, args.max_turns, args.turn_timeout)) else 1
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(redact(f'{type(error).__name__}: {error}', (settings.api_key,)), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
