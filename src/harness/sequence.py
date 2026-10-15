"""Run private multi-round task manifests in one environment and one Agent session."""

from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time

from src.bench.sequence import inside, score_route
from src.dataset.episodes.base import EpisodeSpec
from src.environment.habitat import HabitatEnvironment
from src.environment.tasks.base import ObjectNavTask
from .config import load_config, load_navigation_prompt
from .runtime import PromptLimits, open_session
from .session import NavigationSession, _write
from .tools import ToolSet
from .trace import TraceWriter
from .worker import supervise

REPO = Path(__file__).resolve().parents[2]


def validate_scenario(scenario):
    rounds = scenario.get('rounds', [])
    if not rounds or len({r['id'] for r in rounds}) != len(rounds):
        raise ValueError('A sequence requires nonempty rounds with unique IDs')
    for r in rounds:
        if r['kind'] not in {'explore', 'navigation'} or not r['instruction'].strip():
            raise ValueError('Each round requires a supported kind and an instruction')
        if any(type(r[k]) is not int or r[k] <= 0 for k in ('max_actions', 'max_requests', 'seconds')):
            raise ValueError('Round budgets must be positive integers')
        if r['kind'] == 'navigation' and not r.get('positions'):
            raise ValueError('Navigation requires private accepted viewpoints')
    if rounds[0].get('teleport'):
        raise ValueError('Specify initial pose separately from later relocations')


class SequenceSession(NavigationSession):
    def __init__(self, repo, workspace, private, scenario, cancel_event=None):
        super().__init__(cancel_event)
        self.repo, self.workspace, self.private_dir = Path(repo), Path(workspace), Path(private)
        self.scenario = scenario
        self.config = deepcopy(scenario['environment'])
        self.config['depth'] = False
        self.private_dir.mkdir(parents=True, exist_ok=True)
        self.round_index = -1
        self.mode = 'navigation'
        self.boundaries = []
        if self.config.get('egl_vendor_file'):
            os.environ['__EGL_VENDOR_LIBRARY_FILENAMES'] = self.config['egl_vendor_file']

    def begin(self, index):
        spec = self.scenario['rounds'][index]
        self.round_index = index
        positions = spec.get('positions', [self.scenario['start_position']])
        episode = EpisodeSpec('sequence:' + spec['id'], spec['id'], 'campaign', self.scenario['scene_id'],
                              self.scenario['start_position'], self.scenario['start_rotation'],
                              [{'modality': 'instruction', 'instruction': spec['instruction'], 'positions': positions}],
                              'private campaign scenario')
        task = ObjectNavTask({'max_steps': spec['max_actions'], 'success_distance': spec.get('success_distance', 1),
                              'protocol': self.scenario['protocol']})
        episode.provenance = 'custom_continuous_map_free_campaign'
        before = deepcopy(self.env.observe().pose) if self.env else None
        if self.env is None:
            self.env = HabitatEnvironment(self.config, task, self.private_dir / 'environment')
            transition = self.env.reset(episode)
        else:
            if spec.get('teleport'):
                from habitat_sim.utils.common import quat_from_coeffs
                import numpy as np
                state = self.env.sim.agent.get_state()
                state.position = np.asarray(spec['teleport']['position'], dtype=np.float32)
                state.rotation = quat_from_coeffs(spec['teleport']['rotation_xyzw'])
                self.env.sim.agent.set_state(state, reset_sensors=True)
            self.env.task = task
            task.reset(episode, self.env.sim.position(), self.env.sim.distance(positions))
            self.env.observation = self.env.capture_current()
            self.env.public_task = self.env._public_task()
            self.env._save()
            from src.environment.tasks.base import TaskUpdate
            transition = self.env._transition(TaskUpdate(), {'event': 'round_start'})
        self._save_private_frame(transition.observation)
        value = self._transition(transition)
        value.update(actions={'allowed': list(self.env.movement_actions),
                              'forward_metres': self.config['forward_step'],
                              'turn_degrees': self.config['turn_angle'], 'look_degrees': self.config['turn_angle']},
                     round_number=index + 1, relocated=bool(spec.get('teleport')))
        self.boundaries.append({'round': index + 1, 'before': before, 'after': transition.observation.pose,
                                'teleport': spec.get('teleport'), 'frame': transition.observation.frame_id})
        _write(self.private_dir / 'round_boundaries.json', self.boundaries)
        self._save_result()
        return value

    def _save_private_frame(self, observation):
        from .session import _append
        _append(self.private_dir / 'frames.jsonl', {'id': observation.frame_id,
                'round': self.round_index + 1, 'step': self.env.task.steps,
                'pose': observation.pose, 'camera': observation.camera})

    def _task(self, task):
        value = super()._task(task)
        if value is not None:
            value['goal_id'] = self.round_index
        return value

    def finish(self, goal_id):
        if type(goal_id) is not int or goal_id != self.round_index:
            raise ValueError('goal_id does not match this round')
        return super().finish(0)

    def score(self, reason):
        spec, task = self.scenario['rounds'][self.round_index], self.env.task
        if not task.done:
            self.env.finalize(reason)
        route = score_route(spec, task.trajectory, {r['id']: r for r in self.scenario['private_rooms']})
        result = task.private_state()
        if spec['kind'] == 'explore':
            living = next(r for r in self.scenario['private_rooms'] if r['id'] == spec['scope_room'])
            result['exploration'] = {'scope_maintained': all(inside([p[0], -p[2]], living['polygon'])
                                                           for p in task.trajectory),
                                     'navigation_success': None, 'manual_memory_review_required': True}
        else:
            result['independent'] = {'arrival': bool(task.stopped and task.distance < task.threshold),
                                     'route': route,
                                     'passed': bool(task.stopped and task.distance < task.threshold and route['passed'])}
        self._save_result()
        return result


def request_count(private):
    path = private / 'requests.json'
    return len(json.loads(path.read_text())['attempts']) if path.exists() else 0


def memory_versions(workspace):
    return {p.relative_to(workspace).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (workspace / 'memory').rglob('*') if p.is_file() and not p.is_symlink()}


def run_sequence(config, scenario, directory, cancelled):
    directory = Path(directory).resolve()
    workspace, private = directory / 'workspace', directory / 'private'
    for name in ('scene', 'observations', 'memory', 'scratch'):
        (workspace / name).mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    validate_scenario(scenario)
    scenario = deepcopy(scenario)
    _write(private / 'scenario.json', scenario)
    (workspace / 'scene/index.md').write_text('# Public workspace\n\nOnly actual observations and the current instruction are supplied. No prebuilt map, entity cards or evidence bank.\n\nscene/ and observations/ are read-only; memory/ and scratch/ are writable. Start with empty memory; organize your notes yourself. No exact pose or depth is available.\n')
    trace, session, bridge = TraceWriter(private), None, None
    result = {'model': config['model'], 'protocol': scenario['protocol'], 'rounds': [], 'error': None,
              'input_setting': 'RGB + action feedback; no map/prior memory/true pose/depth', 'case': 'sequence'}
    state, started = {}, time.monotonic()
    _write(directory / 'result.json', result)
    try:
        bridge = SequenceSession(REPO, workspace, private / 'environment', scenario, cancelled)
        # Privately validate accepted target viewpoints before starting the model.
        initial = bridge.begin(0)
        for spec in scenario['rounds']:
            if spec.get('snap_viewpoints'):
                resolved = []
                for point in spec['positions']:
                    snapped = bridge.env.sim.sim.pathfinder.snap_point(point)
                    displacement = math.dist(point, snapped)
                    if not all(math.isfinite(float(v)) for v in snapped) or displacement > .2:
                        raise ValueError(f'Unreliable private viewpoint {spec["id"]}: displacement {displacement}')
                    resolved.append([float(v) for v in snapped])
                spec['positions'] = resolved
        _write(private / 'scenario.json', scenario)
        state.update(current=initial, done=False, goal_changed=False)
        tools = ToolSet(REPO, workspace, bridge, state, cancel_event=cancelled)
        tools.tools.pop('render_map')  # No base map or calibrated memory exists in this condition.
        system_prompt = load_navigation_prompt(REPO)
        session = open_session(REPO, config, directory, workspace, system_prompt, tools,
                               max_requests=scenario['rounds'][0]['max_requests'],
                               trace=trace, cancel_event=cancelled)
        result['session_id'] = session.session_id
        for index, spec in enumerate(scenario['rounds']):
            if cancelled.is_set():
                break
            before, t0 = session.requests, time.monotonic()
            entry = {'round': index + 1, 'id': spec['id'], 'instruction': spec['instruction'],
                     'reason': None, 'answer': None, 'error': None, 'last_error': None,
                     'result': None, 'memory_before': memory_versions(workspace)}
            round_started = index == 0
            settled = {}
            try:
                if index:
                    initial = bridge.begin(index)
                    round_started = True
                    state.update(current=initial, done=False, goal_changed=False)
                    state['revision'] += 1
                    session.update_checkpoint()
                trace.emit('sequence_round_start', {'round': index + 1, 'id': spec['id'],
                           'public': initial, 'requests_before': before, 'request_limit': spec['max_requests'],
                           'memory_versions': entry['memory_before'], 'session_id': result['session_id']})
                print(json.dumps({'round': index + 1, 'id': spec['id'], 'phase': 'running'}), flush=True)
                prompt = (('You have been relocated; previous memories remain but your former pose is stale.\n'
                           if spec.get('teleport') else '') + spec['instruction'] + '\nCurrent public state: '
                           + json.dumps(initial, ensure_ascii=False))
                settled = session.prompt(prompt, initial['images'],
                                         limits=PromptLimits(max_requests=spec['max_requests'], seconds=spec['seconds'])).to_dict()
                reason = state.get('current', {}).get('reason') if state.get('done') else settled.get('reason')
                entry.update(reason=reason, answer=settled.get('answer'), error=settled.get('error'),
                             last_error=(settled.get('last') or {}).get('errorMessage'),
                             prompt_id=settled.get('prompt_id'))
                entry['result'] = bridge.score(reason)
            except Exception as error:
                entry.update(reason='runtime_error', error=f'{type(error).__name__}: {error}')
                if round_started and entry['result'] is None:
                    try:
                        entry['result'] = bridge.score('runtime_error')
                    except Exception as score_error:
                        entry['score_error'] = f'{type(score_error).__name__}: {score_error}'
                raise
            finally:
                entry.update(requests=session.requests - before, seconds=time.monotonic() - t0,
                             memory_after=memory_versions(workspace))
                _write(private / f'round_{index + 1:02d}.json', entry)
                result['rounds'].append(entry)
                trace.emit('sequence_round_end', entry)
                _write(directory / 'result.json', result)
                print(json.dumps({k:entry[k] for k in ('round', 'id', 'reason', 'requests', 'seconds')},ensure_ascii=False),flush=True)
            if cancelled.is_set() or settled.get('reason') in {'model_error', 'runtime_error', 'tool_timeout', 'tool_result_invalid', 'output_limit'}:
                result['error'] = 'Sequence interrupted by ' + str(settled.get('reason'))
                break
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    finally:
        if session:
            try:
                session.close()
            except Exception as error:
                result['cleanup_error'] = str(error)
            if session.cleanup_error:
                result['cleanup_error'] = session.cleanup_error
            result['core_close'] = session.close_status
        elif bridge:
            try:
                bridge.close()
            except Exception as error:
                result['cleanup_error'] = str(error)
        result.update(requests=session.requests if session else request_count(private), elapsed_seconds=time.monotonic() - started,
                      completed_rounds=len(result['rounds']), compactions=len(trace.compactions),
                      finish_reason='sequence_complete' if len(result['rounds']) == len(scenario['rounds']) and not result['error'] else 'sequence_incomplete')
        trace.emit('sequence_finished', result)
        trace.close()
        _write(directory / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/harness-pi/default.yaml')
    parser.add_argument('--scenario', required=True, help='Private multi-round task manifest')
    parser.add_argument('--directory')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    scenario = json.loads(Path(args.scenario).read_text())
    validate_scenario(scenario)
    directory = Path(args.directory) if args.directory else REPO / config['output_dir'] / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-sequence')
    if args.worker:
        cancelled = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: cancelled.set())
        run_sequence(config, scenario, directory, cancelled)
    else:
        command = [sys.executable, '-B', '-u', '-m', 'src.harness.sequence', '--config', args.config,
                   '--scenario', args.scenario, '--directory', str(directory), '--worker']
        print(json.dumps({'directory':str(directory),'phase':'supervised'}),flush=True)
        result = supervise(command, directory, scenario['seconds'], REPO)
        print(json.dumps({'directory':str(directory),'completed_rounds':result.get('completed_rounds'),
                          'requests':result.get('requests'),'finish_reason':result.get('finish_reason'),
                          'error':result.get('error')},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
