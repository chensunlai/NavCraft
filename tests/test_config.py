"""Composition and script wiring checks without model requests or simulation."""

import argparse
import json
import os
from pathlib import Path
import subprocess

import pytest

from src.common.config import merge_config, parse_config_args, read_config

REPO = Path(__file__).resolve().parents[1]


def test_composition_relative_extends_and_list_replacement(tmp_path):
    (tmp_path / 'base.yaml').write_text('environment: {gpus: [0, 1], seed: 0}\nepisodes: [a, b]\n')
    child = tmp_path / 'child.yaml'
    child.write_text('extends: base.yaml\nenvironment: {seed: 7}\n')
    override = tmp_path / 'override.yaml'
    override.write_text('episodes: [c]\n')
    assert read_config([child, override]) == {'environment': {'gpus': [0, 1], 'seed': 7}, 'episodes': ['c']}
    original = {'nested': {'list': [1]}}
    merged = merge_config(original, {})
    merged['nested']['list'].append(2)
    assert original['nested']['list'] == [1]
    child.write_text('extends: child.yaml\n')
    with pytest.raises(ValueError, match='Circular'):
        read_config(child)


def test_config_required_arguments_aliases_types_and_cli_priority(tmp_path):
    config = tmp_path / 'args.yaml'
    config.write_text('credentials: .env.example\nreasoning_effort: high\nconcurrency: 16\ngpus: [0, 1]\n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=Path, required=True)
    parser.add_argument('--effort', required=True)
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--gpus', type=int, choices=[0, 1], nargs='+')
    args = parse_config_args(parser, ['--config', str(config), '--concurrency', '8'],
                             {'credentials': 'env', 'reasoning_effort': 'effort'})
    assert args.env == Path('.env.example')
    assert args.effort == 'high' and args.concurrency == 8 and args.gpus == [0, 1]
    bad = tmp_path / 'bad.yaml'
    bad.write_text('env: example\neffort: high\ngpus: [2]\n')
    fresh = argparse.ArgumentParser()
    fresh.add_argument('--gpus', choices=[0, 1], type=int, nargs='+')
    with pytest.raises(SystemExit):
        parse_config_args(fresh, ['--config', str(bad)])


def test_default_pilot_matches_explicit_composition():
    explicit = read_config([REPO / 'config/models/pilot.yaml', REPO / 'config/environments/habitat.yaml',
                            REPO / 'config/budgets/navigation.yaml', REPO / 'config/tasks/pilot.yaml', REPO / 'config/maps/scene_dataset.yaml'])
    assert read_config(REPO / 'config/pilot.yaml') == explicit
    for name in ('deepseek', 'qwen27', 'sol', 'astra'):
        config = read_config(REPO / f'config/models/{name}.yaml')
        assert 'python' not in config
        assert config['credentials'] == '.env.harness-test'


@pytest.mark.parametrize('script', sorted((REPO / 'scripts').glob('*.sh')), ids=lambda p: p.name)
def test_scripts_module_wiring(script, tmp_path):
    # Replace python on PATH: inspect actual shell expansion without any execution.
    executable = tmp_path / 'python'
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$PWD" "$@" > "$CAPTURE.$$.txt"\n')
    executable.chmod(0o755)
    env = {**os.environ, 'PATH': f'{tmp_path}:{os.environ["PATH"]}', 'CAPTURE': str(tmp_path / 'capture')}
    subprocess.run(['bash', '-n', str(script)], check=True)
    subprocess.run(['bash', str(script)], cwd=tmp_path, env=env, capture_output=True, check=True)
    calls = list(tmp_path.glob('capture.*.txt'))
    assert calls
    for call in calls:
        cwd, *args = call.read_text().splitlines()
        assert Path(cwd).resolve() == REPO.resolve()
        assert args[0] == '-m' and args[1].startswith('src.')
        for i, arg in enumerate(args):
            if arg in {'--config', '--model-config'}:
                assert (REPO / args[i + 1]).is_file()


def test_build_map_plan_with_composed_configs_and_isolated_archive(tmp_path):
    from src.dataset.build_maps.__main__ import main

    (tmp_path / 'asset.glb').touch()
    skill = tmp_path / 'skill'
    (skill / 'scripts').mkdir(parents=True)
    (skill / 'SKILL.md').write_text('---\nname: build-scene-map\n---\nBuild walls.')
    (skill / 'scripts/validate_archive.py').write_text('print("{}")')
    (skill / 'scripts/validate_wall_stage.py').write_text('print("{}")')
    (tmp_path / 'queue.json').write_text(json.dumps({'output_root': 'maps', 'scenes': [
        {'scene_id': 'a', 'dataset': 'hm3d', 'scene_path': 'asset.glb', 'output_dir': 'maps/hm3d/a', 'order': 1}]}))
    (tmp_path / '.env').write_text('NAVCRAFT_TEST_BASE_URL=http://example.test\nNAVCRAFT_TEST_API_KEY=test-key\n')
    config = tmp_path / 'config.yaml'
    config.write_text('credentials: .env\nmodel: test-model\nreasoning_effort: high\nqueue: queue.json\nskill: skill\nstage: walls\nscene_workspaces: true\n')
    args = ['plan', '--repo', str(tmp_path), '--config', str(config), '--codex-home', 'codex',
            '--output', 'batch', '--archive-root', 'comparison/sol']
    assert main(args) == 0
    plan = json.loads((tmp_path / 'batch/plan.json').read_text())
    scene = plan['scenes'][0]
    assert Path(scene['output_dir']) == tmp_path / 'comparison/sol/hm3d/a'
    assert Path(scene['workspace_dir']) == tmp_path / 'comparison/sol/hm3d/a/raw'
    assert not (tmp_path / 'maps').exists()  # The source destination is untouched.
    assert (Path(scene['workspace_dir']) / 'prompt.md').is_file()
    assert not (Path(scene['output_dir']) / 'final').exists()
    assert main([*args, '--resume']) == 0
