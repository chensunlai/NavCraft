"""Published-map loading, public boundaries, and Bench wiring (no simulator/API)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.dataset.maps import load_wall_map, map_directory, prepare_map
from src.bench import samples


def publish(tmp_path):
    directory = tmp_path / 'maps/hm3d/val/00001-house'
    directory.mkdir(parents=True)
    scene = {'scene_id': 'house', 'coordinate_system': 'Habitat (X,-Z), height=Y',
             'levels': [{'id': 'L1', 'floor_height': 0., 'bounds': [[-4, 4], [-5, 5]]},
                        {'id': 'L2', 'floor_height': 3., 'bounds': [[-4, 4], [-5, 5]]}],
             'private_description': 'bed in north room'}
    walls = {'scene_id': 'house', 'segments': [
        {'level': 'L1', 'kind': 'wall', 'start': [-2, 1], 'end': [2, 1], 'evidence_frames': ['secret']},
        {'level': 'L1', 'kind': 'railing', 'start': [0, 0], 'end': [1, 1]},
        {'level': 'L2', 'kind': 'wall', 'start': [-2, 4], 'end': [2, 4]}],
        'openings': [{'id': 'D1', 'label': 'bedroom'}], 'stairs': [{'target': 'upstairs'}]}
    (directory / 'scene.json').write_text(json.dumps(scene))
    (directory / 'walls.json').write_text(json.dumps(walls))
    real = {**walls, 'segments': [walls['segments'][0]]}
    (directory / 'wall_real.json').write_text(json.dumps(real))
    return directory


def test_conversion_selects_floor_without_semantic_leakage(tmp_path):
    directory = publish(tmp_path)
    public, provenance = load_wall_map(directory, [0, .1, 0])
    assert public['walls'] == [[[-2., -1.], [2., -1.]]]
    assert set(public) == {'origin', 'size_m', 'floor_height', 'walls'}
    assert provenance['level'] == 'L1' and provenance['layer'] == 'walls'
    upper, _ = load_wall_map(directory, [0, 3.1, 0])
    assert upper['walls'] == [[[-2., -4.], [2., -4.]]]
    assert public['size_m'] == 20
    assert 'bedroom' not in json.dumps(public)
    raw = (directory / 'walls.json').read_bytes()
    saved = prepare_map(directory, [0, .1, 0], tmp_path / 'run/private')
    assert json.loads(saved.read_text()) == public
    assert (directory / 'walls.json').read_bytes() == raw
    assert (saved.parent / 'map_source.json').is_file()
    assert prepare_map(saved, [0, .1, 0], tmp_path / 'another') == saved
    assert load_wall_map(directory, [0, .1, 0], 'wall_real')[1]['layer'] == 'wall_real'


def test_level_metadata_fallback_and_existing_height_fields(tmp_path):
    directory = publish(tmp_path)
    scene = json.loads((directory / 'scene.json').read_text())
    levels = scene.pop('levels')
    levels[0]['height_m'] = levels[0].pop('floor_height')
    levels[0]['bounds'] = [-4, 4, -5, 5]
    (directory / 'scene.json').write_text(json.dumps(scene))
    (directory / 'levels.json').write_text(json.dumps(levels))
    _, provenance = load_wall_map(directory, [0, .1, 0])
    assert 'levels.json' in provenance['source_sha256']


def test_invalid_or_ambiguous_maps_are_not_silently_replaced(tmp_path):
    directory = publish(tmp_path)
    with pytest.raises(ValueError, match='exactly one floor'):
        load_wall_map(directory, [0, 1.5, 0])
    with pytest.raises(ValueError, match='outside'):
        load_wall_map(directory, [100, .1, 0])
    with pytest.raises(FileNotFoundError):
        load_wall_map(tmp_path / 'absent', [0, 0, 0])
    with pytest.raises(ValueError, match='layer'):
        load_wall_map(directory, [0, 0, 0], 'unknown')
    scene = json.loads((directory / 'scene.json').read_text())
    scene['levels'][1]['floor_height'] = .2
    (directory / 'scene.json').write_text(json.dumps(scene))
    with pytest.raises(ValueError, match='exactly one floor'):
        load_wall_map(directory, [0, .1, 0])
    scene['coordinate_system'] = 'Habitat (X,Z)'
    (directory / 'scene.json').write_text(json.dumps(scene))
    with pytest.raises(ValueError, match='coordinates'):
        load_wall_map(directory, [0, .1, 0])


def test_scene_path_matching():
    assert map_directory('maps', 'data/scene_datasets/mp3d/home/home.glb') == Path('maps/mp3d/home')
    assert map_directory('maps', 'hm3d/val/00001-house/house.basis.glb') == Path('maps/hm3d/val/00001-house')
    with pytest.raises(ValueError, match='relative'):
        map_directory('maps', '../other.glb')


def test_bench_plan_uses_published_map_and_reports_missing(tmp_path, monkeypatch):
    directory = publish(tmp_path)
    episode = SimpleNamespace(id='objnav:val:house:chair:0', benchmark='objnav',
                              scene_id='hm3d/val/00001-house/house.basis.glb', start_position=[0, .1, 0], goals=[{}])
    scene = tmp_path / 'scenes' / episode.scene_id
    scene.parent.mkdir(parents=True)
    scene.touch()
    selection = tmp_path / samples.SOURCES['objnav']['selection_file']
    selection.parent.mkdir(parents=True)
    selection.write_text(json.dumps([episode.id]))
    monkeypatch.setattr(samples, 'load_episodes', lambda repo, ids: [episode])
    config = {'episodes': [episode.id], 'environment': {'gpus': [0, 1], 'scenes': 'scenes'},
              'maps': {'root': 'maps', 'layer': 'walls'}}
    job = samples.plan(tmp_path, config)['jobs'][0]
    assert job['map'] == str(directory) and job['map_error'] is None
    assert job['map_source']['level'] == 'L1'
    assert not (directory / 'map.json').exists()  # Planning is read-only.
    (directory / 'walls.json').unlink()
    job = samples.plan(tmp_path, config)['jobs'][0]
    assert job['map_error'] and job['map_source'] is None


def test_worker_materializes_public_map_before_harness(tmp_path, monkeypatch):
    from src.bench import runner
    from src.dataset.episodes import EpisodeSpec
    from src.environment import Environment
    from src.environment.fake import FakeSimulator
    from src.harness import session

    directory = publish(tmp_path)
    episode = EpisodeSpec('test', '0', 'objnav', 'hm3d/val/00001-house/house.basis.glb',
                          [0, .1, 0], [0, 0, 0, 1],
                          [{'modality': 'object', 'instruction': 'Find a chair.', 'positions': [[0, .1, 0]]}], 'fixture')
    captured = []
    class Session:
        def prompt(self, *args, **kwargs):
            return SimpleNamespace(reason='submitted', submitted=True, answer='', requests=1)
        def close(self):
            pass
    def open_session(repo, config, output, env, source):
        captured.append(json.loads(source.read_text()))
        return Session()
    monkeypatch.setattr(runner, 'Environment', lambda e, c, d: Environment(e, c, d, FakeSimulator))
    monkeypatch.setattr(session, 'open_session', open_session)
    result = runner.run_episode(tmp_path, {'model': 'fixture', 'reasoning_effort': 'high'}, episode,
                                {'gpu': 0}, directory, tmp_path / 'run',
                                {'max_actions': 5, 'max_requests': 2, 'seconds': 10})
    assert result['status'] == 'completed'
    assert captured[0]['walls'] == [[[-2., -1.], [2., -1.]]]
    assert (tmp_path / 'run/private/map_source.json').is_file()


@pytest.mark.parametrize("declaration", ["Habitat (X,-Z); height Y", {"horizontal": "[Habitat X, -Habitat Z]", "height": "Habitat Y"}])
@pytest.mark.parametrize("field", ["coordinates", "coordinate_convention"])
def test_published_coordinates_field(tmp_path, declaration, field):
    directory = publish(tmp_path)
    path = directory / 'scene.json'
    scene = json.loads(path.read_text())
    scene.pop('coordinate_system')
    scene[field] = declaration
    path.write_text(json.dumps(scene))
    public, _ = load_wall_map(directory, [0, .1, 0])
    assert public['walls'] == [[[-2., -1.], [2., -1.]]]
