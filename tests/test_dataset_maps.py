"""Published-map loading, public boundaries, and Bench wiring (no simulator/API)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.dataset.maps import load_wall_map, map_directory, prepare_map, publish_map, publish_tree
from src.bench import samples


def publish(tmp_path):
    directory = tmp_path / 'maps/hm3d/val/00001-house'
    directory.mkdir(parents=True)
    scene = {'scene_id': 'house', 'coordinate_system': 'Habitat (X,-Z), height=Y', 'units': 'm',
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
    publish_map(directory, directory)
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
    publish_map(directory, directory)
    _, provenance = load_wall_map(directory, [0, .1, 0])
    assert 'levels.json' in provenance['source']['sha256']


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
    publish_map(directory, directory)
    with pytest.raises(ValueError, match='exactly one floor'):
        load_wall_map(directory, [0, .1, 0])
    scene['coordinate_system'] = 'Habitat (X,Z)'
    (directory / 'scene.json').write_text(json.dumps(scene))
    with pytest.raises(ValueError, match='coordinates'):
        publish_map(directory, directory)


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
    assert (directory / 'map.json').exists()  # Planning only reads the published artifact.
    (directory / 'map.json').unlink()
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
    publish_map(directory, directory)
    public, _ = load_wall_map(directory, [0, .1, 0])
    assert public['walls'] == [[[-2., -1.], [2., -1.]]]


def test_runtime_uses_frozen_publication_not_raw_files(tmp_path):
    directory = publish(tmp_path)
    before, _ = load_wall_map(directory, [0, 0, 0])
    (directory / 'walls.json').unlink()
    (directory / 'scene.json').unlink()
    after, _ = load_wall_map(directory, [0, 0, 0])
    assert after == before
    serialized = json.dumps(json.loads((directory / 'map.json').read_text()))
    for hidden in ('bedroom', 'evidence_frames', 'private_description', 'openings', 'stairs'):
        assert hidden not in serialized


def test_layers_share_calibration_even_with_different_geometry_bounds(tmp_path):
    directory = publish(tmp_path)
    wall = json.loads((directory / 'wall_real.json').read_text())
    wall['segments'][0]['end'] = [25, 1]
    (directory / 'wall_real.json').write_text(json.dumps(wall))
    publish_map(directory, directory)
    complete, _ = load_wall_map(directory, [0, 0, 0])
    observed, _ = load_wall_map(directory, [0, 0, 0], 'wall_real')
    assert complete['origin'] == observed['origin']
    assert complete['size_m'] == observed['size_m']


def test_publish_rejects_ambiguous_coordinate_declaration(tmp_path):
    directory = publish(tmp_path)
    scene = json.loads((directory / 'scene.json').read_text())
    scene['coordinate_system'] = 'map=(Habitat X,Z); source contains -Z elsewhere'
    (directory / 'scene.json').write_text(json.dumps(scene))
    with pytest.raises(ValueError, match='explicitly declare'):
        publish_map(directory, directory)


def test_publish_queue_indexes_scene_instance_assets(tmp_path):
    source = publish(tmp_path)
    destination = tmp_path / 'published'
    queue = tmp_path / 'queue.json'
    queue.write_text(json.dumps({'scene_root': 'assets', 'output_root': 'archive', 'scenes': [
        {'scene_path': 'assets/hssd-hab/scenes/house.scene_instance.json',
         'output_dir': 'archive/hm3d/val/00001-house'}]}))
    results = publish_tree(tmp_path / 'maps', destination, [queue])
    assert results[0]['status'] == 'published'
    assert map_directory(destination, 'hssd-hab/scenes/house.scene_instance.json') == destination / 'hm3d/val/00001-house'


def test_public_map_input_is_validated(tmp_path):
    source = tmp_path / 'map.json'
    source.write_text(json.dumps({'origin': [0, 0], 'size_m': 0, 'floor_height': 0, 'walls': []}))
    with pytest.raises(ValueError, match='positive'):
        prepare_map(source, [0, 0, 0], tmp_path / 'private')


def test_publish_requires_declared_units_and_accepts_archive_unit_field(tmp_path):
    directory = publish(tmp_path)
    path = directory / 'scene.json'
    scene = json.loads(path.read_text())
    scene['unit'] = scene.pop('units')
    path.write_text(json.dumps(scene))
    publish_map(directory, directory)
    del scene['unit']
    path.write_text(json.dumps(scene))
    with pytest.raises(ValueError, match='metres'):
        publish_map(directory, directory)


def test_publish_excludes_declared_non_floor_and_does_not_invent_missing_layer(tmp_path):
    directory = publish(tmp_path)
    with pytest.raises(ValueError, match='unavailable'):
        load_wall_map(directory, [0, 3, 0], 'wall_real')
    path = directory / 'scene.json'
    scene = json.loads(path.read_text())
    scene['levels'][1]['occupied_floor_confirmed'] = False
    path.write_text(json.dumps(scene))
    publish_map(directory, directory)
    document = json.loads((directory / 'map.json').read_text())
    assert [level['id'] for level in document['levels']] == ['L1']
    assert document['source']['excluded_levels'] == [{'id': 'L2', 'reason': 'source_marks_non_floor'}]
    with pytest.raises(ValueError, match='exactly one floor'):
        load_wall_map(directory, [0, 3, 0])


def test_publish_without_queue_preserves_existing_asset_index(tmp_path):
    directory = publish(tmp_path)
    root = tmp_path / 'maps'
    mapping = {'hssd-hab/scenes/house.scene_instance.json': directory.relative_to(root).as_posix()}
    (root / 'index.json').write_text(json.dumps(mapping))
    publish_tree(root, root)
    assert json.loads((root / 'index.json').read_text()) == mapping
    assert map_directory(root, next(iter(mapping))) == directory


def test_build_run_publishes_only_ready_and_checked_scenes(tmp_path):
    source = publish(tmp_path)
    build = tmp_path / 'build'
    build.mkdir()
    output = tmp_path / 'published'
    scenes, summaries = [], []
    cases = [('house', 'ready_for_review', 'ready_for_review', True),
             ('running', 'running', 'running', False),
             ('stale_summary', 'ready_for_review', 'running', True),
             ('unchecked', 'ready_for_review', 'ready_for_review', False),
             ('pending', 'pending', 'pending', False)]
    for scene_id, summary_status, result_status, checked in cases:
        path = source if scene_id == 'house' else tmp_path / 'maps/hm3d/val' / scene_id
        path.mkdir(parents=True, exist_ok=True)
        if scene_id != 'house':
            # Deliberately invalid: these sources must not be parsed at all.
            (path / 'scene.json').write_text('in-progress write')
        result = build / f'{scene_id}.json'
        result.write_text(json.dumps({'scene_id': scene_id, 'status': result_status,
                                     'validation': {'structural_pass': checked}}))
        scenes.append({'scene_id': scene_id, 'output_dir': str(path)})
        summaries.append({'scene_id': scene_id, 'status': summary_status, 'result': str(result)})
    (build / 'plan.json').write_text(json.dumps({'settings': {'repo': str(tmp_path)}, 'scenes': scenes}))
    (build / 'summary.json').write_text(json.dumps({'scenes': summaries}))
    results = publish_tree(tmp_path / 'maps', output, build_run=build)
    assert sum(row['status'] == 'published' for row in results) == 1
    assert sum(row['status'] == 'skipped' for row in results) == 4
    assert (output / 'hm3d/val/00001-house/map.json').is_file()
    assert not (output / 'hm3d/val/running').exists()
    assert {row['scene'].split('/')[-1] for row in results if row['status'] == 'skipped'} == {
        'running', 'stale_summary', 'unchecked', 'pending'}


def test_publish_rechecks_build_result_before_replacing_frozen_map(tmp_path, monkeypatch):
    from src.dataset import maps
    directory = publish(tmp_path)
    before = (directory / 'map.json').read_bytes()
    result = tmp_path / 'result.json'
    result.write_text(json.dumps({'scene_id': 'house', 'status': 'ready_for_review',
                                 'validation': {'structural_pass': True}}))
    check = maps._ready_result
    checks = []
    def resumed_build(path, scene_id):
        checks.append(path)
        if len(checks) == 2:
            result.write_text(json.dumps({'scene_id': 'house', 'status': 'running'}))
        check(path, scene_id)
    monkeypatch.setattr(maps, '_ready_result', resumed_build)
    with pytest.raises(ValueError, match='not finished'):
        publish_map(directory, directory, build_result=result)
    assert (directory / 'map.json').read_bytes() == before
