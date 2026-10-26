"""Known world points through source walls, RGB-D, map display and map selection."""
import json
import math

import numpy as np
import pytest
from PIL import Image

from src.common.coordinates import map_pose, uv_to_world
from src.dataset.maps import load_wall_map
from src.harness.geometry import Geometry
from src.harness.maps import MapRenderer


@pytest.mark.parametrize('heading', [0, 37, 90, 180, 270])
@pytest.mark.parametrize('view', ['full', {'around_self': 3}, {'ego': 3}])
def test_source_rgb_and_map_selection_agree(tmp_path, heading, view):
    angle = math.radians(heading)
    forward = np.array([math.sin(angle), 0., -math.cos(angle)])
    right = np.array([math.cos(angle), 0., math.sin(angle)])
    camera_position = np.array([1., 2.88, -2.])
    target = camera_position + 2 * forward
    scene = tmp_path / 'source'
    scene.mkdir()
    (scene / 'scene.json').write_text(json.dumps({
        'scene_id': 'fixture', 'coordinate_system': 'Habitat (X,-Z)', 'units': 'm',
        'levels': [{'id': 'L1', 'floor_height': 2., 'bounds': [[-8, 8], [-8, 8]]}]}))
    # An actual source wall endpoint and railing endpoint at the known world point.
    endpoint = [float(target[0]), -float(target[2])]
    (scene / 'walls.json').write_text(json.dumps({'scene_id': 'fixture', 'segments': [
        {'id': 'W1', 'level': 'L1', 'start': endpoint, 'end': [endpoint[0] + .7, endpoint[1]], 'kind': 'wall'},
        {'id': 'W2', 'level': 'L1', 'start': endpoint, 'end': [endpoint[0], endpoint[1] + .8], 'kind': 'railing'}]}))
    original = {p.name: p.read_bytes() for p in scene.iterdir()}
    transform, _ = load_wall_map(scene, [1., 2., -2.])
    assert len(transform['walls']) == 2
    for segment in transform['walls']:
        assert segment[0] == pytest.approx(target[[0, 2]])

    workspace, private = tmp_path / 'workspace', tmp_path / 'private'
    (workspace / 'scene').mkdir(parents=True)
    (workspace / 'observations').mkdir()
    (private / 'depth').mkdir(parents=True)
    (workspace / 'scene/map.json').write_text(json.dumps(transform))
    Image.fromarray(np.full((5, 5), 2000, dtype=np.uint16)).save(private / 'depth/f.png')
    Image.new('RGB', (5, 5)).save(workspace / 'observations/f.png')
    (workspace / 'observations/index.jsonl').write_text(json.dumps({'frame': 'f', 'path': 'observations/f.png'}) + '\n')
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack([right, [0, -1, 0], forward])
    matrix[:3, 3] = camera_position
    camera = {'frame': 'f', 'depth': 'f.png', 'depth_scale': .001,
              'K': [[2, 0, 2], [0, 2, 2], [0, 0, 1]], 'world_from_optical': matrix.tolist(),
              'body_position': [1., 2., -2.], 'body_forward': forward.tolist(),
              'floor_height': 2., 'width': 5, 'height': 5, 'depth_min': .05, 'depth_max': 10.}
    (private / 'depth/index.jsonl').write_text(json.dumps(camera) + '\n')
    geometry = Geometry(workspace, private, transform)
    world, _ = geometry.resolve_frame_point('f', [2, 2])
    assert world == pytest.approx(target)
    projected = geometry.resolve_frame('f', [2, 2])
    assert projected['bearing'] == pytest.approx(0)
    assert projected['distance'] == 2
    assert projected['height'] == .88
    exact_uv = [float((target[i] - transform['origin'][j]) / transform['size_m']) for j, i in enumerate((0, 2))]
    pose = map_pose({'position': [1., 2., -2.],
                     'rotation_xyzw': [0, math.sin(-angle / 2), 0, math.cos(angle / 2)]}, transform)
    result = MapRenderer(workspace).render(pose, [], view=view,
                                          located_points=[{'label': 'P1', 'uv': exact_uv}])
    # Solve independently, then verify both the drawn marker and tool backprojection.
    pixel = np.linalg.solve(np.vstack([result['pixel_to_uv'], [0, 0, 1]]), [*exact_uv, 1])[:2]
    if isinstance(view, dict) and 'ego' in view:
        assert pixel == pytest.approx([384., 380. - 2 * 692 / 6])
    with Image.open(workspace / result['map']) as image:
        x, y = map(round, pixel)
        patch = np.asarray(image.crop((x - 9, y - 9, x + 10, y + 10)))
        assert ((patch == [147, 51, 234]).all(axis=2)).any()
    restored = geometry.resolve_map(result['view_id'], pixel.tolist())['uv']
    assert restored == projected['uv']
    assert uv_to_world(restored, transform, target[1]) == pytest.approx(target, abs=.002)
    assert {p.name: p.read_bytes() for p in scene.iterdir()} == original
