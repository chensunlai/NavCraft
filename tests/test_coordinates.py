import math
from pathlib import Path

import pytest

from src.common.coordinates import (bearing_distance, body_forward, heading_from_forward,
                                    map_pose, project_uv, uv_to_world, validate_map, world_to_uv)
from src.common.scenes import canonical_scene_id, resolve_scene


TRANSFORM = {'origin': [-5., -10.], 'size_m': 20., 'floor_height': 3., 'walls': []}


def test_metric_world_uv_roundtrip_preserves_axis_and_height():
    point = [7., 3.2, -4.]
    assert world_to_uv(point, TRANSFORM) == [.6, .3]
    assert uv_to_world([.6, .3], TRANSFORM, height=3.2) == point
    assert uv_to_world([.6, .3], TRANSFORM)[1] == 3.
    validate_map(TRANSFORM)


@pytest.mark.parametrize('degrees', [0, 30, 90, 180, 270])
def test_habitat_body_quaternion_matches_clockwise_map_heading(degrees):
    half = math.radians(-degrees) / 2
    pose = map_pose({'position': [5, 3, 0], 'rotation_xyzw': [0, math.sin(half), 0, math.cos(half)]}, TRANSFORM)
    assert pose == {'pos': [.5, .5], 'heading': degrees}
    forward = [.5 + .1 * math.sin(math.radians(degrees)), .5 - .1 * math.cos(math.radians(degrees))]
    relative = bearing_distance(pose, forward, TRANSFORM)
    assert relative['bearing_degrees'] == pytest.approx(0)
    assert relative['distance_m'] == pytest.approx(2)


def test_heading_wrap_and_zero_distance():
    assert bearing_distance({'pos': [.5, .5], 'heading': 90}, [.5, .5], TRANSFORM) == {
        'bearing_degrees': 0, 'distance_m': 0}
    assert bearing_distance({'pos': [.5, .5], 'heading': 0}, [.6, .5], TRANSFORM)['bearing_degrees'] == 90


@pytest.mark.parametrize('heading', [0, 37, 90, 180, 359])
@pytest.mark.parametrize('bearing', [-180, -60, 0, 90])
def test_relative_target_roundtrip(heading, bearing):
    pose = {'pos': [.17, .81], 'heading': heading}
    target = project_uv(pose, bearing, 2.3, TRANSFORM)
    result = bearing_distance(pose, target, TRANSFORM)
    assert result['distance_m'] == pytest.approx(2.3)
    assert (result['bearing_degrees'] - bearing + 180) % 360 - 180 == pytest.approx(0)


def test_body_direction_keeps_pitch_but_heading_is_horizontal():
    # +60 degree X rotation: forward tilts upward, map heading remains -Z.
    q = [math.sin(math.pi / 6), 0, 0, math.cos(math.pi / 6)]
    forward = body_forward(q)
    assert forward == pytest.approx([0, math.sqrt(3) / 2, -.5])
    assert heading_from_forward(forward) == 0
    with pytest.raises(ValueError, match='horizontal'):
        heading_from_forward([0, 1, 0])


def test_scene_source_spellings_normalize_once(tmp_path):
    canonical = 'hm3d/val/00001-home/home.basis.glb'
    assert canonical_scene_id('data/scene_datasets/' + canonical) == canonical
    assert canonical_scene_id(canonical.replace('hm3d/', 'hm3d_v0.2/')) == canonical
    asset = tmp_path / canonical
    asset.parent.mkdir(parents=True)
    asset.touch()
    assert resolve_scene(tmp_path, canonical) == asset
    for bad in ('../home.glb', '/private/home.glb', 'hm3d/../../home.glb', '', 'hm3d\\home.glb'):
        with pytest.raises(ValueError):
            canonical_scene_id(bad)
    with pytest.raises(FileNotFoundError):
        resolve_scene(tmp_path, 'hm3d/absent.glb')
