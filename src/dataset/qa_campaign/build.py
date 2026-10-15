"""Prepare the private dataset for the Qa 12-round campaign offline.

Writes a complete scenario and catalog for later loading by navigation or bench.
Archived scene geometry is never copied to the public workspace.
"""

from copy import deepcopy
import json
import math
from pathlib import Path

from .goals import complete_scenario, load_goal_catalog


def build_scenario(repo, goal_catalog=None):
    repo = Path(repo)
    draft = json.loads((repo / 'cache/harness-pi/astra-campaign-preflight/scenario.json').read_text())
    by_id = {r['id']: r for r in draft['rounds']}
    archive = repo / 'data/scene_maps/hm3d/val/00829-QaLdnwvtxbs'
    frames = {f['id']: f for f in json.loads((archive / 'raw/evidence.json').read_text())['frames']}
    rooms = json.loads((archive / 'raw/rooms.json').read_text())['rooms']

    def target(identifier, object_id, evidence, instruction):
        positions = [[frames[f]['camera_position_xyz'][0], draft['start_position'][1],
                      frames[f]['camera_position_xyz'][2]] for f in evidence]
        return dict(id=identifier, kind='navigation', instruction=instruction, max_actions=180,
                    max_requests=40, seconds=1800, positions=positions, success_distance=1.0,
                    private_verification={'target_object_id': object_id, 'source_evidence_ids': evidence})

    exploration = deepcopy(by_id['explore'])
    exploration.update(instruction='先熟悉当前客餐厅的主要物体与可见出入口，留下初步记忆；暂不深入其他房间。',
                       max_actions=90, max_requests=30, seconds=1200, scope_room='R05')
    rounds = [exploration, deepcopy(by_id['find_bed']), deepcopy(by_id['find_bathtub']),
              deepcopy(by_id['find_dining_table']),
              target('bedroom_tv', 'O042', ['render_O042_1', 'render_O042_2'], '去卧室里床对面的电视旁。'),
              deepcopy(by_id['preparation_area_sink']), deepcopy(by_id['hallway_bathroom_sink']),
              target('double_basin', 'O108', ['render_O109_1', 'render_O110_1'], '去带浴缸的浴室，走到双台盆洗手台前。'),
              deepcopy(by_id['living_room_tv']), deepcopy(by_id['relocate_closet']),
              deepcopy(by_id['relocate_hallway']), deepcopy(by_id['relocate_bathroom'])]
    instructions = ['先熟悉当前客餐厅的主要物体与可见出入口，留下初步记忆；暂不深入其他房间。',
                    '找到床，走到床边。', '找到浴缸，走到浴缸旁。', '找到餐桌，走到餐桌旁。',
                    '去卧室里床对面的电视旁。', '找到小型备餐区，走到水槽旁。',
                    '经过入户走廊旁的门，进入独立卫生间，停在洗手盆旁。',
                    '去带浴缸的浴室，走到双台盆洗手台前。', '去客餐厅的电视旁。',
                    '离开当前空间，穿过卧室，进入有浴缸的浴室，停在浴缸旁。',
                    '沿走廊进入客餐厅，经过餐桌，再进入卧室，走到床边。',
                    '离开这里，穿过卧室回到客厅，停在沙发与电视之间的方形木茶几旁。']
    for r, instruction in zip(rounds, instructions):
        r['instruction'] = instruction
    rounds[9]['teleport']['position'] = [3.55, draft['start_position'][1], -3.65]
    tea = target('relocate_bathroom', 'O188', ['render_O188_1'], instructions[11])
    rounds[11]['positions'] = tea['positions']
    rounds[11]['private_verification'].update(tea['private_verification'])
    for r in rounds:
        r['snap_viewpoints'] = r['id'] in {'bedroom_tv', 'double_basin', 'relocate_bathroom'}
        r['route_checks'] = []  # Exact directed crossings are scored separately, not proximity.
        r['route_events'] = [dict(kind='portal', **portal)
                             for portal in r.get('private_verification', {}).get('ordered_portals', [])]
    rounds[10]['route_events'].insert(1, {'kind': 'waypoint', 'id': 'dining_table',
                                       'positions': rounds[3]['positions'], 'radius': 1.5})
    scenario = {'schema_version': 2, 'protocol': 'navcraft-map-free-12-round-exploratory-campaign',
            'scene_id': draft['scene_id'], 'environment': draft['environment'],
            'start_position': [-.7, draft['start_position'][1], -.5],
            'start_rotation': [0, math.sin(math.pi / 4), 0, math.cos(math.pi / 4)],
            'rounds': rounds, 'private_rooms': rooms, 'seconds': sum(r['seconds'] for r in rounds) + 120,
            'limitations': 'One continuous exploratory chain; no benchmark SR, no prebuilt map/memory or live pose/depth provided. STOP and geodesic distance to accepted evidence viewpoints <1m; exact ordered directional portal crossings and dining waypoint scored offline. Exploration scope scored separately.'}
    return complete_scenario(scenario, goal_catalog or load_goal_catalog(repo))



def prepare(repo, output):
    """Build once; consumers only need the resulting two JSON files."""
    catalog = load_goal_catalog(repo)
    scenario = build_scenario(repo, catalog)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (('scenario.json', scenario), ('catalog.json', catalog)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return scenario


def main():
    import argparse
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=repo / 'data/datasets/navcraft/qa_campaign_v2')
    args = parser.parse_args()
    prepare(repo, args.output)
    print(args.output)


if __name__ == '__main__':
    main()
