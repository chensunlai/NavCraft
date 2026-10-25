#!/usr/bin/env python3
"""Render evidence-authored wall layers; never infer walls or certify visual review."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from shapely.geometry import LineString


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def clip_walls(data, half_width):
    """Clip parallel structural faces only; optional wall_ids narrows each portal."""
    result = copy.deepcopy(data)
    result['segments'] = []
    for wall in data['segments']:
        line = LineString([wall['start'], wall['end']])
        if line.length <= 0:
            raise ValueError('zero-length wall: ' + wall['id'])
        dx, dy = (wall['end'][i] - wall['start'][i] for i in (0, 1))
        geom = line
        for door in data.get('openings', []):
            if door['level'] != wall['level']:
                continue
            if 'wall_ids' in door and wall['id'] not in door['wall_ids']:
                continue
            portal = LineString(door['segment'])
            if portal.length <= 0:
                raise ValueError('zero-length portal: ' + door['id'])
            a, b = door['segment']
            cosine = abs(dx * (b[0]-a[0]) + dy * (b[1]-a[1])) / (line.length * portal.length)
            if cosine < .995:
                continue  # Do not erase perpendicular return walls or stair rails.
            geom = geom.difference(portal.buffer(door.get('cut_half_width_m', half_width), cap_style=2))
        parts = list(geom.geoms) if hasattr(geom, 'geoms') else [geom]
        for i, part in enumerate(parts, 1):
            if part.is_empty or part.length < .005:
                continue
            record = copy.deepcopy(wall)
            record.update(id=f"{wall['id']}_{i:02d}", source_wall=wall['id'],
                          start=list(part.coords[0]), end=list(part.coords[-1]))
            result['segments'].append(record)
    result['geometry_stage'] = 'portal_clipped'
    return result


def draw(data, scene, stairs, output, name, dpi):
    levels = scene['levels']
    xlo = min(l['bounds'][0][0] for l in levels)
    xhi = max(l['bounds'][0][1] for l in levels)
    ylo = min(l['bounds'][1][0] for l in levels)
    yhi = max(l['bounds'][1][1] for l in levels)
    if xhi <= xlo or yhi <= ylo:
        raise ValueError('invalid level bounds')
    height = max(5, 14 * (yhi-ylo)/(xhi-xlo))
    fig, axes = plt.subplots(len(levels), 1, figsize=(14, height*len(levels)), squeeze=False, layout='constrained')
    for lev, ax in zip(levels, axes[:, 0]):
        lid = lev['id']
        for area in data.get('structural_voids', []):
            if area['level'] == lid:
                ax.add_patch(Polygon(area['polygon'], facecolor='#eef2f6', edgecolor='#cbd5e1', hatch='///', linewidth=.5))
        for stair in stairs:
            if lid not in (stair['from_level'], stair['to_level']):
                continue
            ax.add_patch(Polygon(stair['polygon'], facecolor='#edf5f8', edgecolor='none', zorder=1))
            for tread in stair['treads']:
                ax.plot(*zip(*tread), color='#74939c', lw=.8, zorder=2)
            up = lid == stair['from_level']
            points = stair['ascent_points'] if up else list(reversed(stair['ascent_points']))
            for a, b in zip(points, points[1:]):
                ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='->', color='#087f8c', lw=1.8), zorder=5)
            pos = stair.get('label_positions', {}).get(lid)
            if pos is None:
                mid = LineString(points).interpolate(.5, normalized=True)
                pos = [mid.x, mid.y + .2]
            label = f"{stair['id']} {'UP TO' if up else 'DOWN TO'} {stair['to_level'] if up else stair['from_level']}"
            ax.text(*pos, label, fontsize=9, ha='center', color='#076673', zorder=6,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=1))
        for wall in data['segments']:
            if wall['level'] == lid:
                ax.plot(*zip(wall['start'], wall['end']), color='#20252c', lw=3.5, solid_capstyle='butt', zorder=4)
        for door in data.get('openings', []):
            if door['level'] != lid:
                continue
            color = {'confirmed_open': '#168272', 'confirmed_closed': '#b7791f'}.get(door['status'], '#88939e')
            ax.plot(*zip(*door['segment']), color=color, lw=1.5, ls=':', zorder=4)
            mid = LineString(door['segment']).interpolate(.5, normalized=True)
            ax.text(mid.x+.04, mid.y+.1, door['id'], fontsize=8, color=color)
        for note in data.get('drawing_notes', []):
            if note['level'] == lid:
                ax.text(*note['position'], note['text'], fontsize=8, color='#64748b', ha='center')
        ax.set(xlim=(xlo, xhi), ylim=(ylo, yhi), aspect='equal', xlabel='Habitat X (m)', ylabel='-Habitat Z (m)',
               title=f"{scene['scene_id']} | {lid} | floor {lev['floor_height']:.3f} m | {name}")
        ax.grid(alpha=.12)
    fig.suptitle(f"{name}: {'evidence-supported geometry' if name == 'wall_real' else 'walls, railings and completed boundaries'} | Blue: stairs | Green: open door | Amber: closed door", fontsize=9)
    for suffix in ('png', 'svg'):
        fig.savefig(output/f'{name}.{suffix}', dpi=dpi)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--walls', type=Path, required=True, help='Authored construction lines, before portal clipping')
    p.add_argument('--real', type=Path, required=True, help='Explicit evidence-only authored lines; never derived by guessing statuses')
    p.add_argument('--stairs', type=Path, help='Optional separate {scene_id, stairs}; otherwise read stairs from wall input')
    p.add_argument('--output', type=Path, required=True, help='Working output directory; input files are never overwritten')
    p.add_argument('--portal-half-width', type=float, default=.18)
    p.add_argument('--dpi', type=int, default=160)
    args = p.parse_args()
    if not 0 < args.portal_half_width <= .5 or not 40 <= args.dpi <= 400:
        p.error('portal half width must be in (0, .5] metres; dpi in [40, 400]')
    inputs = {'scene': args.scene, 'walls': args.walls, 'real': args.real}
    if args.stairs:
        inputs['stairs'] = args.stairs
    outputs = ['walls.json', 'wall_real.json', 'walls.png', 'walls.svg', 'wall_real.png', 'wall_real.svg', 'render_report.json']
    if {x.resolve() for x in inputs.values()} & {(args.output/x).resolve() for x in outputs}:
        p.error('output would overwrite an input; use a separate working directory')
    scene, wall, real = load(args.scene), load(args.walls), load(args.real)
    if any(x.get('scene_id') != scene['scene_id'] for x in (wall, real)):
        p.error('scene IDs differ')
    if wall.get('geometry_stage') == 'portal_clipped' or real.get('geometry_stage') == 'portal_clipped':
        p.error('use construction lines, not previously clipped output')
    known_levels = {l['id'] for l in scene['levels']}
    for data in (wall, real):
        ids = [w['id'] for w in data['segments']]
        if not ids or len(set(ids)) != len(ids):
            p.error('empty layer or duplicate construction-line IDs')
        if any(r['level'] not in known_levels for r in data['segments'] + data.get('openings', [])):
            p.error('wall or opening refers to an unknown level')
    if wall.get('unknown_areas') or any(w.get('status') != 'resolved' for w in wall['segments']):
        p.error('final wall still has pending geometry; make explicit decisions before rendering')
    if any(w.get('origin') != 'observed_schematic' for w in real['segments']):
        p.error('wall_real must contain only explicitly observed_schematic segments')
    stair_data = load(args.stairs) if args.stairs else wall
    if stair_data.get('scene_id') != scene['scene_id']:
        p.error('stair scene ID differs')
    stairs = stair_data.get('stairs', [])
    if any(s['from_level'] not in known_levels or s['to_level'] not in known_levels or s['from_level'] == s['to_level'] for s in stairs):
        p.error('invalid stair level references')
    args.output.mkdir(parents=True, exist_ok=True)
    for data, name in ((wall, 'walls'), (real, 'wall_real')):
        derived = clip_walls(data, args.portal_half_width)
        derived['stairs'] = stairs
        save(args.output/f'{name}.json', derived)
        draw(derived, scene, stairs, args.output, name, args.dpi)
    report = dict(scene_id=scene['scene_id'], inputs={k:dict(path=str(v.resolve()), sha256=sha(v)) for k,v in inputs.items()},
                  outputs_sha256={f:sha(args.output/f) for f in outputs if f != 'render_report.json'},
                  portal_half_width_m=args.portal_half_width, visual_review_performed=False)
    save(args.output/'render_report.json', report)
    print(json.dumps(dict(output=str(args.output), visual_review_performed=False)))


if __name__ == '__main__':
    main()
