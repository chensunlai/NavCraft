#!/usr/bin/env python3
"""Read-only wall-stage archive checks. Does not certify RGB truth or navigation."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PIL import Image
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(root, tolerance=.005):
    root = root.resolve()
    raw = root/'raw'
    errors, warnings = [], []

    def require(ok, kind, detail=None):
        if not ok:
            errors.append(dict(kind=kind, detail=detail))

    def local(base, ref):
        path = (base/ref).resolve()
        if not path.is_relative_to(root):
            raise ValueError('reference escapes archive: ' + str(ref))
        return path

    def read(ref):
        return json.loads(local(raw, ref).read_text())

    def exists(base, ref):
        path = local(base, ref)
        require(path.is_file(), 'missing_reference', str(path.relative_to(root)))
        return path

    def xy(point):
        if len(point) != 2 or not all(math.isfinite(float(v)) for v in point):
            raise ValueError('invalid XY coordinate: ' + str(point))
        return point

    def line(points):
        return LineString([xy(p) for p in points])

    require({p.name for p in root.iterdir()} == {'floor_plan.png','scene_description.md','evidence','evidence_rooms.jpg','raw'}, 'top_level_layout')
    scene, manifest = read('scene.json'), read('manifest.json')
    files = scene.get('files', {})
    wall_name = files.get('wall', files.get('walls', 'walls.json'))
    real_name = files.get('wall_real', 'wall_real.json')
    walls, real = read(wall_name), read(real_name)
    doors = read(files.get('doors','doors.json'))['doors']
    frames = read(files.get('evidence','evidence.json'))['frames']
    stair_data = read(files.get('stairs','stairs.json'))
    require(scene['stage'] == 'walls_only', 'wrong_stage')
    for name in ('regions.json','rooms.json','objects.json'):
        if (raw/name).exists():
            warnings.append(dict(kind='later_stage_present', detail=name))
    # Existing later-stage files are not deleted or treated as empty completed work.
    for ref in files.values():
        exists(raw, ref)
    for data in (walls, real, stair_data, manifest):
        require(data.get('scene_id') == scene['scene_id'], 'scene_id_mismatch')
    if (raw/'wall.json').exists() and (raw/'walls.json').exists():
        require(read('wall.json') == read('walls.json'), 'wall_alias_mismatch')
    levels = {l['id']:l for l in scene['levels']}
    require(len(levels) == len(scene['levels']), 'duplicate_level')
    byframe = {f['id']:f for f in frames}
    require(len(byframe) == len(frames), 'duplicate_frame')
    for f in frames:
        require(bool(f.get('source')), 'missing_frame_source', f['id'])
        exists(raw, f['rgb'])
        for key in ('depth','semantic'):
            if f.get(key):
                exists(raw, f[key])
    for name in ('../floor_plan.png','../evidence_rooms.jpg',files.get('wall_layer','walls.png'),files.get('wall_real_layer','wall_real.png')):
        with Image.open(local(raw, name)) as im:
            im.verify()

    def evidence(record):
        require(bool(record.get('evidence_frames')), 'missing_evidence', record['id'])
        for fid in record.get('evidence_frames', []):
            require(fid in byframe, 'unknown_evidence_frame', [record['id'], fid])
        for ref in record.get('evidence_geometry', []):
            exists(raw, ref)

    all_lines = {}
    for name, data in (('wall',walls),('wall_real',real)):
        records = data['segments']
        require(bool(records), 'empty_wall_layer', name)
        require(len({w['id'] for w in records}) == len(records), 'duplicate_wall_id', name)
        if name == 'wall':
            require(not data.get('unknown_areas'), 'pending_wall_areas')
        lines = []
        for w in records:
            require(w['level'] in levels, 'unknown_level', w['id'])
            shape = line([w['start'], w['end']])
            require(shape.length > tolerance, 'zero_wall', w['id'])
            require(w.get('origin') in ('observed_schematic','inferred_completion'), 'missing_wall_origin', w['id'])
            require(bool(w.get('basis')), 'missing_wall_basis', w['id'])
            if name == 'wall':
                require(w.get('status') == 'resolved', 'pending_wall_geometry', w['id'])
            else:
                require(w.get('origin') == 'observed_schematic', 'estimate_in_wall_real', w['id'])
            evidence(w)
            lines.append((w, shape))
        for i,(a,x) in enumerate(lines):
            for b,y in lines[i+1:]:
                if a['level'] == b['level']:
                    require(x.intersection(y).length <= tolerance, 'duplicate_wall_overlap', [name,a['id'],b['id']])
        all_lines[name] = lines
    for w, shape in all_lines['wall_real']:
        observed = unary_union([g for s,g in all_lines['wall'] if s['level'] == w['level'] and s.get('origin') == 'observed_schematic'])
        require(shape.difference(observed.buffer(tolerance)).length <= tolerance, 'real_geometry_missing_from_wall', w['id'])
    require(walls.get('openings',[]) == real.get('openings',[]), 'dual_layer_opening_mismatch')
    require(len({d['id'] for d in doors}) == len(doors), 'duplicate_door_id')
    wall_doors = {d['id']:d for d in walls.get('openings',[])}
    require(set(wall_doors) == {d['id'] for d in doors}, 'door_set_mismatch')
    for d in doors:
        evidence(d)
        require(d['level'] in levels, 'unknown_door_level', d['id'])
        if d['id'] in wall_doors:
            for key in ('level','segment','status'):
                require(d[key] == wall_doors[d['id']][key], 'door_layer_mismatch', [d['id'],key])
        portal = line(d['segment'])
        require(portal.length > tolerance, 'zero_portal', d['id'])
        inner = LineString([portal.interpolate(.02,normalized=True),portal.interpolate(.98,normalized=True)])
        for name, items in all_lines.items():
            for w, shape in items:
                if w['level'] == d['level']:
                    require(not inner.intersects(shape), 'portal_blocked', [name,d['id'],w['id']])
    stairs = stair_data['stairs']
    require(len({s['id'] for s in stairs}) == len(stairs), 'duplicate_stair_id')
    for s in stairs:
        evidence(s)
        require(s['from_level'] in levels and s['to_level'] in levels and s['from_level'] != s['to_level'], 'invalid_stair_levels', s['id'])
        require(s['to_height'] > s['from_height'], 'invalid_stair_heights', s['id'])
        poly = Polygon([xy(p) for p in s['polygon']])
        require(poly.is_valid and poly.area > tolerance**2, 'invalid_stair_polygon', s['id'])
        ascent = line(s['ascent_points'])
        require(ascent.length > tolerance and poly.buffer(.15).covers(ascent), 'stair_direction_outside_footprint', s['id'])
        require(bool(s['treads']), 'missing_treads', s['id'])
        for tread in s['treads']:
            require(poly.buffer(.15).covers(line(tread)), 'tread_outside_footprint', s['id'])
        for key in ('lower_entry','upper_exit'):
            require(line(s[key]).length > tolerance and poly.buffer(.15).covers(line(s[key])), 'invalid_stair_entry', [s['id'],key])
    # Check present reports against their recorded inputs. Reports remain evidence,
    # not an automatic assertion that their method or claimed pass is sufficient.
    review_name = files.get('review','layout_review.json')
    review = read(review_name)
    aliases = review.get('build_working_input_aliases', {})
    reports = [p.name for p in raw.glob('*check.json')] + ['review_final.json']
    for name in reports:
        if not (raw/name).is_file():
            warnings.append(dict(kind='missing_review_report',detail=name))
            continue
        report = read(name)
        hashes = report.get('inputs_sha256', report.get('checked_inputs_sha256',{}))
        for ref, digest in hashes.items():
            path = exists(raw, aliases.get(ref,ref))
            require(path.is_file() and sha(path) == digest, 'stale_check', [name,ref])
    for name in ('path_check.json','stair_check.json'):
        if (name != 'stair_check.json' or stairs) and not (raw/name).exists():
            warnings.append(dict(kind='navigation_check_not_supplied',detail=name))
    for document in (root/'scene_description.md',raw/'index.html'):
        if not document.exists():
            continue
        text = document.read_text()
        refs = re.findall(r'\]\(([^)]+)\)',text) if document.suffix == '.md' else re.findall(r'(?:href|src)="([^"]+)"',text)
        for ref in refs:
            url = urlsplit(ref.strip('<>'))
            if url.scheme or not url.path:
                continue
            exists(document.parent, unquote(url.path))
    require(bool(manifest.get('checksums')), 'empty_checksums')
    base = local(raw, manifest.get('checksum_path_base','..'))
    for ref,digest in manifest.get('checksums',{}).items():
        path = exists(base, ref)
        require(path.is_file() and sha(path) == digest, 'checksum_mismatch', ref)
    for path in [root/'floor_plan.png', root/'scene_description.md'] + [raw/f for f in (wall_name,real_name,files.get('stairs','stairs.json'))]:
        require(str(path.resolve().relative_to(base)) in manifest.get('checksums',{}), 'critical_file_not_checksummed', str(path.relative_to(root)))
    warnings.append(dict(kind='manual_review_required',detail='Review RGB, scene boundary decisions, stair direction, free endpoints, local and cross-floor paths separately. Structural pass is not scene acceptance.'))
    return dict(scene_id=scene['scene_id'], structural_pass=not errors, errors=errors, warnings=warnings,
                counts=dict(wall_segments=len(walls['segments']),wall_real_segments=len(real['segments']),stairs=len(stairs),frames=len(frames)),
                scope='File, provenance and 2-D geometry consistency only; no visual or navigation certification.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene_directory',type=Path)
    parser.add_argument('--tolerance',type=float,default=.005)
    args = parser.parse_args()
    if not 0 < args.tolerance <= .02:
        parser.error('tolerance must be in (0, .02] metres')
    try:
        report = validate(args.scene_directory,args.tolerance)
    except (OSError,ValueError,KeyError,TypeError,IndexError) as exc:
        report = dict(structural_pass=False,errors=[dict(kind='invalid_archive',detail=str(exc))])
    print(json.dumps(report,ensure_ascii=False,indent=2))
    raise SystemExit(0 if report['structural_pass'] else 1)


if __name__ == '__main__':
    main()
