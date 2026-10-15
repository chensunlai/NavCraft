"""Render the verified goal catalog and re-score audit without running a model.

Uses the previous audit's saved NavMesh/grid; does not initialize Habitat sensors
or import the previous generator (which has executable top-level code).
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import html
import json
from pathlib import Path
import shutil

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle

REPO = Path(__file__).resolve().parents[3]
DEFAULT_AUDIT = REPO / 'outputs/goal-completion-20261015'
DEFAULT_PREVIOUS = REPO / 'outputs/goal-coverage-20261015'
SOURCE_LABELS = {
    'official_instance_viewpoints': '原始 ObjectNav 对应实例的完整视点',
    'rendered_custom_viewpoints': '真实渲染核验的自建实例视点',
}


def load(path):
    if not path.is_file():
        raise FileNotFoundError(f'缺少必要输入：{path}')
    return json.loads(path.read_text())


def state(value):
    return '通过' if value is True else '未通过' if value is False else '不适用 / 未评分'


def fmt_distance(value):
    return '不适用' if value is None else f'{value:.3f} m'


def compute_mask(pathfinder, points, cells, shape, positions, threshold):
    import habitat_sim
    query = habitat_sim.MultiGoalShortestPath()
    query.requested_ends = np.asarray(positions, dtype=np.float32)
    distances = np.full(len(points), np.inf)
    for i, point in enumerate(points):
        query.requested_start = np.asarray(point, dtype=np.float32)
        if pathfinder.find_path(query):
            distances[i] = query.geodesic_distance
    mask = np.zeros(shape, dtype=bool)
    mask[cells[:, 0], cells[:, 1]] = distances < threshold
    return mask, distances


def draw_object(ax, obj, target=False):
    for primitive in obj['primitives']:
        style = dict(facecolor='#fdc46c' if target else '#faf6ed',
                     edgecolor='#b66b05' if target else '#c5b9a8',
                     linewidth=1.5 if target else .55, alpha=.85 if target else .65,
                     zorder=4 if target else 1)
        if primitive['type'] == 'rectangle':
            x, y = primitive['center']
            width, height = primitive['size']
            patch = Rectangle((x - width / 2, y - height / 2), width, height, **style)
            patch.set_transform(transforms.Affine2D().rotate_deg_around(
                x, y, primitive.get('angle_deg', 0)) + ax.transData)
            ax.add_patch(patch)
        elif primitive['type'] == 'polygon':
            ax.add_patch(Polygon(primitive['points'], **style))


def draw(ax, task, geometry, maps, *, zoom=False, small=False):
    bounds, rooms, walls, objects = maps
    ax.set_facecolor('#fafafa')
    for room in rooms:
        ax.fill(*np.asarray(room['polygon']).T, color='#f1f2f4', zorder=0)
    ax.imshow(np.where(geometry['nav'], 1., np.nan), extent=bounds, origin='lower',
              cmap=ListedColormap(['#cdd3da']), vmin=0, vmax=1, interpolation='nearest', zorder=1)
    for obj in objects.values():
        draw_object(ax, obj)
    if task['positions']:
        ax.imshow(np.where(task['mask'], 1., np.nan), extent=bounds, origin='lower',
                  cmap=ListedColormap(['#30b86a']), vmin=0, vmax=1, alpha=.55,
                  interpolation='nearest', zorder=2)
        if task['old_mask'].any():
            ax.contour(geometry['x'], geometry['y'], task['old_mask'].astype(float),
                       levels=[.5], colors=['#a46bd3'], linewidths=1.6,
                       linestyles='dashed', zorder=3)
    elif task.get('scope_room'):
        room = next(room for room in rooms if room['id'] == task['scope_room'])
        ax.add_patch(Polygon(room['polygon'], fill=False, edgecolor='#30b86a',
                             linewidth=2.5, hatch='//', alpha=.5, zorder=3))
    for wall in walls:
        ax.plot(*np.array([wall['start'], wall['end']]).T, color='#343c48', lw=1.15, zorder=5)
    for room in rooms:
        ax.text(*room['label_position'], room['id'] + ' ' + room['name'], ha='center', va='center',
                fontsize=6 if small else 7, color='#626975', zorder=7, clip_on=True,
                bbox=dict(facecolor='white', alpha=.75, edgecolor='none', pad=.3))
    target_xy = []
    for identifier in task['object_ids']:
        obj = objects[identifier]
        draw_object(ax, obj, True)
        centers = [primitive['center'] for primitive in obj['primitives'] if 'center' in primitive]
        if centers:
            center = np.mean(centers, axis=0)
            target_xy.append(center)
            ax.text(*center, identifier, ha='center', fontsize=7, fontweight='bold', zorder=8)
    if task['positions']:
        points = np.asarray(task['positions'])
        ax.scatter(points[:, 0], -points[:, 2], s=9 if len(points) > 30 else 24,
                   c='#1863ba', alpha=.65, linewidths=0, zorder=8)
        old_points = np.asarray(task['old_positions'])
        ax.scatter(old_points[:, 0], -old_points[:, 2], s=40, c='#a46bd3',
                   marker='o', edgecolors='white', linewidths=.7, zorder=9)
        if not small:
            for i, point in enumerate(old_points):
                ax.annotate(f'旧V{i + 1}', (point[0], -point[2]), xytext=(4, 4),
                            textcoords='offset points', fontsize=7, color='#704699', zorder=10)
    for event in task.get('events', []):
        if event['kind'] == 'portal':
            segment = np.asarray(event['segment_xz']) * [1, -1]
            ax.plot(*segment.T, c='#b32c97', lw=3, zorder=6)
            if not small:
                ax.text(*segment.mean(axis=0), event['id'], color='#92207a', fontsize=7, zorder=11)
    if task.get('start') is not None:
        point = task['start']
        ax.scatter(point[0], -point[2], marker='^', c='#525966', s=45, zorder=9)
    if task.get('endpoint') is not None:
        point = task['endpoint']
        ax.scatter(point[0], -point[2], marker='X', c='#e42c49', s=65,
                   edgecolors='white', linewidths=.6, zorder=10)
    ax.set_aspect('equal')
    ax.set_xlim(bounds[:2])
    ax.set_ylim(bounds[2:])
    ax.grid(alpha=.15, lw=.5)
    ax.tick_params(labelsize=7)
    if zoom and target_xy:
        xy = np.asarray(target_xy)
        if task.get('endpoint') is not None:
            xy = np.vstack([xy, [task['endpoint'][0], -task['endpoint'][2]]])
        if task['positions']:
            xy = np.vstack([xy, np.asarray(task['positions'])[:, [0, 2]] * [1, -1]])
        lower, upper = xy.min(axis=0) - .8, xy.max(axis=0) + .8
        ax.set_xlim(max(bounds[0], lower[0]), min(bounds[1], upper[0]))
        ax.set_ylim(max(bounds[2], lower[1]), min(bounds[3], upper[1]))
    if not small:
        ax.set_xlabel('Habitat X / m', fontsize=9)
        ax.set_ylabel('Habitat −Z / m', fontsize=9)


def render_sequence(tasks, geometry, maps, output):
    legend = [
        Patch(facecolor='#fdc46c', edgecolor='#b66b05', label='目标物体示意框（非精确碰撞边界）'),
        Line2D([], [], marker='o', color='none', markerfacecolor='#1863ba', markersize=6, label='补全后的实例视点'),
        Patch(facecolor='#30b86a', alpha=.55, label='新视点集合下测地距离 <1 m（仍需主动提交）'),
        Line2D([], [], color='#a46bd3', ls='--', label='旧证据拍摄站点的 <1 m 范围'),
        Line2D([], [], marker='X', color='none', markerfacecolor='#e42c49', markersize=7, label='旧运行真实终点'),
        Line2D([], [], marker='^', color='none', markerfacecolor='#525966', markersize=6, label='旧运行本轮起点'),
        Patch(facecolor='#cdd3da', label='当前机器人配置下的 NavMesh'),
        Line2D([], [], color='#b32c97', lw=2, label='指定穿越门线；顺序和途经点另行评分'),
    ]
    for task in tasks:
        figure, axes = plt.subplots(1, 2, figsize=(14, 6.8))
        draw(axes[0], task, geometry, maps)
        draw(axes[1], task, geometry, maps, zoom=True)
        axes[0].set_title('全局位置', fontsize=11)
        axes[1].set_title('目标区域放大', fontsize=11)
        if task['positions']:
            detail = (f"旧 {len(task['old_positions'])} 点 → 新 {len(task['positions'])} 点；"
                      f"阈值仍为 <1 m；栅格面积约 {task['acceptance_area_m2']:.2f} m²\n"
                      f"旧终点距离：{fmt_distance(task['reevaluation']['old_distance'])} → "
                      f"{fmt_distance(task['reevaluation']['new_distance'])}；来源：{task['source_label']}")
        else:
            detail = '探索任务：绿色斜线表示原探索范围，没有到达终点或成功视点'
        figure.suptitle(task['title'] + '\n' + detail, fontsize=11)
        figure.legend(handles=legend, loc='lower center', ncol=2, fontsize=8,
                      frameon=False, bbox_to_anchor=(.5, .005))
        figure.subplots_adjust(left=.05, right=.99, top=.80, bottom=.21, wspace=.17)
        figure.savefig(output / task['image'], dpi=145)
        plt.close(figure)
    figure, axes = plt.subplots(3, 4, figsize=(17, 13))
    for task, ax in zip(tasks, axes.flat):
        draw(ax, task, geometry, maps, small=True)
        title = task['short_title']
        count = f"{len(task['old_positions'])} → {len(task['positions'])} 个视点" if task['positions'] else '探索，不评到达'
        ax.set_title(title + '\n' + count, fontsize=9)
    figure.suptitle('12轮目标信息补全与重新核验\n蓝点/绿区：新集合；紫虚线：旧范围；橙框：目标；红叉：原运行终点', fontsize=16)
    figure.tight_layout(rect=[0, 0, 1, .94])
    figure.savefig(output / 'sequence-overview.png', dpi=125)
    plt.close(figure)


def task_description(task):
    if task['group'] == '12轮':
        review = task['reevaluation']
        if not task['positions']:
            return ['客餐厅探索范围，无物体到达目标；不计导航成功率。']
        meta = task['target_info']
        result = [f"目标：{meta['label']}；地图实例 {', '.join(meta['object_ids'])}；"
                  f"语义实例 {', '.join(map(str, meta['semantic_instance_ids']))}；房间 {meta['room_id']}。",
                  f"视点来源：{task['source_label']}。旧 {len(task['old_positions'])} 点，新 {len(task['positions'])} 点；"
                  f"原终点距离 {fmt_distance(review['old_distance'])} → {fmt_distance(review['new_distance'])}。",
                  f"旧完整判定：{state(review['old_passed'])}；新完整判定：{state(review['new_passed'])}。"
                  '完整判定还包括主动提交和要求的路线，不能只看绿色位置范围。',
                  '绿色是新视点集合的 1 米测地距离邻域，可能延伸到目标房间外；不额外约束终点朝向、当前可见性或房间归属。']
        result.append(f"主动提交：{state(review.get('stopped'))}；距离达标：{state(review.get('arrival'))}；"
                      f"终点位于目标房间示意边界内：{state(review.get('endpoint_in_target_room'))}。")
        if review.get('endpoint_in_target_room') is False:
            result.append('该轮终点仍在目标房间外；即使距离判定通过，也不能据此声称已进入目标房间。')
        route = review.get('route') or {}
        if route.get('events'):
            result.append('路线核验：' + '；'.join(
                f"{event['id']} {state(event['passed'])}"
                + (f"（轨迹索引 {event['trajectory_index']}）" if event.get('trajectory_index') is not None else '')
                for event in route['events']) + '。')
            if any(event['id'] == 'dining_table' for event in route['events']):
                result.append('餐桌途经点也替换为 O175 的完整自建视点集合，保留原 1.5 米途经阈值。')
        return result
    info = task['target_info']
    result = [f"目标：{info['target_name_zh']}；实例：{', '.join(info['target_instance_ids']) or '无物体实例，使用指定终点'}。",
              f"来源：{info['source']['kind']}；{info['source']['file']}。",
              f"成功视点 {info['viewpoint_count']} 个；评分目标位置 {info['goal_point_count']} 个；"
              f"本项目距离阈值 <{info['success_distance_m']:g} m，并要求主动提交。"]
    if info.get('target_rooms'):
        result.append('目标位置解释：' + '；'.join(
            f"{room.get('instance_id', '终点')}：{room.get('room_id') or '未分配房间'} "
            f"{room.get('room_name_zh') or ''}（{room.get('basis', '未提供依据')}）" for room in info['target_rooms']))
    result.extend(info.get('information_limitations', []))
    if info.get('image_goal_camera'):
        camera = info['image_goal_camera']
        result.append('图像指令相机位置：' + str(camera['position']) +
                      '；图中紫色菱形仅用于生成目标照片，不替代蓝色成功视点集合。')
    return result


def write_pages(tasks, output, catalog):
    official = sum(item['source_type'] == 'official_instance_viewpoints' for item in catalog['targets'].values())
    custom = len(catalog['targets']) - official
    intro = (f'逐项核对 48 个任务：12 轮自建任务、6 个 ObjectNav、3 个自建 HM3D 指令任务及 27 个 GOAT 子目标。'
             f'12 轮涉及 9 个不同目标，其中 {official} 个采用对应官方实例视点、{custom} 个采用真实渲染核验的自建视点。'
             '没有调用模型生成目标或地图；原运行轨迹和原评分保持归档，新结果单独复评。')
    limits = [
        '自建视点以实际渲染、语义实例和几何规则筛选；采样通过不等于无限连续空间中的完备性证明。',
        '绿色区域按保存的相同 NavMesh 在 0.10 米网格计算最短可行走距离，边界是栅格近似。',
        '12轮保留原1米距离阈值。新旧差异来自视点集合、相关途经点补全和已确认的门方向评分修正，不代表新运行成绩。',
        '即使种子位于目标房间且构造时看见目标，其1米邻域仍可能跨出房间；最终提交位置不自动具有可见性或房间约束。',
        '其余36张图沿用上轮图像和原本评分范围；逐项身份、来源、图片相机及限制已重新核对。3条指令属于自建数据，不能称官方VLN-CE。',
        '本次只覆盖Qa场景所选开发任务，不代表全数据集审计。',
    ]
    md = ['# 导航目标信息补全与逐项核验', '', intro, '',
          '[交互图册](index.html) · [目标目录](catalog.json) · [历史复评](reevaluation.json) · [官方来源核对](official_audit.json)',
          '', '![12轮补全总览](sequence-overview.png)', '', '## 核验边界', '', *['- ' + line for line in limits], '']
    cards = []
    for task in tasks:
        description = task_description(task)
        md.extend(['## ' + task['title'], '', task['instruction'], '', *[text + '\n' for text in description],
                   f"![{task['title']}]({task['image']})", ''])
        preview = task.get('evidence_preview')
        if preview:
            md.extend(['真实候选视点的 RGB 核验图（构造阶段，非旧运行终点观察）：', '',
                       f"![{task['title']}的真实证据]({preview})", ''])
        details = {key: value for key, value in task['target_info'].items() if key != 'positions'}
        if task.get('reevaluation'):
            details['reevaluation'] = task['reevaluation']
        details['accepted_positions_file'] = 'catalog.json' if task['group'] == '12轮' else 'gallery.json'
        cards.append(f'<article data-group="{html.escape(task["group"])}"><h2>{html.escape(task["title"])}</h2>'
                     f'<p class="instruction">{html.escape(task["instruction"])}</p>'
                     + ''.join('<p>' + html.escape(text) + '</p>' for text in description)
                     + f'<a href="{task["image"]}" target="_blank"><img loading="lazy" src="{task["image"]}" alt="{html.escape(task["title"])}"></a>'
                     + (f'<p>真实候选视点 RGB 核验图（构造阶段，非旧运行终点观察）：</p><a href="{preview}" target="_blank"><img style="max-width:500px" loading="lazy" src="{preview}" alt="真实视点证据"></a>' if preview else '')
                     + '<details><summary>逐项源信息与核验记录</summary><pre>'
                     + html.escape(json.dumps(details, ensure_ascii=False, indent=2)) + '</pre></details></article>')
    (output / 'README.md').write_text('\n'.join(md))
    buttons = ''.join(f'<button onclick="filter(\'{group}\')">{group}</button>'
                      for group in ['全部', '12轮', 'ObjectNav', '指令导航', 'GOAT'])
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>导航目标信息补全</title><style>body{max-width:1450px;margin:24px auto;padding:0 18px;font-family:system-ui,sans-serif;color:#253046;background:#f5f6f8}header,article{background:white;padding:20px;border-radius:12px;margin:16px 0}h1{font-size:26px}h2{font-size:20px}p,li{line-height:1.7}img{width:100%;height:auto}nav{position:sticky;top:0;background:#f5f6f8;padding:12px;z-index:20}button{padding:9px 18px;margin:4px;border:1px solid #ccc;border-radius:8px;cursor:pointer}pre{overflow:auto;font-size:12px}.instruction{border-left:3px solid #367fb4;padding:8px 14px;background:#f4f8fc}</style><header><h1>导航目标信息补全与逐项核验</h1><p>'''
    page += html.escape(intro) + '</p><ul>' + ''.join('<li>' + html.escape(text) + '</li>' for text in limits)
    page += '</ul><a href="catalog.json">目标目录</a> · <a href="reevaluation.json">历史复评</a> · <a href="official_audit.json">来源核对</a> · <a href="README.md">Markdown图册</a><img src="sequence-overview.png" alt="12轮总览"></header>'
    page += '<nav>' + buttons + '</nav>' + ''.join(cards)
    page += "<script>function filter(group){document.querySelectorAll('article').forEach(x=>x.hidden=group!=='全部'&&x.dataset.group!==group)}</script></html>"
    (output / 'index.html').write_text(page)


def build(previous, audit, output):
    import habitat_sim
    old = load(previous / 'summary.json')
    catalog = load(audit / 'catalog.json')
    reevaluation = load(audit / 'reevaluation.json')
    official = load(audit / 'official_audit.json')
    reviews = {entry['round']: entry for entry in reevaluation['rounds']}
    officials = {entry['id']: entry for entry in official['tasks']}
    if len(old['tasks']) != 48 or len(reviews) != 12 or len(officials) != 36:
        raise ValueError('图册需要原48个任务、12轮复评与36条来源核对记录')
    raw = REPO / 'data/scene_maps/hm3d/val/00829-QaLdnwvtxbs/raw'
    maps = (old['grid_bounds_x_minus_z'], load(raw / 'rooms.json')['rooms'],
            load(raw / 'walls.json')['segments'], {item['id']: item for item in load(raw / 'objects.json')['objects']})
    with np.load(previous / 'objnav-grid.npz') as data:
        geometry = {name: data[name] for name in ('x', 'y', 'nav', 'points', 'cells')}
        old_masks = {task['id']: data[task['id']] for task in old['tasks'] if task['group'] == '12轮'}
    pathfinder = habitat_sim.PathFinder()
    navmesh = previous / 'objnav.navmesh'
    if not navmesh.is_file() or not pathfinder.load_nav_mesh(str(navmesh)):
        raise FileNotFoundError(f'无法加载原审计NavMesh：{navmesh}')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'images').mkdir(exist_ok=True)
    (output / 'evidence').mkdir(exist_ok=True)
    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.is_file():
        plt.rcParams['font.family'] = FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    cache, tasks = {}, []
    for original in old['tasks']:
        task = deepcopy(original)
        task['image'] = f"images/{task['id']}.png"
        if task['group'] == '12轮':
            number = int(task['id'].split('-')[1])
            task['reevaluation'] = reviews[number]
            task['old_positions'] = deepcopy(task['positions'])
            task['old_events'] = deepcopy(task.get('events', []))
            for event in task.get('events', []):
                if event['kind'] == 'waypoint' and event['id'] == 'dining_table':
                    event['positions'] = catalog['targets']['O175']['positions']
                    event['target_object_id'] = 'O175'
                    event['viewpoint_source'] = catalog['targets']['O175']['source_type']
            task['old_mask'] = old_masks[task['id']]
            if task['object_ids']:
                identifier = task['object_ids'][0]
                target = catalog['targets'][identifier]
                task.update(target_info=target, positions=target['positions'], object_ids=target['object_ids'],
                            source_label=SOURCE_LABELS[target['source_type']],
                            source=target['source'], seed_provenance=target['source_type'],
                            object_label=target['label'],
                            reachable_seed_count=reviews[number]['reachable_viewpoint_count'],
                            recomputed_end_distance_m=reviews[number]['new_distance'],
                            title=f"第{number:02d}轮 · {target['label']}（{identifier}）",
                            short_title=f"第{number:02d}轮 · {target['label']}\n{identifier}")
                for key in ('point_diagnostics', 'official_bed_seed_coverage',
                            'acceptance_vs_official_1m_area_ratio', 'overlay_official_bed'):
                    task.pop(key, None)
                evidence = next((entry for entry in target.get('evidence_refs', []) if entry.get('accepted')), None)
                if evidence:
                    source_file = Path(target['source'])
                    if not source_file.is_absolute():
                        source_file = REPO / source_file
                    evidence_source = source_file.parent / evidence['path']
                    if not evidence_source.is_file():
                        raise FileNotFoundError(f'缺少真实视点核验图片：{evidence_source}')
                    task['evidence_preview'] = 'evidence/' + evidence_source.name
                    shutil.copy2(evidence_source, output / task['evidence_preview'])
                if identifier not in cache:
                    cache[identifier] = compute_mask(pathfinder, geometry['points'], geometry['cells'],
                        geometry['nav'].shape, task['positions'], task['radius'])
                    print(json.dumps({'target': identifier, 'viewpoints': len(task['positions']),
                                      'accepted_cells': int(cache[identifier][0].sum())}), flush=True)
                task['mask'] = cache[identifier][0]
                task['acceptance_area_m2'] = round(float(task['mask'].sum()) * old['grid_resolution_m'] ** 2, 2)
            else:
                task.update(target_info={}, source_label='探索范围', title='第01轮 · 客餐厅探索（R05）',
                            short_title='第01轮 · 客餐厅探索', mask=np.zeros_like(geometry['nav']))
        else:
            target = officials[task['id']]
            task.update(target_info=target, title=task['id'] + ' · ' + target['target_name_zh'])
            source = previous / original['image']
            if not source.is_file():
                raise FileNotFoundError(f'缺少原逐任务图片：{source}')
            shutil.copy2(source, output / task['image'])
        tasks.append(task)
    render_sequence([task for task in tasks if task['group'] == '12轮'], geometry, maps, output)
    for name in ('catalog.json', 'reevaluation.json', 'official_audit.json'):
        if (audit / name).resolve() != (output / name).resolve():
            shutil.copy2(audit / name, output / name)
    write_pages(tasks, output, catalog)
    serial = [{key: value for key, value in task.items() if key not in ('mask', 'old_mask')}
              for task in tasks]
    (output / 'gallery.json').write_text(json.dumps({'scene': old['scene'], 'tasks': serial}, ensure_ascii=False, indent=2) + '\n')
    np.savez_compressed(output / 'completed-grid.npz', **geometry,
                        **{identifier: pair[0] for identifier, pair in cache.items()})
    from PIL import Image
    for task in tasks:
        with Image.open(output / task['image']) as image:
            image.verify()
        if task.get('evidence_preview'):
            with Image.open(output / task['evidence_preview']) as image:
                image.verify()
    validation = {'task_count': len(tasks), 'image_count': len(tasks), 'sequence_count': 12,
                  'official_and_synthetic_count': 36, 'all_images_readable': True,
                  'evidence_image_count': len({task['evidence_preview'] for task in tasks if task.get('evidence_preview')}),
                  'original_audit_modified': False, 'original_run_modified': False,
                  'model_requests': 0, 'grid_resolution_m': old['grid_resolution_m']}
    (output / 'gallery_validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(output), **validation}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument('--audit-dir', type=Path, default=DEFAULT_AUDIT)
    parser.add_argument('--output', type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    try:
        build(args.previous.resolve(), args.audit_dir.resolve(), args.output.resolve())
    except (FileNotFoundError, ValueError, KeyError) as error:
        parser.exit(1, f'目标图册生成失败：{error}\n')


if __name__ == '__main__':
    main()
