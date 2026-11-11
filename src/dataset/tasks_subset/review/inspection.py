"""Static inspection page for the collection owner; contains private labels."""
from html import escape
import json
from pathlib import Path
from urllib.parse import quote

from src.common.io import read_json


def _detail(title, value):
    return f"<details><summary>{escape(title)}</summary><pre>{escape(json.dumps(value, indent=2, ensure_ascii=False))}</pre></details>"


def _map(state, frames):
    def xy(pos):
        return tuple(float(p)*280 for p in pos)
    trajectory = " ".join("%g,%g" % xy(p["pos"]) for p in state.get("public_trajectory", []))
    dots = []
    for node, keys in state["node_frames"].items():
        if keys:
            x, y = xy(frames[keys[0]]["public_pose"]["pos"])
            dots.append(f'<circle cx="{x}" cy="{y}" r="4"/><text x="{x+5}" y="{y}">{escape(node)}</text>')
    x, y = xy(frames[state["current_frame"]]["public_pose"]["pos"])
    return (f'<svg viewBox="0 0 280 280" role="img" aria-label="Public prefix map" width="280">'
            f'<rect width="280" height="280" fill="#edf1f5"/><polyline points="{trajectory}" fill="none" '
            f'stroke="#245b91" stroke-width="2"/>{"".join(dots)}<circle cx="{x}" cy="{y}" r="5" fill="#bf392d"/></svg>')


def write_inspection(root, report, bundle_path=None):
    root = Path(root).resolve()
    bundle = read_json(bundle_path or root / "private/bundle.json")
    states = {s["id"]: s for s in bundle["states"]}
    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>任务样本检查</title>',
             '<style>body{font:16px system-ui;max-width:80rem;margin:2rem auto;padding:1rem}article{border-top:1px solid #aaa}'
             'figure{display:inline-block;margin:.5rem}img{max-width:340px;max-height:260px}pre{white-space:pre-wrap}'
             'td,th{padding:.3rem .6rem;text-align:left}</style>',
             '<h1>任务样本检查</h1><p>仅供检查者：页面包含私有答案、来源与试走记录。</p>',
             f'<p>{escape(report["scene_id"])}</p>']
    columns = ("constructor_registered", "proposed", "produced", "structural", "physical", "visual", "model_tried", "training_ready")
    parts.append("<table><tr><th>题型</th>" + "".join(f"<th>{x}</th>" for x in columns) + "</tr>")
    for kind, values in report["coverage"].items():
        parts.append(f"<tr><td>{escape(kind)}</td>" + "".join(f"<td>{values[x]}</td>" for x in columns) + "</tr>")
    parts.append("</table>")
    for result in report["cases"]:
        path = root / result["path"]
        parts.append(f"<article><h2>{escape(path.parent.name)}</h2>")
        if result.get("invalidated"):
            parts.append(f'<p>已失效：{escape(result["invalidated"])}</p>')
        if result.get("error"):
            parts.append(f'<p>检查问题：{escape(result["error"])}</p>')
        try:
            task = read_json(path)
            parts.append(f'<pre>{escape(task["instruction"])}</pre>')
            state = states[task["provenance"]["state_id"]]
            if result["structural"]:
                parts.append(_map(state, bundle["frames"]))
                for image in task["images"]:
                    src = (path.parent / image["path"]).relative_to(root).as_posix()
                    parts.append(f'<figure><img loading="lazy" src="{quote(src)}"><figcaption>{escape(image["caption"])}</figcaption></figure>')
                parts.append(_detail("私有标签", read_json(path.parent / task["label"])))
                parts.append(_detail("构造来源", read_json(path.with_name("construction.json"))))
        except (OSError, KeyError, ValueError, TypeError) as exc:
            parts.append(f"<p>{escape(str(exc))}</p>")
        parts.append("</article>")
    parts.append("</html>")
    output = root / "index.html"
    output.write_text("\n".join(parts), encoding="utf-8")
    return output
