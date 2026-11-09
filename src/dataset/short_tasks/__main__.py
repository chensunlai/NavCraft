"""Build a small reproducible set; no model calls and no benchmark-time generation."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import traceback

from PIL import Image

from src.common.io import write_json
from src.common.progress import Progress
from src.harness.memory import MemoryStore
from src.harness.observations import Observations
from src.common.coordinates import validate_map


KINDS = {"ground": "观察目标并落图", "recognize": "跨视角复认与维护",
         "revisit": "利用记忆短程回访", "recover": "局部受阻后调整"}


def read(path):
    return json.loads(Path(path).read_text())


def validate_case(directory):
    """Check frozen input integrity, separately from semantic/physical acceptance."""
    directory = Path(directory)
    task = read(directory / "task.json")
    snapshot = directory / task["snapshot"]
    state = read(snapshot / "state.json")
    workspace = snapshot / state["workspace"]
    transform = read(workspace / "scene/map.json")
    validate_map(transform)
    observations = Observations(workspace, transform)
    if state.get("history") or state.get("session_file") or state.get("target_frames"):
        raise ValueError("Short cases start without a previous model answer or navigation goal")
    observations.get(state["current_frame"])
    for frame in observations.items:
        row = observations.image(frame)
        with Image.open(workspace / row["path"]) as image:
            image.verify()
    memory = MemoryStore(workspace, snapshot / state["private_dir"])
    result = memory.read(ids=[entity["id"] for entity in memory.entities])
    if "error" in result:
        raise ValueError(result["error"])
    label = read(directory / task["label"])
    if task["kind"] in {"ground", "recognize"}:
        for frame, relative in label["evidence_masks"].items():
            observations.get(frame)
            if not (directory / relative).is_file():
                raise ValueError(f"Missing private semantic evidence: {relative}")
    else:
        if not label.get("reference_replayed") or not label.get("positions"):
            raise ValueError("Motion cases need an independently replayed route and accepted positions")
        if task["kind"] == "recover" and not label.get("initial_blocked"):
            raise ValueError("A collision alone is not a recovery case")
    for path in workspace.rglob("*"):
        if path.suffix == ".npy" or path.name.startswith("depth_"):
            raise ValueError(f"Private sampling data in public workspace: {path}")
    read(directory / task["episode"])
    return {"input_integrity": True, "images": len(observations.items),
            "initial_entities": len(memory.entities)}


def render_index(output, tasks, status):
    cards = []
    for task in tasks:
        directory = output / task["path"]
        base = directory.parent
        state = read(base / task["snapshot"] / "state.json")
        workspace = base / task["snapshot"] / state["workspace"]
        rows = [json.loads(line) for line in (workspace / "observations/index.jsonl").read_text().splitlines()]
        frames = {row["frame"]: row for row in rows}
        pictures = [("初始观察", workspace / frames[state["current_frame"]]["path"])]
        for item in task.get("images", []):
            path = workspace / frames[item]["path"] if item in frames else base / item
            pictures.append(("指令附图", path))
        memory = read(workspace / "memory/entities.json")
        evidence = list(dict.fromkeys(frame for entity in memory for frame in entity.get("evidence", [])
                                      if frame != state["current_frame"]))
        for frame in evidence[:2]:
            pictures.append((f"已有记忆证据 {frame}", workspace / frames[frame]["path"]))
        maps = sorted((workspace / "observations/maps").glob("m_*.png"))
        if maps:
            pictures.append(("已保存地图（图中位姿对应绘图时刻）", maps[-1]))
        figures = "".join(f'<figure><img loading="lazy" src="{html.escape(p.relative_to(output).as_posix(), quote=True)}">'
                          f'<figcaption>{html.escape(title)}</figcaption></figure>' for title, p in pictures)
        label = read(base / task["label"])
        # This page is for human inspection; it is never added to model workspace.
        cards.append(f'<article><h2>{html.escape(task["id"])} · {KINDS[task["kind"]]}</h2>'
                     f'<p>{html.escape(task.get("instruction_zh", task["instruction"]))}</p>'
                     f'<div class="images">{figures}</div>'
                     f'<details><summary>模型请求与预算</summary><pre>{html.escape(json.dumps({k:task[k] for k in ("instruction","limits")},ensure_ascii=False,indent=2))}</pre></details>'
                     f'<details><summary>初始公开记忆</summary><pre>{html.escape(json.dumps(memory,ensure_ascii=False,indent=2))}</pre></details>'
                     f'<details><summary>私有验收标签（模型不可见）</summary><pre>{html.escape(json.dumps(label,ensure_ascii=False,indent=2))}</pre></details>'
                     f'<a href="{html.escape(task["path"],quote=True)}">冻结任务文件</a></article>')
    counts = Counter(task["kind"] for task in tasks)
    text = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>短任务构建检查</title><style>body{{font:16px/1.6 system-ui;margin:32px auto;max-width:1100px;padding:0 20px;background:#f7f7f8;color:#202124}}article{{background:white;border:1px solid #ddd;border-radius:12px;margin:24px 0;padding:20px}}h1,h2{{line-height:1.3}}h2{{font-size:21px}}.images{{display:flex;gap:12px;flex-wrap:wrap}}figure{{margin:0;max-width:48%}}img{{width:100%;height:auto}}pre{{white-space:pre-wrap;word-break:break-word;background:#f6f6f6;padding:12px;font-size:13px}}summary{{cursor:pointer}}a{{color:#006e62}}</style>
<h1>四类短任务 · 固定状态回归样例</h1><p>状态：{html.escape(status)}。{'；'.join(f'{name} {counts[kind]}' for kind,name in KINDS.items())}。</p>
<p>本页包含私有标签，仅供数据核查。参考动作来自程序辅助真实重放，不是模型示范；尚未报告模型通过率。</p>
{''.join(cards)}</html>'''
    (output / "index.html").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("all", "memory", "motion", "check"), default="all")
    parser.add_argument("--per-kind", type=int, default=4)
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if args.per_kind != 4:
        parser.error("This fixed pilot contains four cases per kind")
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256((source / "snapshots.jsonl").read_bytes()).hexdigest()
    config = {"source": str(source), "source_snapshots_sha256": source_hash,
              "per_kind": args.per_kind, "gpu": args.gpu, "model_calls": 0}
    config_path = output / "build_config.json"
    if config_path.exists() and read(config_path) != config:
        raise ValueError("An existing collection has different inputs; choose a new output directory")
    write_json(config_path, config)
    existing = [dict(read(p), path=p.relative_to(output).as_posix()) for p in sorted(output.glob("cases/*/task.json"))]
    tasks = {task["id"]: task for task in existing}
    kinds = list(KINDS) if args.phase in {"all", "check"} else list(KINDS)[:2] if args.phase == "memory" else list(KINDS)[2:]
    keys = [f"{kind}-{i:02d}" for kind in kinds for i in range(1, 5)]
    progress = Progress("Short cases", keys, {key: "finished" for key in tasks if key in keys})

    def save_status(status, error=None):
        rows = list(tasks.values())
        counts = dict(Counter(row["kind"] for row in rows))
        write_json(output / "manifest.json", {"tasks": rows, "counts": counts,
                   "source": str(source), "reference_source": "program-assisted, not model demonstrations"})
        write_json(output / "status.json", {"status": status, "counts": counts, "planned": 16,
                   "model_calls": 0, "pid": os.getpid(), "updated_at": datetime.now(timezone.utc).isoformat(),
                   **({"error": error} if error else {})})
        render_index(output, rows, status)

    def accepted(task):
        checks = validate_case(output / task["path"].removesuffix("task.json"))
        write_json(output / Path(task["path"]).parent / "private/input_check.json", checks)
        tasks[task["id"]] = task
        progress.set(task["id"], "finished")
        save_status("running")

    save_status("running")
    try:
        for task in existing:
            validate_case(output / Path(task["path"]).parent)
        if args.phase != "check":
            if args.phase in {"all", "memory"} and any(f"{k}-{i:02d}" not in tasks for k in kinds[:2] for i in range(1, 5)):
                from .memory_cases import build_memory_cases
                build_memory_cases(source, output, per_kind=4, on_case=accepted)
            if args.phase in {"all", "motion"} and any(f"{k}-{i:02d}" not in tasks for k in ("revisit", "recover") for i in range(1, 5)):
                from .motion_cases import build_motion_cases
                for key in keys:
                    if key not in tasks:
                        progress.set(key, "running")
                build_motion_cases(source, output, per_kind=4, gpu=args.gpu, on_case=accepted)
        missing = [key for key in keys if key not in tasks]
        for key in missing:
            progress.set(key, "partial")
        status = "completed" if len(tasks) == 16 else "partial" if missing else "stage_completed"
        save_status(status)
        print(output / "index.html", flush=True)
    except Exception as error:
        (output / "error.txt").write_text(traceback.format_exc())
        save_status("failed", f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
