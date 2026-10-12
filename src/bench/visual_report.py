"""Offline visual-Harness results, costs and failure diagnosis from saved runs."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

from ..common.config import ROOT, resolve_path
from ..common.io import write_json
from ..harness.trace import TraceReader

CACHE = "cache/visual-harness-v1"


def read_object(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def number(value, default=0):
    return (
        value
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        else default
    )


def top_usage(usage: dict) -> dict:
    """Do not double-count per-item attribution embedded in service usage."""
    details = usage.get("input_tokens_details") or {}
    return {
        "input_tokens": number(usage.get("input_tokens")),
        "output_tokens": number(usage.get("output_tokens")),
        "cached_tokens": number(
            details.get("cached_tokens", usage.get("cached_tokens"))
        ),
    }


def trace_costs(path: Path) -> dict:
    result = {
        "requests": 0,
        "responses": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "response_latency_seconds": 0.0,
        "latency_measured_responses": 0,
        "compactions": 0,
        "image_prunes": 0,
    }
    pending = {}
    try:
        for row in TraceReader(path).events():
            event = row.get("event")
            if event == "model_request":
                result["requests"] += 1
                pending[row.get("request_id")] = number(row.get("timestamp"), None)
            elif event == "model_response":
                result["responses"] += 1
                for key, value in top_usage(row.get("usage") or {}).items():
                    result[key] += value
                start = pending.pop(row.get("request_id"), None)
                latency = number(row.get("latency_seconds"), None)
                if latency is None and start is not None:
                    end = number(row.get("timestamp"), None)
                    latency = max(0, end - start) if end is not None else None
                if latency is not None:
                    result["response_latency_seconds"] += latency
                    result["latency_measured_responses"] += 1
            elif event == "context_compacted":
                result["compactions"] += 1
            elif event == "context_pruned":
                result["image_prunes"] += 1
    except (OSError, ValueError):
        # A running trace may end in a partial JSON line. Completed prefix is useful.
        result["trace_incomplete"] = True
    return result


def category(result: dict, task: dict) -> str:
    metrics = result.get("metrics") or {}
    if (
        number(
            metrics.get("composite_success")
            if result.get("benchmark") == "goat"
            else metrics.get("success")
        )
        >= 1
    ):
        return "goal_finish"
    status = result.get("status", "unknown")
    if status == "model_incomplete":
        return "model_incomplete"
    if status in {"model_error", "context_compaction_error"}:
        return "service_error"
    if status in {
        "setup_error",
        "environment_error",
        "runtime_error",
        "evaluation_error",
        "cleanup_error",
        "worker_error",
        "worker_timeout",
    }:
        return "infrastructure"
    threshold = number(
        metrics.get("success_distance", task.get("success_distance")), None
    )
    distance = number(metrics.get("navigation_error"), None)
    if (
        result.get("benchmark") == "objnav"
        and status == "terminated"
        and threshold == 0.1
        and distance is not None
        and threshold <= distance <= 0.5
    ):
        return "precision_stop"
    return "navigation_failure"


def run_stage(config: dict, name: str) -> str:
    label = config.get("run", {}).get("name", name)
    for stage in ("smoke", "develop", "validate"):
        if f"visual-{stage}" in label:
            return stage
    return "other"


def collect_run(directory: Path) -> dict:
    config = read_object(directory / "config.json", {})
    selected = read_object(directory / "selection.json", [])
    if not isinstance(selected, list):
        raise ValueError(f"Invalid selection.json in {directory.name}")
    saved_summary = read_object(directory / "summary.json", {})
    diagnostic = read_object(directory / "diagnostic.json", {})
    task = config.get("task", {})
    family = {"objectnav": "objnav", "vlnce": "instruction", "goat": "goat"}.get(
        task.get("name"), "unknown"
    )
    manifest_ref = config.get("episodes", {}).get("manifest")
    manifest = read_object(Path(resolve_path(manifest_ref)), {}) if manifest_ref else {}
    goals = {
        row["id"]: len(row.get("goals", [])) for row in manifest.get("episodes", [])
    }
    aggregate = read_object(directory / "results.json", [])
    aggregate = (
        {row.get("episode_id"): row for row in aggregate if isinstance(row, dict)}
        if isinstance(aggregate, list)
        else {}
    )
    episodes, pending = [], []
    for index, episode_id in enumerate(selected):
        episode_dir = directory / "episodes" / f"{index:04d}"
        row = read_object(episode_dir / "result.json") or aggregate.get(episode_id)
        if not isinstance(row, dict) or row.get("episode_id") != episode_id:
            pending.append(
                {
                    "episode_id": episode_id,
                    "costs": trace_costs(episode_dir / "trace.jsonl"),
                    "state": (
                        "running_or_incomplete"
                        if episode_dir.exists()
                        else "not_started"
                    ),
                }
            )
            continue
        metrics = row.get("metrics") or {}
        costs = trace_costs(episode_dir / "trace.jsonl")
        agent = row.get("agent") or {}
        if isinstance(agent.get("usage"), dict):
            costs.update(top_usage(agent["usage"]))
        costs["requests"] = number(agent.get("requests"), costs["requests"])
        subtask = metrics.get("subtask_success") or []
        episodes.append(
            {
                "episode_id": episode_id,
                "status": row.get("status", "unknown"),
                "success": number(
                    metrics.get("composite_success")
                    if family == "goat"
                    else metrics.get("success")
                ),
                "category": category(row, task),
                "navigation_error": number(metrics.get("navigation_error"), None),
                "steps": number(metrics.get("steps")),
                "elapsed_seconds": number(row.get("elapsed_seconds")),
                "planned_goals": goals.get(
                    episode_id, number(row.get("planned_goals"), 1)
                ),
                "successful_subgoals": sum(number(value) for value in subtask),
                "route_constraint_completion": number(
                    metrics.get("route_constraint_completion"), None
                ),
                "costs": costs,
                "artifact": f"runs/{directory.name}/episodes/{index:04d}/result.json",
            }
        )
    all_finished = len(episodes) == len(selected)
    successful = sum(row["success"] for row in episodes)
    planned = sum(goals.get(identifier, 0) for identifier in selected)
    if not planned:
        planned = number(saved_summary.get("planned_subgoals"), None)
    sub_success = sum(row["successful_subgoals"] for row in episodes)
    costs = {
        key: sum(row["costs"].get(key, 0) for row in [*episodes, *pending])
        for key in (
            "requests",
            "responses",
            "input_tokens",
            "output_tokens",
            "cached_tokens",
            "response_latency_seconds",
            "latency_measured_responses",
            "compactions",
            "image_prunes",
        )
    }
    return {
        "run": f"runs/{directory.name}",
        "stage": run_stage(config, directory.name),
        "family": family,
        "model": config.get("model", {}).get("model", "unknown"),
        "harness_version": config.get("harness", {}).get("version", "unrecorded"),
        "diagnostic_note": diagnostic.get("reason", ""),
        "protocol": task.get("protocol", "unrecorded"),
        "success_distance": task.get("success_distance"),
        "selected": len(selected),
        "completed": len(episodes),
        "pending": pending,
        "complete": all_finished,
        "successful_episodes": successful,
        "sr_completed": successful / len(episodes) if episodes else None,
        "sr_selected": (
            successful / len(selected) if all_finished and selected else None
        ),
        "planned_subgoals": planned if family == "goat" else None,
        "successful_subgoals": sub_success if family == "goat" else None,
        "goat_micro_selected": (
            sub_success / planned if family == "goat" and planned else None
        ),
        "goat_chain_sr": (
            successful / len(episodes) if family == "goat" and episodes else None
        ),
        "categories": dict(Counter(row["category"] for row in episodes)),
        "costs": costs,
        "episodes": episodes,
    }


def collect_report(
    runs: list[Path], *, cache: Path | None = None, suites: list[Path] | None = None
) -> dict:
    cache = cache or ROOT / CACHE
    rows = [collect_run(path) for path in runs]
    suites = (
        suites
        if suites is not None
        else sorted((cache / "experiments").glob("*/experiment.json"))
    )
    experiments = []
    for path in suites:
        raw = read_object(path if path.is_file() else path / "experiment.json", {})
        # An acceptance flag is necessary but not sufficient: re-check the saved evidence.
        from .visual import acceptance_qualifies

        passed = bool(
            raw.get("acceptance_passed")
            and raw.get("stage") == "validate"
            and not raw.get("diagnostic")
            and acceptance_qualifies(raw.get("runs", []))
        )
        experiments.append(
            {"path": str(path), "stage": raw.get("stage"), "acceptance_passed": passed}
        )
    summary = read_object(cache / "summary.json", {})
    fixed = [
        {
            "name": key,
            "episodes": number(value.get("count")),
            "goals": number(value.get("goals")),
            "validated": bool(value.get("validated")),
        }
        for key, value in summary.items()
        if isinstance(value, dict)
    ]
    probes = read_object(cache / "probes" / "summary.json", {}).get("results", [])
    probes = [
        {
            key: row.get(key)
            for key in (
                "model",
                "status",
                "requests",
                "elapsed_seconds",
                "error",
                "trace",
            )
        }
        for row in probes
        if isinstance(row, dict)
    ]
    overlaps = []
    seen = {}
    for row in rows:
        for episode in row["episodes"]:
            key = (row["model"], episode["episode_id"])
            if key in seen:
                overlaps.append(
                    {
                        "model": key[0],
                        "episode_id": key[1],
                        "runs": [seen[key], row["run"]],
                    }
                )
            else:
                seen[key] = row["run"]
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_setting": "complete_scene_map_and_rgb_initial_pose_only",
        "acceptance_passed": any(row["acceptance_passed"] for row in experiments),
        "runs": rows,
        "fixed_tasks": fixed,
        "compatibility_probes": probes,
        "experiments": experiments,
        "overlapping_attempts": overlaps,
    }


def pct(value):
    return "—" if value is None else f"{value:.1%}"


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_report(report: dict, output: Path) -> str:
    def link(label, target):
        path = Path(target)
        if not path.is_absolute():
            path = ROOT / path
        return f"[{cell(label)}]({os.path.relpath(path, output.parent)})"

    lines = [
        "# Visual Harness 实验结果",
        "",
        f"生成时间：{report['generated_utc']}。",
        "",
        "输入为完整场景平面图、描述与 RGB 证据，真实位姿只用于初始箭头；后续由模型维护位置估计。",
        "",
        "本报告逐个 run 展示结果，smoke、develop、validate 分开记录；重复任务是独立尝试，不合并为正式基准成绩。"
        "各 run 按其保存的配置计分，历史结果不随当前配置改变。当前 ObjectNav 和 GOAT 为成功视点 1 m，指令为 3 m。"
        "terminated 仅表示任务结束，不能等同于成功。",
        "",
        "Sol 历史开发轨迹按统一 1 m 判据的独立复评见 "
        + link("Sol 1 m 成功率", "docs/Experiments.md")
        + "；这不是新运行结果。",
        "",
        "**历史 GOAT 3 m 成绩采用自定义封装协议，不是官方 GOAT SR。** "
        "官方公开代码使用到成功视点的 0.25 m；ObjectNav 的 0.1 m 也指成功视点距离。"
        + "详见 "
        + link("评测协议核查", "docs/Experiments.md")
        + "。",
        "",
        "**验收结论："
        + (
            "已有完整冻结实验通过双轮验收。"
            if report["acceptance_passed"]
            else "尚未有完整冻结实验通过双轮验收，不能声称达到 80%。"
        )
        + "**",
        "",
        "## 固定任务与兼容性",
        "",
        "| 集合 | 起点／任务链 | 子目标 | 已验证 |",
        "|---|---:|---:|---|",
    ]
    for row in report["fixed_tasks"]:
        lines.append(
            f"| {row['name']} | {row['episodes']} | {row['goals']} | {row['validated']} |"
        )
    lines.extend(
        [
            "",
            f"固定集合共 {sum(row['episodes'] for row in report['fixed_tasks'])} 个起点／任务链。自建 HM3D 指令任务不能当作官方 MP3D VLN-CE 成绩。",
            "",
            "| 模型 | 多图＋工具续接探针 | 请求 | 耗时（秒） |",
            "|---|---|---:|---:|",
        ]
    )
    for row in report["compatibility_probes"]:
        status = cell(row["status"])
        if row.get("error"):
            status += "（" + cell(row["error"]) + "）"
        lines.append(
            f"| {cell(row['model'])} | {status} | {row['requests']} | {number(row['elapsed_seconds']):.1f} |"
        )
    lines.extend(
        [
            "",
            "探针仅验证接口兼容，不是导航成功率。",
            "",
            "## 逐次运行",
            "",
            "完成分母只包含已有最终 result.json 的任务；运行中及未开始任务单列。"
            "GOAT 的子目标分母包含整份选定清单中的所有预定子目标；未结束 run 的数值是当前下界，不能当作最终成绩。",
            "",
            "| Run／阶段／模型 | 任务 | 完成／选定 | 成功／完成 | 完整选定 SR | GOAT 成功／预定子目标 | 整串 SR |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["runs"]:
        label = link(row["run"].split("/")[-1], row["run"] + "/summary.json")
        if row.get("diagnostic_note"):
            label += "<br>仅诊断：" + cell(row["diagnostic_note"])
        sub = "—"
        if row["family"] == "goat":
            sub = f"{row['successful_subgoals']}/{row['planned_subgoals']} ({pct(row['goat_micro_selected'])})"
            if not row["complete"]:
                sub += " 当前下界"
        lines.append(
            f"| {label}<br>{row['stage']} / {cell(row['model'])}<br>版本 `{row['harness_version']}` | {row['family']} | "
            f"{row['completed']}/{row['selected']} | {row['successful_episodes']:g}/{row['completed']} ({pct(row['sr_completed'])}) | "
            f"{pct(row['sr_selected'])} | {sub} | {pct(row['goat_chain_sr'])} |"
        )
    lines.extend(
        [
            "",
            "## 成本与失败诊断",
            "",
            "| Run | 请求 | 输入／输出 token | 缓存 token | 响应耗时和（秒） | 剪枝／压缩 | 分类 |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["runs"]:
        cost = row["costs"]
        categories = (
            ", ".join(
                f"{key}: {value}" for key, value in sorted(row["categories"].items())
            )
            or "尚无完成任务"
        )
        lines.append(
            f"| {cell(row['run'].split('/')[-1])} | {cost['requests']} | {cost['input_tokens']:g}/{cost['output_tokens']:g} | "
            f"{cost['cached_tokens']:g} | {cost['response_latency_seconds']:.1f}（{cost['latency_measured_responses']} 响应） | "
            f"{cost['image_prunes']}/{cost['compactions']} | {categories} |"
        )
    lines.extend(
        [
            "",
            "响应耗时优先读取 latency_seconds，旧轨迹用匹配请求与响应的时间戳计算；无响应的超时不计入响应耗时，已计入 episode 总耗时。"
            "成本包含未完成 episode 已记录的请求与返回用量；没有返回用量的请求无法估算 token。"
            "token 仅累计用量顶层字段，未重复累计 attribution。未配置价格，因此不估算金额。",
            "",
            "分类含义：goal_finish 为评测成功；service_error 为模型请求／摘要失败；model_incomplete 为服务返回未完成；"
            "precision_stop 为 ObjectNav 在 0.1～0.5 m 间 STOP 的失败；navigation_failure 为其他导航失败；infrastructure 为环境、进程或评测故障。"
            "precision_stop 仍严格计失败，不改变阈值。",
            "",
            "## 已完成任务与待完成任务",
            "",
            "| Run | 任务 ID | 状态／分类 | 成功 | 距离（m） | 动作 | 总耗时（秒） | 途经约束完成 |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["runs"]:
        for episode in row["episodes"]:
            distance = (
                "—"
                if episode["navigation_error"] is None
                else f"{episode['navigation_error']:.4f}"
            )
            route = pct(episode["route_constraint_completion"])
            lines.append(
                f"| {cell(row['run'].split('/')[-1])} | {link(episode['episode_id'], episode['artifact'])} | "
                f"{episode['status']} / {episode['category']} | {episode['success']:g} | {distance} | {episode['steps']:g} | "
                f"{episode['elapsed_seconds']:.1f} | {route} |"
            )
        for episode in row["pending"]:
            lines.append(
                f"| {cell(row['run'].split('/')[-1])} | {cell(episode['episode_id'])} | {episode['state']} | — | — | — | — | — |"
            )
    lines.extend(
        [
            "",
            f"同一模型与任务出现 {len(report['overlapping_attempts'])} 次重叠尝试，均保留独立 run，未取最佳结果。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", nargs="+", help="Run directories; default: runs/*visual-*"
    )
    parser.add_argument(
        "--suite", nargs="+", help="Experiment JSON files or directories"
    )
    parser.add_argument("--output", default=f"{CACHE}/VisualHarnessResults.md")
    parser.add_argument("--snapshot", default=f"{CACHE}/report.json")
    args = parser.parse_args(argv)
    runs = (
        [Path(resolve_path(path)) for path in args.run]
        if args.run
        else sorted((ROOT / "runs").glob("*visual-*"))
    )
    suites = [Path(resolve_path(path)) for path in args.suite] if args.suite else None
    report = collect_report(runs, suites=suites)
    output = Path(resolve_path(args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(report, output))
    write_json(Path(resolve_path(args.snapshot)), report)
    print(
        json.dumps(
            {
                "output": str(output),
                "runs": len(report["runs"]),
                "acceptance_passed": report["acceptance_passed"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
