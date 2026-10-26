"""Package real frozen task examples for the read-only task catalogue."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw


TASK_IDS = ("5a", "5c", "6a", "6c", "13d", "1b", "1n", "14b")


def _chinese_prompt(value):
    """Display translations only; original model requests remain unchanged."""
    if not value:
        return ""
    fixed = {
        "Find the same physical object shown in the reference image in the current image. Submit [x,y] in original 640x480 pixels, or none if absent.":
            "在当前图像中找出参考图展示的同一个物理实例。请提交该物体在原始640×480图像中的像素坐标[x,y]；如果没有出现，提交none。",
        "Which spatial-memory entity is the object inside the red box? Submit its entity ID, or new if it has no record. Other objects outside the box are irrelevant.":
            "红框内的物体对应空间记忆中的哪个实体？请提交它的实体ID；如果尚未记录，提交new。不要考虑红框外的其他物体。",
        "Which direction is map-up relative to your current body heading? Submit exactly one of front, front-left, left, back-left, back, back-right, right, front-right.":
            "相对于你当前身体的朝向，地图上方在哪个方向？请只提交以下一个标签：front（前）、front-left（左前）、left（左）、back-left（左后）、back（后）、back-right（右后）、right（右）、front-right（右前）。",
    }
    if value in fixed:
        return fixed[value]
    match = re.fullmatch(r"Do spatial-memory records (\S+) and (\S+) refer to the same physical object\? Submit yes or no\.", value)
    if match:
        return f"空间记忆中的记录{match[1]}和{match[2]}是否对应同一个物理实例？请提交yes（是）或no（否）。"
    categories = {"cabinet": "柜子", "chair": "椅子", "desk": "书桌", "oven": "烤箱",
                  "shelf": "搁架", "tv": "电视", "bathtub": "浴缸"}
    match = re.fullmatch(r"Is a (\w+) visible in the current image\? If yes, point to any visible instance\. Submit \[x,y\] in original 640x480 pixels, or none if absent\.", value)
    if match and match[1] in categories:
        return (f"当前图像中能看到{categories[match[1]]}吗？如果能，请指出任意一个可见实例。"
                "提交原始640×480图像中的像素坐标[x,y]；如果没有，提交none。")
    match = re.fullmatch(r"Go next to the (\w+) marked in the reference image, then submit\.", value)
    if match and match[1] in categories:
        return f"走到参考图中标出的{categories[match[1]]}旁边，然后提交。"
    match = re.fullmatch(r"Find the (\w+) in the reference image and go next to it\.", value)
    if match and match[1] in categories:
        return f"找到参考图中的{categories[match[1]]}，并走到它旁边。"
    raise ValueError(f"Missing Chinese display translation: {value}")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _inside(root, ref):
    path = (root / ref).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or external showcase source: {ref}")
    return path


def _image(path, output, source, label, scope="public"):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    relative = f"assets/{digest}.jpg"
    destination = output / relative
    with Image.open(path) as opened:
        size = list(opened.size)
        if not destination.exists():
            image = opened.convert("RGB")
            image.thumbnail((1200, 1000))
            image.save(destination, quality=83, optimize=True)
        with Image.open(destination) as saved:
            displayed = list(saved.size)
    return {"src": relative, "label": label, "scope": scope, "source": source,
            "sha256": digest, "width": displayed[0], "height": displayed[1], "original_size": size}


def _answer(root, sample, label):
    task = sample["task_id"]
    result = {"review_only": True}
    if task in {"5a", "5c"}:
        if label.get("answer_kind") == "none":
            result.update(answer="none", summary="当前图没有题目所问类别；提交 none。")
        else:
            mask = np.load(_inside(root, label["mask_file"]), allow_pickle=False)
            mask = mask == label["instance_id"] if "instance_id" in label else mask.astype(bool)
            points = np.argwhere(mask)
            center = np.median(points, axis=0)
            y, x = points[np.argmin(np.sum((points - center) ** 2, axis=1))]
            result.update(answer=f"[{int(x)}, {int(y)}]", summary="示例有效点；不是唯一答案。原始640×480图上落在允许实例掩码内的点均可。",
                          frame=label.get("frame"), accepted_pixels=int(mask.sum()))
    elif task == "6a":
        result.update(answer=" / ".join(label["accepted_ids"]), summary="按切点的记忆实例对应评分；没有记录时答 new。")
    elif task == "6c":
        result.update(answer="yes" if label["same_instance"] else "no", summary="比较真实物理实例，不以同类别或相邻位置代替实例身份。")
    elif task == "13d":
        result.update(answer=label["direction"], summary="根据切点身体朝向，将固定地图上方换算为八方向之一。")
    elif task == "1b":
        result.update(answer="新建对应物体记忆；ID可以自行选择。", summary="窗口末应覆盖新出现的真实实例，位置与证据正确，同时保留已有实例。",
                      target_instance=label["target_instance"], baseline_bindings=label["baseline_bindings"])
    elif task == "1n":
        result.update(answer="不重复登记；允许补充描述、位置和真实证据。", summary="同一真实实例最终只有一条有效记录；正确补充不算误改，其他已有实例不能丢失。",
                      target_instance=label["target_instance"], baseline_bindings=label["baseline_bindings"])
    else:
        result.update(answer="实际走到核验过的接近区域内，再调用 submit。", summary="提交时到核验接近点集合的测地距离≤0.35m，且未超过动作、真实路程及请求预算。不是仅看见目标或隔墙直线接近。",
                      target_instance=label["target_instance"], approach_tolerance=label["approach_tolerance"])
    return result


def _reference(root, directory, snapshot, output):
    path = directory / "private/reference.json"
    if not path.exists():
        return None
    ref = _json(path)
    result = {"source": "scripted_verification", "trainable": False, "passed": ref["passed"],
              "actions": snapshot["attributes"]["reference_actions"], "steps": ref["steps"],
              "path_length": ref["path_length"], "distance": ref["distance"],
              "note": "真实动作重放验证轨迹，不是模型示范；橙线为真实微步路径，绿色为终点。"}
    map_path = directory / "review-map.png"
    if map_path.exists():
        transform = _json(directory / "workspace/scene/map.json")
        with Image.open(map_path) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        origin, scale = transform["origin"], transform["size_m"]
        points = [((pos[0] - origin[0]) / scale * 680 + 20,
                   (-pos[2] - origin[1]) / scale * 680 + 20) for pos in ref["motion_trajectory"]]
        if len(points) > 1:
            draw.line(points, fill="#f27c32", width=5)
            x, y = points[-1]
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="#258863", outline="white", width=2)
        source_hash = hashlib.sha256((path.read_bytes() + map_path.read_bytes())).hexdigest()
        relative = f"assets/route-{source_hash}.jpg"
        destination = output / relative
        image.save(destination, quality=88, optimize=True)
        result["image"] = {"src": relative, "label": "私有核验：参考动作真实路径", "scope": "review",
                           "source": path.relative_to(root).as_posix(), "width": image.width, "height": image.height}
    return result


def _teacher(row, repo, output):
    run_dir = Path(row["run_dir"])
    run = (repo / run_dir).resolve()
    runs_root = (repo / "runs").resolve()
    if not run.is_relative_to(runs_root):
        raise ValueError(f"Teacher run is outside configured runs directory: {run_dir}")
    result = _json(run / "result.json")
    outcome = result.get("outcome", {})
    data = {"model": result.get("model"), "effort": result.get("reasoning_effort"), "attempt": row["attempt"],
            "status": row["status"], "success": row["score"]["success"], "reason": row["score"]["reason"],
            "metrics": row["score"].get("metrics", {}), "answer": outcome.get("answer"),
            "requests": outcome.get("requests"), "actions": outcome.get("actions"),
            "path_length": outcome.get("path_length"), "run": row["run_dir"],
            "tools": [], "visible_outputs": [], "images": [], "final_memory": outcome.get("entities", [])}
    trace = run / "private/trace.jsonl"
    if trace.exists():
        for event in _rows(trace):
            payload = event.get("data", {})
            if event["type"] == "runtime_started":
                data.update(model=payload.get("model"), effort=payload.get("thinking"))
            elif event["type"] == "message_end":
                message = payload.get("message", {})
                if message.get("role") == "assistant":
                    text = "\n".join(item.get("text", "") for item in message.get("content", []) if item.get("type") == "text")
                    if text:
                        data["visible_outputs"].append(text)
            elif event["type"] == "public_tool_reply":
                result = payload.get("result", {})
                texts = [item["text"] for item in result.get("content", []) if item.get("type") == "text"]
                data["tools"].append({"name": payload["name"], "arguments": payload.get("args", {}),
                                      "reply": "\n".join(texts), "is_error": result.get("isError", False)})
    index_path = run / "workspace/observations/index.jsonl"
    if index_path.exists():
        observations = [obs for obs in _rows(index_path) if obs.get("kind") == "observation" and (obs.get("step") or 0) > 0]
        selected = observations if len(observations) <= 6 else [observations[i] for i in sorted({0, len(observations)//4, len(observations)//2, 3*len(observations)//4, len(observations)-1})]
        for obs in selected:
            path = _inside(run / "workspace", obs["path"])
            data["images"].append(_image(path, output, (run_dir / "workspace" / obs["path"]).as_posix(),
                                         f"真实教师动作后：{obs['frame']} · step {obs['step']}", "teacher"))
    return data


def build_examples(collection: Path, output: Path) -> dict:
    """Build an offline manifest and optimized allowlisted image assets; no model calls."""
    root = Path(collection).resolve()
    output = Path(output)
    (output / "assets").mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[3]
    samples = _rows(root / "samples.jsonl")
    snapshots = {row["id"]: row for row in _rows(root / "snapshots.jsonl")}
    selected = set(_json(root / "teacher_selection.json"))
    solutions = _rows(root / "solutions.jsonl") if (root / "solutions.jsonl").exists() else []
    tasks = {identifier: [] for identifier in TASK_IDS}
    for sample in sorted(samples, key=lambda row: (row["id"] not in selected, row["id"])):
        snapshot = snapshots[sample["snapshot"]]
        directory = root / snapshot["directory"]
        state = _json(directory / "state.json")
        workspace = directory / state["workspace"]
        index = {row["frame"]: row for row in _rows(workspace / "observations/index.jsonl")}
        current = index[state["current_frame"]]
        images = []
        for frame, caption in [(state["current_frame"], "模型切点实际看到的当前RGB"),
                               *((frame, "保留的原导航目标图") for frame in state.get("target_frames", []))]:
            source = _inside(workspace, index[frame]["path"])
            images.append(_image(source, output, source.relative_to(root).as_posix(), f"{caption} · {frame}"))
        for ref in sample["inputs"].get("images", []):
            source = _inside(root, ref)
            caption = "题目附图：目标实例参考图" if sample["task_id"] == "5c" else "题目附图：红框标出待对应／接近的物体"
            images.append(_image(source, output, ref, caption))
        map_path = directory / "review-map.png"
        if map_path.exists():
            images.append(_image(map_path, output, map_path.relative_to(root).as_posix(),
                                 "核验地图：切点位置与朝向；模型可经 map_view 查询地图", "review"))
        label = _json(_inside(root, sample["label_file"]))
        memory = _json(_inside(root, sample["inputs"]["memory_file"]))
        prefix_tools = [item.get("name") for message in state.get("history", [])
                        if message.get("role") == "assistant" and message.get("model") == "script"
                        for item in message.get("content", []) if item.get("type") == "toolCall"]
        example = {"id": sample["id"], "task_id": sample["task_id"], "scene": sample["attributes"]["scene_id"],
                   "split": sample["split"], "form": sample["form"], "entry": sample["entry"],
                   "question": sample["question"], "question_zh": _chinese_prompt(sample["question"]),
                   "active_request": state.get("current_request", ""),
                   "active_request_zh": _chinese_prompt(state.get("current_request", "")),
                   "frame": state["current_frame"], "pose": current.get("pose"), "limits": sample["limits"],
                   "tool_tier": sample["attributes"].get("tool_tier", "super"), "images": images,
                   "memory": memory, "label": _answer(root, sample, label), "group_id": sample["group_id"],
                   "permissions": ("只可查询既有资料、计算和提交答案；不能移动、拍新图或修改正式记忆。"
                                   if sample["form"] == "qa" else "可使用当前super工具观察、移动、查询并维护正式记忆。"),
                   "prefix": {"source": "script", "trainable": False, "tools": prefix_tools,
                              "note": "切点前是实际工具执行产生的脚本前缀，不是教师推理或训练目标。"},
                   "teacher_selected": sample["id"] in selected,
                   "teacher": [_teacher(row, repo, output) for row in solutions
                               if row["sample_id"] == sample["id"] and row.get("model_started")]}
        if sample["task_id"] == "14b":
            example["reference"] = _reference(root, directory, snapshot, output)
        if sample["form"] == "qa":
            example["budget_note"] = "冻结记录中的max_actions=1是通用运行器的配置下限；QA权限仍禁止所有移动，不代表允许走一步。"
        if (snapshot["attributes"]["target_instance"] == "cabinet_49"
                or "cabinet_49" in label.get("baseline_bindings", {}).values()):
            example["quality_note"] = "此切点涉及的一个柜子仅在画面边缘露出窄条；已有记忆描述也较弱。可检查真实数据，但不宜据此单独归因模型能力不足。"
        tasks[sample["task_id"]].append(example)
    report_path = root / "exports/report.json"
    export = _json(report_path) if report_path.exists() else {}
    manifest = {"collection": root.name, "notes": [
        "以下80条均来自真实场景采样；16条被固定选入教师验收，其余未调用教师。",
        "私有答案和核验地图仅供网页检查，不属于模型初始输入。图片经过展示压缩，像素答案仍以原始640×480图为准。",
        "真实教师输出只包含可见正文及工具调用，不展示或伪造不可见推理；失败也保留。",
        "教师列表只列实际调用过模型的尝试；模型启动前的基础设施故障不当作教师作答。",
        "三场景工程验收数据，不代表完整目录已构建，也不作为跨场景泛化结论。"],
        "counts": {"samples": len(samples), "by_task": dict(Counter(sample["task_id"] for sample in samples)),
                   "scenes": len({sample["attributes"]["scene_id"] for sample in samples}),
                   "teacher_selected": len(selected), "teacher_attempts": sum(bool(row.get("model_started")) for row in solutions),
                   "verified_samples": export.get("verified_samples"),
                   "exported_conversations": export.get("exported_conversations"), "exported_decisions": export.get("exported_decisions")},
        "tasks": tasks}
    (output / "examples.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
