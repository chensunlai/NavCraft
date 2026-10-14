"""Python-owned episode runner, initially beside the existing Pi implementation."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import signal
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import yaml

from core_client import CoreClient
from bridge import Bridge, _write
import memory
import verify
from tools import ToolSet
from worker import supervise


def _copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _prepare_query(repo, job, workspace):
    sample = repo / job["sample"]
    task = json.loads((sample / "task.json").read_text())
    resources = {}
    for identifier, ref in task["resources"].items():
        target = ref if ref.startswith("observations/") else "scene/sample/" + ref
        _copy(sample / ref, workspace / target)
        resources[identifier] = target
    _write(workspace / "scene/resources.json", resources)
    observation = {"id": "initial_observation", "path": resources["initial_observation"],
                   "step": 0, "source": "observation", "view": None, "derived_from": []}
    (workspace / "observations/index.jsonl").write_text(json.dumps(observation) + "\n")
    images = [ref if ref.startswith("observations/") else "scene/sample/" + ref
              for ref in task["initial_images"]]
    initial = {"task": {"goal_id": 0, "modality": "qa", "instruction": task["question"]},
               "observation": observation}
    prompt = (task["question"] + '\nResources: scene/resources.json. Only these fixed documents/images are '
              'available; no fresh observations. Final answer must be JSON '
              '{"choice":"<opening ID>","anchor":"initial_observation"}.')
    return initial, images, prompt


def finish_reason(state, settled, error=None):
    if state.get("done"):
        return state.get("current", {}).get("reason", "task_ended")
    if settled.get("reason") in {"request_budget", "timeout", "tool_timeout", "model_error", "output_limit", "runtime_error"}:
        return settled["reason"]
    if settled.get("reason") == "aborted":
        return "timeout"
    last = settled.get("last") or {}
    if last.get("stopReason") == "length":
        return "output_limit"
    if last.get("stopReason") == "error":
        return "model_error"
    if last.get("stopReason") == "aborted":
        return "timeout"
    return "runtime_error" if error else "model_finished"


def run_case(repo, config, name, max_requests=None, seconds=None, directory=None, cancel_event=None):
    repo = Path(repo).resolve()
    job = yaml.safe_load((repo / "config/harness-pi/smoke.yaml").read_text())["cases"][name]
    started = time.monotonic()
    seconds = seconds or job["seconds"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = Path(directory) if directory else repo / config["output_dir"] / f"{stamp}-python-{name}"
    workspace, private = directory / "workspace", directory / "private"
    for folder in ("scene", "observations", "memory", "scratch"):
        (workspace / folder).mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"directory": str(directory), "case": name, "phase": "initializing"}), flush=True)
    bridge = None
    client = None
    settled = {}
    state = {"done": False, "goal_changed": False}
    events = []
    error = None
    result = {"case": name, "sample": job.get("sample"), "episode_id": job.get("episode_id"),
              "goal_limit": job.get("goal_limit"), "model": config["model"], "api": config["api"],
              "software_only": False}
    cancel_event = cancel_event or threading.Event()

    def on_event(message):
        events.append(message)
        event = message.get("event", {})
        if event.get("type") == "turn_start":
            toolset.turn_start()
        if event.get("type") == "compaction_end" and (event.get("data") or {}).get("result"):
            client.request("configure", {"compaction": {"enabled": True, "reserveTokens": 16384,
                                                        "keepRecentTokens": 20000}})

    try:
        if name == "query":
            initial, images, prompt = _prepare_query(repo, job, workspace)
        else:
            partial = name == "mapping"
            memory.prepare(str(repo / config["scene_archive"]), str(workspace),
                           mode="partial" if partial else "prebuilt",
                           allowed_ids=["R01", "R03"] if partial else None)
            if partial:
                shutil.rmtree(workspace / "scene/evidence")
                (workspace / "scene/evidence").mkdir()
                (workspace / "scene/evidence/index.jsonl").write_text("")
                import re
                for folder in ("memory/entities", "scene/initial/entities"):
                    for card in (workspace / folder).glob("*.md"):
                        text = re.sub(r"(?m)^- 证据：.*$", "- 证据：无", card.read_text())
                        text = re.sub(r"(?m)^- 描述：.*$", "- 描述：区域范围由初始图提供，具体布局与出入口需要现场观察。", text)
                        card.write_text(text)
            _copy(Path(memory.__file__), workspace / "scene/tools/memory.py")
            bridge = Bridge()
            instruction = {"mapping": "局部空间记忆构建：根据真实观察建立一个出入口记录，取得新视角后修订，并在后续判断中使用。",
                           "compaction": "局部空间记忆与压缩续接检查；保持真实观察引用和待核实判断。"}.get(name)
            initial = bridge.init(repo=str(repo), workspace=str(workspace),
                        private_dir=str(private / "environment"),
                        env_config=job.get("env_config", "config/environment/objectnav_visual.yaml"),
                        task_config=job.get("task_config"), episode_id=job.get("episode_id"),
                        atomic_sample=job.get("sample"), goal_limit=job.get("goal_limit"),
                        scene_archive=config["scene_archive"], purpose=job.get("purpose", "navigation"),
                        instruction_override=instruction)
            saved = json.loads((workspace / "memory/state.json").read_text())
            saved.update(goal_id=initial["task"]["goal_id"],
                         pose_estimate={**initial["initial_hint"], "status": "initial_hint"})
            _write(workspace / "memory/state.json", saved)
            rendered = memory.render(str(workspace), initial["initial_hint"]["map_id"])
            state.update(latest_map=rendered["images"][0], latest_map_metadata=rendered["data"])
            images = initial["images"] + rendered["images"]
            if name == "local":
                _copy(repo / job["sample"] / "task_map.png", workspace / "scene/task/task_map.png")
                images.append("scene/task/task_map.png")
            prompt = json.dumps(initial, ensure_ascii=False) + "\nMemory entry: scene/index.md. Attached images include initial RGB and the initialization map."
            if name == "local":
                prompt += "\nThe attached scene/task/task_map.png shows the requested T point."
            if partial:
                prompt += ('\nUse actual observations to identify one doorway between supplied regions. Create a portal and '
                           'card with real evidence, check/render it, obtain a new viewpoint, revise the SAME record, '
                           'read/render it and explain how to approach it. Keep within 24 actions and 4 metres. '
                           'Geometry may initially be null, but the final record must contain an approximate '
                           'geometry:{"segment":[[x1,y1],[x2,y2]]} in original-map normalized coordinates. '
                           'Do not invent evidence or claim metric precision. Report remaining uncertainty.')
            if name == "compaction":
                prompt += "\nInspect nearby map/card evidence. Save a tentative observation-to-map judgment with real frame references and uncertainty to memory/notes.md, read it back, and answer briefly. At most two movement actions; no goal submission."
        state["current"] = initial
        toolset = ToolSet(repo, workspace, bridge=bridge, state=state,
                          allow_finish=bool(job.get("task_config")), cancel_event=cancel_event)
        system_prompt = ("Answer the spatial question using the fixed public resource bank. Use read or view_images. Preserve the answer anchor."
                         if name == "query" else (repo / "src/harness-pi/navigation_prompt.md").read_text())
        client = CoreClient(repo, private)
        client.request("init", {"workspace": str(workspace), "directory": str(directory),
                       "budgetFile": str(private / "requests.json"), "totalLimit": None,
                       "maxRequests": max_requests or job["requests"],
                       "credentialFile": str(repo / config["credentials"]), "modelId": config["model"],
                       "api": config["api"], "python": config["python"], "systemPrompt": system_prompt,
                       "tools": toolset.specs(), "compact": True, "checkpoint": toolset.checkpoint()})
        def remaining():
            return max(0.01, seconds - (time.monotonic() - started))
        settled = client.run_prompt(prompt, images, toolset.call, remaining(), on_event, cancel_event)
        if (name == "compaction" and not state.get("done") and not cancel_event.is_set()
                and remaining() > 0.1 and settled.get("requests", 0) < (max_requests or job["requests"])
                and settled.get("reason") == "model_finished"):
            client.request("configure", {"compaction": {"enabled": True, "reserveTokens": 65000,
                                                        "keepRecentTokens": 1200}})
            settled = client.run_prompt("Continue after compression: reopen your note and its evidence image, call nav_observe(single), update the note distinguishing old estimates from the new frame, then give a brief answer. Do not claim tentative spatial correspondence is confirmed without evidence.",
                                         [], toolset.call, remaining(), on_event, cancel_event)
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
    finally:
        reason = "cancelled" if cancel_event.is_set() and not state.get("done") else finish_reason(state, settled, error)
        if bridge:
            try:
                state["current"] = {**state.get("current", {}), **bridge.finalize(reason)}
                state["done"] = bool(state["current"].get("terminated") or state["current"].get("truncated"))
            except Exception as failure:
                error = error or f"finalize: {failure}"
            try:
                bridge.close()
            except Exception as failure:
                error = error or f"close: {failure}"
        if client:
            try:
                client.close()
            except Exception as failure:
                result["cleanup_error"] = f"Core close: {failure}"
        result.update(answer=settled.get("answer", ""), requests=settled.get("requests", 0),
                      finish_reason=reason, error=error, elapsed_seconds=time.monotonic() - started,
                      terminated=bool(state.get("current", {}).get("terminated")),
                      truncated=bool(state.get("current", {}).get("truncated")),
                      last_stop_reason=(settled.get("last") or {}).get("stopReason"),
                      last_error=(settled.get("last") or {}).get("errorMessage"))
        env_result = private / "environment/result.json"
        if env_result.exists():
            try:
                result["environment"] = json.loads(env_result.read_text())
            except (OSError, ValueError) as failure:
                result["archive_error"] = f"Environment result: {failure}"
        try:
            if name == "query":
                result["verification"] = verify.qa(workspace=workspace, sample_dir=repo / job["sample"], answer=result["answer"])
            if name == "mapping" and not error:
                result["verification"] = verify.mapping(workspace=workspace, scene_dir=repo / config["scene_archive"],
                                            trace_dir=private, environment_private_dir=private / "environment")
        except Exception as failure:
            result["scoring_error"] = str(failure)
        if name == "compaction":
            result["compaction"] = {"compacted": any(e.get("event", {}).get("type") == "compaction_end" and
                                                    e["event"].get("data", {}).get("result") for e in events)}
        _write(directory / "result.json", result)
        with (private / "python_events.jsonl").open("w") as stream:
            for event in events:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(json.dumps({"directory": str(directory), **result}, ensure_ascii=False, indent=2), flush=True)
    return directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/harness-pi/default.yaml")
    parser.add_argument("--case", default="local")
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--directory")
    args = parser.parse_args()
    config = yaml.safe_load((REPO / args.config).read_text())
    if args.worker:
        cancelled = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: cancelled.set())
        run_case(REPO, config, args.case, args.max_requests, directory=args.directory, cancel_event=cancelled)
    else:
        job = yaml.safe_load((REPO / "config/harness-pi/smoke.yaml").read_text())["cases"][args.case]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        directory = REPO / config["output_dir"] / f"{stamp}-python-{args.case}"
        print(json.dumps({"directory": str(directory), "case": args.case, "phase": "supervised"}), flush=True)
        command = [sys.executable, "-B", "-u", str(Path(__file__)), "--worker", "--directory", str(directory),
                   "--config", args.config, "--case", args.case]
        if args.max_requests:
            command.extend(["--max-requests", str(args.max_requests)])
        result = supervise(command, directory, job["seconds"], REPO)
        print(json.dumps({"directory": str(directory), **result}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
