"""Python-owned episode runner, initially beside the existing Pi implementation."""

from datetime import datetime, timezone
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import yaml

from .session import NavigationSession, _write
from . import memory
from . import moving_compaction
from src.bench import harness_verify as verify
from .tools import ToolSet
from .config import context_options, credentials, load_navigation_prompt
from .runtime import PromptLimits, open_session
from .trace import TraceWriter
from src.bench.harness_cases import case_config


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
    (workspace / "scene/index.md").write_text(
        "# Public workspace\n\nFixed resources: scene/resources.json. "
        "Only the listed documents/images are available; no fresh observations or movement. "
        "No structured map memory or map renderer is supplied. "
        "Use only pose hints explicitly present in the task resources. "
        "Keep answers tied to the frame requested by the task.\n")
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
    if settled.get("reason") in {"request_budget", "timeout", "tool_timeout", "tool_result_invalid", "model_error", "output_limit", "runtime_error"}:
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
    job = case_config(repo, name)
    context = context_options(config)
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
    session = None
    settled = {}
    state = {"done": False, "goal_changed": False}
    trace = TraceWriter(private)
    diagnostic = ({"kind": "forced_moving_compaction; not a natural 100k trigger",
                   "normal_config": deepcopy(context), "case": name}
                  if name == "moving_compaction" else None)
    error = None
    result = {"case": name, "sample": job.get("sample"), "episode_id": job.get("episode_id"),
              "goal_limit": job.get("goal_limit"), "model": config["model"], "api": config["api"],
              "software_only": False}
    cancel_event = cancel_event or threading.Event()

    def on_event(message):
        event = message.get("event", {})
        if (name in {"compaction", "moving_compaction"} and event.get("type") == "compaction_end"
                and (event.get("data") or {}).get("result")):
            receipt = session.client.request("configure", {"compaction": {"enabled": True, **context["compaction"]}})
            if diagnostic is not None and "restored_config" not in diagnostic:
                diagnostic["restored_config"] = deepcopy(context)
                diagnostic["restored_receipt"] = receipt
                diagnostic["at_compaction"] = moving_compaction.snapshot(bridge, workspace, toolset.checkpoint())
                trace.emit("moving_compaction_config_restored", diagnostic["restored_config"])
                moving_compaction.save(private, diagnostic)

    try:
        trace.redact_with(credentials(config, repo)["apiKey"])
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
            bridge = NavigationSession(cancel_event=cancel_event)
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
            if name in {"local", "moving_compaction"}:
                _copy(repo / job["sample"] / "task_map.png", workspace / "scene/task/task_map.png")
                images.append("scene/task/task_map.png")
            prompt = json.dumps(initial, ensure_ascii=False) + "\nMemory entry: scene/index.md. Attached images include initial RGB and the initialization map."
            if name in {"local", "moving_compaction"}:
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
            if name == "moving_compaction":
                prompt += ("\nThis is the FIRST phase of a same-session diagnostic. Keep the original D01/T local goal pending. "
                           "Choose observations and actions yourself: adjust direction as needed, then make a small real "
                           "translation to obtain a new viewpoint (about one forward step; issue one atomic action per call). "
                           "Pause well before completing the crossing/arrival event. Save a tentative spatial judgment to "
                           f"{moving_compaction.NOTE}, citing the actual latest frame ID and step, the unchanged goal, "
                           "your uncertainty and what remains to do. Read the note back, then answer briefly and pause. "
                           "Do not claim the navigation task is complete; the next user message will resume this same goal.")
        state["current"] = initial
        toolset = ToolSet(repo, workspace, bridge=bridge, state=state,
                          allow_finish=bool(job.get("task_config")), cancel_event=cancel_event)
        if name == "query":
            toolset.tools.pop("render_map")  # Fixed resources have no renderable map memory.
        system_prompt = load_navigation_prompt(repo)
        session = open_session(repo, config, directory, workspace, system_prompt, toolset,
                               max_requests=max_requests or job["requests"], trace=trace,
                               on_event=on_event, cancel_event=cancel_event)
        def remaining():
            return max(0.01, seconds - (time.monotonic() - started))
        def prompt_limits():
            return PromptLimits(max(1, (max_requests or job["requests"]) - session.requests), remaining())
        settled = session.prompt(prompt, images, prompt_limits()).to_dict()
        if diagnostic is not None:
            diagnostic["phase1"] = moving_compaction.snapshot(bridge, workspace, toolset.checkpoint())
            diagnostic["phase1_settled"] = settled
            diagnostic["phase1_gate"] = moving_compaction.ready(diagnostic["phase1"])
            moving_compaction.save(private, diagnostic)
            if (diagnostic["phase1_gate"]["passed"] and not cancel_event.is_set() and remaining() > .1
                    and settled.get("requests", 0) < (max_requests or job["requests"])
                    and settled.get("reason") == "model_finished"):
                forced = {"enabled": True, "reserveTokens": context["contextWindow"] - 512,
                          "keepRecentTokens": 1200}
                diagnostic["forced_config"] = {"contextWindow": context["contextWindow"], "compaction": forced}
                diagnostic["forced_receipt"] = session.client.request("configure", {"compaction": forced})
                trace.emit("moving_compaction_boundary", {"phase1_gate": diagnostic["phase1_gate"],
                           "steps": diagnostic["phase1"]["state"]["metrics"]["steps"],
                           "forced_config": diagnostic["forced_config"]})
                moving_compaction.save(private, diagnostic)
                settled = session.prompt(
                    "Resume the SAME D01/T local goal after the diagnostic pause and compression. "
                    f"Reopen {moving_compaction.NOTE} and its real evidence; distinguish old position estimates "
                    "from current evidence. Decide how to observe and move, then actually cross D01 and reach T. "
                    "The local-event detector will end the task; a verbal completion claim is insufficient.",
                    [], prompt_limits()).to_dict()
        if (name == "compaction" and not state.get("done") and not cancel_event.is_set()
                and remaining() > 0.1 and settled.get("requests", 0) < (max_requests or job["requests"])
                and settled.get("reason") == "model_finished"):
            # Lower only this diagnostic's trigger to 512 tokens, regardless of model capacity.
            session.client.request("configure", {"compaction": {"enabled": True, "reserveTokens": context["contextWindow"] - 512,
                                                        "keepRecentTokens": 1200}})
            settled = session.prompt("Continue after compression: reopen your note and its evidence image, call nav_observe(single), update the note distinguishing old estimates from the new frame, then give a brief answer. Do not claim tentative spatial correspondence is confirmed without evidence.",
                                     [], prompt_limits()).to_dict()
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
    finally:
        error = error or settled.get("error")
        reason = "cancelled" if cancel_event.is_set() and not state.get("done") else finish_reason(state, settled, error)
        if bridge:
            try:
                state["current"] = {**state.get("current", {}), **bridge.finalize(reason)}
                state["done"] = bool(state["current"].get("terminated") or state["current"].get("truncated"))
            except Exception as failure:
                error = error or f"finalize: {failure}"
            if session is None:
                try:
                    bridge.close()
                except Exception as failure:
                    error = error or f"close: {failure}"
        if session:
            try:
                session.close()
                if session.cleanup_error:
                    result["cleanup_error"] = session.cleanup_error
                close_status = session.close_status
                result["core_close"] = close_status if isinstance(close_status, dict) else None
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
        if name in {"compaction", "moving_compaction"}:
            result["compaction"] = {"compacted": bool(trace.compactions), "count": len(trace.compactions)}
        if diagnostic is not None:
            diagnostic["final_settled"] = settled
            moving_compaction.save(private, diagnostic)
            try:
                result["verification"] = verify.verify_moving_compaction(
                    workspace=workspace, sample_dir=repo / job["sample"], trace_dir=private,
                    environment_private_dir=private / "environment", diagnostic=diagnostic,
                    finish_reason=reason, cleanup_error=result.get("cleanup_error"), runtime_error=error,
                    core_close=result.get("core_close"))
            except Exception as failure:
                result["scoring_error"] = str(failure)
        ledger = private / "requests.json"
        if ledger.exists():
            try:
                result["requests"] = len(json.loads(ledger.read_text())["attempts"])
            except (OSError, ValueError, KeyError):
                result["archive_error"] = "Request ledger could not be read"
        result = trace.sanitize(result)
        trace.emit("run_finished", result)
        trace.close()
        _write(directory / "result.json", result)
    return directory
