"""Run one frozen task from the selected spatial task subset."""
import argparse
import hashlib
import json
from pathlib import Path

from src.common.config import read_config
from src.common.io import read_json, redact, write_json
from src.dataset.episodes import EpisodeSpec
from src.dataset.tasks_subset.bundle import snapshot_digest
from src.dataset.tasks_subset.judgment.scoring import score_answer
from src.dataset.tasks_subset.review.model import run_fixed_evidence
from src.dataset.tasks_subset.navigation.scoring import score_navigation
from src.environment import Environment
from src.harness import PromptLimits, open_session
from src.harness.runtime.config import credentials
from src.harness.runtime.workspace import read_snapshot
from .runner import INFRASTRUCTURE_REASONS


REPO = Path(__file__).resolve().parents[2]


def _status(reason):
    if reason in INFRASTRUCTURE_REASONS:
        return "infrastructure_error"
    if reason.endswith("_budget") or reason in {"output_limit", "turn_limit"}:
        return "budget_exhausted"
    return "completed"


def _navigation(repo, config, output, task, case, label, limits, gpu, images):
    state = read_snapshot(case / task["snapshot"])
    episode = EpisodeSpec(**read_json(case / task["episode"]))
    spec = read_json(case / task["environment"])
    spec = {key: str((repo / spec[key]).resolve()) if key in {"scenes", "scene_dataset_config"} else spec[key]
            for key in spec if key in {"scenes", "scene_dataset_config", "seed"}}
    environment = Environment(episode, {**spec, "gpu": gpu, "depth": False}, output / "private/environment")
    session = None
    try:
        session = open_session(repo, {**config, "toolsets": ["navigation", "node_memory"]},
                               output, environment, Path(state["workspace"]) / "scene/map.json",
                               initial_state=state)
        nav = session.navigation
        initial = {"steps": environment.steps, "path_length": environment.path_length,
                   "position": environment.position(), "memory": nav.memory.state,
                   "nodes": [n["id"] for n in nav.memory.nodes],
                   "panoramas": {f.get("group") for f in nav.observations.items.values()},
                   "receipt_ids": set(nav._calls)}
        instruction = "\n".join([task["instruction"], *[
            f"Image {index}: {image['caption']}" for index, image in enumerate(images, 2)]])
        prompt = session.prompt(instruction, images=[str(i["path"]) for i in images], limits=limits)
        status = _status(prompt.reason)
        result = {"status": status, "reason": prompt.reason, "prompt": prompt.to_dict()}
        if status != "infrastructure_error":
            score = score_navigation(environment, nav, label, initial, prompt)
            if task["kind"] == "nav_04":
                from src.dataset.tasks_subset.review.execution import capture_execution_review
                capture_execution_review(case / "task.json", output, nav, initial, score)
            over_path = (limits.max_path_length is not None and
                         score["details"]["path_length"] > limits.max_path_length + 1e-6)
            if over_path or score["details"]["actions"] > limits.max_actions:
                result.update(status="budget_exhausted", reason="motion_budget")
            result["score"] = score if result["status"] == "completed" else {
                **score, "success": None, "status": "budget_exhausted"}
        return result
    finally:
        try:
            if session is not None:
                session.close()
        finally:
            environment.close()


def run_task(repo, task_path, config, output, gpu=0):
    repo, task_path, output = Path(repo).resolve(), Path(task_path).resolve(), Path(output).resolve()
    if task_path.is_dir():
        task_path /= "task.json"
    if type(gpu) is not int or gpu not in (0, 1):
        raise ValueError("Subset navigation requires GPU 0 or 1")
    case = task_path.parent
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "starting"}
    secrets = ()
    inputs = {}
    expected = {}
    try:
        def bind(name, path, digest=None):
            if digest is None:
                digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            inputs[name] = (path, digest)
            expected[name] = digest(path)
            result[name] = expected[name]

        bind("task_sha256", task_path)
        task = read_json(task_path)
        result.update(id=task["id"], kind=task["kind"], mode=task["mode"])
        try:
            from src.dataset.tasks_subset.bundle import validate_frozen_inputs
            validate_frozen_inputs(task, case)
        except (ValueError, OSError, KeyError) as error:
            result.update(status="infrastructure_error", reason="frozen_input_mismatch", error=str(error))
            write_json(output / "result.json", result)
            return result
        bind("label_sha256", case / task["label"])
        result["snapshot_sha256"] = None
        if task.get("snapshot"):
            bind("snapshot_sha256", case / task["snapshot"], snapshot_digest)
        if task.get("episode"):
            bind("episode_sha256", case / task["episode"])
        if task.get("environment"):
            bind("environment_sha256", case / task["environment"])
        for index, image in enumerate(task["images"], 1):
            bind(f"image_{index:02d}_sha256", case / image["path"])
        write_json(output / "result.json", result)
        label = read_json(case / task["label"])
        limits = PromptLimits(**task["limits"])
        images = [{**row, "path": (case / row["path"]).resolve(strict=True)} for row in task["images"]]
        auth = credentials(config, repo)
        secrets = (auth["apiKey"],)
        config = {**config, "model": auth["model"]}
        result.update(model=config["model"], reasoning_effort=config.get("reasoning_effort", "high"))
        if task["mode"] == "navigation":
            result.update(_navigation(repo, config, output, task, case, label, limits, gpu, images))
        elif task["mode"] == "judgment":
            text = task["instruction"] + "\nReturn only the JSON value that satisfies this schema. For a string, include JSON quotation marks. For a number, array, or object, return that value directly. Do not copy the schema, wrap your answer, or use Markdown.\nSchema: " + json.dumps(task["answer_schema"], ensure_ascii=False)
            prompt = run_fixed_evidence(repo, config, output, text, images, limits)
            reason = prompt["reason"]
            result.update(prompt=prompt, reason=reason, status=_status(reason))
            if result["status"] == "completed":
                try:
                    answer = json.loads(prompt.get("answer", ""))
                except (TypeError, ValueError):
                    answer = object()
                result["score"] = score_answer(answer, label)
        else:
            raise ValueError(f"Unknown task mode: {task['mode']}")
    except Exception as error:
        result.update(status="infrastructure_error", reason="worker_error", error=str(error))
    changed = []
    for name, initial in expected.items():
        path, digest = inputs[name]
        try:
            if digest(path) != initial:
                changed.append(name)
        except (OSError, ValueError):
            changed.append(name)
    if changed:
        result.update(status="infrastructure_error", reason="frozen_input_changed", changed_inputs=changed)
        result.pop("score", None)
    result = redact(result, secrets)
    write_json(output / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--config", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=[0, 1], default=0)
    parser.add_argument("--reasoning-effort", choices=["off", "minimal", "low", "medium", "high", "xhigh", "max"])
    args = parser.parse_args(argv)
    config = read_config([REPO / "config/harness/node.yaml", *args.config])
    if args.reasoning_effort:
        config["reasoning_effort"] = args.reasoning_effort
    result = run_task(REPO, args.task, config, args.output, args.gpu)
    print(json.dumps({key: result.get(key) for key in ("id", "status", "reason", "score")}
                     | {"result": str(args.output / "result.json")}, ensure_ascii=False))
    return int(result["status"] == "infrastructure_error")


if __name__ == "__main__":
    raise SystemExit(main())
