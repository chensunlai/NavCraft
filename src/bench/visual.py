"""Fixed visual-map experiments and an explicit, recorded model escalation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from ..common.config import ROOT, load_config
from ..common.io import write_json
from .run import run_experiment

MODELS = {
    "qwen27": "config/model/qwen27.yaml",
    "qwen35": "config/model/qwen35.yaml",
    "deepseek": "config/model/deepseek.yaml",
    "sol": "config/model/sol.yaml",
}
ENVIRONMENTS = {"objnav": "objectnav", "instruction": "r2r", "goat": "goat"}


def harness_version() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*")):
        if path.suffix in {".py", ".md"}:
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    digest.update((ROOT / "config/harness/visual_map.yaml").read_bytes())
    return digest.hexdigest()[:16]


def conditions_fingerprint(config):
    snapshot = config.snapshot()
    snapshot.pop("sources", None)
    digest = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode())
    maps = Path(config.harness["scene_maps"])
    for description in sorted(maps.glob("*/*/*/scene_description.md")):
        archive = description.parent
        for path in sorted(archive.rglob("*")):
            if path.is_file() and (
                path.name
                in {"floor_plan.png", "scene_description.md", "evidence_rooms.jpg"}
                or path.relative_to(archive).as_posix()
                in {"raw/scene.json", "raw/regions.json", "raw/floor_plan.svg"}
                or "evidence" in path.relative_to(archive).parts
                and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ):
                digest.update(str(path.relative_to(maps)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def config_for(family, stage, model, limit=None, max_requests=None, fine_actions=False):
    phase = "test" if stage == "validate" else "dev"
    config = load_config(
        {
            "TASK_CONFIG": f"config/task/{family}_{phase}_visual.yaml",
            "ENV_CONFIG": (
                "config/environment/objectnav_visual_fine.yaml"
                if fine_actions and family == "objnav"
                else f"config/environment/{ENVIRONMENTS[family]}_visual.yaml"
            ),
            "MODEL_CONFIG": MODELS[model],
            "HARNESS_CONFIG": "config/harness/visual_map.yaml",
            "RUN_CONFIG": "config/run/visual.yaml",
        }
    )
    config.run["name"] = f"visual-{stage}-{family}-{model}"
    if fine_actions and family == "objnav":
        config.run["name"] += "-fine"
    if limit is not None:
        config.episodes["limit"] = limit
    if max_requests is not None:
        config.run["max_requests"] = max_requests
    config.harness["version"] = harness_version()
    return config


def score(summary, family):
    if family == "goat" and summary.get("subgoal_success_over_planned") is not None:
        return summary["subgoal_success_over_planned"]
    return summary.get("score_over_selected") or 0.0


def run_batch(
    model,
    stage,
    families,
    rounds,
    limit=None,
    max_requests=None,
    fine_actions=False,
    *,
    on_completed=None,
):
    rows = []
    frozen = {
        family: hashlib.sha256(
            Path(config_for(family, stage, model).episodes["manifest"]).read_bytes()
        ).hexdigest()
        for family in families
    }
    version = harness_version()
    conditions = {
        family: conditions_fingerprint(
            config_for(family, stage, model, limit, max_requests, fine_actions)
        )
        for family in families
    }

    def verify(config, family, phase):
        digest = hashlib.sha256(
            Path(config.episodes["manifest"]).read_bytes()
        ).hexdigest()
        fingerprint = conditions_fingerprint(config)
        if (
            digest != frozen[family]
            or harness_version() != version
            or fingerprint != conditions[family]
        ):
            raise RuntimeError(
                "Frozen task manifest or harness changed between rounds "
                f"({phase} {model}/{family}, round {round_index + 1})"
            )
        return digest, fingerprint

    for round_index in range(rounds):
        for family in families:
            config = config_for(family, stage, model, limit, max_requests, fine_actions)
            digest, fingerprint = verify(config, family, "before")
            directory, summary = run_experiment(config)
            row = {
                "model": model,
                "family": family,
                "round": round_index + 1,
                "run": "runs/" + directory.name,
                "status": "completed",
                "valid": True,
                "score": score(summary, family),
                "manifest_sha256": digest,
                "harness_version": version,
                "model_id": config.model["model"],
                "conditions_sha256": fingerprint,
                "summary": summary,
            }
            try:
                # Reload settings: workers may have loaded edits made during a run.
                current = config_for(
                    family, stage, model, limit, max_requests, fine_actions
                )
                verify(current, family, "during")
            except (KeyboardInterrupt, Exception) as error:
                row.update(
                    status="invalidated",
                    valid=False,
                    score=None,
                    invalidation_reason=str(error)
                    or "Post-run frozen-condition verification interrupted",
                )
                rows.append(row)
                if on_completed is not None:
                    on_completed(row)
                if isinstance(error, KeyboardInterrupt):
                    raise
                raise RuntimeError(row["invalidation_reason"]) from error
            rows.append(row)
            if on_completed is not None:
                on_completed(row)
    return rows


def qualifies(rows, threshold=0.8):
    return bool(rows) and all(
        row.get("valid", True) and row["score"] >= threshold for row in rows
    )


def acceptance_qualifies(rows):
    if not qualifies(rows):
        return False
    if (
        len({row.get("harness_version") for row in rows}) != 1
        or len({row.get("model_id") for row in rows}) != 1
    ):
        return False
    for family in ENVIRONMENTS:
        batch = [row for row in rows if row["family"] == family]
        if (
            len(batch) != 2
            or {row.get("round") for row in batch} != {1, 2}
            or len({row.get("manifest_sha256") for row in batch}) != 1
            or len({row.get("conditions_sha256") for row in batch}) != 1
        ):
            return False
        if any(
            row["summary"]["selected"] != 20 or row["summary"]["finished"] != 20
            for row in batch
        ):
            return False
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["smoke", "develop", "validate"], default="develop"
    )
    parser.add_argument("--model", choices=list(MODELS), default="qwen27")
    parser.add_argument(
        "--fine-actions",
        action="store_true",
        help="Use 0.1m ObjectNav forward actions; success threshold is unchanged",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Develop: Qwen27, conditional Qwen35, then DeepSeek/Sol when needed",
    )
    parser.add_argument(
        "--families", nargs="+", choices=list(ENVIRONMENTS), default=list(ENVIRONMENTS)
    )
    parser.add_argument("--rounds", type=int)
    parser.add_argument(
        "--limit",
        type=int,
        help="Development/smoke only; validation always uses the frozen full set",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        help="Diagnostic runs only; excludes acceptance claims",
    )
    args = parser.parse_args(argv)
    if len(args.families) != len(set(args.families)):
        parser.error("families must be unique")
    if args.stage == "validate" and (
        args.auto or args.limit is not None or args.max_requests is not None
    ):
        parser.error(
            "Validation uses a frozen model, full selection and planned budgets"
        )
    if args.auto and set(args.families) != set(ENVIRONMENTS):
        parser.error("Automatic model escalation requires all three families")
    rounds = (
        args.rounds
        if args.rounds is not None
        else (2 if args.stage == "validate" else 1)
    )
    if (
        rounds < 1
        or (args.limit is not None and args.limit < 1)
        or (args.max_requests is not None and args.max_requests < 1)
    ):
        parser.error("rounds, limit and max-requests must be positive")
    limit = (
        args.limit if args.limit is not None else (1 if args.stage == "smoke" else None)
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = (
        ROOT
        / "cache/visual-harness-v1/experiments"
        / f"{stamp}-{args.stage}-{args.model}"
    )
    rows = []
    result = {
        "stage": args.stage,
        "status": "running",
        "harness_version": harness_version(),
        "diagnostic": args.stage == "smoke"
        or limit is not None
        or args.max_requests is not None,
        "runs": rows,
        "acceptance_passed": False,
    }
    write_json(destination / "experiment.json", result)

    def completed(row):
        rows.append(row)
        write_json(destination / "experiment.json", result)

    def evaluate(model):
        result["active_model"] = model
        write_json(destination / "experiment.json", result)
        batch = run_batch(
            model,
            args.stage,
            args.families,
            rounds,
            limit,
            args.max_requests,
            args.fine_actions,
            on_completed=completed,
        )
        return batch

    try:
        if args.auto:
            baseline = evaluate("qwen27")
            candidates = [baseline]
            if qualifies(baseline, 0.6):
                candidates.append(evaluate("qwen35"))
            if not any(qualifies(batch) for batch in candidates):
                candidates.append(evaluate("deepseek"))
            if not any(qualifies(batch) for batch in candidates):
                candidates.append(evaluate("sol"))
            result["qualified_development_models"] = [
                batch[0]["model"] for batch in candidates if qualifies(batch)
            ]
            result["next_step"] = (
                "Freeze candidate and run validate twice"
                if result["qualified_development_models"]
                else "Inspect failures and make shared harness corrections; model upgrades have been tested"
            )
        else:
            batch = evaluate(args.model)
            result["acceptance_passed"] = (
                args.stage == "validate" and acceptance_qualifies(batch)
            )
    except (KeyboardInterrupt, Exception) as error:
        result.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            acceptance_passed=False,
            error={
                "type": type(error).__name__,
                "message": str(error) or "Interrupted by user",
            },
        )
        write_json(destination / "experiment.json", result)
        raise
    result["status"] = "completed"
    result.pop("active_model", None)
    write_json(destination / "experiment.json", result)
    print(
        json.dumps(
            {
                "experiment": str(destination.relative_to(ROOT)),
                "scores": [
                    {k: row[k] for k in ("model", "family", "round", "score")}
                    for row in rows
                ],
                "acceptance_passed": result["acceptance_passed"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    main()
