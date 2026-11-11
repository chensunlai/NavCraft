"""Run or import bounded evidence reviews and retain their provenance."""
from hashlib import sha256
import json
from pathlib import Path

from src.common.config import read_config

from ..bundle import write_json
from .evidence import apply_reviews, model_review, review_candidates


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def run_review(bundle, root, source, config, candidates=(), imported=None):
    root = Path(root)
    if Path(source).resolve() == (root / "private/reviewed_bundle.json").resolve():
        raise ValueError("Review requires a source distinct from its output")
    private = root / "private"
    rows = list(review_candidates(bundle, root))
    indexed = {row["candidate_id"]: row for row in rows}
    write_json(private / "review_candidates.json", {
        "bundle_sha256": sha256(Path(source).read_bytes()).hexdigest(), "candidates": rows})
    if set(candidates) - indexed.keys():
        raise ValueError("Unknown review candidate ID")
    log = private / "reviews.jsonl"
    records = _rows(log)
    keys = {(r["candidate_id"], r["reviewer"]) for r in records}

    def append(record):
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        records.append(record)
        keys.add((record["candidate_id"], record["reviewer"]))

    if imported is not None:
        imported = Path(imported)
        if not imported.is_file():
            raise FileNotFoundError(imported)
        additions = []
        seen = keys.copy()
        for row in _rows(imported):
            key = row["candidate_id"], row["reviewer"]
            if key not in seen:
                additions.append(row)
                seen.add(key)
        apply_reviews(bundle, root, records + additions)
        for row in additions:
            append(row)
    else:
        apply_reviews(bundle, root, records)
        settings = config.get("review", {})
        if candidates:
            selected = [indexed[key] for key in dict.fromkeys(candidates)]
        else:
            selected = rows[:settings.get("limit", 3)]
        for config_path in settings.get("models", []):
            model_config = read_config(config_path)
            reviewer = model_config["model"]
            effort = model_config.get("reasoning_effort", "high")
            for candidate in selected:
                key = candidate["candidate_id"], reviewer
                if key in keys:
                    continue
                directory = private / "review_attempts" / key[0] / sha256(reviewer.encode()).hexdigest()[:12]
                directory.mkdir(parents=True, exist_ok=False)
                try:
                    record = model_review(Path.cwd(), model_config, directory, root, bundle, candidate, reviewer)
                    record["status"] = "completed"
                except Exception as exc:
                    record = {"candidate_id": key[0], "kind": candidate["kind"], "source": candidate["source"],
                              "reviewer": reviewer, "decision": "uncertain", "checks": {},
                              "status": "service_or_runtime_error", "reason": type(exc).__name__}
                record.update(reasoning_effort=effort, attempt=str(directory.relative_to(root)))
                write_json(directory / "review.json", record)
                append(record)
    result = apply_reviews(bundle, root, records)
    write_json(private / "reviewed_bundle.json", result)
    return result
