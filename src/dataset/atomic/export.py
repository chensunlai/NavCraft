"""Portable training records and a self-contained local review gallery."""

from collections import Counter
from copy import deepcopy
import html
import json
from pathlib import Path

from .schema import COVERAGE, FORMS, write_json


def portable(value, root):
    if isinstance(value, list):
        return [portable(v, root) for v in value]
    if isinstance(value, dict):
        return {k: portable(v, root) for k, v in value.items()}
    if isinstance(value, str) and value.startswith(str(root.resolve()) + "/"):
        return str(Path(value).relative_to(root.resolve()))
    return value


def visible_messages(items):
    """Opaque/summarized server reasoning is not a training target or prompt."""
    cleaned = []
    for item in items:
        if item.get("type") == "reasoning":
            continue
        item = deepcopy(item)
        for key in ("id", "status", "encrypted_content"):
            item.pop(key, None)
        cleaned.append(item)
    return cleaned


def check_dialogue(record, root):
    """Check tool pairing, prefix chronology, and image accessibility."""
    messages = record["messages"]
    pending, completed = set(), set()
    for item in messages:
        kind = item.get("type")
        if kind == "function_call":
            identifier = item["call_id"]
            if identifier in pending or identifier in completed:
                raise ValueError("Duplicate call ID")
            pending.add(identifier)
        elif kind == "function_call_output":
            identifier = item["call_id"]
            if identifier not in pending:
                raise ValueError("Unpaired tool output")
            pending.remove(identifier)
            completed.add(identifier)
        for part in item.get("content", []) if isinstance(item.get("content"), list) else []:
            if part.get("type") == "input_image":
                path = Path(part["image_path"])
                path = path if path.is_absolute() else root / path
                if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
                    raise ValueError(f"Unportable image: {path}")
    if pending:
        raise ValueError("Unanswered tool calls")
    for decision in record["decisions"]:
        prefix = decision["input"]
        if messages[:len(prefix)] != prefix:
            raise ValueError("Decision includes future or altered context")
        if messages[len(prefix):len(prefix) + len(decision["output"])] != decision["output"]:
            raise ValueError("Decision target differs from actual conversation")


def export_build(root):
    root = Path(root).resolve()
    examples, decisions, cards, manifest = [], [], [], []
    for sample in sorted((root / "samples").iterdir()):
        if not (sample / "selected.json").exists():
            continue
        selection = json.loads((sample / "selected.json").read_text())
        record = json.loads((sample / selection["record"]).read_text())
        if not record["validation"]["passed"]:
            raise ValueError(f"Selected failed sample: {sample.name}")
        check_dialogue(record, root)
        task = json.loads((sample / "task.json").read_text())
        labels = json.loads((sample / "private" / "labels.json").read_text())
        excluded_calls = [c["call"]["call_id"] for c in record["calls"] if not c["result"]["ok"]]
        example = {"id": task["id"], "cluster": task["cluster"], "form": task["form"],
                   "observation": task["observation"], "source": record["source"],
                   "tools": record["tools"], "messages": visible_messages(record["messages"]),
                   "training_targets": "assistant visible answers and valid function_call items only; user/tool inputs and rejected calls have zero loss",
                   "excluded_training_call_ids": excluded_calls,
                   "review_ref": str((sample / "selected.json").relative_to(root))}
        examples.append(portable(example, root))
        for decision in record["decisions"]:
            target = visible_messages(decision["output"])
            if not target:
                continue
            decisions.append(portable({"id": f"{task['id']}/{decision['index']}", "sample_id": task["id"],
                                       "source": record["source"], "tools": decision["tools"],
                                       "input": visible_messages(decision["input"]), "target": target,
                                       "trainable": not any(i.get("call_id") in excluded_calls for i in target),
                                       "target_item_mask": [i.get("call_id") not in excluded_calls for i in target]}, root))
        meta = {**task, "source": record["source"], "selected_attempt": record["attempt"],
                "validation": record["validation"], "sample_directory": str(sample.relative_to(root)),
                "revised": record["attempt"] != "teacher" or (sample / "private/design_revision.json").exists(),
                "source_components": {"sampling": "rule_generated", "demonstration": record["source"],
                                      "question_revision": "agent_assisted" if (sample / "private/design_revision.json").exists() else "rule_template"},
                "attempts": [{"path": str(p.relative_to(root)), "source": json.loads(p.read_text())["source"],
                              "validation": json.loads(p.read_text())["validation"]}
                             for p in sorted((sample / "attempts").glob("*/record.json"))],
                "selection": selection}
        manifest.append(meta)
        cards.append(portable({"meta": meta, "messages": visible_messages(record["messages"]),
                               "decisions": [{"index": d["index"], "output": visible_messages(d["output"])} for d in record["decisions"]],
                               "labels": labels, "tools": record["tools"], "excluded_training_call_ids": excluded_calls}, root))
    for name, records in [("conversations.jsonl", examples), ("decisions.jsonl", decisions), ("manifest.jsonl", manifest)]:
        (root / name).write_text("".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in records))
    counts = Counter((r["cluster"], r["form"], r["observation"]) for r in manifest)
    expected = {(cluster, form, observation) for cluster, (_, _, forms) in COVERAGE.items()
                for form in forms for observation in ("single", "panorama")}
    report = {"samples": len(examples), "decisions": len(decisions),
              "coverage_complete": set(counts) == expected and all(v == 1 for v in counts.values()),
              "forms": dict(Counter(r["form"] for r in manifest)),
              "sources": dict(Counter(r["source"] for r in manifest)),
              "revised": sum(r["revised"] for r in manifest),
              "requests": sum(1 for _ in (root / "requests.jsonl").open()) if (root / "requests.jsonl").exists() else 0,
              "limitations": ["Single curated scene; no model success-rate or generalization claim.",
                              "Schematic maps are not exact occupancy maps.",
                              "Reasoning state is excluded from training export; raw responses remain private."]}
    write_json(root / "validation.json", report)
    payload = json.dumps({"cards": cards, "report": report, "forms": FORMS,
                          "clusters": {k: v[0] for k, v in COVERAGE.items()}}, ensure_ascii=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("gallery.html").read_text()
    (root / "index.html").write_text(template.replace("__PAYLOAD__", payload))
    return report
