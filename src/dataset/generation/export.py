"""Export verified teacher decisions from the inputs and outputs actually recorded."""

from collections import Counter
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re

from PIL import Image

from src.common.io import write_json
from src.harness.trace import read_trace

from .records import load_records, save_records, stable_id, update_manifest
from .teacher import model_started


class _Assets:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.images = {}
        self.files = {}

    def _write(self, relative, content):
        path = self.directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(content)
        elif path.read_bytes() != content:
            raise ValueError(f"Export asset disagrees with its hash: {relative}")

    def image(self, content, expected=None):
        digest = hashlib.sha256(content).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError("Image trace blob hash mismatch")
        if digest not in self.images:
            with Image.open(io.BytesIO(content)) as image:
                extension = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "GIF": "gif"}.get(image.format)
                if extension is None:
                    raise ValueError(f"Unsupported recorded image format: {image.format}")
                dimensions = list(image.size)
                image.verify()
            relative = f"images/{digest}.{extension}"
            self._write(relative, content)
            self.images[digest] = {"path": relative, "sha256": digest, "bytes": len(content),
                                   "size": dimensions}
        return self.images[digest]["path"]

    def file(self, private, digest):
        content = self.blob(private, digest)
        relative = f"files/{digest}"
        self._write(relative, content)
        self.files[digest] = {"path": relative, "sha256": digest, "bytes": len(content)}
        return relative

    @staticmethod
    def blob(private, digest):
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Invalid trace blob digest")
        path = Path(private) / "blobs" / digest
        if path.is_symlink():
            raise ValueError("Trace blobs cannot be symlinks")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("Trace blob hash mismatch")
        return content

    def externalize(self, value, private):
        if isinstance(value, list):
            return [self.externalize(child, private) for child in value]
        if not isinstance(value, dict):
            if isinstance(value, str) and value.startswith("data:image/"):
                raise ValueError("Inline image payload found instead of a recorded trace blob")
            return value
        if "$trace_blob" in value:
            if value.get("encoding") not in {"base64", "data-url"}:
                raise ValueError("Unexpected unresolved non-image trace blob")
            digest = value["$trace_blob"]
            return {"path": self.image(self.blob(private, digest), digest)}
        result = {}
        for key, child in value.items():
            if value.get("type") == "image" and key == "data":
                if not isinstance(child, dict) or "$trace_blob" not in child:
                    raise ValueError("Image payload was not archived as a trace blob")
                result["path"] = self.externalize(child, private)["path"]
            else:
                result[key] = self.externalize(child, private)
        return result

    def public_image(self, workspace, reference):
        path = (Path(workspace) / reference).resolve()
        if not path.is_relative_to(Path(workspace).resolve()) or not path.is_file():
            raise ValueError(f"Missing or external public image: {reference}")
        return self.image(path.read_bytes())


def _visible_target(message):
    """Only visible text and executable calls are supervised output."""
    content = []
    for block in message.get("content", []):
        if block.get("type") == "text" and str(block.get("text", "")).strip():
            content.append({"type": "text", "text": block["text"]})
        elif block.get("type") == "toolCall":
            content.append({key: deepcopy(block[key]) for key in ("type", "id", "name", "arguments")})
    if not content:
        return None
    return {"role": "assistant", "content": content}


def _calls(message):
    return {block["id"]: block["name"] for block in message.get("content", [])
            if isinstance(block, dict) and block.get("type") == "toolCall"}


def _pairing_errors(messages):
    pending, errors = {}, []
    for message in messages:
        if message.get("role") == "assistant":
            for call_id, name in _calls(message).items():
                if call_id in pending:
                    errors.append(f"Duplicate tool call: {call_id}")
                pending[call_id] = name
        elif message.get("role") == "toolResult":
            call_id = message.get("toolCallId")
            expected = pending.pop(call_id, None)
            if expected is None:
                errors.append(f"Unpaired tool result: {call_id}")
            elif message.get("toolName") != expected:
                errors.append(f"Wrong tool result name: {call_id}")
    errors.extend(f"Missing tool result: {call_id}" for call_id in pending)
    return errors


def _one_run(solution, sample, repo, assets):
    reference = Path(solution["run_dir"])
    if reference.is_absolute() or ".." in reference.parts:
        raise ValueError("Teacher run directory must be repository-relative")
    if not reference.parts or reference.parts[0] != "runs":
        raise ValueError("Teacher run directory must be under runs")
    run = (Path(repo) / reference).resolve()
    if not run.is_relative_to((Path(repo) / "runs").resolve()):
        raise ValueError("Teacher run directory escapes trusted runs directory")
    private, workspace = run / "private", run / "workspace"
    pending, initial, decisions, timeline, receipts, calls = None, None, [], [], {}, {}
    changes, skipped = [], Counter()
    for index, event in enumerate(read_trace(private / "trace.jsonl")):
        kind, data = event["type"], event.get("data", {})
        if kind == "model_input":
            if pending is not None:
                skipped["request_without_assistant"] += 1
            pending = (index, event)
            continue
        if kind == "workspace_changes":
            for change in data.get("changes", []):
                row = dict(change)
                for key in ("before", "after"):
                    if row.get(key):
                        row[key + "_file"] = assets.file(private, row[key])
                changes.append(row)
            continue
        if kind == "public_tool_reply":
            call_id = data.get("call_id")
            if call_id not in calls or call_id in receipts:
                continue
            result = data["result"]
            content = assets.externalize(result.get("content", []), private)
            labels = result.get("image_labels", [])
            for image_index, ref in enumerate(result.get("images", [])):
                if image_index < len(labels):
                    content.append({"type": "text", "text": labels[image_index]})
                content.append({"type": "image", "path": assets.public_image(workspace, ref)})
            receipts[call_id] = {"role": "toolResult", "toolCallId": call_id,
                                 "toolName": data["name"], "content": content,
                                 "isError": result.get("isError", False), "trainable": False}
            timeline.append({"receipt": call_id})
            continue
        if kind != "message_end":
            continue
        message = data.get("message", {})
        if message.get("role") == "toolResult":
            call_id = message.get("toolCallId")
            if call_id not in calls:
                continue
            if call_id not in receipts:
                timeline.append({"receipt": call_id})
            receipts[call_id] = {**assets.externalize(message, private), "trainable": False}
            continue
        if message.get("role") != "assistant" or pending is None:
            continue
        request_index, request = pending
        pending = None
        if message.get("stopReason") in {"error", "aborted", "length"}:
            skipped["failed_assistant"] += 1
            continue
        target = _visible_target(message)
        if target is None:
            skipped["no_visible_output"] += 1
            continue
        request_data = request["data"]
        context = assets.externalize(request_data["context"], private)
        if initial is None:
            initial = [{**deepcopy(row), "trainable": False} for row in context.get("messages", [])]
        decision_id = stable_id(sample["id"], solution.get("attempt", 1), request_index)
        files = {name: {"sha256": digest, "path": assets.file(private, digest)}
                 for name, digest in request_data.get("files", {}).items()}
        decisions.append({"id": decision_id, "sample_id": sample["id"], "task_id": sample["task_id"],
                          "form": sample["form"], "group_id": sample["group_id"], "split": sample.get("split", "train"),
                          "source": "teacher", "attempt": solution.get("attempt", 1),
                          "model": request_data.get("model"), "api": request_data.get("api"),
                          "reasoning_effort": request_data.get("reasoning"),
                          "input": context, "input_trainable": False,
                          "target": target, "trainable": True,
                          "files": files,
                          "record": {"run_dir": str(reference), "input_event": request_index,
                                     "output_event": index, "prompt_id": request.get("prompt_id")}})
        timeline.append({**deepcopy(target), "trainable": True, "decision_id": decision_id})
        calls.update(_calls(target))
    if pending is not None:
        skipped["request_without_assistant"] += 1
    if not decisions:
        raise ValueError("No paired, visible successful teacher decision in the trace")
    messages = list(initial or [])
    for item in timeline:
        if "receipt" in item:
            messages.append(receipts[item["receipt"]])
        else:
            messages.append(item)
    errors = _pairing_errors(messages)
    if errors:
        raise ValueError("; ".join(errors))
    conversation = {"sample_id": sample["id"], "task_id": sample["task_id"], "form": sample["form"],
                    "group_id": sample["group_id"], "split": sample.get("split", "train"),
                    "source": "teacher", "attempt": solution.get("attempt", 1),
                    "model": decisions[0]["model"], "reasoning_effort": decisions[0]["reasoning_effort"],
                    "messages": messages, "decision_ids": [row["id"] for row in decisions],
                    "workspace_changes": changes, "skipped_events": dict(skipped),
                    "record": {"run_dir": str(reference)}}
    return conversation, decisions


def export(root, config, repo):
    """Write first successful train attempts; keep dev/test records evaluation-only."""
    root, repo = Path(root), Path(repo).resolve()
    output = root / "exports"
    output.mkdir(parents=True, exist_ok=True)
    samples = {row["id"]: row for row in load_records(root / "samples.jsonl")}
    solutions_file = root / "solutions.jsonl"
    solutions = load_records(solutions_file) if solutions_file.exists() else []
    selected = {}
    for row in solutions:
        if row.get("score", {}).get("success") is True:
            selected.setdefault(row["sample_id"], row)
    assets, conversations, decisions, rejected = _Assets(output), [], [], []
    excluded_by_split = Counter()
    for sample_id, solution in selected.items():
        try:
            if sample_id not in samples:
                raise ValueError("Solution references a missing sample")
            split = samples[sample_id].get("split", "train")
            if split != "train":
                excluded_by_split[split] += 1
                continue
            conversation, rows = _one_run(solution, samples[sample_id], repo, assets)
            conversations.append(conversation)
            decisions.extend(rows)
        except (OSError, ValueError, KeyError, TypeError) as error:
            rejected.append({"sample_id": sample_id, "run_dir": solution.get("run_dir"), "reason": str(error)})
    save_records(output / "conversations.jsonl", conversations)
    save_records(output / "decisions.jsonl", decisions)
    actual_attempts = sum(model_started(Path(repo) / row["run_dir"]) for row in solutions)
    report = {"samples": len(samples), "attempt_records": len(solutions), "teacher_attempts": actual_attempts,
              "verified_samples": len(selected), "exported_conversations": len(conversations),
              "exported_decisions": len(decisions), "rejected": rejected,
              "excluded_by_split": dict(excluded_by_split),
              "by_task": dict(Counter(row["task_id"] for row in conversations)),
              "images": list(assets.images.values()), "files": list(assets.files.values()),
              "training_targets": "Only visible teacher text and tool calls; all input history and tool results are conditions."}
    write_json(output / "report.json", report)
    validation = check_exports(output)
    report["validation"] = validation
    write_json(output / "report.json", report)
    update_manifest(root, exports={"conversations": len(conversations), "decisions": len(decisions),
                                   "rejected": len(rejected), "valid": validation["valid"]})
    return report


def check_exports(directory):
    """Verify exported assets, tool pairing and references without model access."""
    directory = Path(directory).resolve()
    report = json.loads((directory / "report.json").read_text())
    errors = []
    for item in [*report.get("images", []), *report.get("files", [])]:
        path = (directory / item["path"]).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            errors.append(f"Missing or external asset: {item['path']}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            errors.append(f"Asset hash mismatch: {item['path']}")

    def references(value):
        if isinstance(value, list):
            for child in value:
                references(child)
        elif isinstance(value, dict):
            if "$trace_blob" in value:
                errors.append("Unresolved trace blob in training export")
            if value.get("type") == "image":
                path = (directory / value.get("path", "")).resolve()
                if "data" in value or not path.is_relative_to(directory) or not path.is_file():
                    errors.append("Invalid exported image reference")
            for child in value.values():
                references(child)
        elif isinstance(value, str) and value.startswith("data:image/"):
            errors.append("Inline image data in training export")

    decisions = load_records(directory / "decisions.jsonl")
    known = {row["id"] for row in decisions}
    for row in decisions:
        references(row)
        if row.get("split") != "train":
            errors.append(f"Non-training split in training decisions: {row['id']}")
        if not row.get("trainable") or row["target"].get("role") != "assistant" or any(
                block.get("type") not in {"text", "toolCall"} for block in row["target"]["content"]):
            errors.append(f"Invalid training target: {row['id']}")
    for row in load_records(directory / "conversations.jsonl"):
        references(row)
        if row.get("split") != "train":
            errors.append(f"Non-training split in training conversations: {row['sample_id']}")
        errors.extend(_pairing_errors(row["messages"]))
        if not set(row["decision_ids"]) <= known:
            errors.append(f"Missing decision reference: {row['sample_id']}")
        if any(message.get("trainable") for message in row["messages"] if message.get("role") != "assistant"):
            errors.append(f"Non-assistant training target: {row['sample_id']}")
    return {"valid": not errors, "errors": errors}
