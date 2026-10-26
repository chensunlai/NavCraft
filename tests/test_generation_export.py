from contextlib import closing
import hashlib
import io

from PIL import Image

from src.common.io import write_json
from src.dataset.generation.export import check_exports, export
from src.dataset.generation.records import load_records, save_records
from src.harness.trace import TraceWriter


def blob(private, content):
    digest = hashlib.sha256(content).hexdigest()
    path = private / "blobs" / digest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return digest


def make_run(repo, name="attempt-1", *, broken_pair=False, fallback=False, failed_first=False):
    run = repo / "runs" / name
    private = run / "private"
    workspace = run / "workspace"
    workspace.mkdir(parents=True)
    picture = io.BytesIO()
    Image.new("RGB", (16, 12), "purple").save(picture, format="PNG")
    image_hash = blob(private, picture.getvalue())
    image = {"type": "image", "data": {"$trace_blob": image_hash, "encoding": "base64"}, "mimeType": "image/png"}
    memory_hash = blob(private, b'[{"id":"O01","type":"object"}]')
    write_json(private / "requests.json", {"attempts": [{"number": 1}]})
    initial = [
        {"role": "system", "content": "Use the real tools."},
        {"role": "user", "content": [{"type": "text", "text": "Find the chair."}]},
        {"role": "assistant", "content": [{"type": "toolCall", "id": "script-observe", "name": "nav_observe", "arguments": {}}]},
        {"role": "toolResult", "toolCallId": "script-observe", "toolName": "nav_observe", "content": [image]},
    ]
    first = {"role": "assistant", "stopReason": "toolUse", "content": [
        {"type": "thinking", "thinking": "Opaque private reasoning", "thinkingSignature": "signed-state"},
        {"type": "text", "text": "I will inspect the existing record.", "textSignature": "opaque-text-state"},
        {"type": "toolCall", "id": "read-1", "name": "memory_read", "arguments": {"ids": ["O01"]}},
    ]}
    receipt = {"role": "toolResult", "toolCallId": "read-1", "toolName": "memory_read",
               "content": [{"type": "text", "text": '{"entities":[{"id":"O01"}]}'}]}
    second = {"role": "assistant", "stopReason": "toolUse", "content": [
        {"type": "toolCall", "id": "submit-1", "name": "submit", "arguments": {"answer": "O01"}}]}
    with closing(TraceWriter(private)) as trace:
        trace.emit("workspace_changes", {"label": "initial", "changes": [
            {"path": "memory/entities.json", "after": memory_hash}]})
        # Summary requests and failed outputs cannot create supervised decisions.
        trace.emit("checkpoint_summary_input", {"context": {"messages": []}})
        trace.emit("model_input", {"model": "gpt-6.1-sol", "context": {"messages": []}})
        trace.emit("checkpoint_summary_output", {"role": "assistant", "content": [{"type": "text", "text": "summary-only"}]})
        if failed_first:
            trace.emit("model_input", {"model": "gpt-6.1-sol", "context": {"messages": initial}})
            trace.emit("message_end", {"message": {"role": "assistant", "stopReason": "error",
                        "content": [{"type": "text", "text": "failed output must not train"}]}})
        trace.emit("model_input", {"model": "gpt-6.1-sol", "reasoning": "high", "api": "openai-responses",
                    "context": {"messages": initial, "tools": [{"name": "memory_read"}]},
                    "files": {"memory/entities.json": memory_hash}})
        trace.emit("message_end", {"message": first})
        if fallback:
            trace.emit("public_tool_reply", {"call_id": "read-1", "name": "memory_read",
                        "result": {"content": receipt["content"], "images": [], "isError": False}})
        else:
            trace.emit("message_end", {"message": receipt})
        trace.emit("model_input", {"model": "gpt-6.1-sol", "reasoning": "high", "api": "openai-responses",
                    "context": {"messages": [*initial, first, receipt]}, "files": {}})
        trace.emit("message_end", {"message": second})
        if not broken_pair:
            trace.emit("message_end", {"message": {"role": "toolResult", "toolCallId": "submit-1",
                        "toolName": "submit", "content": [{"type": "text", "text": '{"submitted":true}'}]}})
        trace.emit("private_score", {"correct_answer": "NEVER-PUT-THIS-IN-TRAINING"})
    return run.relative_to(repo).as_posix(), image_hash


def collection(root):
    root.mkdir()
    save_records(root / "samples.jsonl", [{"id": "sample-1", "task_id": "6a", "form": "qa",
                 "group_id": "scene-1-cut-1", "split": "train"}])
    return root


def test_export_actual_decisions_images_and_loss_boundaries(tmp_path):
    repo = tmp_path / "repo"
    root = collection(tmp_path / "collection")
    run, image_hash = make_run(repo)
    save_records(root / "solutions.jsonl", [{"sample_id": "sample-1", "task_id": "6a", "attempt": 1,
                 "run_dir": run, "score": {"success": True}, "status": "completed"}])
    report = export(root, {}, repo)
    assert report["exported_conversations"] == 1
    assert report["exported_decisions"] == 2
    assert report["validation"]["valid"]
    assert len(report["images"]) == 1
    assert report["images"][0]["sha256"] == image_hash
    assert report["images"][0]["path"].endswith(".png")
    decisions = load_records(root / "exports/decisions.jsonl")
    assert decisions[0]["model"] == "gpt-6.1-sol"
    assert decisions[0]["reasoning_effort"] == "high"
    assert [row["type"] for row in decisions[0]["target"]["content"]] == ["text", "toolCall"]
    assert decisions[0]["target"]["content"][0] == {"type": "text", "text": "I will inspect the existing record."}
    assert decisions[0]["input_trainable"] is False
    assert len(decisions[0]["input"]["messages"]) == 4
    assert len(decisions[1]["input"]["messages"]) == 6
    serialized = (root / "exports/decisions.jsonl").read_text()
    assert "$trace_blob" not in serialized and '"data":' not in serialized
    assert "NEVER-PUT-THIS-IN-TRAINING" not in serialized
    conv = load_records(root / "exports/conversations.jsonl")[0]
    assert len([m for m in conv["messages"] if m.get("trainable")]) == 2
    assert not any(m["trainable"] for m in conv["messages"][:4])
    assert all(not m["trainable"] for m in conv["messages"] if m["role"] == "toolResult")
    assert conv["workspace_changes"][0]["after_file"].startswith("files/")


def test_export_selects_first_successful_attempt_and_keeps_failures_out(tmp_path):
    repo = tmp_path / "repo"
    root = collection(tmp_path / "collection")
    run, _ = make_run(repo, "attempt-2")
    save_records(root / "solutions.jsonl", [
        {"sample_id": "sample-1", "attempt": 1, "run_dir": "missing-failed", "score": {"success": False}},
        {"sample_id": "sample-1", "attempt": 2, "run_dir": run, "score": {"success": True}},
        {"sample_id": "sample-1", "attempt": 3, "run_dir": "missing-later", "score": {"success": True}},
    ])
    report = export(root, {}, repo)
    assert report["attempt_records"] == 3
    assert report["teacher_attempts"] == 1
    assert report["exported_conversations"] == 1
    assert not report["rejected"]
    assert load_records(root / "exports/conversations.jsonl")[0]["attempt"] == 2
    assert len(load_records(root / "solutions.jsonl")) == 3


def test_unpaired_call_is_rejected_instead_of_fabricating_receipt(tmp_path):
    repo = tmp_path / "repo"
    root = collection(tmp_path / "collection")
    run, _ = make_run(repo, broken_pair=True)
    save_records(root / "solutions.jsonl", [{"sample_id": "sample-1", "run_dir": run, "score": {"success": True}}])
    report = export(root, {}, repo)
    assert not report["exported_conversations"]
    assert "Missing tool result: submit-1" in report["rejected"][0]["reason"]


def test_actual_public_receipt_is_usable_when_native_event_is_absent(tmp_path):
    repo = tmp_path / "repo"
    root = collection(tmp_path / "collection")
    run, _ = make_run(repo, fallback=True)
    save_records(root / "solutions.jsonl", [{"sample_id": "sample-1", "run_dir": run, "score": {"success": True}}])
    report = export(root, {}, repo)
    assert report["exported_conversations"] == 1
    assert report["validation"]["valid"]


def test_checker_detects_tampered_exported_image(tmp_path):
    repo = tmp_path / "repo"
    root = collection(tmp_path / "collection")
    run, _ = make_run(repo)
    save_records(root / "solutions.jsonl", [{"sample_id": "sample-1", "run_dir": run, "score": {"success": True}}])
    report = export(root, {}, repo)
    (root / "exports" / report["images"][0]["path"]).write_bytes(b"wrong")
    checked = check_exports(root / "exports")
    assert not checked["valid"]
    assert any("hash mismatch" in message for message in checked["errors"])


def test_failed_or_absent_solutions_do_not_create_training_targets(tmp_path):
    repo = tmp_path / "repo"
    root = collection(tmp_path / "collection")
    report = export(root, {}, repo)
    assert report["exported_decisions"] == 0
    assert report["validation"]["valid"]
    assert not (root / "exports/decisions.jsonl").read_text()


def test_failed_responses_and_summary_requests_are_not_supervised(tmp_path):
    repo = tmp_path / "repo"
    root = collection(tmp_path / "collection")
    run, _ = make_run(repo, failed_first=True)
    save_records(root / "solutions.jsonl", [{"sample_id": "sample-1", "run_dir": run, "score": {"success": True}}])
    report = export(root, {}, repo)
    assert report["exported_decisions"] == 2
    exported = (root / "exports/decisions.jsonl").read_text()
    assert "failed output must not train" not in exported
    assert "summary-only" not in exported
    skipped = load_records(root / "exports/conversations.jsonl")[0]["skipped_events"]
    assert skipped["failed_assistant"] == 1
    assert skipped["request_without_assistant"] == 1


def test_successful_dev_and_test_solutions_are_evaluation_only(tmp_path):
    repo = tmp_path / "repo"
    root = collection(tmp_path / "collection")
    run, _ = make_run(repo)
    train = load_records(root / "samples.jsonl")[0]
    save_records(root / "samples.jsonl", [train, {**train, "id": "dev-1", "split": "dev", "group_id": "dev-group"},
                                           {**train, "id": "test-1", "split": "test", "group_id": "test-group"}])
    solutions = [{"sample_id": "sample-1", "run_dir": run, "score": {"success": True}},
                 {"sample_id": "dev-1", "run_dir": "evaluation-dev-not-loaded", "score": {"success": True}},
                 {"sample_id": "test-1", "run_dir": "evaluation-test-not-loaded", "score": {"success": True}}]
    save_records(root / "solutions.jsonl", solutions)
    report = export(root, {}, repo)
    assert report["verified_samples"] == 3
    assert report["excluded_by_split"] == {"dev": 1, "test": 1}
    assert report["exported_conversations"] == 1
    assert not report["rejected"]
    decisions = load_records(root / "exports/decisions.jsonl")
    assert all(row["sample_id"] == "sample-1" and row["split"] == "train" for row in decisions)
    assert load_records(root / "solutions.jsonl") == solutions
    decisions[0]["split"] = "dev"
    save_records(root / "exports/decisions.jsonl", decisions)
    checked = check_exports(root / "exports")
    assert not checked["valid"]
    assert any("Non-training split" in message for message in checked["errors"])


def test_trusted_runs_root_may_be_a_symlink_outside_repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    storage = tmp_path / "external-run-storage"
    storage.mkdir()
    (repo / "runs").symlink_to(storage, target_is_directory=True)
    root = collection(tmp_path / "collection")
    run, _ = make_run(repo)
    save_records(root / "solutions.jsonl", [{"sample_id": "sample-1", "run_dir": run, "score": {"success": True}}])
    report = export(root, {}, repo)
    assert report["exported_conversations"] == 1
    assert report["validation"]["valid"]
    assert not report["rejected"]


def test_teacher_run_cannot_escape_trusted_runs_root(tmp_path):
    repo = tmp_path / "repo"
    (repo / "runs").mkdir(parents=True)
    storage = tmp_path / "untrusted"
    storage.mkdir()
    (repo / "runs/escape").symlink_to(storage, target_is_directory=True)
    root = collection(tmp_path / "collection")
    for run_dir, message in [("runs/../escape", "repository-relative"),
                             (str(storage), "repository-relative"),
                             ("other/place", "under runs"),
                             ("runs/escape", "escapes trusted runs directory")]:
        save_records(root / "solutions.jsonl", [{"sample_id": "sample-1", "run_dir": run_dir, "score": {"success": True}}])
        report = export(root, {}, repo)
        assert report["exported_conversations"] == 0
        assert message in report["rejected"][0]["reason"]
