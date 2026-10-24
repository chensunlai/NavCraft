import hashlib
import json

from PIL import Image
import pytest

from src.harness.trace import TraceWriter
from src.viewer.export import TEXT_LIMIT, export_sequence


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture_run(tmp_path, *, memory_events=True):
    run = tmp_path / "run"
    model = run / "sol"
    workspace = model / "workspace"
    observation = workspace / "observations/f_0001.jpg"
    observation.parent.mkdir(parents=True)
    Image.new("RGB", (1600, 1200), "red").save(observation)
    second = observation.with_name("f_0002.jpg")
    second.write_bytes(observation.read_bytes())
    frames = [{"frame": f"f_{index:04d}", "step": index - 1, "pose": {"pos": [.4, .5], "heading": 90},
               "path": f"observations/f_{index:04d}.jpg", "kind": "observation"} for index in (1, 2)]
    (workspace / "observations/index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in frames))
    scene_map = {"origin": [-10, -10], "size_m": 20, "walls": [[[0, 0], [1, 0]]], "floor_height": .05}
    write_json(workspace / "scene/map.json", scene_map)
    memory1 = [{"id": "O01", "type": "object", "point": [.5, .5], "name": "chair", "evidence": ["f_0001"]}]
    memory2 = [{**memory1[0], "name": "armchair", "evidence": ["f_0001", "f_0002"]}]
    write_json(workspace / "memory/entities.json", memory2)
    specifications = [{"id": "explore", "kind": "explore", "instruction": "Explore public room"},
                      {"id": "navigate", "kind": "navigation", "instruction": "Approach the chair", "hidden_answer": "PRIVATE_LABEL"}]
    write_json(model / "private/scenario.json", {"scene_id": "fixture.glb", "protocol": "fixture", "rounds": specifications})
    rows = []
    for index, specification in enumerate(specifications, 1):
        row = {"round": index, "id": specification["id"], "kind": specification["kind"], "submitted": True,
               "reason": "submitted", "actions": index - 1, "requests": 1, "path_length": .25 * (index - 1),
               "memory": {"entities": 1, "types": {"object": 1}, "tools": {"memory_read": 1, "memory_edit": 1},
                          "operations": {"create": index == 1, "update": index == 2}, "tool_errors": 0}}
        if index == 2:
            row.update(success=True, arrived=True, navigation_error=.1, spl=.8, route={"passed": True})
        rows.append(row)
        write_json(model / f"private/rounds/{index:02d}-{row['id']}.json", {**row, "trajectory": [[0, 0, 0], [.25, 0, 0]]})
    write_json(model / "result.json", {"model": "test-model", "reasoning_effort": "high", "status": "completed",
               "planned_navigation": 1, "metrics": {"navigation_successes": 1}, "rounds": rows})
    trace = TraceWriter(model / "private")
    trace.emit("http_request", {"authorization": "SECRET_TOKEN", "body": "HTTP_PRIVATE"})
    trace.emit("model_input", {"hidden_answer": "PRIVATE_LABEL"})
    for index in (1, 2):
        trace.emit("prompt_start", {"text": specifications[index - 1]["instruction"], "images": [frames[index - 1]["path"]],
                                   "limits": {"max_requests": 30}})
        trace.emit("message_end", {"message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "PRIVATE_THINKING", "thinkingSignature": "OPAQUE_SIGNATURE"},
            {"type": "text", "text": f"Visible decision {index}", "textSignature": "PRIVATE_SIGNATURE"}]}})
        trace.emit("tool_execution_start", {"toolCallId": f"call{index}", "toolName": "nav_observe", "args": {"mode": "current"}})
        public = {"content": [{"type": "text", "text": json.dumps({"frame": frames[index - 1]["frame"], "step": index - 1})}],
                  "images": [frames[index - 1]["path"]], "image_labels": ["[CURRENT]"], "isError": False}
        trace.emit("public_tool_reply", {"call_id": f"call{index}", "name": "nav_observe", "args": {"mode": "current"},
                                        "result": public, "checkpoint": {"images": [frames[index - 1]["path"]]}})
        trace.emit("tool_execution_end", {"toolCallId": f"call{index}", "toolName": "nav_observe", "result": public})
        trace.emit("tool_execution_start", {"toolCallId": f"native{index}", "toolName": "bash", "args": {"command": "printf test"}})
        trace.emit("tool_execution_end", {"toolCallId": f"native{index}", "toolName": "bash", "result": {
            "content": [{"type": "text", "text": "X" * (TEXT_LIMIT + 20)}]}, "isError": False})
        if memory_events:
            trace.emit("memory_changed", {"before": trace.store([] if index == 1 else memory1),
                                          "after": trace.store(memory1 if index == 1 else memory2)})
        trace.emit("prompt_end", {"submitted": True})
    trace.close()
    return run


def test_export_preserves_actual_public_events_and_exact_memory_without_private_inputs(tmp_path):
    run = fixture_run(tmp_path)
    output = tmp_path / "site"
    export_sequence(run, output)
    index = json.loads((output / "data/index.json").read_text())
    model = index["models"][0]
    assert model["successes"] == model["navigation_total"] == 1
    assert model["rounds"][0]["success"] is None
    assert model["rounds"][1]["success"] is True
    assert model["memory_reads"] == model["memory_writes"] == 2
    first, second = [json.loads((output / row["detail"]).read_text()) for row in model["rounds"]]
    assert [row["id"] for row in first["frames"]] == ["f_0001"]
    assert [row["id"] for row in second["frames"]] == ["f_0002"]
    assert first["memory_before"] == [] and first["memory_changes"]["created"] == ["O01"]
    assert second["memory_before"] == first["memory_after"]
    assert second["memory_changes"]["revised"] == ["O01"]
    assert set(second["evidence"]) == {"f_0001", "f_0002"}
    assert all((output / item["url"]).is_file() for item in second["evidence"].values())
    tools = [event for event in first["events"] if event["type"] == "tool"]
    assert [tool["name"] for tool in tools] == ["nav_observe", "bash"]
    assert tools[1]["args"] == {"command": "printf test"} and tools[1]["truncated"]
    assert "truncated" in tools[1]["result"]
    text = "".join(path.read_text() for path in (output / "data").rglob("*.json"))
    for forbidden in ("PRIVATE_LABEL", "SECRET_TOKEN", "HTTP_PRIVATE", "PRIVATE_THINKING", "OPAQUE_SIGNATURE", "PRIVATE_SIGNATURE", "max_requests"):
        assert forbidden not in text
    assert len(list((output / "assets").glob("*.jpg"))) == 1
    for detail in (first, second):
        for image in detail["frames"] + [image for event in detail["events"] for image in event["images"]]:
            with Image.open(output / image["url"]) as picture:
                assert max(picture.size) <= 960 and picture.format == "JPEG"


def test_absent_snapshots_do_not_pass_final_memory_off_as_earlier_memory(tmp_path):
    run = fixture_run(tmp_path, memory_events=False)
    output = tmp_path / "site"
    export_sequence(run, output)
    first = json.loads((output / "data/rounds/sol-01.json").read_text())
    second = json.loads((output / "data/rounds/sol-02.json").read_text())
    assert first["memory_before"] == first["memory_after"] == []
    assert "unavailable" in first["memory_status"]
    assert len(second["memory_after"]) == 1
    assert "Only the final" in second["memory_status"]


def test_native_image_blobs_are_copied_without_exporting_base64(tmp_path):
    run = fixture_run(tmp_path)
    private = run / "sol/private"
    image = run / "sol/workspace/observations/f_0001.jpg"
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    (private / "blobs" / digest).write_bytes(image.read_bytes())
    lines = (private / "trace.jsonl").read_text().splitlines()
    event = {"type": "tool_execution_end", "data": {"toolCallId": "read-image", "toolName": "read", "result": {
        "content": [{"type": "image", "data": {"$trace_blob": digest, "encoding": "base64"}, "mimeType": "image/jpeg"}]}}}
    first_prompt = next(i for i, line in enumerate(lines) if json.loads(line)["type"] == "prompt_start")
    lines.insert(first_prompt + 1, json.dumps(event))
    (private / "trace.jsonl").write_text("\n".join(lines) + "\n")
    output = tmp_path / "site"
    export_sequence(run, output)
    first = json.loads((output / "data/rounds/sol-01.json").read_text())
    native = next(event for event in first["events"] if event.get("name") == "read")
    assert len(native["images"]) == 1 and (output / native["images"][0]["url"]).is_file()
    assert "$trace_blob" not in json.dumps(first)


def test_export_refuses_the_source_directory(tmp_path):
    run = fixture_run(tmp_path)
    with pytest.raises(ValueError, match="separate"):
        export_sequence(run, run)


def test_explicit_empty_initial_memory_is_known_before_the_first_edit(tmp_path):
    run = fixture_run(tmp_path, memory_events=False)
    path = run / "sol/private/trace.jsonl"
    path.write_text(json.dumps({"type": "application_started", "data": {"memory_source": "empty"}}) + "\n" + path.read_text())
    output = tmp_path / "site"
    export_sequence(run, output)
    first = json.loads((output / "data/rounds/sol-01.json").read_text())
    assert first["memory_before"] == first["memory_after"] == []
    assert "Inherited" in first["memory_status"]
