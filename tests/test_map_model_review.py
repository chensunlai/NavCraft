"""CPU protocol fixtures. These tests never call a model or initialize a GPU."""

import hashlib
import io
import json
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.error

import numpy as np
import jsonschema
from src.data.map_model_review import run_suite


def test_selected_experiment_does_not_repeat_other_model_requests(tmp_path):
    class Client:
        calls = []

        def review(self, prompt, images, **kwargs):
            self.calls.append((prompt, kwargs))
            return {"ok": True, "parsed": {"verdict": "ok", "issues": []}}

    client = Client()
    result = run_suite(tmp_path, client=client, model="Qwen3.8-27B", modes=["baseline"])
    assert len(client.calls) == 1
    assert set(result["methods"]) == {"baseline"}


def test_active_review_receives_world_calibration_not_only_unlocated_map(tmp_path):
    class Client:
        calls = []

        def review(self, prompt, images, **kwargs):
            self.calls.append(prompt)
            return {"ok": True, "parsed": {"verdict": "ok", "issues": []}}

    client = Client()
    result = run_suite(tmp_path, geometry_summary={"origin_xz": [-10, -5],
        "safe_initial_world_xz": [2.5, -1.2]}, client=client, model="Qwen3.8-27B",
        modes=["active"], tools=[{"name": "render_panorama"}])
    assert set(result["methods"]) == {"active"}
    assert '"safe_initial_world_xz": [2.5, -1.2]' in client.calls[0]
from PIL import Image
import pytest

from src.data.map_model_review import (
    MODEL_PROFILES, POINT_SCHEMA, REVIEW_SCHEMA, ReviewClient, aggregate_votes,
    backproject_pixel, parse_structured, project_point_votes, run_suite,
)


ANSWER = {"verdict": "inspect", "issues": [{"type": "opening", "image": "walls.png",
          "pixel_hint": [10, 20], "confidence": 0.7, "reason": "Visible gap needs geometry check"}]}
TOOL = {"type": "function", "name": "render_view", "description": "Render a calibrated view",
        "strict": True, "parameters": {"type": "object", "properties": {"x": {"type": "number"}},
                                       "required": ["x"], "additionalProperties": False}}


class Response:
    def __init__(self, payload, status=200):
        self.payload = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


def response(answer=ANSWER, *, text=None, output=None, status="completed"):
    return {"id": "resp-fixture", "status": status, "output": output if output is not None else [
        {"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": text if text is not None else json.dumps(answer)}]}],
        "usage": {"input_tokens": 20, "output_tokens": 10}}


def function_call(identifier="call-1", arguments='{"x": 2}'):
    return {"type": "function_call", "call_id": identifier, "name": "render_view", "arguments": arguments}


@pytest.fixture
def client(tmp_path):
    env = tmp_path / "config.env"
    env.write_text('NAVCRAFT_TEST_BASE_URL="https://fixture.invalid/v1"\n'
                   "NAVCRAFT_TEST_API_KEY='unit-test-secret'\n")
    return ReviewClient(env, tmp_path / "calls")


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "rgb.png"
    Image.new("RGB", (8, 6), (60, 90, 120)).save(path)
    return path


def fake_http(monkeypatch, replies):
    seen = []
    replies = iter(replies)

    def open_request(request, timeout):
        seen.append((request, timeout))
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return Response(reply)

    monkeypatch.setattr("urllib.request.urlopen", open_request)
    return seen


def audit(result):
    return json.loads(Path(result["record_path"]).read_text())


def body(result):
    return json.loads((Path(result["record_path"]).parent / "request.json").read_text())


def test_responses_request_logs_original_image_and_never_authorization(client, image, monkeypatch):
    seen = fake_http(monkeypatch, [response(), response()])
    first = client.request("Inspect image", [image], model="gpt-6.1-sol", max_output_tokens=9000)
    second = client.request("Inspect image", [image], model="gpt-6.1-sol")
    assert first["ok"] and first["parsed"] == ANSWER
    assert second["ok"] and first["request_id"] != second["request_id"]
    assert seen[0][0].full_url == "https://fixture.invalid/v1/responses"
    assert seen[0][1] == 300
    assert seen[0][0].headers["Authorization"] == "Bearer unit-test-secret"
    request = body(first)
    assert request["model"] == "gpt-6.1-sol" and request["reasoning"]["effort"] == "high"
    assert request["max_output_tokens"] == 9000
    assert request["text"]["format"]["schema"] == REVIEW_SCHEMA
    assert request["input"][0]["content"][-1]["type"] == "input_image"
    assert request["input"][0]["content"][-1]["detail"] == "high"
    record = audit(first)
    assert record["task_prompt"] == "Inspect image"
    assert record["prompt"] == request["input"][0]["content"][0]["text"]
    assert record["prompt"].startswith("Inspect image\n\n")
    assert json.dumps(REVIEW_SCHEMA, ensure_ascii=False, allow_nan=False, sort_keys=True) in record["prompt"]
    assert record["images"][0]["path"] == str(image)
    assert record["images"][0]["sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()
    assert record["usage"] == {"input_tokens": 20, "output_tokens": 10}
    assert record["latency_seconds"] >= 0
    assert record["response_sha256"] == hashlib.sha256(Path(first["response_saved"]).read_bytes()).hexdigest()
    for path in client.output.rglob("*"):
        if path.is_file():
            assert "unit-test-secret" not in path.read_text()


@pytest.mark.parametrize("text", [
    '<｜DSML｜function_calls><｜DSML｜invoke name="render_view">',
    '```json\n{"verdict":"ok","issues":[]}\n```',
    'Answer: {"verdict":"ok","issues":[]}',
    '{"verdict":"ok","verdict":"inspect","issues":[]}',
    '{"verdict":"ok","issues":[],"extra":true}',
    '{"verdict":"wrong","issues":[]}',
    '{"verdict":"ok","issues":[{"type":"opening","image":"x","pixel_hint":[NaN,2],"confidence":0.4,"reason":"x"}]}',
    '{"verdict":"ok","issues":[{"type":"opening","image":"x","pixel_hint":[1e999,2],"confidence":0.4,"reason":"x"}]}',
])
def test_invalid_model_answers_remain_protocol_failures(client, monkeypatch, text):
    seen = fake_http(monkeypatch, [response(text=text)])
    result = client.request("Return schema JSON")
    assert result["ok"] is False and "error" in result
    assert len(seen) == 1  # Parsing failure must not trigger a route fallback.
    saved = json.loads(Path(result["response_saved"]).read_text())
    assert saved["output"][0]["content"][0]["text"] == text
    assert audit(result)["ok"] is False


def test_reasoning_text_is_not_a_final_answer(client, monkeypatch):
    fake_http(monkeypatch, [response(output=[{"type": "reasoning", "summary": [
        {"type": "summary_text", "text": json.dumps(ANSWER)}]}])])
    result = client.request("Return schema JSON")
    assert not result["ok"]
    assert "parsed" not in result


@pytest.mark.parametrize("route", ["responses", "chat"])
def test_audit_records_actual_response_model_separately(client, monkeypatch, route):
    actual = "deepseek-v4.1-flash-1017"
    payload = response() if route == "responses" else {
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(ANSWER)}}]}
    payload["model"] = actual
    fake_http(monkeypatch, [payload])
    result = client.request("Inspect", model="deepseek-flash", route=route)
    assert result["ok"] and result["requested_model"] == "deepseek-flash"
    assert result["response_model"] == actual
    assert audit(result)["response_model"] == actual
    assert body(result)["model"] == "deepseek-flash"


@pytest.mark.parametrize("payload", [
    response(text="AI provider temporarily unavailable (upstream error: 502). Please try again shortly."),
    b"AI provider temporarily unavailable (upstream error: 502)",
    {"error": {"code": "503", "message": "Provider temporarily unavailable"}},
])
def test_provider_outage_notice_is_not_a_json_protocol_error(client, monkeypatch, payload):
    seen = fake_http(monkeypatch, [payload])
    result = client.request("Inspect")
    assert not result["ok"] and result["error"]["type"] == "ProviderUnavailable"
    assert audit(result)["error"]["type"] == "ProviderUnavailable"
    assert result["http_status"] == 200 and len(seen) == 1


@pytest.mark.parametrize("count", [0, 1, 7])
def test_point_schema_rejects_empty_insufficient_or_excess_annotations(count):
    point = {"image": "view.png", "pixel": [2, 1], "type": "wall", "confidence": 0.5, "reason": "visible wall"}
    with pytest.raises(jsonschema.ValidationError):
        parse_structured(json.dumps({"points": [point] * count}), POINT_SCHEMA)


@pytest.mark.parametrize("status", ["incomplete", "failed", "in_progress"])
def test_incomplete_responses_are_not_accepted(client, monkeypatch, status):
    fake_http(monkeypatch, [response(status=status)])
    result = client.request("Inspect")
    assert not result["ok"]
    assert "not complete" in result["error"]["message"]


def test_chat_fallback_has_two_audited_attempts_and_normalized_schema(client, image, monkeypatch):
    error = urllib.error.HTTPError("https://fixture.invalid/v1/responses", 404, "No route", {}, io.BytesIO(b"route missing"))
    chat = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(ANSWER)}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20}}
    seen = fake_http(monkeypatch, [error, chat])
    result = client.request("Inspect", [image], model="Qwen3.8-27B")
    assert result["ok"] and result["route"] == "chat" and result["parsed"] == ANSWER
    assert len(result["attempts"]) == 2
    assert json.loads(Path(result["attempts"][0]).read_text())["http_status"] == 404
    assert seen[1][0].full_url == "https://fixture.invalid/v1/chat/completions"
    request = body(result)
    assert request["reasoning_effort"] == "medium"
    assert request["max_tokens"] == 8192
    assert request["messages"][0]["content"][-1]["type"] == "image_url"
    assert request["response_format"]["json_schema"]["strict"] is True


@pytest.mark.parametrize("code", [400, 401, 429, 500])
def test_http_errors_are_logged_without_retry_or_secret_leak(client, monkeypatch, code):
    error = urllib.error.HTTPError("https://fixture.invalid/v1/responses", code, "bad", {},
                                  io.BytesIO(b'{"error":"unit-test-secret"}'))
    seen = fake_http(monkeypatch, [error])
    result = client.request("Inspect")
    assert not result["ok"] and result["http_status"] == code and len(seen) == 1
    assert audit(result)["response_redacted"] is True
    assert "unit-test-secret" not in Path(result["response_saved"]).read_text()
    assert "[REDACTED]" in Path(result["response_saved"]).read_text()


def test_network_and_preparation_errors_keep_audit(client, monkeypatch, tmp_path):
    fake_http(monkeypatch, [TimeoutError("local fixture timeout")])
    result = client.request("Inspect")
    assert not result["ok"] and result["error"]["type"] == "TimeoutError"
    assert audit(result)["latency_seconds"] >= 0
    missing = client.request("Inspect", [tmp_path / "absent.png"])
    assert not missing["ok"] and audit(missing)["error"]["type"] == "FileNotFoundError"
    client.env_file = tmp_path / "absent.env"
    config = client.request("Inspect")
    assert not config["ok"] and Path(config["record_path"]).exists()


@pytest.mark.parametrize("model,effort", [("deepseek-v4.1-flash-1017", None), ("deepseek-flash", "xhigh")])
def test_only_user_approved_model_profiles_are_used(client, monkeypatch, model, effort):
    seen = fake_http(monkeypatch, [])
    result = client.request("Inspect", model=model, reasoning_effort=effort)
    assert not result["ok"] and not seen


def test_list_models_uses_audited_get(client, monkeypatch):
    seen = fake_http(monkeypatch, [{"data": [{"id": "deepseek-flash"}, {"id": "Qwen3.8-27B"}]}])
    result = client.list_models()
    assert result["ok"] and result["parsed"]["data"][0]["id"] == "deepseek-flash"
    assert seen[0][0].method == "GET" and seen[0][0].data is None
    assert seen[0][0].full_url == "https://fixture.invalid/v1/models"
    assert audit(result)["model"] is None


def test_native_tool_rounds_send_new_rendered_images_and_preserve_call_ids(client, image, tmp_path, monkeypatch):
    rendered = []
    for index in range(2):
        path = tmp_path / f"view-{index}.png"
        Image.new("RGB", (8, 6), (index, 40, 90)).save(path)
        rendered.append(path)
    seen = fake_http(monkeypatch, [response(output=[function_call("c1")]),
                                  response(output=[function_call("c2", '{"x":3}')]), response()])
    called = []

    def render(args):
        called.append(args)
        return {"data": {"camera_x": args["x"]}, "images": [rendered[len(called) - 1]]}

    result = client.review("Actively review", [image], tools=[TOOL],
                           tool_callbacks={"render_view": render}, require_tool=True)
    assert result["ok"] and result["parsed"] == ANSWER
    assert called == [{"x": 2}, {"x": 3}] and len(result["attempts"]) == 3
    bodies = [json.loads(request.data) for request, _ in seen]
    assert bodies[0]["tool_choice"] == "required"
    for index, identifier in [(1, "c1"), (2, "c2")]:
        history = bodies[index]["input"]
        assert history[0]["role"] == "system"
        assert json.dumps(REVIEW_SCHEMA, ensure_ascii=False, allow_nan=False, sort_keys=True) in history[0]["content"]
        assert any(item.get("type") == "function_call" and item["call_id"] == identifier for item in history)
        assert any(item.get("type") == "function_call_output" and item["call_id"] == identifier for item in history)
        assert history[-1]["content"][-1]["type"] == "input_image"
    assert {Path(item["path"]).name for item in audit(result)["images"]} == {"rgb.png", "view-0.png", "view-1.png"}
    assert len(result["tool_events"]) == 2
    assert (Path(result["attempts"][0]).parent / "tool_events.json").exists()


@pytest.mark.parametrize("arguments", ['<DSML/>', '{"x":"not-a-number"}', '{"x":2,"unknown":3}'])
def test_invalid_tool_arguments_never_invoke_callback(client, monkeypatch, arguments):
    fake_http(monkeypatch, [response(output=[function_call(arguments=arguments)])])
    called = []
    result = client.review("Inspect", tools=[TOOL], tool_callbacks={"render_view": called.append})
    assert not result["ok"] and called == []


def test_callback_error_and_tool_budget_have_structured_review_failures(client, monkeypatch):
    fake_http(monkeypatch, [response(output=[function_call()]), response(output=[function_call()])])
    result = client.review("Inspect", tools=[TOOL], tool_callbacks={})
    assert not result["ok"] and result["error"]["type"] == "ValueError"
    assert (Path(result["record_path"]).parent / "review_result.json").exists()
    limited = client.review("Inspect", tools=[TOOL], max_rounds=1)
    assert not limited["ok"] and limited["error"]["type"] == "ToolRoundLimit"


@pytest.mark.parametrize("model", MODEL_PROFILES)
def test_model_concurrency_caps_are_shared_between_clients(client, monkeypatch, model):
    lock = threading.Lock()
    active = peak = 0

    def network(request, timeout):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.015)
        with lock:
            active -= 1
        return Response(response())

    monkeypatch.setattr("urllib.request.urlopen", network)
    other = ReviewClient(client.env_file, client.output)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit((client if index % 2 else other).request, "Inspect", model=model) for index in range(8)]
        assert all(future.result()["ok"] for future in futures)
    assert peak <= MODEL_PROFILES[model]["concurrency"]


def test_backprojection_uses_calibrated_optical_axes_and_rigid_world_transform():
    intrinsics = [[2, 0, 1], [0, 2, 1], [0, 0, 1]]
    transform = np.array([[1, 0, 0, 10], [0, -1, 0, 2], [0, 0, -1, 20], [0, 0, 0, 1]])
    depth = np.full((3, 3), 4.0)
    assert backproject_pixel([2, 0], depth, intrinsics, transform) == [12, 4, 16]
    assert backproject_pixel([1, 1], 3, intrinsics, transform) == [10, 2, 17]


@pytest.mark.parametrize("pixel,depth", [([-1, 0], np.ones((3, 3))), ([3, 0], np.ones((3, 3))),
                                        ([1, 1], 0), ([1, 1], float("nan")), ([float("inf"), 1], 3)])
def test_bad_depth_or_pixel_is_not_projected(pixel, depth):
    with pytest.raises(ValueError):
        backproject_pixel(pixel, depth, np.eye(3), np.eye(4))


def test_point_projection_retains_invalid_and_unknown_image_evidence(image):
    views = [{"image": str(image), "depth": np.full((6, 8), 2),
              "intrinsics": np.eye(3), "world_from_optical": np.eye(4)}]
    points = [{"image": image.name, "pixel": [2, 1], "type": "wall"},
              {"image": "missing.png", "pixel": [2, 1], "type": "wall"},
              {"image": image.name, "pixel": [200, 1], "type": "wall"}]
    projected = project_point_votes(points, views)
    assert projected[0]["world_xyz"] == [4, 2, 2]
    assert [item["ok"] for item in projected] == [True, False, False]


def test_vote_denominator_includes_failures_and_duplicate_issue_is_one_vote():
    repeated = dict(ANSWER, issues=ANSWER["issues"] * 2)
    results = [{"ok": True, "parsed": repeated}, {"ok": False}, {"ok": False}]
    vote = aggregate_votes(results)
    assert vote["groups"][0]["votes"] == 1
    assert not vote["groups"][0]["consensus"]
    results[1] = {"ok": True, "parsed": ANSWER}
    assert aggregate_votes(results)["groups"][0]["consensus"]


def test_suite_executes_all_six_distinct_protocols_without_geometry_edits(client, image, tmp_path, monkeypatch):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    for name in ["walls.png", "overlay.png"]:
        Image.open(image).save(candidate / name)
    before = {path.name: path.read_bytes() for path in candidate.iterdir()}
    points = {"points": [{"image": image.name, "pixel": [2, 1], "type": "door",
                          "confidence": 0.7, "reason": "visible door"},
                         {"image": image.name, "pixel": [3, 1], "type": "wall",
                          "confidence": 0.7, "reason": "visible wall"}]}
    analyst = {"findings": [{"cause": "height", "evidence": "measured gap", "proposed_rule": "sample more heights",
                             "validation": "held-out geometry checks"}]}
    replies = [response(), response(points), response(output=[function_call()]), response(),
               response(), response(), response(), response(analyst), response()]
    seen = fake_http(monkeypatch, replies)
    view = {"image": str(image), "depth": np.ones((6, 8)),
            "intrinsics": np.eye(3).tolist(), "world_from_optical": np.eye(4).tolist()}
    result = run_suite(candidate, [view], {"flagged": ["gap"]}, client=client,
                       tools=[TOOL], tool_callbacks={"render_view": lambda args: {"images": [image]}})
    methods = result["methods"]
    assert set(methods) == {"baseline", "point_depth", "active", "repeatvote", "analyst", "hybrid"}
    assert all(value["ok"] for value in methods.values()) and len(seen) == 9
    assert methods["point_depth"]["projected_points"][0]["world_xyz"] == [2, 1, 1]
    assert methods["repeatvote"]["vote"]["attempted"] == 3
    modes = [json.loads(path.read_text())["metadata"]["mode"] for path in client.output.glob("*/record.json")]
    assert set(modes) == set(methods)
    assert before == {path.name: path.read_bytes() for path in candidate.iterdir()}
