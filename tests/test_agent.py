"""Protocol preservation, action authority and bounded execution regressions."""

import base64
import copy
import json

import httpx
import pytest

from src.harness.agent.base import AgentLimits, ModelRequest, ToolResult, ToolSpec
from src.harness.agent.context import AgentContext
from src.harness.agent.loop import AgentRuntime
from src.harness.agent.model import ModelError, ResponsesModel, ScriptedModel


class Trace:
    def __init__(self):
        self.events = []

    def emit(self, event, **fields):
        self.events.append((event, copy.deepcopy(fields)))


class Tools:
    def __init__(self, *, terminate=False, image=None):
        self.executed = []
        self.terminate = terminate
        self.image = image

    def specs(self):
        return [ToolSpec("nav_step", "Move one step", {"type": "object"})]

    def call(self, call):
        try:
            arguments = (
                json.loads(call.arguments)
                if isinstance(call.arguments, str)
                else call.arguments
            )
        except json.JSONDecodeError:
            return ToolResult(call.call_id, False, {"error": "Invalid JSON arguments"})
        if not isinstance(arguments, dict):
            return ToolResult(call.call_id, False, {"error": "Expected argument object"})
        self.executed.append(call)
        return ToolResult(
            call.call_id,
            True,
            {"frame_id": len(self.executed)},
            [self.image] if self.image else [],
            terminated=self.terminate,
        )


def function(call_id="call-1", arguments="{}"):
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": "nav_step",
        "arguments": arguments,
    }


def backend(handler, **config):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return ResponsesModel(
        {
            "base_url": "https://gateway.test/v1",
            "api_key": "secret-key",
            "model": "test-vision",
            "max_retries": 0,
            **config,
        },
        client=client,
    )


def test_response_reasoning_survives_next_request_and_images_are_not_inlined_in_trace(tmp_path):
    image = tmp_path / "frame.png"
    image.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aY9cAAAAASUVORK5CYII="
    ))
    reasoning = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [],
        "encrypted_content": "opaque-protocol-state",
    }
    outbound = []

    def handler(request):
        assert request.headers["authorization"] == "Bearer secret-key"
        assert request.url.path == "/v1/responses"
        payload = json.loads(request.content)
        outbound.append(payload)
        if len(outbound) == 1:
            return httpx.Response(200, json={
                "output": [reasoning, function()],
                "status": "completed",
                "usage": {"input_tokens": 4, "output_tokens": 3},
            })
        return httpx.Response(200, json={
            "output": [{"type": "message", "role": "assistant", "content": []}],
            "usage": {"input_tokens": 7, "output_tokens": 2},
        })

    model = backend(handler, include_encrypted_reasoning=True, reasoning_effort="high")
    tools = Tools(image=image)
    trace = Trace()
    initial = [{"role": "user", "content": [
        {"type": "input_text", "text": "Find a chair"},
        {"type": "input_image", "image_path": str(image)},
    ]}]
    result = AgentRuntime(model, {}).run(initial, tools, trace, AgentLimits())
    assert result.reason == "model_finished"
    assert len(tools.executed) == 1
    assert result.usage == {"input_tokens": 11, "output_tokens": 5}
    assert outbound[1]["input"][1] == reasoning
    assert outbound[1]["input"][2] == function()
    assert outbound[1]["input"][3]["type"] == "function_call_output"
    assert outbound[1]["input"][3]["call_id"] == "call-1"
    assert outbound[1]["input"][4]["content"][1]["image_url"].startswith("data:image/png;base64,")
    assert outbound[0]["include"] == ["reasoning.encrypted_content"]
    assert outbound[0]["store"] is False
    assert initial[0]["content"][1]["image_path"] == str(image)
    requests = [fields for event, fields in trace.events if event == "model_request"]
    assert requests[1]["items"][0]["content"][1]["image_path"] == str(image)
    assert "base64" not in json.dumps(requests)
    assert requests[0]["options"]["model"] == "test-vision"


def test_model_completion_does_not_issue_stop():
    model = ScriptedModel({"turns": [{"output": [{
        "type": "message", "role": "assistant",
        "content": [{"type": "output_text", "text": "I have reached the target."}],
    }]}]})
    tools = Tools()
    result = AgentRuntime(model, {}).run([], tools, Trace(), AgentLimits())
    assert result.reason == "model_finished"
    assert tools.executed == []


def test_invalid_arguments_are_observed_by_model_before_retry():
    model = ScriptedModel({"turns": [
        {"output": [function(arguments="{not-json")]},
        {"output": [function("call-2")]},
    ]})
    tools = Tools(terminate=True)
    runtime = AgentRuntime(model, {})
    result = runtime.run([], tools, Trace(), AgentLimits())
    assert result.reason == "terminated"
    assert [call.call_id for call in tools.executed] == ["call-2"]
    tool_outputs = [item for item in runtime.context.items if item.get("type") == "function_call_output"]
    assert json.loads(tool_outputs[0]["output"])["ok"] is False
    assert json.loads(tool_outputs[1]["output"])["ok"] is True


def test_multiple_calls_after_episode_end_are_not_executed():
    model = ScriptedModel({"turns": [{"output": [function(), function("call-2")]}]})
    tools = Tools(terminate=True)
    trace = Trace()
    result = AgentRuntime(model, {}).run([], tools, trace, AgentLimits())
    assert result.reason == "terminated"
    assert [call.call_id for call in tools.executed] == ["call-1"]
    skipped = [fields for event, fields in trace.events if event == "tool_skipped"]
    assert skipped[0]["call"]["call_id"] == "call-2"


def test_context_and_request_budgets_stop_without_more_requests():
    tools = Tools()
    trace = Trace()
    model = ScriptedModel({"turns": [{"output": [function()]}]})
    result = AgentRuntime(model, {}).run([], tools, trace, AgentLimits(max_requests=1))
    assert result.reason == "request_budget"
    assert result.requests == 1
    assert len(tools.executed) == 1
    result = AgentRuntime(model, {}).run(
        [{"role": "user", "content": "x" * 100}],
        tools,
        Trace(),
        AgentLimits(max_context_chars=10),
    )
    assert result.reason == "context_budget"
    assert result.requests == 0


def test_transport_retries_do_not_strip_reasoning_and_http_errors_redact_keys():
    attempts = []

    def retry(request):
        attempts.append(json.loads(request.content))
        if len(attempts) == 1:
            raise httpx.ConnectError("connection lost", request=request)
        return httpx.Response(200, json={"output": []})

    model = backend(retry, max_retries=1)
    model.complete(ModelRequest("req", [], []))
    assert attempts[0] == attempts[1]

    def forbidden(request):
        return httpx.Response(401, text="Bad credentials secret-key")

    with pytest.raises(ModelError) as error:
        backend(forbidden).complete(ModelRequest("req", [], []))
    assert "secret-key" not in str(error.value)
    assert "[REDACTED]" in str(error.value)


def test_model_error_and_incomplete_response_do_not_execute_actions():
    def invalid(request):
        return httpx.Response(200, text="invalid json")

    trace = Trace()
    tools = Tools()
    result = AgentRuntime(backend(invalid), {}).run([], tools, trace, AgentLimits())
    assert result.reason == "model_error"
    assert "invalid JSON" in result.error
    result = AgentRuntime(ScriptedModel({"turns": [{
        "output": [function()], "status": "incomplete",
    }]}), {}).run([], tools, Trace(), AgentLimits())
    assert result.reason == "model_incomplete"
    assert tools.executed == []


def test_context_owns_input_and_opaque_output_copies():
    initial = [{"role": "user", "content": "start"}]
    output = [{"type": "reasoning", "encrypted_content": "state"}]
    context = AgentContext(initial)
    context.append_output(output)
    initial[0]["content"] = "changed"
    output[0]["encrypted_content"] = "changed"
    snapshot = context.snapshot()
    assert snapshot[0]["content"] == "start"
    assert snapshot[1]["encrypted_content"] == "state"


def test_missing_response_output_is_a_protocol_error():
    def missing(request):
        return httpx.Response(200, json={"status": "completed"})

    with pytest.raises(ModelError, match="missing output"):
        backend(missing).complete(ModelRequest("request-1", [], []))


def test_image_detail_is_explicit_for_paths_and_urls_without_overriding_preference(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"test image content")
    parts = [
        {"type": "input_image", "image_path": str(image)},
        {"type": "input_image", "image_url": "https://example.test/frame.png"},
        {"type": "input_image", "image_path": str(image), "detail": "high"},
    ]

    def strict_server(request):
        content = json.loads(request.content)["input"][0]["content"]
        assert [part["detail"] for part in content] == ["auto", "auto", "high"]
        assert content[0]["image_url"].startswith("data:image/png;base64,")
        assert content[1]["image_url"] == "https://example.test/frame.png"
        assert all("image_path" not in part for part in content)
        return httpx.Response(200, json={"output": []})

    backend(strict_server).complete(ModelRequest("request-1", [{"role": "user", "content": parts}], []))
    assert "detail" not in parts[0]
    assert "detail" not in parts[1]
    assert parts[2]["detail"] == "high"


def test_environment_failure_is_recorded_and_timeout_prevents_tool_execution(monkeypatch):
    class BrokenTools(Tools):
        def call(self, call):
            raise RuntimeError("renderer unavailable")

    trace = Trace()
    result = AgentRuntime(ScriptedModel({"turns": [{"output": [function()]}]}), {}).run(
        [], BrokenTools(), trace, AgentLimits()
    )
    assert result.reason == "environment_error"
    assert "renderer unavailable" in result.error
    assert trace.events[-1][0] == "agent_end"

    clock = [0.0]

    class SlowModel(ScriptedModel):
        def complete(self, request):
            assert request.timeout_seconds <= 1
            clock[0] = 2
            return super().complete(request)

    monkeypatch.setattr("src.harness.agent.loop.time.monotonic", lambda: clock[0])
    tools = Tools()
    result = AgentRuntime(SlowModel({"turns": [{"output": [function()]}]}), {}).run(
        [], tools, Trace(), AgentLimits(timeout_seconds=1)
    )
    assert result.reason == "timeout"
    assert tools.executed == []
