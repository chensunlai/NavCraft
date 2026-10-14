"""Incremental context keeps image evidence and complete tool protocol groups."""

import copy
import json
from pathlib import Path

import pytest

from src.legacy.harness.agent.base import AgentLimits, ToolResult, ToolSpec
from src.legacy.harness.agent.context import AgentContext
from src.legacy.harness.agent.loop import AgentRuntime
from src.legacy.harness.agent.model import ScriptedModel


def function(call_id):
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": "nav_step",
        "arguments": "{}",
    }


def reply(text):
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def image_paths(items):
    return [
        part["image_path"]
        for item in items
        if isinstance(item.get("content"), list)
        for part in item["content"]
        if part.get("type") == "input_image"
    ]


class Trace:
    def __init__(self):
        self.events = []

    def emit(self, event, **fields):
        self.events.append((event, copy.deepcopy(fields)))


class RecordingModel(ScriptedModel):
    def __init__(self, turns):
        super().__init__({"turns": turns})
        self.requests = []

    def complete(self, request):
        self.requests.append(copy.deepcopy(request))
        return super().complete(request)


class Tools:
    def __init__(self, *, goal_change=False):
        self.calls = []
        self.summaries = []
        self.goal_change = goal_change

    def specs(self):
        return [ToolSpec("nav_step", "Move", {"type": "object"})]

    def call(self, call):
        self.calls.append(call.call_id)
        return ToolResult(
            call.call_id,
            True,
            {
                "task": {"goal_id": 2, "instruction": "Find table"},
                "frame_id": f"frame-{len(self.calls)}",
                "feedback": "x" * 80,
            },
            goal_changed=self.goal_change and len(self.calls) == 1,
        )

    def save_summary(self, text):
        self.summaries.append(text)
        return "summary-1.txt"


def test_prune_images_in_batches_and_pin_latest_map_and_goal(tmp_path):
    paths = [tmp_path / f"frame-{i}.png" for i in range(12)]
    for path in paths:
        path.write_bytes(b"rgb")
    initial_map, old_goal, old_rgb = (
        tmp_path / name for name in ("map-0.png", "goal-0.png", "initial.png")
    )
    context = AgentContext(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_path": str(initial_map),
                        "image_role": "map",
                    },
                    {
                        "type": "input_image",
                        "image_path": str(old_goal),
                        "image_role": "goal",
                    },
                    {"type": "input_image", "image_path": str(old_rgb)},
                ],
            }
        ]
    )
    new_map = tmp_path / "map-1.png"
    for i in range(11):
        images = [paths[i]] + ([new_map] if i == 0 else [])
        context.append_output([function(f"call-{i}")])
        context.append_tool_result(
            ToolResult(
                f"call-{i}",
                True,
                {"frame_id": f"frame-{i}"},
                images,
                image_roles={str(new_map): "map"},
            )
        )
        if i < 10:
            assert context.prune_images(10, 5) is None
    pruned = context.prune_images(10, 5)
    snapshot = context.snapshot()
    kept = image_paths(snapshot)
    assert str(initial_map) not in kept
    assert str(old_rgb) not in kept
    assert str(new_map) in kept
    assert str(old_goal) in kept
    assert all(str(path) in kept for path in paths[6:11])
    assert all(str(path) not in kept for path in paths[:6])
    assert "image_role" not in json.dumps(snapshot)
    assert "_context_exchange" not in json.dumps(snapshot)
    assert len([i for i in snapshot if i.get("type") == "function_call_output"]) == 11
    assert pruned["retained_exchanges"] == 5
    assert all(path.exists() for path in paths)
    assert context.prune_images(10, 5) is None


def test_goal_change_unpins_old_goal_and_preserves_new_goal():
    context = AgentContext(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_path": "old-goal.png",
                        "image_role": "goal",
                    },
                ],
            }
        ]
    )
    context.append_output([function("first")])
    context.append_tool_result(
        ToolResult(
            "first",
            True,
            {},
            [Path("new-goal.png")],
            goal_changed=True,
            image_roles={"new-goal.png": "goal"},
        )
    )
    context.append_output([function("second")])
    context.append_tool_result(ToolResult("second", True, {}, [Path("last.png")]))
    context.prune_images(1, 1)
    assert image_paths(context.snapshot()) == ["new-goal.png", "last.png"]


def test_image_limit_counts_individual_images_and_preserves_protocol():
    context = AgentContext([{"role": "user", "content": [
        {"type": "input_image", "image_path": "map.png", "image_role": "map"},
        {"type": "input_image", "image_path": "goal.png", "image_role": "goal"},
    ]}])
    for i in range(5):
        context.append_output([function(f"call-{i}")])
        context.append_tool_result(ToolResult(
            f"call-{i}", True, {},
            [Path(f"frame-{3*i+j}.png") for j in range(3)],
        ))
        context.prune_images(10, 10, max_images=10)
    snapshot = context.snapshot()
    assert image_paths(snapshot) == [
        "map.png", "goal.png", *[f"frame-{i}.png" for i in range(5, 15)]
    ]
    assert len([item for item in snapshot if item.get("type") == "function_call_output"]) == 5
    assert context.prune_images(10, 10, max_images=10) is None


@pytest.mark.parametrize("input_tokens,expected_summaries", [(99999, 0), (100000, 1)])
def test_default_compaction_waits_until_100k(input_tokens, expected_summaries):
    turns = [
        {"output": [function("first")]},
        {"output": [function("second")], "usage": {"input_tokens": input_tokens}},
    ]
    if expected_summaries:
        turns.append({"output": [reply("history summary")]})
    turns.append({"output": [reply("done")]})
    model, tools = RecordingModel(turns), Tools()
    result = AgentRuntime(model, {"context": {"keep_turns": 1}}).run(
        [], tools, Trace(), AgentLimits()
    )
    assert result.reason == "model_finished"
    assert len(tools.summaries) == expected_summaries


def test_compaction_preserves_recent_whole_protocol_groups_and_pinned_images():
    context = AgentContext(
        [
            {"role": "system", "content": "Navigate"},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "initial task"},
                    {"type": "input_image", "image_path": "initial.png"},
                ],
            },
        ]
    )
    for i in range(4):
        context.append_output(
            [
                {"type": "reasoning", "encrypted_content": f"opaque-{i}"},
                function(f"call-{i}"),
            ]
        )
        context.append_tool_result(
            ToolResult(
                f"call-{i}",
                True,
                {"task": {"instruction": "new target"}},
                [Path("map.png")] if i == 0 else [],
                image_roles={"map.png": "map"},
            )
        )
    source = context.compaction_source(2)
    assert any(item.get("call_id") == "call-1" for item in source)
    assert not any(item.get("call_id") == "call-2" for item in source)
    context.compact(
        "Near door, heading uncertain; map.png supports identity", 2, "summary.txt"
    )
    items = context.snapshot()
    assert items[0] == {"role": "system", "content": "Navigate"}
    assert "initial.png" not in image_paths(items)
    assert "map.png" in image_paths(items)
    assert any(item.get("encrypted_content") == "opaque-2" for item in items)
    assert not any(item.get("encrypted_content") == "opaque-1" for item in items)
    calls = {i["call_id"] for i in items if i.get("type") == "function_call"}
    outputs = {i["call_id"] for i in items if i.get("type") == "function_call_output"}
    assert calls == outputs == {"call-2", "call-3"}
    assert "Public task at compaction" in items[2]["content"][0]["text"]
    assert "newer tool results supersede it" in items[2]["content"][0]["text"]
    assert "summary.txt" in items[2]["content"][0]["text"]


def test_runtime_compaction_is_logged_counted_and_summary_written():
    model = RecordingModel(
        [
            {"output": [function("one")], "usage": {"input_tokens": 5}},
            {"output": [function("two")], "usage": {"input_tokens": 100000}},
            {
                "output": [reply("Location uncertain; inspect frame-1 before moving")],
                "usage": {"input_tokens": 10, "output_tokens": 7},
            },
            {"output": [reply("done")], "usage": {"input_tokens": 2}},
        ]
    )
    trace, tools = Trace(), Tools()
    runtime = AgentRuntime(model, {"context": {"keep_turns": 1}})
    result = runtime.run(
        [{"role": "system", "content": "Navigate"}], tools, trace, AgentLimits()
    )
    assert result.requests == 4
    assert result.usage == {"input_tokens": 100017, "output_tokens": 7}
    assert model.requests[2].tools == []
    summary_instruction = model.requests[2].items[-1]["content"]
    assert "already" in summary_instruction
    assert "Absence of view_images does not" in summary_instruction
    assert "no explicit interpretation was recorded" in summary_instruction
    assert any(item.get("call_id") == "one" for item in model.requests[2].items)
    assert any(item.get("call_id") == "two" for item in model.requests[2].items)
    assert any(item.get("call_id") == "two" for item in model.requests[3].items)
    assert not any(item.get("call_id") == "one" for item in model.requests[3].items)
    assert len(tools.summaries) == 1
    assert any(event == "context_compacted" for event, _ in trace.events)
    requests = [fields for event, fields in trace.events if event == "model_request"]
    responses = [fields for event, fields in trace.events if event == "model_response"]
    assert [r["request_id"] for r in requests] == [r["request_id"] for r in responses]
    assert requests[2]["purpose"] == "context_compaction"


def test_compaction_failure_does_not_rewrite_context_or_fabricate_summary():
    model = RecordingModel(
        [
            {"output": [function("one")]},
            {"output": [function("two")]},
            {"output": []},
        ]
    )
    runtime = AgentRuntime(model, {"context": {"keep_turns": 1, "compact_chars": 1}})
    tools, trace = Tools(), Trace()
    result = runtime.run([], tools, trace, AgentLimits())
    assert result.reason == "context_compaction_error"
    assert tools.summaries == []
    assert [
        i["call_id"] for i in runtime.context.items if i.get("type") == "function_call"
    ] == ["one", "two"]
    assert any(event == "context_compaction_failed" for event, _ in trace.events)


def test_runtime_summary_sees_current_rgb_and_keeps_five_complete_recent_groups(
    tmp_path,
):
    class ImageTools(Tools):
        def call(self, call):
            result = super().call(call)
            path = tmp_path / f"frame-{len(self.calls)}.png"
            path.write_bytes(b"fixture RGB")
            result.images = [path]
            return result

    turns = [
        {
            "output": [
                {"type": "reasoning", "encrypted_content": f"opaque-{i}"},
                function(f"call-{i}"),
            ],
            "usage": {"input_tokens": 100000 if i == 6 else 100},
        }
        for i in range(1, 7)
    ]
    turns.extend(
        [
            {
                "output": [
                    reply(
                        "Current frame-6 identity remains uncertain; inspect evidence."
                    )
                ]
            },
            {"output": [reply("done")]},
        ]
    )
    model, trace, tools = RecordingModel(turns), Trace(), ImageTools()
    runtime = AgentRuntime(model, {"context": {}})
    result = runtime.run([], tools, trace, AgentLimits())
    assert result.requests == 8
    summary_request = model.requests[6]
    assert summary_request.tools == []
    assert str(tmp_path / "frame-6.png") in image_paths(summary_request.items)
    summary_calls = {
        item["call_id"]
        for item in summary_request.items
        if item.get("type") == "function_call"
    }
    summary_outputs = {
        item["call_id"]
        for item in summary_request.items
        if item.get("type") == "function_call_output"
    }
    assert summary_calls == summary_outputs == {f"call-{i}" for i in range(1, 7)}
    instruction = summary_request.items[-1]["content"]
    assert "entire completed navigation transcript" in instruction
    assert "Do not promote plans or" in instruction
    assert "hypotheses with their uncertainty" in instruction
    final_items = model.requests[7].items
    calls = {
        item["call_id"] for item in final_items if item.get("type") == "function_call"
    }
    outputs = {
        item["call_id"]
        for item in final_items
        if item.get("type") == "function_call_output"
    }
    assert calls == outputs == {f"call-{i}" for i in range(2, 7)}
    assert all(
        str(tmp_path / f"frame-{i}.png") in image_paths(final_items)
        for i in range(2, 7)
    )
    assert str(tmp_path / "frame-1.png") not in image_paths(final_items)
    assert all(
        any(item.get("encrypted_content") == f"opaque-{i}" for item in final_items)
        for i in range(2, 7)
    )
    assert len(tools.summaries) == 1


def test_goal_change_skips_stale_calls_then_requests_new_decision():
    model = RecordingModel(
        [
            {"output": [function("finish"), function("stale")]},
            {"output": [reply("new goal decision")]},
        ]
    )
    runtime = AgentRuntime(model, {})
    tools, trace = Tools(goal_change=True), Trace()
    result = runtime.run([], tools, trace, AgentLimits())
    assert result.reason == "model_finished"
    assert result.requests == 2
    assert tools.calls == ["finish"]
    skipped = [fields for event, fields in trace.events if event == "tool_skipped"]
    assert skipped[0]["reason"] == "goal_changed"
    outputs = [
        i for i in model.requests[1].items if i.get("type") == "function_call_output"
    ]
    assert len(outputs) == 2
    assert json.loads(outputs[1]["output"])["ok"] is False


def test_disabled_context_policy_is_append_only_and_summary_consumes_request_budget():
    model = RecordingModel(
        [
            {"output": [function("one")]},
            {"output": [function("two")], "usage": {"input_tokens": 50000}},
            {"output": [reply("done")]},
        ]
    )
    result = AgentRuntime(model, {}).run([], Tools(), Trace(), AgentLimits())
    assert result.requests == 3
    assert any(item.get("call_id") == "one" for item in model.requests[2].items)
    bounded = RecordingModel(
        [
            {"output": [function("one")]},
            {"output": [function("two")]},
            {"output": [reply("summary")]},
        ]
    )
    result = AgentRuntime(
        bounded, {"context": {"keep_turns": 1, "compact_chars": 1}}
    ).run([], Tools(), Trace(), AgentLimits(max_requests=3))
    assert result.reason == "request_budget"
    assert result.requests == 3
    assert bounded.requests[-1].tools == []


@pytest.mark.parametrize(
    "policy", [{"image_keep": 11}, {"compact_tokens": 0}, {"keep_turns": -1}]
)
def test_invalid_context_policy_fails_before_running(policy):
    with pytest.raises(ValueError):
        AgentRuntime(ScriptedModel({}), {"context": policy})
