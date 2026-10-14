"""Compatibility probes use local fixture servers, never a paid model API."""

import json

import httpx
import pytest
import yaml

from src.bench.probe_models import load_model_config, probe_model, scene_images
from src.legacy.harness.agent.model import ResponsesModel


def fixtures(tmp_path):
    images = [tmp_path / f"image-{i}.png" for i in range(3)]
    for image in images:
        image.write_bytes(b"image bytes")
    return images


def test_multimodal_tool_probe_preserves_reasoning_and_saves_no_base64_or_key(tmp_path):
    sent = []
    reasoning = {"type": "reasoning", "encrypted_content": "opaque-state"}

    def handler(request):
        payload = json.loads(request.content)
        sent.append(payload)
        if len(sent) == 1:
            assert (
                len(
                    [
                        part
                        for part in payload["input"][0]["content"]
                        if part["type"] == "input_image"
                    ]
                )
                == 3
            )
            return httpx.Response(
                200,
                json={
                    "output": [
                        reasoning,
                        {
                            "type": "function_call",
                            "call_id": "call-1",
                            "name": "probe_result",
                            "arguments": json.dumps(
                                {"image_count": 3, "map_title": "Indoor floor plan"}
                            ),
                        },
                    ],
                    "usage": {"input_tokens": 7, "output_tokens": 2},
                },
            )
        assert reasoning in payload["input"]
        assert any(
            item.get("type") == "function_call_output" for item in payload["input"]
        )
        assert payload["tools"] == []
        return httpx.Response(
            200,
            json={
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "probe complete"},
                        ],
                    }
                ],
                "usage": {"input_tokens": 9, "output_tokens": 2},
            },
        )

    config = {
        "model": "fixture-vision",
        "base_url": "https://fixture.test/v1",
        "api_key": "secret-key",
        "max_retries": 0,
    }
    model = ResponsesModel(
        config, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    result = probe_model(config, fixtures(tmp_path), tmp_path / "run", backend=model)
    assert result["status"] == "passed"
    assert result["requests"] == 2
    assert result["usage"]["input_tokens"] == 16
    assert result["navigation_metric"] is False
    trace = (tmp_path / "run" / "trace.jsonl").read_text()
    assert "opaque-state" in trace
    assert "base64" not in trace
    assert "secret-key" not in trace
    assert (
        len(
            [
                line
                for line in trace.splitlines()
                if json.loads(line)["event"] == "model_request"
            ]
        )
        == 2
    )


def test_invalid_first_call_fails_once_and_redacts_service_error(tmp_path):
    def handler(request):
        return httpx.Response(401, text="bad secret-key")

    config = {
        "model": "fixture",
        "base_url": "https://fixture.test/v1",
        "api_key": "secret-key",
        "max_retries": 0,
    }
    model = ResponsesModel(
        config, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    result = probe_model(config, fixtures(tmp_path), tmp_path / "run", backend=model)
    assert result["status"] == "failed"
    assert result["requests"] == 1
    assert "secret-key" not in result["error"]
    assert "[REDACTED]" in result["error"]


def test_model_key_resolution_and_scene_uses_three_public_images(tmp_path, monkeypatch):
    credentials = tmp_path / "credentials.yaml"
    credentials.write_text(yaml.safe_dump({"model": {"api_key": "file-secret"}}))
    model_file = tmp_path / "model.yaml"
    model_file.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "name": "responses",
                    "model": "fixture",
                    "api_key_env": "PROBE_TEST_KEY",
                    "api_key_file": str(credentials),
                }
            }
        )
    )
    monkeypatch.delenv("PROBE_TEST_KEY", raising=False)
    assert load_model_config(model_file)["api_key"] == "file-secret"
    monkeypatch.setenv("PROBE_TEST_KEY", "env-secret")
    assert load_model_config(model_file)["api_key"] == "env-secret"
    root = tmp_path / "scene"
    (root / "evidence" / "frames").mkdir(parents=True)
    for path in (
        root / "floor_plan.png",
        root / "evidence_rooms.jpg",
        root / "evidence" / "frames" / "frame.jpg",
    ):
        path.write_bytes(b"rgb")
    assert len(scene_images(root)) == 3
    assert all("raw" not in path.parts for path in scene_images(root))
    with pytest.raises(ValueError):
        probe_model({"model": "fixture"}, [], tmp_path / "invalid")
