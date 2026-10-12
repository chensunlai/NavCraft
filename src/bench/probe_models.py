"""Bounded multi-image/tool-continuation probes, not navigation evaluation."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import yaml

from src.common.config import ROOT, resolve_path
from src.common.io import redact, write_json
from src.harness.agent.base import ModelBackend, ModelRequest, ToolResult, ToolSpec
from src.harness.agent.context import AgentContext
from src.harness.agent.model import ResponsesModel
from src.harness.trace import TraceWriter

DEFAULT_SCENE = "data/scene_maps/hm3d/val/00829-QaLdnwvtxbs"


def load_model_config(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(resolve_path(str(path))).read_text())
    if (
        not isinstance(raw, dict)
        or set(raw) != {"model"}
        or not isinstance(raw["model"], dict)
    ):
        raise ValueError("Model config must contain exactly a model mapping")
    config = dict(raw["model"])
    if config.get("name") != "responses":
        raise ValueError("Compatibility probes require a Responses model")
    if not config.get("api_key"):
        key = os.environ.get(config.get("api_key_env", "OMNI_API_KEY"), "")
        if not key and config.get("api_key_file"):
            credentials = yaml.safe_load(
                Path(resolve_path(config["api_key_file"])).read_text()
            )
            key = credentials.get("model", credentials).get("api_key", "")
        config["api_key"] = key
    # A probe is bounded to two actual requests, with no automatic retry loop.
    config["max_retries"] = 0
    return config


def scene_images(scene: str | Path) -> list[Path]:
    root = Path(resolve_path(str(scene)))
    evidence = sorted(
        path
        for path in (root / "evidence").rglob("*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"} and path.is_file()
    )
    if not evidence:
        raise ValueError("Scene has no archived RGB evidence image")
    images = [root / "floor_plan.png", root / "evidence_rooms.jpg", evidence[0]]
    if not all(path.is_file() for path in images):
        raise ValueError(
            "Probe requires floor_plan.png, evidence_rooms.jpg and one RGB evidence frame"
        )
    if any("raw" in path.relative_to(root).parts for path in images):
        raise ValueError("Raw scene assets cannot be used by this probe")
    return images


def probe_model(
    config: dict[str, Any],
    images: list[Path],
    directory: Path,
    *,
    timeout_seconds: float = 120,
    backend: ModelBackend | None = None,
) -> dict:
    if len(images) != 3 or not all(path.is_file() for path in images):
        raise ValueError("Exactly three existing RGB/map image files are required")
    if timeout_seconds <= 0:
        raise ValueError("Probe timeout must be positive")
    secrets = (config.get("api_key", ""),)
    model = backend or ResponsesModel(
        {**config, "max_retries": 0, "timeout_seconds": timeout_seconds}
    )
    trace = TraceWriter(directory, "compatibility-probe", secrets)
    started = time.monotonic()
    request_usages = []
    result = {
        "kind": "model_compatibility_probe",
        "navigation_metric": False,
        "model": config["model"],
        "status": "failed",
        "requests": 0,
        "images": [str(path) for path in images],
        "trace": str(trace.path),
    }
    tool = ToolSpec(
        "probe_result",
        "Report the number of input images and a short map title.",
        {
            "type": "object",
            "properties": {
                "image_count": {"type": "integer", "enum": [3]},
                "map_title": {"type": "string"},
            },
            "required": ["image_count", "map_title"],
            "additionalProperties": False,
        },
    )
    context = AgentContext(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "This is a compatibility probe, not navigation. Exactly three images follow: "
                            "a floor plan, room evidence contact sheet, and one RGB evidence frame. "
                            "Inspect the images and call probe_result exactly once with image_count=3 "
                            "and a short map_title describing the first image. Do not answer with prose yet."
                        ),
                    },
                    *[
                        {"type": "input_image", "image_path": str(path)}
                        for path in images
                    ],
                ],
            }
        ]
    )

    def complete(items, tools):
        result["requests"] += 1
        request = ModelRequest(
            f"request-{result['requests']:06d}", items, tools, timeout_seconds
        )
        trace.emit(
            "model_request",
            request_id=request.request_id,
            items=items,
            tools=tools,
            timeout_seconds=timeout_seconds,
            options=(
                model.request_options() if hasattr(model, "request_options") else {}
            ),
            purpose="compatibility_probe",
        )
        request_started = time.monotonic()
        turn = model.complete(request)
        latency = time.monotonic() - request_started
        request_usages.append(turn.usage)
        trace.emit(
            "model_response",
            request_id=request.request_id,
            items=turn.items,
            calls=[asdict(call) for call in turn.calls],
            usage=turn.usage,
            status=turn.status,
            latency_seconds=latency,
            purpose="compatibility_probe",
        )
        return turn

    try:
        first = complete(context.snapshot(), [tool.to_dict()])
        if first.status != "completed" or len(first.calls) != 1:
            raise ValueError(
                "First response must contain exactly one completed tool call"
            )
        call = first.calls[0]
        arguments = (
            json.loads(call.arguments)
            if isinstance(call.arguments, str)
            else call.arguments
        )
        if (
            call.name != "probe_result"
            or not isinstance(arguments, dict)
            or set(arguments) != {"image_count", "map_title"}
            or type(arguments.get("image_count")) is not int
            or arguments["image_count"] != 3
            or not isinstance(arguments.get("map_title"), str)
            or not 0 < len(arguments["map_title"].strip()) <= 200
        ):
            raise ValueError("Invalid probe_result call or image_count")
        context.append_output(first.items)
        context.append_tool_result(
            ToolResult(call.call_id, True, {"accepted": True, "image_count": 3})
        )
        # Preserve all first-turn protocol items, including encrypted reasoning.
        context.items.append(
            {
                "role": "user",
                "content": "Tool result accepted. Now reply only with the plain text: probe complete",
            }
        )
        second = complete(context.snapshot(), [])
        texts = [
            part.get("text", "")
            for item in second.items
            if item.get("type") == "message" and item.get("role") == "assistant"
            for part in item.get("content", [])
            if isinstance(part, dict) and part.get("type") == "output_text"
        ]
        if (
            second.status != "completed"
            or second.calls
            or "probe complete" not in " ".join(texts).lower()
        ):
            raise ValueError(
                "Second response did not complete the plain-text tool continuation"
            )
        result.update(
            status="passed",
            map_title=arguments["map_title"],
            continuation=" ".join(texts),
        )
    except Exception as error:
        result["error"] = redact(f"{type(error).__name__}: {error}", secrets)
        trace.emit("probe_error", error=result["error"])
    finally:
        model.close()
        result["elapsed_seconds"] = time.monotonic() - started
        result["request_usage"] = request_usages
        result["usage"] = {
            key: sum(
                usage[key]
                for usage in request_usages
                if isinstance(usage.get(key), (int, float))
                and not isinstance(usage[key], bool)
            )
            for key in ("input_tokens", "output_tokens", "total_tokens")
        }
        result = redact(result, secrets)
        trace.emit("probe_end", result=result)
        trace.close()
        write_json(directory / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config", nargs="+", default=["config/model/deepseek.yaml"]
    )
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--output-dir", default="cache/visual-harness-v1/probes")
    parser.add_argument("--timeout-seconds", type=float, default=120)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = Path(resolve_path(args.output_dir)) / stamp
    images = scene_images(args.scene)
    results = []
    for path in args.model_config:
        config = load_model_config(path)
        slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", config["model"])
        result = probe_model(
            config, images, directory / slug, timeout_seconds=args.timeout_seconds
        )
        results.append(result)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ("model", "status", "requests", "elapsed_seconds")
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    summary = directory / "summary.json"
    write_json(
        summary,
        {
            "kind": "model_compatibility_probe",
            "navigation_metric": False,
            "results": results,
        },
    )
    print(
        f"Results: {summary.relative_to(ROOT) if summary.is_relative_to(ROOT) else summary}"
    )


if __name__ == "__main__":
    main()
