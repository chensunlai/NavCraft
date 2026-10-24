"""Audited model review experiments; proposals never modify a map.

Responses is preferred. An unsupported HTTP route may fall back to Chat; invalid
model output never triggers a retry or becomes a geometry label. Every attempt
gets a fresh audit directory, including preparation, HTTP and parsing failures.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import mimetypes
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

import jsonschema
import numpy as np


MODEL_PROFILES = {
    "deepseek-flash": {"reasoning_effort": "high", "concurrency": 1},
    "Qwen3.8-27B": {"reasoning_effort": "medium", "concurrency": 4},
    "gpt-6.1-sol": {"reasoning_effort": "high", "concurrency": 2},
}
_LIMITS = {name: threading.BoundedSemaphore(profile["concurrency"])
           for name, profile in MODEL_PROFILES.items()}
KINDS = ["wall", "missing_wall", "overdrawn_wall", "opening", "door", "window", "furniture", "unknown"]
PIXEL_SCHEMA = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
REVIEW_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["ok", "inspect"]},
        "issues": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "type": {"type": "string", "enum": KINDS},
                "image": {"type": "string"}, "pixel_hint": PIXEL_SCHEMA,
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"},
            }, "required": ["type", "image", "pixel_hint", "confidence", "reason"],
        }},
    }, "required": ["verdict", "issues"],
}
POINT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"points": {"type": "array", "minItems": 2, "maxItems": 6, "items": {
        "type": "object", "additionalProperties": False,
        "properties": {"image": {"type": "string"}, "pixel": PIXEL_SCHEMA,
                       "type": {"type": "string", "enum": KINDS},
                       "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                       "reason": {"type": "string"}},
        "required": ["image", "pixel", "type", "confidence", "reason"],
    }}}, "required": ["points"],
}
ANALYST_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"findings": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {"cause": {"type": "string"}, "evidence": {"type": "string"},
                       "proposed_rule": {"type": "string"}, "validation": {"type": "string"}},
        "required": ["cause", "evidence", "proposed_rule", "validation"],
    }}}, "required": ["findings"],
}


def _env(path: str | Path) -> dict[str, str]:
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"\s*(?:export\s+)?([A-Z_][A-Z_0-9]*)\s*=\s*(.*?)\s*", line)
        if match:
            values[match[1]] = match[2].strip("\"'")
    return values


def _write(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _strict_json(text: str):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            invalid_constant(value)
        return result

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant,
                      parse_float=finite_float)


def parse_structured(text: str, schema: dict) -> dict:
    """Accept only an entire JSON value matching the declared schema.

    In particular, DSML, Markdown fences, extracted JSON fragments, NaN and
    duplicate keys are failures. Reasoning fields are never parsed as answers.
    """
    value = _strict_json(text)
    jsonschema.Draft202012Validator(schema).validate(value)
    return value


class ProviderUnavailable(RuntimeError):
    """The service returned an outage notice instead of a model answer."""


def _schema_instruction(schema):
    return ("Return only JSON matching this complete JSON Schema. All required fields "
            "must be present; this schema is also sent through the API format parameter.\n"
            + json.dumps(schema, ensure_ascii=False, allow_nan=False, sort_keys=True))


def _provider_unavailable(value):
    if isinstance(value, dict):
        return (str(value.get("code")) in {"502", "503", "504"}
                or _provider_unavailable(value.get("message", "")))
    return isinstance(value, str) and re.match(
        r"\s*(?:error:\s*)?(?:ai\s+)?provider\s+(?:is\s+)?temporarily\s+unavailable\b",
        value, re.IGNORECASE) is not None


def _image(path: str | Path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    mime = mimetypes.guess_type(path.name)[0]
    if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
        raise ValueError(f"Unsupported image format: {path.name}")
    manifest = {"path": str(path), "name": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw), "mime_type": mime}
    content = {"type": "input_image", "detail": "high",
               "image_url": f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")}
    return manifest, content


def _chat_messages(items):
    messages = []
    for item in items:
        kind = item.get("type", "message")
        if kind == "reasoning":
            continue
        if kind == "function_call":
            call = {"id": item["call_id"], "type": "function",
                    "function": {"name": item["name"], "arguments": item["arguments"]}}
            if messages and messages[-1]["role"] == "assistant" and messages[-1].get("tool_calls"):
                messages[-1]["tool_calls"].append(call)
            else:
                messages.append({"role": "assistant", "tool_calls": [call]})
        elif kind == "function_call_output":
            messages.append({"role": "tool", "tool_call_id": item["call_id"], "content": item["output"]})
        elif kind == "message":
            content = item.get("content", [])
            if isinstance(content, str):
                messages.append({"role": item["role"], "content": content})
                continue
            parts = []
            for part in content:
                if part["type"] in {"input_text", "output_text"}:
                    parts.append({"type": "text", "text": part["text"]})
                elif part["type"] == "input_image":
                    parts.append({"type": "image_url", "image_url": {
                        "url": part["image_url"], "detail": part.get("detail", "high")}})
                else:
                    raise ValueError(f"Unsupported message content: {part['type']}")
            messages.append({"role": item["role"], "content": parts})
        else:
            raise ValueError(f"Unsupported conversation item: {kind}")
    return messages


def _response_items(response, route):
    if route == "responses":
        if response.get("status") in {"incomplete", "failed", "cancelled", "queued", "in_progress"}:
            raise ValueError(f"Response is not complete: {response.get('status')}")
        if response.get("error"):
            raise ValueError("API returned an error object")
        items = response.get("output")
        if not isinstance(items, list):
            raise ValueError("Responses payload has no output array")
        return items
    choices = response.get("choices", [])
    if len(choices) != 1:
        raise ValueError("Chat payload must contain exactly one choice")
    choice = choices[0]
    if choice.get("finish_reason") not in {"stop", "tool_calls"}:
        raise ValueError(f"Chat response is not complete: {choice.get('finish_reason')}")
    message = choice["message"]
    items = []
    if message.get("refusal"):
        raise ValueError("Model refused the request")
    if message.get("content"):
        if not isinstance(message["content"], str):
            raise ValueError("Expected textual Chat answer")
        items.append({"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": message["content"]}]})
    for call in message.get("tool_calls", []):
        items.append({"type": "function_call", "call_id": call["id"],
                      "name": call["function"]["name"], "arguments": call["function"]["arguments"]})
    return items


class ReviewClient:
    """Small synchronous client; all instances share the model concurrency limits."""

    def __init__(self, env_file=".env.harness", output="cache/map-exploration/model_calls",
                 *, timeout=300, max_output_tokens=8192, chat_fallback=True):
        self.env_file = Path(env_file)
        self.output = Path(output)
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.chat_fallback = chat_fallback

    def request(self, prompt, images=(), *, model="deepseek-flash", reasoning_effort=None,
                schema=REVIEW_SCHEMA, tools=(), metadata=None, input_items=None,
                image_manifest=(), route="responses", max_output_tokens=None,
                tool_choice=None):
        """Return ``ok, parsed, tool_calls, record_path`` or a logged structured error.

        ``input_items`` is the complete Responses-format history for continuations.
        Ordinary callers should use ``review`` for native tool rounds. Route
        fallback is only for HTTP 404/405/501 and produces a separate audit entry.
        """
        result = self._attempt(prompt, images, model, reasoning_effort, schema, tools,
                               metadata, input_items, image_manifest, route, max_output_tokens, tool_choice)
        attempts = [result["record_path"]]
        if (self.chat_fallback and route == "responses" and
                result.get("http_status") in {404, 405, 501}):
            result = self._attempt(prompt, images, model, reasoning_effort, schema, tools,
                                   metadata, input_items, image_manifest, "chat", max_output_tokens, tool_choice)
            attempts.append(result["record_path"])
        result["attempts"] = attempts
        return result

    def list_models(self):
        """List the configured service's actual model IDs with a normal audit entry."""
        schema = {"type": "object", "required": ["data"], "properties": {
            "data": {"type": "array", "items": {
                "type": "object", "required": ["id"], "properties": {"id": {"type": "string"}}}}}}
        return self.request("List available model IDs", route="models", schema=schema,
                            metadata={"operation": "list_models"})

    def _attempt(self, prompt, images, model, effort, schema, tools, metadata,
                 input_items, image_manifest, route, max_output_tokens, tool_choice):
        identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex
        directory = self.output / identifier
        directory.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        key = ""
        actual_prompt = prompt + "\n\n" + _schema_instruction(schema) if route != "models" else prompt
        record = {"request_id": identifier,
                  "created_utc": datetime.now(timezone.utc).isoformat(), "model": model,
                  "requested_model": model, "response_model": None,
                  "reasoning_effort": effort or MODEL_PROFILES.get(model, {}).get("reasoning_effort"),
                  "route": route, "task_prompt": prompt, "prompt": actual_prompt, "images": list(image_manifest),
                  "metadata": metadata or {}, "http_status": None, "usage": None, "ok": False}
        if route == "models":
            record.update(model=None, requested_model=None, reasoning_effort=None)
        result = {"ok": False, "request_id": identifier, "record_path": str(directory / "record.json"),
                  "response_saved": str(directory / "response.txt"), "route": route}

        def redact(value):
            if key and isinstance(value, str):
                return value.replace(key, "[REDACTED]")
            if isinstance(value, dict):
                return {name: redact(item) for name, item in value.items()}
            if isinstance(value, list):
                return [redact(item) for item in value]
            return value

        try:
            # Reserve an audit record before reading credentials or image files.
            _write(directory / "record.json", record)
            if model not in MODEL_PROFILES:
                raise ValueError("Model is not in the approved experiment profiles")
            if route not in {"responses", "chat", "models"}:
                raise ValueError("route must be responses, chat or models")
            if effort is not None and effort != MODEL_PROFILES[model]["reasoning_effort"]:
                raise ValueError("reasoning_effort must match the approved experiment profile")
            jsonschema.Draft202012Validator.check_schema(schema)
            values = _env(self.env_file)
            base = values.get("NAVCRAFT_TEST_BASE_URL", "").rstrip("/")
            key = values.get("NAVCRAFT_TEST_API_KEY", "")
            url = urllib.parse.urlsplit(base)
            if not key or url.scheme not in {"http", "https"} or not url.netloc:
                raise ValueError("API endpoint configuration is incomplete")
            if url.username or url.password or url.query or url.fragment:
                raise ValueError("API base URL must not contain credentials, query or fragment")
            endpoint = base + ("" if base.endswith("/v1") else "/v1")
            endpoint += {"responses": "/responses", "chat": "/chat/completions", "models": "/models"}[route]
            record["endpoint"] = endpoint
            record["method"] = "GET" if route == "models" else "POST"
            content = [{"type": "input_text", "text": actual_prompt}]
            for path in images:
                manifest, item = _image(path)
                record["images"].append(manifest)
                content.append({"type": "input_text", "text": f"Image: {manifest['name']}"})
                content.append(item)
            items = list(input_items) if input_items is not None else [{"role": "user", "content": content}]
            if input_items is not None:
                # Stateless tool continuations also need the schema in visible text.
                items.insert(0, {"role": "system", "content": _schema_instruction(schema)})
            limit = max_output_tokens or self.max_output_tokens
            fmt = {"type": "json_schema", "name": "map_review", "schema": schema, "strict": True}
            if route == "models":
                body = None
            elif route == "responses":
                body = {"model": model, "input": items, "reasoning": {"effort": record["reasoning_effort"]},
                        "max_output_tokens": limit, "store": False, "text": {"format": fmt}}
                if tools:
                    body["tools"] = list(tools)
                    body["parallel_tool_calls"] = False
                    if tool_choice:
                        body["tool_choice"] = tool_choice
                if model == "gpt-6.1-sol":
                    body["include"] = ["reasoning.encrypted_content"]
            else:
                body = {"model": model, "messages": _chat_messages(items),
                        "reasoning_effort": record["reasoning_effort"], "max_tokens": limit,
                        "response_format": {"type": "json_schema", "json_schema": {
                            name: value for name, value in fmt.items() if name != "type"}}}
                if tools:
                    body["tools"] = [{"type": "function", "function": {
                        name: value for name, value in tool.items() if name != "type"}} for tool in tools]
                    body["parallel_tool_calls"] = False
                    if tool_choice:
                        body["tool_choice"] = tool_choice
            payload = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8") if body is not None else b""
            record["request_sha256"] = hashlib.sha256(payload).hexdigest()
            record["request_bytes"] = len(payload)
            # No authorization headers are persisted. The body has no credentials.
            safe_payload = redact(payload.decode("utf-8"))
            record["request_redacted"] = safe_payload != payload.decode("utf-8")
            (directory / "request.json").write_text(safe_payload if body is not None else "null", encoding="utf-8")
            _write(directory / "record.json", redact(record))
            request = urllib.request.Request(endpoint, data=payload if body is not None else None,
                                             method=record["method"], headers={
                "Authorization": "Bearer " + key, "Content-Type": "application/json"})
            queued = time.monotonic()
            with _LIMITS[model]:
                record["queue_seconds"] = time.monotonic() - queued
                network_started = time.monotonic()
                try:
                    with urllib.request.urlopen(request, timeout=self.timeout) as response:
                        raw = response.read()
                        record["http_status"] = response.status
                except urllib.error.HTTPError as error:
                    record["http_status"] = error.code
                    raw = error.read()
                finally:
                    record["latency_seconds"] = time.monotonic() - network_started
            record["response_sha256"] = hashlib.sha256(raw).hexdigest()
            response_text = raw.decode("utf-8", errors="replace")
            safe_text = redact(response_text)
            record["response_redacted"] = safe_text != response_text
            (directory / "response.txt").write_text(safe_text, encoding="utf-8")
            if not 200 <= record["http_status"] < 300:
                if record["http_status"] in {502, 503, 504}:
                    raise ProviderUnavailable(f"HTTP {record['http_status']}")
                raise RuntimeError(f"HTTP {record['http_status']}")
            if _provider_unavailable(response_text):
                raise ProviderUnavailable(response_text)
            response = _strict_json(response_text)
            if not isinstance(response, dict):
                raise ValueError("API response must be a JSON object")
            record["usage"] = response.get("usage")
            record["response_id"] = response.get("id")
            record["response_model"] = response.get("model")
            if _provider_unavailable(response.get("error")):
                raise ProviderUnavailable("API returned a provider outage error")
            output = [] if route == "models" else _response_items(response, route)
            calls = [item for item in output if item.get("type") == "function_call"]
            text_parts = []
            for item in output:
                if item.get("type") == "message":
                    for part in item.get("content", []):
                        if part.get("type") == "refusal":
                            raise ValueError("Model refused the request")
                        if part.get("type") == "output_text":
                            text_parts.append(part["text"])
            if _provider_unavailable("".join(text_parts)):
                raise ProviderUnavailable("".join(text_parts))
            if calls:
                declarations = {tool["name"]: tool for tool in tools}
                ids = set()
                for call in calls:
                    if not call.get("call_id") or call["call_id"] in ids:
                        raise ValueError("Missing or duplicate tool call_id")
                    ids.add(call["call_id"])
                    if call.get("name") not in declarations:
                        raise ValueError("Model requested an undeclared tool")
                    arguments = parse_structured(call["arguments"], declarations[call["name"]]["parameters"])
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be a JSON object")
                result.update(tool_calls=calls, output_items=output, parsed=None)
            else:
                parsed = parse_structured(response_text if route == "models" else "".join(text_parts), schema)
                result.update(tool_calls=[], output_items=output, parsed=parsed)
                record["parsed"] = parsed
            result["ok"] = record["ok"] = True
        except Exception as error:
            failure = {"type": type(error).__name__, "message": redact(str(error))}
            record["error"] = result["error"] = failure
        finally:
            record["seconds"] = time.monotonic() - started
            _write(directory / "record.json", redact(record))
        result.update(http_status=record["http_status"], usage=record["usage"],
                      seconds=record["seconds"], images=record["images"],
                      requested_model=record["requested_model"], response_model=record["response_model"])
        return redact(result)

    def review(self, prompt, images=(), *, tools=(), tool_callbacks=None, max_rounds=4,
               require_tool=False, **kwargs):
        """Run real native tool calls; callbacks receive the decoded argument dict.

        Callback result: ``{text?: str, data?: JSON, images?: [local paths]}``.
        Rendered images are sent as image content in the next request. A callback
        error or exhausted round budget is a failed review, never a final verdict.
        """
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        history = None
        manifest = []
        attempts, events = [], []
        route = "responses"
        result = None
        for index in range(max_rounds):
            result = self.request(prompt, images if history is None else (), tools=tools,
                                  input_items=history, image_manifest=manifest, route=route,
                                  tool_choice="required" if require_tool and index == 0 else None,
                                  metadata={**(kwargs.get("metadata") or {}), "round": index},
                                  **{key: value for key, value in kwargs.items() if key != "metadata"})
            attempts.extend(result["attempts"])
            route = result["route"]
            if result["ok"] and require_tool and index == 0 and not result["tool_calls"]:
                result.update(ok=False, error={"type": "MissingToolCall", "message": "Active review did not request a view"})
            if not result["ok"] or not result["tool_calls"]:
                break
            if index + 1 == max_rounds:
                result.update(ok=False, error={"type": "ToolRoundLimit", "message": "Tool round budget exhausted"})
                break
            if history is None:
                history = [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]
                for path in images:
                    info, content = _image(path)
                    history[0]["content"].extend([{"type": "input_text", "text": f"Image: {info['name']}"}, content])
            manifest = list(result["images"])
            history.extend(result["output_items"])
            try:
                for call in result["tool_calls"]:
                    event = {"call_id": call["call_id"], "name": call["name"],
                             "arguments": _strict_json(call["arguments"])}
                    events.append(event)
                    callback = (tool_callbacks or {}).get(call["name"])
                    if callback is None:
                        raise ValueError("No callback for declared tool: " + call["name"])
                    value = callback(event["arguments"])
                    if not isinstance(value, dict):
                        raise ValueError("Tool callback must return a dictionary")
                    public = {key: value[key] for key in ("text", "data") if key in value}
                    history.append({"type": "function_call_output", "call_id": call["call_id"],
                                    "output": json.dumps(public, ensure_ascii=False, allow_nan=False)})
                    rendered = []
                    for path in value.get("images", []):
                        info, content = _image(path)
                        manifest.append(info)
                        rendered.extend([{"type": "input_text", "text": f"Tool {call['call_id']} image: {info['name']}"}, content])
                    if rendered:
                        history.append({"role": "user", "content": rendered})
                    event["result"] = public
                    event["images"] = manifest[len(result["images"]):]
            except Exception as error:
                result.update(ok=False, error={"type": type(error).__name__, "message": str(error)})
                break
            finally:
                _write(Path(result["record_path"]).parent / "tool_events.json", events)
        result["attempts"] = attempts
        result["tool_events"] = events
        # Preserve the distinction between transport success and complete review.
        _write(Path(result["record_path"]).parent / "review_result.json", {
            key: value for key, value in result.items() if key != "output_items"})
        return result


def backproject_pixel(pixel_xy, depth, intrinsics, world_from_optical):
    """Backproject optical +X right, +Y down, +Z forward, using metric Z-depth.

    Pixels use the supplied calibration's original resolution. No implicit image
    scaling, axis flip or range-to-Z conversion is performed. Invalid depths,
    outside-image pixels and non-rigid extrinsics are rejected.
    """
    pixel = np.asarray(pixel_xy, dtype=float)
    values = np.asarray(depth, dtype=float)
    matrix = np.asarray(intrinsics, dtype=float)
    transform = np.asarray(world_from_optical, dtype=float)
    if pixel.shape != (2,) or not np.isfinite(pixel).all():
        raise ValueError("pixel_xy must contain two finite numbers")
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.allclose(matrix[2], [0, 0, 1]):
        raise ValueError("intrinsics must be a finite 3x3 pinhole calibration")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("intrinsics must have positive focal lengths")
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
            or not math.isclose(np.linalg.det(transform[:3, :3]), 1, abs_tol=1e-6)):
        raise ValueError("world_from_optical must be a rigid 4x4 transform")
    if values.ndim == 2:
        x, y = pixel
        if not (0 <= x <= values.shape[1] - 1 and 0 <= y <= values.shape[0] - 1):
            raise ValueError("pixel lies outside the depth image")
        z = float(values[int(math.floor(y + 0.5)), int(math.floor(x + 0.5))])
    elif values.ndim == 0:
        z = float(values)
    else:
        raise ValueError("depth must be a Z-depth scalar or a 2D image")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("depth must be positive and finite")
    optical = np.linalg.solve(matrix, [pixel[0], pixel[1], 1]) * z
    return (transform @ np.append(optical, 1))[:3].tolist()


def project_point_votes(points, views):
    """Resolve point annotations to world XYZ; retain individual invalid points.

    A view has ``image``, ``depth`` (array or .npy path), ``intrinsics`` and
    ``world_from_optical``. All annotations stay proposals for geometry checks.
    """
    lookup = {str(view["image"]): view for view in views}
    for view in views:
        name = Path(view["image"]).name
        if sum(Path(other["image"]).name == name for other in views) == 1:
            lookup[name] = view
    result = []
    for point in points:
        record = dict(point)
        try:
            view = lookup[point["image"]]
            depth = view["depth"]
            if isinstance(depth, (str, Path)):
                depth = np.load(depth, allow_pickle=False)
            record["world_xyz"] = backproject_pixel(point["pixel"], depth, view["intrinsics"],
                                                    view["world_from_optical"])
            record["ok"] = True
        except (KeyError, ValueError, OSError) as error:
            record.update(ok=False, error={"type": type(error).__name__, "message": str(error)})
        result.append(record)
    return result


def aggregate_votes(reviews, *, pixel_tolerance=12.0):
    """Count spatially matching issues once per valid review; no geometry edits.

    Failed reviews and a minority of positive votes cannot become a consensus.
    Review indices are retained so the actual requests can be audited.
    """
    if not math.isfinite(pixel_tolerance) or pixel_tolerance <= 0:
        raise ValueError("pixel_tolerance must be positive and finite")
    groups = []
    valid = []
    for index, result in enumerate(reviews):
        if not result.get("ok"):
            continue
        valid.append(index)
        for issue in result["parsed"]["issues"]:
            match = next((group for group in groups if group["type"] == issue["type"]
                          and group["image"] == issue["image"]
                          and np.linalg.norm(np.asarray(group["pixel_hint"]) - issue["pixel_hint"]) <= pixel_tolerance), None)
            if match is None:
                match = {"type": issue["type"], "image": issue["image"],
                         "pixel_hint": issue["pixel_hint"], "reviews": [], "evidence": []}
                groups.append(match)
            if index not in match["reviews"]:
                match["reviews"].append(index)
                match["evidence"].append(issue)
    threshold = len(reviews) // 2 + 1
    for group in groups:
        group.update(votes=len(group["reviews"]), consensus=len(group["reviews"]) >= threshold)
    return {"attempted": len(reviews), "valid": len(valid), "required_votes": threshold,
            "groups": groups, "verdicts": dict(Counter(reviews[i]["parsed"]["verdict"] for i in valid))}


_BASE_PROMPT = (
    "Review a private indoor wall map using the supplied images. This is a proposal, "
    "not a map edit: every change needs an independent geometry check. Distinguish "
    "walls, furniture, doors, windows, scan holes and traversable openings. Do not "
    "invent room names or certainty. Return only JSON matching the supplied schema. "
    "Use exact image filenames and original-image pixel [x,y] coordinates."
)


def review(candidate, *, env_file=".env.harness", model="deepseek-flash",
           output="cache/map-exploration/model_calls", reasoning_effort=None):
    """Audited single-turn baseline entry point; never edits maps."""
    candidate = Path(candidate).resolve(strict=True)
    return ReviewClient(env_file, output).review(
        _BASE_PROMPT + " Identify at most three likely map errors.",
        [candidate / "walls.png", candidate / "overlay.png"], model=model,
        reasoning_effort=reasoning_effort, metadata={"mode": "baseline", "candidate": str(candidate)})


def run_suite(candidate, views=(), geometry_summary=None, *, client=None,
              model="deepseek-flash", tools=(), tool_callbacks=None, repeat_count=3, modes=None):
    """Execute six distinct experiments sequentially using actual model responses.

    ``views`` supplies real RGB paths and depth calibration to the point method.
    ``geometry_summary`` is private context only; the caller verifies all outputs
    geometrically and measures effects. Hybrid sees only explicitly flagged cases.
    This function does not render, change maps, or claim that a proposal succeeded.
    """
    if repeat_count < 2:
        raise ValueError("repeat_count must be at least two")
    client = client or ReviewClient()
    selected = set(modes or ("baseline", "point_depth", "active", "repeatvote", "analyst", "hybrid"))
    if not selected <= {"baseline", "point_depth", "active", "repeatvote", "analyst", "hybrid"}:
        raise ValueError("Unknown model experiment mode")
    candidate = Path(candidate).resolve(strict=True)
    maps = [candidate / "walls.png", candidate / "overlay.png"]
    rgb = [view["image"] for view in views]
    common = {"model": model}
    results = {}

    def ask(mode, prompt, images, schema=REVIEW_SCHEMA, **kwargs):
        if mode not in selected:
            return {"ok": False, "not_requested": True}
        return client.review(_BASE_PROMPT + " " + prompt, images, schema=schema, **common,
                             metadata={"mode": mode, "candidate": str(candidate)}, **kwargs)

    results["baseline"] = ask("baseline", "Identify at most three likely map errors from this overview.", maps)
    points = ask("point_depth", "In the perspective RGB images, identify 2–6 visible wall-surface "
                 "points across at least two different viewpoints. Door, window and opening surfaces "
                 "may also be included when visible. Use actual observed surfaces with measured "
                 "depth; avoid occluded boundaries. Image filenames identify each source.", rgb, POINT_SCHEMA)
    if points["ok"]:
        points["projected_points"] = project_point_votes(points["parsed"]["points"], views)
    results["point_depth"] = points
    if tools and "active" in selected:
        calibration = {key: (geometry_summary or {}).get(key) for key in
                       ("origin_xz", "side_m", "metres_per_pixel", "axes", "safe_initial_world_xz")}
        results["active"] = ask("active", "Actively inspect ambiguous regions. Request rendered "
                                 "views using the supplied tools before returning a final review. "
                                 "Use more views when the first view is inconclusive. Map calibration and a "
                                 "known walkable viewpoint (world X/Z, not image pixels): "
                                 + json.dumps(calibration, ensure_ascii=False), maps,
                                 tools=tools, tool_callbacks=tool_callbacks, require_tool=True)
    else:
        results["active"] = {"ok": False, "error": {"type": "MissingRenderTools",
                                                    "message": "Active experiment requires real render callbacks"}}
    repeats = [ask("repeatvote", f"Independent review {index + 1}. " + question, maps + rgb[index:index + 1])
               for index, question in enumerate((
                   "Inspect openings and possible blocked routes.",
                   "Inspect missing walls and furniture mistaken for walls.",
                   "Inspect wall continuity and whether gaps are doors, windows or scan holes.",
               ) * math.ceil(repeat_count / 3)) if index < repeat_count and "repeatvote" in selected]
    results["repeatvote"] = {"reviews": repeats, "vote": aggregate_votes(repeats),
                             "ok": sum(item["ok"] for item in repeats) >= repeat_count // 2 + 1}
    context = json.dumps(geometry_summary or {}, ensure_ascii=False, allow_nan=False)
    results["analyst"] = ask("analyst", "Analyze these measured geometry failures. Propose testable "
                             "general rules and validation checks; do not claim improvements "
                             "without measurements. Private geometry summary: " + context,
                             maps + rgb, ANALYST_SCHEMA)
    results["hybrid"] = ask("hybrid", "Geometry supplies the base map. Review only the ambiguous "
                            "or failed cases explicitly flagged in this private geometry summary: "
                            + context + " If none are flagged, return no issues.", maps + rgb)
    return {"model": model, "reasoning_effort": MODEL_PROFILES[model]["reasoning_effort"],
            "candidate": str(candidate), "methods": {k: v for k, v in results.items() if k in selected},
            "note": "All outputs are proposals; geometry acceptance and outcome metrics are external."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--env", default=".env.harness")
    parser.add_argument("--model", choices=MODEL_PROFILES, default="deepseek-flash")
    parser.add_argument("--output", default="cache/map-exploration/model_calls")
    parser.add_argument("--reasoning-effort")
    args = parser.parse_args()
    print(json.dumps(review(args.candidate, env_file=args.env, model=args.model,
                            output=args.output, reasoning_effort=args.reasoning_effort),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
