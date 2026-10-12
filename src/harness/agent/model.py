"""Direct Responses HTTP backend; image encoding happens only at send time."""

import base64
import copy
import mimetypes
from pathlib import Path
import time
from typing import Any

import httpx

from src.common.register import model_registry

from .base import ModelBackend, ModelRequest, ModelTurn, ToolCall


class ModelError(RuntimeError):
    """A model request failed, with credentials removed from its message."""


@model_registry.register("responses")
class ResponsesModel(ModelBackend):
    def __init__(
        self, config: dict[str, Any], *, client: httpx.Client | None = None
    ) -> None:
        self.config = dict(config)
        self.model = str(config["model"])
        self.api_key = str(config.get("api_key", ""))
        self.url = str(config["base_url"]).rstrip("/")
        if not self.url.endswith("/responses"):
            self.url += "/responses"
        self.timeout_seconds = float(config.get("timeout_seconds", 120))
        self.max_retries = int(config.get("max_retries", 1))
        if self.timeout_seconds <= 0 or self.max_retries < 0:
            raise ValueError("Model timeout must be positive and retries nonnegative")
        self._client = client or httpx.Client()
        self._owns_client = client is None

    def _redact(self, value: str) -> str:
        return value.replace(self.api_key, "[REDACTED]") if self.api_key else value

    def request_options(self) -> dict[str, Any]:
        """Nonsecret options needed to reconstruct the outbound request."""
        options: dict[str, Any] = {
            "model": self.model,
            "store": bool(self.config.get("store", False)),
            "stream": False,
            "parallel_tool_calls": False,
        }
        if self.config.get("max_output_tokens") is not None:
            options["max_output_tokens"] = int(self.config["max_output_tokens"])
        if self.config.get("reasoning_effort"):
            options["reasoning"] = {"effort": self.config["reasoning_effort"]}
        if self.config.get("include_encrypted_reasoning", False):
            options["include"] = ["reasoning.encrypted_content"]
        return options

    def _encode_images(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        encoded = copy.deepcopy(items)
        for item in encoded:
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "input_image":
                    continue
                part.pop("image_role", None)
                part.pop("_context_exchange", None)
                # Some compatible servers validate detail as a required field.
                part.setdefault("detail", "auto")
                if "image_path" not in part:
                    continue
                path = Path(part.pop("image_path"))
                mime = mimetypes.guess_type(path.name)[0] or "image/png"
                if not mime.startswith("image/"):
                    raise ModelError(f"Unsupported image file: {path}")
                payload = base64.b64encode(path.read_bytes()).decode("ascii")
                part["image_url"] = f"data:{mime};base64,{payload}"
        return encoded

    def complete(self, request: ModelRequest) -> ModelTurn:
        deadline = time.monotonic() + (
            request.timeout_seconds
            if request.timeout_seconds is not None
            else self.timeout_seconds * (self.max_retries + 1)
        )
        try:
            payload = {
                **self.request_options(),
                "input": self._encode_images(request.items),
                "tools": request.tools,
            }
        except (OSError, TypeError, ValueError) as exc:
            raise ModelError(self._redact(str(exc))) from exc
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ModelError("Model request time budget exhausted") from last_error
            try:
                response = self._client.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    timeout=min(self.timeout_seconds, remaining),
                )
                response.raise_for_status()
                document = response.json()
                break
            except httpx.TransportError as exc:
                # Only transport failures are retried; no navigation tool runs here.
                last_error = exc
                if attempt == self.max_retries:
                    raise ModelError(self._redact(str(exc))) from exc
            except httpx.HTTPStatusError as exc:
                detail = response.text[:2000]
                raise ModelError(
                    self._redact(f"Responses HTTP {response.status_code}: {detail}")
                ) from exc
            except ValueError as exc:
                raise ModelError("Responses service returned invalid JSON") from exc
        if not isinstance(document, dict):
            raise ModelError("Responses service returned a non-object response")
        if document.get("error"):
            raise ModelError(self._redact(str(document["error"])))
        if "output" not in document:
            raise ModelError("Responses service response is missing output")
        items = document["output"]
        if not isinstance(items, list) or any(
            not isinstance(item, dict) for item in items
        ):
            raise ModelError("Responses output must be a list of protocol items")
        calls = []
        for item in items:
            if item.get("type") != "function_call":
                continue
            if not item.get("call_id") or not item.get("name"):
                raise ModelError("Responses function call is missing call_id or name")
            arguments = item.get("arguments", "{}")
            if not isinstance(arguments, (str, dict)):
                raise ModelError(
                    "Responses function arguments must be JSON text or object"
                )
            calls.append(ToolCall(item["call_id"], item["name"], arguments))
        usage = document.get("usage") or {}
        if not isinstance(usage, dict):
            raise ModelError("Responses usage must be an object")
        return ModelTurn(
            items=items,
            calls=calls,
            usage=usage,
            status=document.get("status") or "completed",
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


@model_registry.register("scripted")
class ScriptedModel(ModelBackend):
    """Deterministic protocol turns for environment and runtime smoke checks."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.turns = copy.deepcopy(config.get("turns", []))
        self._index = 0

    def complete(self, request: ModelRequest) -> ModelTurn:
        if self._index >= len(self.turns):
            return ModelTurn([], [])
        document = self.turns[self._index]
        self._index += 1
        items = document.get("output", document.get("items", []))
        calls = [
            ToolCall(item["call_id"], item["name"], item.get("arguments", "{}"))
            for item in items
            if item.get("type") == "function_call"
        ]
        return ModelTurn(
            items, calls, document.get("usage", {}), document.get("status", "completed")
        )
