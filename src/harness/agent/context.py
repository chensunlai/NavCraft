"""Append-only semantic context with durable image references."""

import copy
import json
from typing import Any

from .base import ToolResult


class AgentContext:
    def __init__(self, initial: list[dict[str, Any]]) -> None:
        self.items = copy.deepcopy(initial)

    @property
    def size_chars(self) -> int:
        return len(json.dumps(self.items, ensure_ascii=False))

    def snapshot(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self.items)

    def append_output(self, items: list[dict[str, Any]]) -> None:
        # Reasoning items are protocol state, even when their content is opaque.
        self.items.extend(copy.deepcopy(items))

    def append_tool_result(self, result: ToolResult) -> None:
        self.items.append(
            {
                "type": "function_call_output",
                "call_id": result.call_id,
                "output": json.dumps(
                    {**result.data, "ok": result.ok}, ensure_ascii=False
                ),
            }
        )
        if result.images:
            self.items.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Observation images from tool call {result.call_id}.",
                        },
                        *[
                            {
                                "type": "input_image",
                                "image_path": str(path),
                            }
                            for path in result.images
                        ],
                    ],
                }
            )
