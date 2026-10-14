"""Incremental protocol context with optional image pruning and compaction."""

import copy
import json
from typing import Any

from .base import ToolResult

_INTERNAL_IMAGE_KEYS = {"image_role", "_context_exchange"}


def _protocol(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bookkeeping belongs to the runtime, never the model wire protocol."""
    snapshot = copy.deepcopy(items)
    for item in snapshot:
        content = item.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    for key in _INTERNAL_IMAGE_KEYS:
                        part.pop(key, None)
    return snapshot


class AgentContext:
    def __init__(self, initial: list[dict[str, Any]]) -> None:
        self.items = copy.deepcopy(initial)
        self._initial_end = len(self.items)
        self._turn_starts: list[int] = []
        self._visual_exchanges: list[int] = []
        self._next_exchange = 0
        self._pinned: dict[str, dict[str, Any]] = {}
        self.latest_task: dict[str, Any] | None = None
        for part in self._images():
            part["_context_exchange"] = 0
            role = part.get("image_role", "observation")
            if role in ("map", "goal"):
                self._pinned[role] = part

    def _images(self):
        for item in self.items:
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "input_image":
                        yield part

    @property
    def size_chars(self) -> int:
        return len(json.dumps(self.snapshot(), ensure_ascii=False))

    def snapshot(self) -> list[dict[str, Any]]:
        return _protocol(self.items)

    def append_output(self, items: list[dict[str, Any]]) -> None:
        # Starts mark whole assistant/tool groups; opaque reasoning stays intact.
        self._turn_starts.append(len(self.items))
        self.items.extend(copy.deepcopy(items))

    def append_tool_result(self, result: ToolResult) -> None:
        if result.goal_changed:
            self._pinned.pop("goal", None)
        task = result.data.get("task")
        if isinstance(task, dict):
            self.latest_task = copy.deepcopy(task)
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
            self._next_exchange += 1
            exchange = self._next_exchange
            self._visual_exchanges.append(exchange)
            content = [
                {
                    "type": "input_text",
                    "text": f"Images from tool call {result.call_id}; paths remain reloadable.",
                }
            ]
            for path in result.images:
                role = result.image_roles.get(str(path), "observation")
                part = {
                    "type": "input_image",
                    "image_path": str(path),
                    "image_role": role,
                    "_context_exchange": exchange,
                }
                content.append(part)
                if role in ("map", "goal"):
                    self._pinned[role] = part
            self.items.append({"role": "user", "content": content})

    def prune_images(
        self, after: int, keep: int, *, max_images: int | None = None
    ) -> dict[str, Any] | None:
        """Prune in batches; resource files and textual tool outputs are untouched."""
        pinned = {id(part) for part in self._pinned.values()}
        if max_images is not None:
            images = [part for part in self._images() if id(part) not in pinned]
            if len(images) <= max_images:
                return None
            retained_ids = pinned | {id(part) for part in images[-max_images:]}
            removed = []
            for item in self.items:
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                retained = []
                for part in content:
                    if (
                        isinstance(part, dict)
                        and part.get("type") == "input_image"
                        and id(part) not in retained_ids
                    ):
                        removed.append(str(part.get("image_path", part.get("image_url", ""))))
                    else:
                        retained.append(part)
                item["content"] = retained
            surviving = {part.get("_context_exchange") for part in self._images()}
            self._visual_exchanges = [
                value for value in self._visual_exchanges if value in surviving
            ]
            return {
                "removed_images": removed,
                "retained_images": max_images,
                "retained_exchanges": len(self._visual_exchanges),
            }
        if len(self._visual_exchanges) <= after:
            return None
        recent = set(self._visual_exchanges[-keep:])
        removed: list[str] = []
        for item in self.items:
            content = item.get("content")
            if not isinstance(content, list):
                continue
            retained = []
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "input_image"
                    and id(part) not in pinned
                    and part.get("_context_exchange", 0) not in recent
                ):
                    removed.append(
                        str(part.get("image_path", part.get("image_url", "")))
                    )
                else:
                    retained.append(part)
            item["content"] = retained
        self._visual_exchanges = self._visual_exchanges[-keep:]
        return {"removed_images": removed, "retained_exchanges": len(recent)}

    def compaction_source(self, keep_turns: int) -> list[dict[str, Any]] | None:
        """Only return a prefix ending before a complete assistant/tool group."""
        if len(self._turn_starts) <= keep_turns:
            return None
        cut = self._turn_starts[-keep_turns]
        return _protocol(self.items[:cut])

    def compact(self, summary: str, keep_turns: int, reference: Any = None) -> None:
        """Replace an old prefix only after a successful summary request."""
        if len(self._turn_starts) <= keep_turns:
            raise ValueError("No complete older turns available for compaction")
        cut = self._turn_starts[-keep_turns]
        initial = self.items[: self._initial_end]
        recent = self.items[cut:]
        # Initial prose stays, but stale initial images do not consume the budget.
        pinned = {id(part) for part in self._pinned.values()}
        for item in initial:
            if isinstance(item.get("content"), list):
                item["content"] = [
                    part
                    for part in item["content"]
                    if not (
                        isinstance(part, dict)
                        and part.get("type") == "input_image"
                        and id(part) not in pinned
                    )
                ]
        kept_ids = {
            id(part)
            for item in [*initial, *recent]
            if isinstance(item.get("content"), list)
            for part in item["content"]
            if isinstance(part, dict)
        }
        text = (
            "Navigation history summary (earlier original interactions are archived):\n"
            + summary
        )
        if reference is not None:
            text += "\nReadable summary resource: " + json.dumps(
                reference, ensure_ascii=False
            )
        if self.latest_task is not None:
            text += (
                "\nPublic task at compaction (historical snapshot; newer tool results supersede it): "
                + json.dumps(self.latest_task, ensure_ascii=False)
            )
        summary_parts = [{"type": "input_text", "text": text}]
        summary_parts.extend(
            part for part in self._pinned.values() if id(part) not in kept_ids
        )
        self.items = [*initial, {"role": "user", "content": summary_parts}, *recent]
        prefix_size = len(initial) + 1
        self._turn_starts = [
            prefix_size + start - cut for start in self._turn_starts[-keep_turns:]
        ]
        surviving = {part.get("_context_exchange") for part in self._images()}
        self._visual_exchanges = [
            value for value in self._visual_exchanges if value in surviving
        ]
