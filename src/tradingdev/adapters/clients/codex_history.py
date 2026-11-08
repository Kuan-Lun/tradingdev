"""Verify source history without replacing the original observed conversation."""

import re
from itertools import groupby
from typing import TYPE_CHECKING, Any

from pydantic_core import to_json

from tradingdev.adapters.clients.codex_protocol import (
    CodexProtocolError,
    object_value,
    text_value,
    visible_item,
)

if TYPE_CHECKING:
    from tradingdev.domain.conversations import ConversationItem, ConversationTurn


def _result_matches(saved: Any, source: Any) -> bool:
    if saved == source:
        return True
    # Codex can rehydrate successful MCP results as one text block containing
    # the wire result, with a counted middle omission. Never treat that marker
    # alone as permission to ignore changed results or replace our full record.
    if not isinstance(saved, dict) or not isinstance(source, dict):
        return False
    if (
        set(saved) - {"content", "structuredContent", "_meta"}
        or set(source) - {"content", "structuredContent", "_meta"}
        or saved.get("_meta") is not None
        or source.get("_meta") is not None
        or source.get("structuredContent") is not None
    ):
        return False
    content = source.get("content")
    if (
        not isinstance(content, list)
        or len(content) != 1
        or not isinstance(content[0], dict)
        or set(content[0]) != {"type", "text"}
        or content[0]["type"] != "text"
        or not isinstance(content[0]["text"], str)
    ):
        return False
    wire = {"content": saved.get("content")}
    if saved.get("structuredContent") is not None:
        wire["structuredContent"] = saved["structuredContent"]
    wire["isError"] = False
    original = to_json(wire).decode("utf-8")
    projected = content[0]["text"]
    if projected == original:
        return True
    markers = list(re.finditer(r"…([1-9][0-9]*) chars truncated…", projected))
    if len(markers) != 1:
        return False
    marker = markers[0]
    prefix, suffix = projected[: marker.start()], projected[marker.end() :]
    return (
        bool(prefix and suffix)
        and original.startswith(prefix)
        and original.endswith(suffix)
        and len(original) - len(prefix) - len(suffix) == int(marker[1])
    )


def _matches(saved: ConversationItem, source: ConversationItem) -> bool:
    if (saved.source_id, saved.kind) != (source.source_id, source.kind):
        return False
    if not saved.complete:
        if saved.kind == "tool":
            return all(
                saved.data.get(k) == source.data.get(k)
                for k in ("server", "tool", "arguments")
            )
        return saved.kind != "user" or saved.text == source.text
    if saved.text != source.text:
        return False
    keys = ("server", "tool", "arguments", "error", "status", "phase", "questions")
    if any(saved.data.get(key) != source.data.get(key) for key in keys):
        return False
    a, b = saved.data.get("result"), source.data.get("result")
    if a == b:
        return True
    return (
        saved.kind == "tool"
        and saved.data.get("status") == "completed"
        and saved.data.get("error") is None
        and _result_matches(a, b)
    )


def _ordered(items: list[ConversationItem]) -> list[ConversationItem]:
    # Parallel calls may be replayed in a different order. Canonicalize only
    # adjacent tool groups, retaining all message boundaries and source IDs.
    ordered = []
    for tools, group in groupby(items, key=lambda item: item.kind == "tool"):
        members = list(group)
        ordered.extend(
            sorted(members, key=lambda item: item.source_id) if tools else members
        )
    return ordered


def verify_items(previous: ConversationTurn, items: list[Any], server: str) -> None:
    sources = [object_value(value, "historical item") for value in items]
    source_ids = {
        source["id"] for source in sources if isinstance(source.get("id"), str)
    }
    local = [item for item in previous.items if "source_type" in item.data]
    if not previous.capture_complete:
        last_complete = max(
            (i for i, item in enumerate(local) if item.complete), default=-1
        )
        # Incomplete trailing output may be absent from native history. Retain
        # partial messages that separate completed evidence before filtering;
        # otherwise tool calls could cross an observed message boundary.
        local = [
            item
            for i, item in enumerate(local)
            if item.complete
            or item.kind == "user"
            or item.source_id in source_ids
            or (item.kind != "tool" and i < last_complete)
        ]
    saved_items = {item.source_id: item for item in local}
    observed = []
    for source in sources:
        identifier = text_value(source.get("id"), "historical item ID")
        if not previous.capture_complete and identifier not in saved_items:
            continue
        saved = saved_items.get(identifier)
        complete = saved is None or saved.complete
        if (
            not complete
            and source.get("type") == "mcpToolCall"
            and source.get("status") not in {"inProgress", "completed", "failed"}
        ):
            raise CodexProtocolError("Unknown historical tool status")
        item = visible_item(source, complete=complete, server=server)
        if item is not None:
            observed.append(item)
    if len(observed) != len(local) or not all(
        _matches(saved, source)
        for saved, source in zip(_ordered(local), _ordered(observed), strict=False)
    ):
        raise CodexProtocolError(
            "Codex historical content differs from the saved conversation"
        )
