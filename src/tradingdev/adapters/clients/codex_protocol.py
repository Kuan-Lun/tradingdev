"""Translate Codex app-server public events into durable conversation items."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import JsonValue, TypeAdapter

from tradingdev.adapters.storage.filesystem import now_iso
from tradingdev.domain.conversations import ConversationItem, Outcome

if TYPE_CHECKING:
    from tradingdev.app.conversation_service import ConversationService
    from tradingdev.domain.conversations import ConversationRecord


class CodexProtocolError(RuntimeError):
    """An unsupported or inconsistent event prevents a complete capture claim."""


def object_value(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CodexProtocolError(f"Expected a JSON object for {label}")
    return value


def text_value(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CodexProtocolError(f"Expected nonempty text for {label}")
    return value


def visible_item(
    item: dict[str, Any], *, complete: bool, server: str
) -> ConversationItem | None:
    kind = item.get("type")
    if kind == "reasoning":
        return None
    identifier = text_value(item.get("id"), "item ID")
    text = ""
    fields: tuple[str, ...]
    if kind == "userMessage":
        role = "user"
        content = item.get("content")
        if not isinstance(content, list) or not content:
            raise CodexProtocolError("Missing source user input")
        parts = []
        for part in content:
            part = object_value(part, "user input")
            if part.get("type") != "text" or not isinstance(part.get("text"), str):
                raise CodexProtocolError(
                    "This Codex recorder supports text inputs only"
                )
            parts.append(part["text"])
        text = "".join(parts)
        fields = ("content", "clientId")
    elif kind in {"agentMessage", "plan"}:
        role = "assistant" if kind == "agentMessage" else "plan"
        value = item.get("text")
        if not isinstance(value, str):
            raise CodexProtocolError("Missing visible message text")
        text = value
        fields = (
            ("phase", "questions", "memoryCitation", "delivery")
            if role == "assistant"
            else ()
        )
    elif kind == "mcpToolCall":
        role = "tool"
        if item.get("server") != server:
            raise CodexProtocolError("A tool call targeted an unconfigured MCP server")
        text_value(item.get("tool"), "MCP tool")
        if "arguments" not in item:
            raise CodexProtocolError("Missing MCP arguments")
        if complete and item.get("status") not in {"completed", "failed"}:
            raise CodexProtocolError("A completed MCP item has no terminal status")
        if complete and item.get("result") is None and item.get("error") is None:
            raise CodexProtocolError(
                "A completed MCP item has neither result nor error"
            )
        fields = (
            "server",
            "tool",
            "arguments",
            "result",
            "error",
            "status",
            "durationMs",
            "appContext",
            "mcpAppUi",
            "mcpAppResourceUri",
            "pluginId",
            "readOnlyHint",
        )
    elif kind == "contextCompaction":
        role = "notice"
        text = (
            "Codex compressed its model context; "
            "the saved conversation remains unchanged."
        )
        fields = ()
    else:
        raise CodexProtocolError(f"Unsupported Codex item type: {kind!r}")
    return ConversationItem.model_validate(
        {
            "source_id": identifier,
            "kind": role,
            "complete": complete,
            "text": text,
            "data": {
                "source_type": kind,
                **{key: item[key] for key in fields if key in item},
            },
        }
    )


_METADATA_NOTIFICATIONS = {
    "remoteControl/status/changed",
    "account/updated",
    "account/gatewayOAuth/changed",
    "account/login/completed",
    "app/list/updated",
    "skills/changed",
    "model/safetyBuffering/updated",
    "thread/started",
    "thread/status/changed",
    "thread/readState/changed",
    "thread/goal/cleared",
    "thread/queue/changed",
    "thread/name/updated",
    "thread/tokenUsage/updated",
    "turn/moderationMetadata",
    "account/rateLimits/updated",
    "mcpServer/startupStatus/updated",
    "mcpServer/oauthLogin/completed",
    "serverRequest/resolved",
}


class CodexRecorder:
    def __init__(
        self, service: ConversationService, record: ConversationRecord, server: str
    ) -> None:
        self.service, self.record, self.server = service, record, server
        self.terminal = False

    def bind_turn(self, source_id: str) -> None:
        candidate = self.record.model_copy(deep=True)
        turn = candidate.turns[-1]
        if turn.source_turn_id is not None:
            if source_id != turn.source_turn_id:
                raise CodexProtocolError("Unexpected source turn ID")
            return
        turn.source_turn_id = source_id
        self.record = self.service.save(candidate)

    def check_turn(self, params: dict[str, Any]) -> None:
        if (
            self.terminal
            or not self.record.turns
            or self.record.turns[-1].outcome != "running"
        ):
            raise CodexProtocolError("Event outside the requested running turn")
        if params.get("threadId") != self.record.source_thread_id:
            raise CodexProtocolError("Unexpected source thread ID")
        self.bind_turn(text_value(params.get("turnId"), "turn ID"))

    def notice(self, identifier: str, text: str, data: dict[str, Any]) -> None:
        candidate = self.record.model_copy(deep=True)
        candidate.turns[-1].items.append(
            ConversationItem(
                source_id=identifier,
                kind="notice",
                complete=True,
                text=text,
                data=TypeAdapter(dict[str, JsonValue]).validate_python(data),
            )
        )
        self.record = self.service.save(candidate)

    def end(self, outcome: Outcome, issue: str | None = None) -> None:
        candidate = self.record.model_copy(deep=True)
        turn = candidate.turns[-1]
        if turn.outcome != "running":
            return
        if issue:
            turn.issues.append(issue)
        if not any(item.kind == "user" for item in turn.items):
            turn.issues.append(
                "The source did not acknowledge the submitted user message"
            )
        if any(not item.complete for item in turn.items):
            turn.issues.append("Some visible items have no completion event")
        turn.outcome = outcome
        turn.finished_at = now_iso()
        turn.capture_complete = not turn.issues and turn.source_turn_id is not None
        self.record = self.service.save(candidate)
        self.terminal = True

    def event(self, method: str, params: dict[str, Any]) -> None:
        if method in _METADATA_NOTIFICATIONS or method.startswith("item/reasoning/"):
            return
        if method == "turn/started":
            source_turn = object_value(params.get("turn"), "started turn")
            self.check_turn({**params, "turnId": source_turn.get("id")})
            return
        if method in {"item/started", "item/completed"}:
            self.check_turn(params)
            item = visible_item(
                object_value(params.get("item"), "item"),
                complete=method == "item/completed",
                server=self.server,
            )
            if item is None:
                return
            candidate = self.record.model_copy(deep=True)
            turn = candidate.turns[-1]
            if item.kind == "user" and item.text != turn.input:
                raise CodexProtocolError(
                    "The source user message differs from the submitted input"
                )
            position = next(
                (
                    index
                    for index, old in enumerate(turn.items)
                    if old.source_id == item.source_id
                ),
                None,
            )
            if position is None:
                if item.complete:
                    turn.issues.append(f"Missing item start: {item.source_id}")
                turn.items.append(item)
            else:
                previous = turn.items[position]
                if previous.kind != item.kind:
                    raise CodexProtocolError("Source item type changed")
                if "progress" in previous.data:
                    item.data["progress"] = previous.data["progress"]
                if previous.complete:
                    if previous != item:
                        raise CodexProtocolError("Completed source item changed")
                    return
                if not item.complete:
                    raise CodexProtocolError("Duplicate item start")
                turn.items[position] = item
            self.record = self.service.save(candidate)
        elif method in {
            "item/agentMessage/delta",
            "item/plan/delta",
            "item/mcpToolCall/progress",
        }:
            self.check_turn(params)
            candidate = self.record.model_copy(deep=True)
            item = next(
                (
                    item
                    for item in candidate.turns[-1].items
                    if item.source_id == params.get("itemId")
                ),
                None,
            )
            if item is None or item.complete:
                raise CodexProtocolError(
                    "Delta/progress without a running visible item"
                )
            if method == "item/mcpToolCall/progress":
                if item.kind != "tool":
                    raise CodexProtocolError("Tool progress targeted a non-tool item")
                progress = item.data.setdefault("progress", [])
                if not isinstance(progress, list):
                    raise CodexProtocolError("Invalid tool progress state")
                progress.append(TypeAdapter(JsonValue).validate_python(params))
            else:
                expected = (
                    "assistant" if method == "item/agentMessage/delta" else "plan"
                )
                if item.kind != expected or not isinstance(params.get("delta"), str):
                    raise CodexProtocolError("Invalid visible text delta")
                item.text += params["delta"]
            self.record = self.service.save(candidate)
        elif method == "error":
            self.check_turn(params)
            self.notice(f"error-{len(self.record.turns[-1].items)}", method, params)
        elif method in {
            "warning",
            "configWarning",
            "deprecationNotice",
            "guardianWarning",
            "windows/worldWritableWarning",
        }:
            source_thread = params.get("threadId")
            if method == "guardianWarning":
                text_value(source_thread, "warning thread ID")
            if (
                source_thread is not None
                and self.record.source_thread_id is not None
                and source_thread != self.record.source_thread_id
            ):
                raise CodexProtocolError("Warning belongs to a different source thread")
            if self.record.turns and self.record.turns[-1].outcome == "running":
                self.notice(
                    f"{method}-{len(self.record.turns[-1].items)}", method, params
                )
        elif method == "turn/completed":
            source_turn = object_value(params.get("turn"), "completed turn")
            self.check_turn({**params, "turnId": source_turn.get("id")})
            status = source_turn.get("status")
            if status not in {"completed", "failed", "interrupted"}:
                raise CodexProtocolError("Invalid terminal turn status")
            if source_turn.get("error") is not None:
                self.notice(
                    f"turn-error-{source_turn['id']}",
                    "Codex turn error",
                    {"error": source_turn["error"]},
                )
            self.end(status)
        else:
            raise CodexProtocolError(f"Unsupported Codex notification: {method}")
