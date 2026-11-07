"""One automatically recorded Codex CLI turn, with explicit managed-session resume."""

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from filelock import FileLock

from tradingdev.adapters.clients.codex_protocol import (
    CodexProtocolError,
    CodexRecorder,
    object_value,
    text_value,
    visible_item,
)
from tradingdev.adapters.clients.codex_transport import CodexTransport, codex_transport
from tradingdev.adapters.storage.sqlite import get_sqlite_store
from tradingdev.adapters.storage.workspace_files import WorkspaceFiles
from tradingdev.app.conversation_service import ConversationService

if TYPE_CHECKING:
    from tradingdev.adapters.storage.filesystem import WorkspacePaths
    from tradingdev.domain.conversations import ConversationItem, ConversationTurn

PROVIDER = "codex-cli"
CLIENT_PROTOCOL = 1


def _item_evidence(item: ConversationItem) -> tuple[Any, ...]:
    # Timing/UI metadata may differ when Codex rehydrates history; these fields
    # identify the visible message or tool execution actually observed.
    return (
        item.source_id,
        item.kind,
        item.text,
        *(
            item.data.get(key)
            for key in (
                "server",
                "tool",
                "arguments",
                "result",
                "error",
                "status",
                "phase",
                "questions",
            )
        ),
    )


def _verify_items(previous: ConversationTurn, items: list[Any], server: str) -> None:
    local = [item for item in previous.items if "source_type" in item.data]
    completed = {item.source_id for item in local if item.complete}
    observed = []
    for value in items:
        source = object_value(value, "historical item")
        # An interrupted source may finish more work after capture stops. Keep
        # that gap explicit, but still verify every completed item we did see.
        if not previous.capture_complete and source.get("id") not in completed:
            continue
        item = visible_item(source, complete=True, server=server)
        if item is not None:
            observed.append(item)
    expected = (
        local
        if previous.capture_complete
        else [item for item in local if item.complete]
    )
    if [_item_evidence(item) for item in observed] != [
        _item_evidence(item) for item in expected
    ]:
        raise CodexProtocolError(
            "Codex historical content differs from the saved conversation"
        )


def resolve_binary() -> str:
    configured = os.environ.get("TRADINGDEV_CODEX_BIN", "codex")
    binary = shutil.which(str(Path(configured).expanduser()))
    if binary is None:
        raise ValueError(
            "Codex CLI not found; install it on PATH or set "
            "TRADINGDEV_CODEX_BIN to its executable"
        )
    return str(Path(binary).resolve())


def command(binary: str, root: Path, model: str, effort: str) -> list[str]:
    config: dict[str, Any] = {
        "features.shell_tool": False,
        "features.shell_snapshot": False,
        "features.plugins": False,
        "features.remote_plugin": False,
        "features.apps": False,
        "features.hooks": False,
        "features.memories": False,
        "features.multi_agent": False,
        "features.code_mode": False,
        "features.js_repl": False,
        "features.sleep_tool": False,
        "features.skip_host_skill_discovery": True,
        "model": model,
        "model_reasoning_effort": effort,
        "web_search": "disabled",
        "notify": [],
        "developer_instructions": "",
        "project_doc_max_bytes": 0,
        "history.persistence": "none",
        "log_dir": str(root / "logs"),
        "sqlite_home": str(root / "state"),
    }
    result = [binary, "app-server", "--listen", "stdio://"]
    for key, value in config.items():
        result += ["-c", f"{key}={json.dumps(value)}"]
    return result


class CodexSession:
    def __init__(self, transport: CodexTransport, recorder: CodexRecorder) -> None:
        self.transport, self.recorder = transport, recorder
        self.sequence = 0
        self.requests: set[str | int] = set()

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.sequence += 1
        identifier = f"tradingdev-{self.sequence}"
        await self.transport.send(
            {"id": identifier, "method": method, "params": params}
        )
        while True:
            message = await self.transport.read()
            if "method" in message:
                await self.dispatch(message)
                continue
            if message.get("id") != identifier:
                raise CodexProtocolError("Unexpected Codex RPC response identity")
            if "error" in message:
                raise CodexProtocolError(f"Codex {method} failed: {message['error']}")
            return object_value(message.get("result"), f"{method} result")

    async def dispatch(self, message: dict[str, Any]) -> None:
        method = text_value(message.get("method"), "notification method")
        params = object_value(message.get("params", {}), "notification parameters")
        if "id" not in message:
            self.recorder.event(method, params)
            return
        identifier = message["id"]
        if (
            not isinstance(identifier, str | int)
            or isinstance(identifier, bool)
            or identifier in self.requests
        ):
            raise CodexProtocolError("Invalid or repeated server request ID")
        self.requests.add(identifier)
        # This one-turn CLI never invents an answer to an interactive form or
        # permission request. Preserve both request and explicit client response.
        if (
            method == "mcpServer/elicitation/request"
            and params.get("serverName") == self.recorder.server
            and params.get("threadId") == self.recorder.record.source_thread_id
            and self.recorder.record.turns
            and self.recorder.record.turns[-1].outcome == "running"
        ):
            source_turn_id = params.get("turnId")
            if source_turn_id is not None:
                try:
                    self.recorder.check_turn(params)
                except CodexProtocolError:
                    await self.transport.send(
                        {
                            "id": identifier,
                            "error": {
                                "code": -32602,
                                "message": (
                                    "Elicitation turn identity differs; "
                                    "no permission granted"
                                ),
                            },
                        }
                    )
                    raise
            answer = {"action": "cancel", "content": None}
            self.recorder.notice(
                f"request-{identifier}",
                "The noninteractive client cancelled an elicitation; "
                "no user consent was supplied.",
                {
                    "method": method,
                    "request": params,
                    "response": answer,
                    "source_turn_correlation": "standalone"
                    if source_turn_id is None
                    else "current_turn",
                },
            )
            await self.transport.send({"id": identifier, "result": answer})
            return
        await self.transport.send(
            {
                "id": identifier,
                "error": {
                    "code": -32601,
                    "message": "Unsupported request; no permission granted",
                },
            }
        )
        raise CodexProtocolError(f"Unsupported Codex server request: {method}")

    async def verify_history(self) -> None:
        record = self.recorder.record
        expected = [turn.source_turn_id for turn in record.turns]
        if any(value is None for value in expected):
            raise CodexProtocolError(
                "An interrupted turn has no source ID; "
                "export it and start a new conversation"
            )
        cursor: str | None = None
        seen_cursors: set[str] = set()
        index = 0
        while True:
            page = await self.request(
                "thread/turns/list",
                {
                    "threadId": record.source_thread_id,
                    "itemsView": "full",
                    "sortDirection": "asc",
                    "limit": 5,
                    "cursor": cursor,
                },
            )
            turns = page.get("data")
            if not isinstance(turns, list):
                raise CodexProtocolError("Invalid Codex history page")
            for source in turns:
                source = object_value(source, "historical turn")
                if index >= len(expected) or source.get("id") != expected[index]:
                    raise CodexProtocolError(
                        "Codex history changed outside this recorder; "
                        "refusing to append"
                    )
                previous = record.turns[index]
                if source.get("itemsView", "full") != "full":
                    raise CodexProtocolError(
                        "Codex did not return full historical items"
                    )
                status = text_value(source.get("status"), "historical turn status")
                if status == "inProgress":
                    raise CodexProtocolError(
                        "The source turn is still running; retry after it stops"
                    )
                if status not in {"completed", "failed", "interrupted"}:
                    raise CodexProtocolError("Unknown historical turn status")
                if previous.capture_complete and status != previous.outcome:
                    raise CodexProtocolError(
                        "Historical turn outcome differs from the saved outcome"
                    )
                items = source.get("items")
                if not isinstance(items, list):
                    raise CodexProtocolError("Missing historical items")
                _verify_items(previous, items, self.recorder.server)
                index += 1
            next_cursor = page.get("nextCursor")
            if next_cursor is None:
                break
            cursor = text_value(next_cursor, "history cursor")
            if cursor in seen_cursors:
                raise CodexProtocolError("Codex repeated a history cursor")
            seen_cursors.add(cursor)
        if index != len(expected):
            raise CodexProtocolError("Codex history is missing saved turns")

    async def start(self, workspace: WorkspacePaths, root: Path, effort: str) -> None:
        record = self.recorder.record
        await self.request(
            "initialize",
            {
                "clientInfo": {"name": "tradingdev_recorder", "version": "1.0"},
                "capabilities": {"experimentalApi": True},
            },
        )
        await self.transport.send({"method": "initialized"})
        effective = await self.request("config/read", {"includeLayers": False})
        servers = object_value(
            object_value(effective.get("config"), "Codex config").get(
                "mcp_servers", {}
            ),
            "configured MCP servers",
        )
        name = "tradingdev_recorder"
        # A stable server name is part of persisted tool-call identity.
        if name in servers:
            raise CodexProtocolError(
                "The global tradingdev_recorder MCP entry conflicts "
                "with the name reserved by this client"
            )
        self.recorder.server = name
        environment = {
            key: value
            for key, value in os.environ.items()
            if key.startswith("TRADINGDEV_")
            or key in {"PYTHONPATH", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG"}
        }
        environment["TRADINGDEV_WORKSPACE"] = str(workspace.root)
        mcp = {
            "command": sys.executable,
            "args": ["-m", "tradingdev.mcp.server", "--workspace", str(workspace.root)],
            "cwd": str(workspace.root),
            "env": environment,
            "enabled": True,
            "required": True,
            "startup_timeout_sec": 60,
            "tool_timeout_sec": 180,
            "default_tools_approval_mode": "approve",
        }
        params: dict[str, Any] = {
            "cwd": str(root),
            "model": record.model,
            "sandbox": "read-only",
            "approvalPolicy": {
                "granular": {
                    "sandbox_approval": False,
                    "rules": False,
                    "mcp_elicitations": True,
                    "request_permissions": False,
                    "skill_approval": False,
                }
            },
            "approvalsReviewer": "user",
            "allowProviderModelFallback": False,
            "developerInstructions": (
                "Use only the configured TradingDev MCP tools. "
                "Reply in Traditional Chinese."
            ),
            "config": {
                "mcp_servers": {
                    **{server: {"enabled": False} for server in servers},
                    name: mcp,
                }
            },
        }
        if record.source_thread_id:
            params["threadId"] = record.source_thread_id
            params["excludeTurns"] = True
            response = await self.request("thread/resume", params)
        else:
            params["ephemeral"] = False
            response = await self.request("thread/start", params)
        if (
            response.get("model") != record.model
            or response.get("reasoningEffort") != effort
        ):
            raise CodexProtocolError(
                "Codex selected a different model or reasoning effort"
            )
        thread_id = text_value(
            object_value(response.get("thread"), "thread").get("id"), "thread ID"
        )
        if record.source_thread_id and thread_id != record.source_thread_id:
            raise CodexProtocolError("Codex resumed a different thread")
        if record.source_thread_id is None:
            record = record.model_copy(update={"source_thread_id": thread_id})
            self.recorder.record = self.recorder.service.save(record)
        else:
            await self.verify_history()


async def record_turn(
    *,
    workspace: WorkspacePaths,
    prompt: str,
    model: str | None = None,
    reasoning_effort: str | None = None,
    conversation_id: str | None = None,
    timeout: float = 1200,
) -> dict[str, Any]:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Codex timeout must be finite and positive")
    if not prompt.strip() or len(prompt) > 65536:
        raise ValueError("Provide a nonempty text prompt of at most 65536 characters")
    binary = resolve_binary()
    service = ConversationService(workspace, get_sqlite_store(workspace))
    if conversation_id is None:
        if not model:
            raise ValueError("Select --model when starting a conversation")
        effort = reasoning_effort or "medium"
        record = service.create(
            PROVIDER,
            model,
            settings={"reasoning_effort": effort, "client_protocol": CLIENT_PROTOCOL},
        )
    else:
        record = service.read(conversation_id)
        if (
            record.provider != PROVIDER
            or record.settings.get("client_protocol") != CLIENT_PROTOCOL
        ):
            raise ValueError(
                "This record is not a supported managed Codex conversation"
            )
        effort = text_value(
            record.settings.get("reasoning_effort"), "saved reasoning effort"
        )
        if model not in {None, record.model} or reasoning_effort not in {None, effort}:
            raise ValueError("Resume must use the saved model and reasoning effort")
    files = WorkspaceFiles(workspace.root)
    parent = files.safe_path(workspace.root / ".conversation-clients")
    parent.mkdir(exist_ok=True)
    root = files.safe_path(parent / record.conversation_id)
    root.mkdir(exist_ok=True)
    with FileLock(
        files.safe_path(parent / f"{record.conversation_id}.lock"), timeout=0
    ):
        # Reload under the lifetime lock; no second command can drive this source.
        record = service.read(record.conversation_id)
        recorder = CodexRecorder(service, record, "tradingdev_recorder")
        if record.turns and record.turns[-1].outcome == "running":
            recorder.end(
                "interrupted",
                "Previous recorder ended without a terminal event; "
                "some content may be missing",
            )
        error: str | None = None
        try:
            async with (
                asyncio.timeout(timeout),
                codex_transport(
                    command(binary, root, record.model, effort),
                    cwd=root,
                    stderr_path=files.safe_path(root / "stderr.log"),
                ) as transport,
            ):
                session = CodexSession(transport, recorder)
                await session.start(workspace, root, effort)
                recorder.record = service.start_turn(recorder.record, prompt)
                recorder.terminal = False
                response = await session.request(
                    "turn/start",
                    {
                        "threadId": recorder.record.source_thread_id,
                        "input": [
                            {"type": "text", "text": prompt, "text_elements": []}
                        ],
                    },
                )
                recorder.bind_turn(
                    text_value(
                        object_value(response.get("turn"), "turn acknowledgement").get(
                            "id"
                        ),
                        "turn ID",
                    )
                )
                while not recorder.terminal:
                    await session.dispatch(await transport.read())
        except (Exception, asyncio.CancelledError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            if recorder.record.turns and recorder.record.turns[-1].outcome == "running":
                try:
                    recorder.end("interrupted", error)
                except Exception as save_error:
                    error += f"; could not seal partial capture: {save_error}"
        current = service.read(record.conversation_id)
        last = current.turns[-1] if current.turns else None
        result: dict[str, Any] = {
            "success": error is None
            and last is not None
            and last.outcome == "completed"
            and last.capture_complete,
            "conversation_id": current.conversation_id,
            "version": current.version,
            "source_thread_id": current.source_thread_id,
            "outcome": last.outcome if last else None,
            "capture_complete": last.capture_complete if last else False,
            "error": error,
        }
        try:
            result["export"] = service.export(
                current.conversation_id, current.version
            ).model_dump(mode="json")
        except Exception as export_error:
            result["success"] = False
            result["export"] = None
            result["export_error"] = str(export_error)
        result["assistant_text"] = (
            "\n\n".join(
                item.text
                for item in last.items
                if item.kind == "assistant" and item.complete
            )
            if last
            else ""
        )
        return result
