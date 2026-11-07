"""Offline Codex protocol, resume identity and subprocess cleanup regression tests."""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import psutil
import pytest
from filelock import FileLock, Timeout

from tradingdev.adapters.clients import codex_session
from tradingdev.adapters.clients.codex_protocol import CodexProtocolError, CodexRecorder
from tradingdev.adapters.clients.codex_session import record_turn
from tradingdev.adapters.clients.codex_transport import CodexTransport, codex_transport
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.conversation_service import ConversationService


def recorder_at(root: Path) -> CodexRecorder:
    workspace = WorkspacePaths(root)
    service = ConversationService(workspace, SQLiteStore(workspace))
    record = service.create("codex-cli", "test-model")
    record.source_thread_id = "thread"
    record = service.save(record)
    record = service.start_turn(record, "請先討論")
    return CodexRecorder(service, record, "tradingdev_recorder")


def event(recorder: CodexRecorder, method: str, **params: Any) -> None:
    recorder.event(method, {"threadId": "thread", "turnId": "turn", **params})


def source_user(prompt: str = "請先討論") -> dict[str, Any]:
    return {
        "id": "user",
        "type": "userMessage",
        "content": [{"type": "text", "text": prompt, "text_elements": []}],
    }


def acknowledge(recorder: CodexRecorder) -> None:
    for method in ("item/started", "item/completed"):
        event(recorder, method, item=source_user())


def test_streaming_is_durable_final_is_authoritative_and_reasoning_is_excluded(
    tmp_path: Path,
) -> None:
    recorder = recorder_at(tmp_path)
    acknowledge(recorder)
    event(
        recorder,
        "item/started",
        item={
            "type": "agentMessage",
            "id": "answer",
            "text": "",
            "phase": "final_answer",
        },
    )
    event(recorder, "item/agentMessage/delta", itemId="answer", delta="部分文字")
    saved = recorder.service.read(recorder.record.conversation_id)
    assert saved.turns[0].items[-1].text == "部分文字"
    event(
        recorder,
        "item/completed",
        item={
            "type": "reasoning",
            "id": "reasoning",
            "content": ["PRIVATE REASONING"],
            "summary": ["PRIVATE SUMMARY"],
        },
    )
    event(
        recorder,
        "item/completed",
        item={
            "type": "agentMessage",
            "id": "answer",
            "text": "來源最後定稿",
            "phase": "final_answer",
        },
    )
    event(
        recorder,
        "turn/completed",
        turn={"id": "turn", "status": "completed", "items": [], "itemsView": "summary"},
    )
    record = recorder.service.read(saved.conversation_id)
    assert record.turns[0].capture_complete
    assert record.turns[0].items[-1].text == "來源最後定稿"
    assert "PRIVATE" not in record.model_dump_json()


@pytest.mark.parametrize(
    "failure",
    [
        "wrong_thread",
        "wrong_turn",
        "unknown_item",
        "unstarted_delta",
        "changed_input",
        "unknown_notification",
    ],
)
def test_protocol_mismatch_leaves_honest_partial_capture(
    tmp_path: Path, failure: str
) -> None:
    recorder = recorder_at(tmp_path)
    acknowledge(recorder)
    with pytest.raises(CodexProtocolError):
        if failure == "wrong_thread":
            event(recorder, "item/started", threadId="other", item={"id": "a"})
        elif failure == "wrong_turn":
            event(recorder, "item/started", turnId="other", item={"id": "a"})
        elif failure == "unknown_item":
            event(
                recorder, "item/started", item={"id": "a", "type": "commandExecution"}
            )
        elif failure == "unstarted_delta":
            event(recorder, "item/agentMessage/delta", itemId="missing", delta="text")
        elif failure == "changed_input":
            event(recorder, "item/started", item=source_user("另一個輸入"))
        else:
            event(recorder, "item/futureVisibleText", text="not supported")
    recorder.end("interrupted", "protocol mismatch")
    assert not recorder.record.turns[0].capture_complete
    assert recorder.record.turns[0].input == "請先討論"


def test_missing_item_start_or_completion_is_not_complete(tmp_path: Path) -> None:
    recorder = recorder_at(tmp_path)
    acknowledge(recorder)
    event(
        recorder,
        "item/completed",
        item={"id": "a", "type": "agentMessage", "text": "final"},
    )
    event(
        recorder,
        "item/started",
        item={"id": "b", "type": "agentMessage", "text": "partial"},
    )
    event(recorder, "turn/completed", turn={"id": "turn", "status": "completed"})
    assert not recorder.record.turns[0].capture_complete
    assert len(recorder.record.turns[0].issues) == 2


def test_tool_failure_and_progress_are_preserved_without_turn_failure(
    tmp_path: Path,
) -> None:
    recorder = recorder_at(tmp_path)
    acknowledge(recorder)
    tool = {
        "id": "tool",
        "type": "mcpToolCall",
        "server": "tradingdev_recorder",
        "tool": "get_strategy",
        "arguments": {"strategy_id": "missing"},
        "status": "inProgress",
    }
    event(recorder, "item/started", item=tool)
    event(recorder, "item/mcpToolCall/progress", itemId="tool", message="reading")
    event(
        recorder,
        "item/completed",
        item={**tool, "status": "failed", "error": {"message": "not found"}},
    )
    event(recorder, "turn/completed", turn={"id": "turn", "status": "completed"})
    item = recorder.record.turns[0].items[-1]
    assert item.data["error"] == {"message": "not found"}
    assert item.data["progress"] == [
        {"threadId": "thread", "turnId": "turn", "itemId": "tool", "message": "reading"}
    ]
    assert recorder.record.turns[0].capture_complete


class FakeCodex(CodexTransport):
    """Replace only the external server, exercising production session orchestration."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.sent: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        self.mode = "normal"
        self.closed = 0

    async def read(self) -> dict[str, Any]:
        if not self.messages:
            if self.mode == "timeout":
                await asyncio.Event().wait()
            raise CodexProtocolError("simulated disconnect")
        return self.messages.pop(0)

    async def send(self, message: dict[str, Any]) -> None:
        self.sent.append(deepcopy(message))
        method = message.get("method")
        params = message.get("params", {})
        result: dict[str, Any] = {}
        if method == "initialize":
            self.messages.append(
                {
                    "method": "remoteControl/status/changed",
                    "params": {"status": "disabled"},
                }
            )
        if method == "initialized":
            return
        if method == "config/read":
            result = {"config": {"mcp_servers": {"other": {"enabled": True}}}}
        elif method in {"thread/start", "thread/resume"}:
            if method == "thread/resume":
                for notification in ("thread/goal/cleared", "thread/queue/changed"):
                    self.messages.append(
                        {"method": notification, "params": {"threadId": "thread"}}
                    )
            result = {
                "thread": {"id": "thread"},
                "model": "test-model",
                "reasoningEffort": "medium",
            }
        elif method == "thread/turns/list":
            result = {"data": self.history, "nextCursor": None}
        elif method == "turn/start":
            turn_id = f"turn-{len(self.history) + 1}"
            user = source_user(params["input"][0]["text"])
            answer = {
                "id": "answer",
                "type": "agentMessage",
                "text": "可見回覆",
                "phase": "final_answer",
            }

            def emit(method: str, **values: Any) -> None:
                self.messages.append(
                    {
                        "method": method,
                        "params": {"threadId": "thread", "turnId": turn_id, **values},
                    }
                )

            # Deliberately stream before the RPC acknowledgement.
            emit("turn/started", turn={"id": turn_id, "status": "inProgress"})
            emit("item/started", item=user)
            emit("item/completed", item=user)
            emit("item/started", item={**answer, "text": ""})
            emit("item/agentMessage/delta", itemId="answer", delta="可見")
            if self.mode == "normal":
                emit("item/completed", item=answer)
                emit("turn/completed", turn={"id": turn_id, "status": "completed"})
                self.history.append(
                    {
                        "id": turn_id,
                        "status": "completed",
                        "itemsView": "full",
                        "items": [user, answer],
                    }
                )
            result = {"turn": {"id": turn_id, "status": "inProgress"}}
            if self.mode in {"disconnect", "timeout"}:
                return
        self.messages.append({"id": message["id"], "result": result})


def install_fake(monkeypatch: pytest.MonkeyPatch, fake: FakeCodex) -> None:
    monkeypatch.setenv("TRADINGDEV_CODEX_BIN", sys.executable)

    @asynccontextmanager
    async def transport(*args: Any, **kwargs: Any) -> Any:
        try:
            yield fake
        finally:
            fake.closed += 1

    monkeypatch.setattr(codex_session, "codex_transport", transport)


def test_managed_resume_verifies_history_disables_other_servers_and_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeCodex()
    install_fake(monkeypatch, fake)
    workspace = WorkspacePaths(tmp_path)
    first = asyncio.run(
        record_turn(workspace=workspace, model="test-model", prompt="先討論")
    )
    assert first["success"] and fake.closed == 1
    first_bytes = Path(first["export"]["json_path"]).read_bytes()
    second = asyncio.run(
        record_turn(
            workspace=workspace,
            conversation_id=first["conversation_id"],
            prompt="繼續釐清",
        )
    )
    assert second["success"] and fake.closed == 2
    record = json.loads(Path(second["export"]["json_path"]).read_text())
    assert [turn["input"] for turn in record["turns"]] == ["先討論", "繼續釐清"]
    assert Path(first["export"]["json_path"]).read_bytes() == first_bytes
    started = next(
        message for message in fake.sent if message.get("method") == "thread/start"
    )
    assert started["params"]["config"]["mcp_servers"]["other"]["enabled"] is False
    fake.history[0]["items"][1]["text"] = "externally changed"
    third = asyncio.run(
        record_turn(
            workspace=workspace,
            conversation_id=first["conversation_id"],
            prompt="不要偷偷改歷史",
        )
    )
    assert not third["success"] and "differs" in third["error"]
    assert (
        len([message for message in fake.sent if message.get("method") == "turn/start"])
        == 2
    )


@pytest.mark.parametrize("mode", ["disconnect", "timeout"])
def test_failure_and_timeout_export_partial_text_and_close_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    fake = FakeCodex()
    fake.mode = mode
    install_fake(monkeypatch, fake)
    result = asyncio.run(
        record_turn(
            workspace=WorkspacePaths(tmp_path),
            model="test-model",
            prompt="input",
            timeout=0.05,
        )
    )
    assert not result["success"] and not result["capture_complete"]
    assert result["outcome"] == "interrupted" and fake.closed == 1
    data = json.loads(Path(result["export"]["json_path"]).read_text())
    assert data["turns"][0]["items"][-1]["text"] == "可見"
    assert "部分" in Path(result["export"]["html_path"]).read_text()


@pytest.mark.parametrize("mode", ["normal", "error", "timeout"])
def test_real_stdio_transport_reaps_client_group_on_every_exit(
    tmp_path: Path, mode: str
) -> None:
    # A real child starts another child in its group. No Codex/model is called.
    program = tmp_path / "server.py"
    program.write_text("""import json, os, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
print(json.dumps({'pid': os.getpid(), 'child': child.pid}), flush=True)
time.sleep(120)
""")
    observed: list[psutil.Process] = []

    async def run() -> None:
        async with codex_transport(
            [sys.executable, str(program)],
            cwd=tmp_path,
            stderr_path=tmp_path / "stderr.log",
        ) as transport:
            message = await transport.read()
            observed.extend(psutil.Process(message[key]) for key in ("pid", "child"))
            if mode == "error":
                raise ValueError("simulated failure")
            if mode == "timeout":
                async with asyncio.timeout(0.05):
                    await transport.read()

    if mode == "normal":
        asyncio.run(run())
    else:
        with pytest.raises(ValueError if mode == "error" else TimeoutError):
            asyncio.run(run())
    assert len(observed) == 2
    for process in observed:
        assert not process.is_running() or process.status() == psutil.STATUS_ZOMBIE


def test_noninteractive_elicitation_does_not_invent_user_consent(
    tmp_path: Path,
) -> None:
    recorder = recorder_at(tmp_path)
    acknowledge(recorder)
    fake = FakeCodex()
    session = codex_session.CodexSession(fake, recorder)
    asyncio.run(
        session.dispatch(
            {
                "id": "request",
                "method": "mcpServer/elicitation/request",
                "params": {
                    "threadId": "thread",
                    "turnId": "turn",
                    "serverName": "tradingdev_recorder",
                    "mode": "form",
                    "message": "同意？",
                    "requestedSchema": {"type": "object"},
                },
            }
        )
    )
    assert fake.sent[-1]["result"] == {"action": "cancel", "content": None}
    assert "no user consent" in recorder.record.turns[-1].items[-1].text


def test_known_interrupted_turn_can_resume_without_rewriting_missing_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeCodex()
    fake.mode = "disconnect"
    install_fake(monkeypatch, fake)
    workspace = WorkspacePaths(tmp_path)
    first = asyncio.run(
        record_turn(workspace=workspace, model="test-model", prompt="input")
    )
    first_record = json.loads(Path(first["export"]["json_path"]).read_text())
    fake.history = [
        {
            "id": "turn-1",
            "status": "interrupted",
            "itemsView": "full",
            "items": [
                source_user("input"),
                {
                    "id": "uncompleted-tool",
                    "type": "mcpToolCall",
                    "status": "inProgress",
                },
            ],
        }
    ]
    fake.mode = "normal"
    result = asyncio.run(
        record_turn(
            workspace=workspace, conversation_id=first["conversation_id"], prompt="繼續"
        )
    )
    assert result["success"]
    current = json.loads(Path(result["export"]["json_path"]).read_text())
    assert current["turns"][0] == first_record["turns"][0]
    assert not current["turns"][0]["capture_complete"]
    assert current["turns"][1]["capture_complete"]
    assert "含未完成或缺漏" in Path(result["export"]["html_path"]).read_text()


def test_lifetime_lock_and_model_binding_fail_before_new_model_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeCodex()
    install_fake(monkeypatch, fake)
    workspace = WorkspacePaths(tmp_path)
    result = asyncio.run(
        record_turn(workspace=workspace, model="test-model", prompt="first")
    )
    identifier = result["conversation_id"]
    count = len(fake.sent)
    with (
        FileLock(tmp_path / ".conversation-clients" / f"{identifier}.lock"),
        pytest.raises(Timeout),
    ):
        asyncio.run(
            record_turn(
                workspace=workspace, conversation_id=identifier, prompt="collision"
            )
        )
    with pytest.raises(ValueError, match="saved model"):
        asyncio.run(
            record_turn(
                workspace=workspace,
                conversation_id=identifier,
                model="other",
                prompt="changed",
            )
        )
    assert len(fake.sent) == count


def test_failed_export_still_returns_saved_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeCodex()
    install_fake(monkeypatch, fake)

    def fail_export(*args: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(ConversationService, "export", fail_export)
    result = asyncio.run(
        record_turn(
            workspace=WorkspacePaths(tmp_path), model="test-model", prompt="first"
        )
    )
    assert not result["success"] and result["export"] is None
    assert result["export_error"] == "disk full"
    workspace = WorkspacePaths(tmp_path)
    saved = ConversationService(workspace, SQLiteStore(workspace)).read(
        result["conversation_id"]
    )
    assert saved.turns[-1].capture_complete


def test_partial_history_cannot_rewrite_completed_user_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeCodex()
    fake.mode = "disconnect"
    install_fake(monkeypatch, fake)
    workspace = WorkspacePaths(tmp_path)
    first = asyncio.run(
        record_turn(workspace=workspace, model="test-model", prompt="original input")
    )
    fake.history = [
        {
            "id": "turn-1",
            "status": "interrupted",
            "itemsView": "full",
            "items": [source_user("EXTERNALLY REPLACED INPUT")],
        }
    ]
    fake.mode = "normal"
    second = asyncio.run(
        record_turn(
            workspace=workspace,
            conversation_id=first["conversation_id"],
            prompt="next input",
        )
    )
    assert not second["success"] and "differs" in second["error"]
    assert sum(message.get("method") == "turn/start" for message in fake.sent) == 1
    stored = json.loads(Path(second["export"]["json_path"]).read_text())
    assert stored["turns"][0]["input"] == "original input"


@pytest.mark.parametrize("status", [None, "unknown", "inProgress", "failed"])
def test_history_must_have_known_consistent_terminal_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str | None
) -> None:
    fake = FakeCodex()
    install_fake(monkeypatch, fake)
    workspace = WorkspacePaths(tmp_path)
    first = asyncio.run(
        record_turn(workspace=workspace, model="test-model", prompt="original")
    )
    if status is None:
        del fake.history[0]["status"]
    else:
        fake.history[0]["status"] = status
    second = asyncio.run(
        record_turn(
            workspace=workspace, conversation_id=first["conversation_id"], prompt="next"
        )
    )
    assert not second["success"]
    assert sum(message.get("method") == "turn/start" for message in fake.sent) == 1


@pytest.mark.parametrize("thread_id,turn_id", [("other", "turn"), ("thread", "other")])
def test_error_cannot_be_recorded_under_another_identity(
    tmp_path: Path, thread_id: str, turn_id: str
) -> None:
    recorder = recorder_at(tmp_path)
    acknowledge(recorder)
    before = recorder.record.model_dump_json()
    with pytest.raises(CodexProtocolError):
        recorder.event(
            "error",
            {
                "threadId": thread_id,
                "turnId": turn_id,
                "error": {"message": "foreign error"},
                "willRetry": False,
            },
        )
    assert recorder.record.model_dump_json() == before


def test_warning_scope_and_standalone_elicitation_are_explicit(tmp_path: Path) -> None:
    recorder = recorder_at(tmp_path)
    acknowledge(recorder)
    recorder.event("warning", {"threadId": None, "message": "global warning"})
    with pytest.raises(CodexProtocolError, match="different source thread"):
        recorder.event("guardianWarning", {"threadId": "other", "message": "foreign"})
    fake = FakeCodex()
    session = codex_session.CodexSession(fake, recorder)
    params = {
        "threadId": "thread",
        "serverName": "tradingdev_recorder",
        "mode": "form",
        "message": "question",
        "requestedSchema": {"type": "object"},
    }
    before = len(recorder.record.turns[-1].items)
    with pytest.raises(CodexProtocolError, match="turn ID"):
        asyncio.run(
            session.dispatch(
                {
                    "id": "wrong-turn",
                    "method": "mcpServer/elicitation/request",
                    "params": {**params, "turnId": "other"},
                }
            )
        )
    assert len(recorder.record.turns[-1].items) == before
    assert "error" in fake.sent[-1]
    asyncio.run(
        session.dispatch(
            {
                "id": "standalone",
                "method": "mcpServer/elicitation/request",
                "params": {**params, "turnId": None},
            }
        )
    )
    assert (
        recorder.record.turns[-1].items[-1].data["source_turn_correlation"]
        == "standalone"
    )
    assert fake.sent[-1]["result"]["action"] == "cancel"
