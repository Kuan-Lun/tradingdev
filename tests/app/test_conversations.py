"""Recorded text, immutable history, partial capture and offline reading evidence."""

from __future__ import annotations

import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tradingdev.adapters.cli.conversations import main
from tradingdev.adapters.storage.conversations import ConversationConflictError
from tradingdev.adapters.storage.filesystem import WorkspacePaths, now_iso
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.conversation_service import ConversationService
from tradingdev.domain.conversations import ConversationItem, ConversationRecord


def service_at(root: Path) -> ConversationService:
    workspace = WorkspacePaths(root)
    return ConversationService(workspace, SQLiteStore(workspace))


def running(service: ConversationService) -> ConversationRecord:
    record = service.create("test-client", "test-model")
    record.source_thread_id = "thread-1"
    record = service.save(record)
    record = service.start_turn(
        record, "我想用均線；<script>alert('x')</script>\n先討論。"
    )
    record.turns[-1].source_turn_id = "source-turn-1"
    return service.save(record)


def finish(
    service: ConversationService, record: ConversationRecord
) -> ConversationRecord:
    turn = record.turns[-1]
    turn.items = [
        ConversationItem(
            source_id="message-1",
            kind="assistant",
            text="請先選擇期間。",
            complete=True,
            data={"phase": "commentary"},
        ),
        ConversationItem(
            source_id="tool-1",
            kind="tool",
            complete=True,
            data={
                "tool": "explore",
                "arguments": {"offset": 0},
                "result": {"text": "完整結果 <img src=x onerror=alert(1)>"},
            },
        ),
        ConversationItem(
            source_id="message-2",
            kind="assistant",
            complete=True,
            text="尚未執行。",
            data={"phase": "final_answer"},
        ),
    ]
    turn.outcome = "completed"
    turn.capture_complete = True
    turn.finished_at = now_iso()
    return service.save(record)


def test_resume_immutable_export_and_chronological_selectable_text(
    tmp_path: Path,
) -> None:
    service = service_at(tmp_path)
    record = finish(service, running(service))
    snapshot = service.export(record.conversation_id, record.version)
    html = Path(snapshot.html_path).read_text()
    original = Path(snapshot.json_path).read_bytes()
    assert "完整對話紀錄" in html and "&lt;script&gt;" in html
    assert "<img src=x" not in html
    assert html.index("請先選擇期間") < html.index("完整結果") < html.index("尚未執行")
    assert (
        html.index("AI 代理 · 過程訊息")
        < html.index("請先選擇期間")
        < html.index("完整結果")
        < html.index("AI 代理 · 本輪回覆")
        < html.index("尚未執行")
    )
    assert json.loads(original) == record.model_dump(mode="json")
    artifact = service.exports.store.get_artifact(snapshot.artifact_id)
    assert artifact is not None and artifact["sha256"] == snapshot.sha256
    restarted = service_at(tmp_path)
    record2 = restarted.start_turn(restarted.read(record.conversation_id), "先看半年。")
    assert record2.turns[0] == record.turns[0]
    assert Path(snapshot.json_path).read_bytes() == original
    assert (
        service.export(record.conversation_id, record2.version).export_id
        != snapshot.export_id
    )
    with pytest.raises(ValueError, match="changed"):
        service.export(record.conversation_id, record.version)


def test_partial_record_retained_and_cannot_claim_complete(tmp_path: Path) -> None:
    service = service_at(tmp_path)
    record = running(service)
    turn = record.turns[-1]
    turn.items = [ConversationItem(source_id="m", kind="assistant", text="未完成回覆")]
    record = service.save(record)
    html = Path(
        service.export(record.conversation_id, record.version).html_path
    ).read_text()
    assert "未完成回覆" in html and "含未完成或缺漏" in html
    record.turns[-1].capture_complete = True
    with pytest.raises(ValueError, match="Complete capture"):
        service.save(record)
    assert service.read(record.conversation_id).version == record.version
    record.turns[-1].capture_complete = False
    record.turns[-1].outcome = "interrupted"
    record.turns[-1].finished_at = now_iso()
    record.turns[-1].issues.append("連線中斷，未收到完成事件")
    record = service.save(record)
    assert (
        "連線中斷"
        in Path(
            service.export(record.conversation_id, record.version).html_path
        ).read_text()
    )


@pytest.mark.parametrize("phase", [None, "future_phase", {"unexpected": "value"}])
def test_unrecognized_phase_is_visible_without_guessing_an_answer(
    tmp_path: Path, phase: object
) -> None:
    service = service_at(tmp_path)
    record = running(service)
    record.turns[0].items = [
        ConversationItem(
            source_id="unknown-phase",
            kind="assistant",
            text="原始文字 <em>x</em>",
            complete=True,
            data={"phase": phase},
        )
    ]
    record = service.save(record)
    snapshot = service.export(record.conversation_id, record.version)
    html = Path(snapshot.html_path).read_text()
    assert "AI 代理 · 階段未標示" in html
    assert "AI 代理 · 本輪回覆" not in html
    assert "原始文字 &lt;em&gt;x&lt;/em&gt;" in html
    assert json.loads(Path(snapshot.json_path).read_text()) == record.model_dump(
        mode="json"
    )


def test_partial_final_answer_keeps_phase_and_capture_status_separate(
    tmp_path: Path,
) -> None:
    service = service_at(tmp_path)
    record = running(service)
    record.turns[0].items = [
        ConversationItem(
            source_id="partial-final",
            kind="assistant",
            text="尚未傳完",
            data={"phase": "final_answer"},
        )
    ]
    record = service.save(record)
    html = Path(
        service.export(record.conversation_id, record.version).html_path
    ).read_text()
    assert "AI 代理 · 本輪回覆" in html and "訊息紀錄：部分" in html
    assert "含未完成或缺漏" in html


def test_process_updates_and_tools_share_a_block_without_reordering_replies(
    tmp_path: Path,
) -> None:
    service = service_at(tmp_path)
    record = running(service)
    turn = record.turns[0]
    turn.items = [
        ConversationItem(
            source_id="p1",
            kind="assistant",
            text="先查資料",
            data={"phase": "commentary"},
        ),
        ConversationItem(source_id="t1", kind="tool", text="查詢結果"),
        ConversationItem(
            source_id="p2",
            kind="assistant",
            text="再核對成本",
            data={"phase": "commentary"},
        ),
        ConversationItem(
            source_id="a1",
            kind="assistant",
            text="正式回覆",
            data={"phase": "final_answer"},
        ),
        ConversationItem(
            source_id="p3",
            kind="assistant",
            text="後續過程",
            data={"phase": "commentary"},
        ),
    ]
    record = service.save(record)
    html = Path(
        service.export(record.conversation_id, record.version).html_path
    ).read_text()
    blocks = re.findall(r"<article class='([^']*)'>(.*?)</article>", html, re.S)
    assert len(blocks) == 3
    assert [kind for kind, _ in blocks] == [
        "process-message",
        "answer-message",
        "process-message",
    ]
    process = blocks[0][1]
    assert process.count("AI 代理 · 過程訊息") == 1
    assert (
        process.index("先查資料")
        < process.index("查詢結果")
        < process.index("再核對成本")
    )
    assert "正式回覆" in blocks[1][1] and "後續過程" in blocks[2][1]


def test_render_keeps_text_and_identity_when_provider_keys_collide(
    tmp_path: Path,
) -> None:
    service = service_at(tmp_path)
    record = running(service)
    record.turns[0].items = [
        ConversationItem(
            source_id="tool-original-id",
            kind="tool",
            text="original tool text",
            data={"text": "provider text"},
        ),
        ConversationItem(
            source_id="assistant-original-id",
            kind="assistant",
            text="answer",
            data={"source_id": "provider-id"},
        ),
    ]
    record = service.save(record)
    html = Path(
        service.export(record.conversation_id, record.version).html_path
    ).read_text()
    for expected in (
        "original tool text",
        "provider text",
        "assistant-original-id",
        "provider-id",
    ):
        assert expected in html


def test_save_cannot_bypass_source_thread_binding(tmp_path: Path) -> None:
    service = service_at(tmp_path)
    example = finish(service, running(service))
    unbound = service.create("client", "model")
    unbound.turns = example.turns
    with pytest.raises(ValueError, match="bound source thread"):
        service.save(unbound)
    saved = service.read(unbound.conversation_id)
    assert saved.source_thread_id is None and not saved.turns


@pytest.mark.parametrize(
    "change", ["past_text", "input", "thread", "model", "remove", "turn_id"]
)
def test_cannot_rewrite_finished_history(tmp_path: Path, change: str) -> None:
    service = service_at(tmp_path)
    record = finish(service, running(service))
    before = record.model_dump_json()
    if change == "past_text":
        record.turns[0].items[0].text = "rewritten"
    elif change == "input":
        record.turns[0].input = "rewritten"
    elif change == "thread":
        record.source_thread_id = "other-thread"
    elif change == "model":
        record.model = "other-model"
    elif change == "remove":
        record.turns = []
    else:
        record.turns[0].source_turn_id = "other-turn"
    with pytest.raises(ConversationConflictError):
        service.save(record)
    assert service.read(record.conversation_id).model_dump_json() == before


def test_completed_item_immutable_while_turn_still_running(tmp_path: Path) -> None:
    service = service_at(tmp_path)
    record = running(service)
    record.turns[0].items.append(
        ConversationItem(
            source_id="m", kind="assistant", text="final item", complete=True
        )
    )
    record = service.save(record)
    record.turns[0].items[0].text = "changed"
    with pytest.raises(ConversationConflictError, match="Completed items"):
        service.save(record)


def test_concurrent_updates_have_one_winner(tmp_path: Path) -> None:
    service = service_at(tmp_path)
    record = running(service)

    def save() -> bool:
        try:
            service.save(record)
            return True
        except ConversationConflictError:
            return False

    with ThreadPoolExecutor(2) as pool:
        assert sorted(pool.map(lambda _: save(), range(2))) == [False, True]


def test_capacity_failure_keeps_previous_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = service_at(tmp_path)
    record = running(service)
    from tradingdev.adapters.storage import conversations

    size = len(conversations.conversation_bytes(record))
    monkeypatch.setattr(conversations, "MAX_CONVERSATION_BYTES", size + 10)
    record.turns[0].items.append(
        ConversationItem(source_id="m", kind="assistant", text="內容" * 100)
    )
    with pytest.raises(ValueError, match="exceeds"):
        service.save(record)
    assert not service.read(record.conversation_id).turns[0].items


def test_export_rejects_tampering_and_symlinks(tmp_path: Path) -> None:
    service = service_at(tmp_path)
    record = finish(service, running(service))
    saved = service.export(record.conversation_id, record.version)
    path = Path(saved.html_path)
    path.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        service.export(record.conversation_id, record.version)
    path.unlink()
    path.symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="symlink"):
        service.export(record.conversation_id, record.version)


def test_export_registration_failure_cleans_only_new_snapshot(tmp_path: Path) -> None:
    service = service_at(tmp_path)
    record = finish(service, running(service))
    with service.records.store.connect() as conn:
        conn.execute(
            "create trigger reject_conversation before insert on artifacts "
            "begin select raise(abort,'rejected'); end"
        )
    with pytest.raises(sqlite3.IntegrityError, match="rejected"):
        service.export(record.conversation_id, record.version)
    assert not [
        path for path in (tmp_path / "conversations").iterdir() if path.is_dir()
    ]
    assert service.read(record.conversation_id) == record


def test_cli_discovery_read_export_and_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    service = service_at(tmp_path)
    record = finish(service, running(service))
    service.create("second", "model")
    prefix = ["--workspace", str(tmp_path)]
    main([*prefix, "list", "--limit", "1"])
    page = json.loads(capsys.readouterr().out)["result"]
    assert page["total"] == 2 and page["next_offset"] == 1
    main([*prefix, "read", record.conversation_id])
    assert (
        json.loads(capsys.readouterr().out)["result"]["turns"][0]["input"]
        == record.turns[0].input
    )
    main(
        [
            *prefix,
            "export",
            record.conversation_id,
            "--expected-version",
            str(record.version),
        ]
    )
    assert Path(json.loads(capsys.readouterr().out)["result"]["html_path"]).is_file()
    with pytest.raises(SystemExit):
        main([*prefix, "read", "../../unknown"])
    assert json.loads(capsys.readouterr().out)["success"] is False
