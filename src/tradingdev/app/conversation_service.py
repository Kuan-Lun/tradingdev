"""Shared recording lifecycle; provider adapters supply observed public items."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from tradingdev.adapters.storage.conversation_exports import ConversationExports
from tradingdev.adapters.storage.conversations import ConversationStore
from tradingdev.adapters.storage.filesystem import now_iso
from tradingdev.app.contracts.common import Page, PageRequest
from tradingdev.app.contracts.conversations import ConversationSummary
from tradingdev.domain.conversations import ConversationRecord, ConversationTurn

if TYPE_CHECKING:
    from pydantic import JsonValue

    from tradingdev.adapters.storage.filesystem import WorkspacePaths
    from tradingdev.adapters.storage.sqlite import SQLiteStore
    from tradingdev.app.contracts.conversations import ConversationExport


class ConversationService:
    def __init__(self, workspace: WorkspacePaths, store: SQLiteStore) -> None:
        self.records = ConversationStore(store)
        self.exports = ConversationExports(workspace, store)

    def create(
        self, provider: str, model: str, *, settings: dict[str, JsonValue] | None = None
    ) -> ConversationRecord:
        now = now_iso()
        record = ConversationRecord(
            conversation_id=uuid4().hex,
            provider=provider,
            model=model,
            settings=settings or {},
            created_at=now,
            updated_at=now,
        )
        self.records.save(record, None)
        return record

    def read(self, conversation_id: str) -> ConversationRecord:
        return self.records.load(conversation_id)

    def save(self, record: ConversationRecord) -> ConversationRecord:
        candidate = ConversationRecord.model_validate(
            {
                **record.model_dump(),
                "version": record.version + 1,
                "updated_at": now_iso(),
            }
        )
        self.records.save(candidate, record.version)
        return candidate

    def start_turn(self, record: ConversationRecord, prompt: str) -> ConversationRecord:
        if not prompt.strip():
            raise ValueError("A conversation input must contain text")
        if not record.source_thread_id:
            raise ValueError("Bind a source thread before starting a turn")
        turn = ConversationTurn(turn_id=uuid4().hex, input=prompt, started_at=now_iso())
        return self.save(record.model_copy(update={"turns": [*record.turns, turn]}))

    def list(self, offset: int = 0, limit: int = 20) -> Page[ConversationSummary]:
        request = PageRequest(offset=offset, limit=limit)
        rows, total = self.records.list(request.offset, request.limit)
        end = offset + len(rows)
        return Page[ConversationSummary](
            items=rows,
            total=total,
            offset=offset,
            limit=limit,
            next_offset=end if end < total else None,
        )

    def export(self, conversation_id: str, expected_version: int) -> ConversationExport:
        record = self.read(conversation_id)
        if record.version != expected_version:
            raise ValueError(
                "Conversation changed; select its current version before exporting"
            )
        return self.exports.publish(record)
