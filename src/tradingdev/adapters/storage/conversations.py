"""Atomic, version-checked conversation recording with immutable finished turns."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from tradingdev.app.contracts.conversations import ConversationSummary
from tradingdev.domain.conversations import (
    MAX_CONVERSATION_BYTES,
    ConversationId,
    ConversationRecord,
)

if TYPE_CHECKING:
    from tradingdev.adapters.storage.sqlite import SQLiteStore


class ConversationConflictError(ValueError):
    """The recorder selected a stale version or attempted to rewrite history."""


def conversation_bytes(record: ConversationRecord) -> bytes:
    content = record.model_dump_json().encode("utf-8")
    if len(content) > MAX_CONVERSATION_BYTES:
        raise ValueError(
            "Conversation exceeds 32 MiB; retain it and start a new conversation"
        )
    return content


class ConversationStore:
    def __init__(self, store: SQLiteStore) -> None:
        self.store = store
        with store.connect() as conn:
            conn.execute("""create table if not exists conversations (
                conversation_id text primary key, version integer not null,
                updated_at text not null, document text not null
            )""")

    def load(self, conversation_id: str) -> ConversationRecord:
        TypeAdapter(ConversationId).validate_python(conversation_id)
        with self.store.connect() as conn:
            row = conn.execute(
                "select * from conversations where conversation_id=?",
                (conversation_id,),
            ).fetchone()
        if row is None:
            raise ValueError("Conversation not found")
        record = ConversationRecord.model_validate_json(row["document"])
        conversation_bytes(record)
        if (record.conversation_id, record.version, record.updated_at) != (
            conversation_id,
            row["version"],
            row["updated_at"],
        ):
            raise ValueError("Conversation identity differs from its stored index")
        return record

    def save(self, record: ConversationRecord, expected_version: int | None) -> None:
        record = ConversationRecord.model_validate(record)
        document = conversation_bytes(record).decode("utf-8")
        with self.store.connect() as conn:
            conn.execute("begin immediate")
            row = conn.execute(
                "select document from conversations where conversation_id=?",
                (record.conversation_id,),
            ).fetchone()
            if expected_version is None:
                if (
                    row is not None
                    or record.version != 1
                    or record.turns
                    or record.source_thread_id
                ):
                    raise ConversationConflictError(
                        "A new conversation must be empty and have a new identity"
                    )
                conn.execute(
                    "insert into conversations values (?,?,?,?)",
                    (
                        record.conversation_id,
                        record.version,
                        record.updated_at,
                        document,
                    ),
                )
                return
            if row is None:
                raise ConversationConflictError("Conversation disappeared")
            previous = ConversationRecord.model_validate_json(row["document"])
            if (
                previous.version != expected_version
                or record.version != expected_version + 1
            ):
                raise ConversationConflictError(
                    "Conversation changed; reload before recording"
                )
            self._check_history(previous, record)
            conn.execute(
                "update conversations set version=?,updated_at=?,document=? "
                "where conversation_id=?",
                (record.version, record.updated_at, document, record.conversation_id),
            )

    @staticmethod
    def _check_history(
        previous: ConversationRecord, record: ConversationRecord
    ) -> None:
        for field in ("provider", "model", "created_at"):
            if getattr(record, field) != getattr(previous, field):
                raise ConversationConflictError("Conversation identity cannot change")
        if (
            previous.source_thread_id
            and record.source_thread_id != previous.source_thread_id
        ):
            raise ConversationConflictError("Source thread cannot change")
        if not len(previous.turns) <= len(record.turns) <= len(previous.turns) + 1:
            raise ConversationConflictError(
                "Record one turn at a time; history cannot be removed"
            )
        for before, after in zip(previous.turns, record.turns, strict=False):
            if before.outcome != "running":
                if before != after:
                    raise ConversationConflictError("Finished turns are immutable")
                continue
            if (before.turn_id, before.input, before.started_at) != (
                after.turn_id,
                after.input,
                after.started_at,
            ):
                raise ConversationConflictError(
                    "Turn identity and original input cannot change"
                )
            if before.source_turn_id and after.source_turn_id != before.source_turn_id:
                raise ConversationConflictError("Source turn cannot change")
            if (
                len(after.items) < len(before.items)
                or after.issues[: len(before.issues)] != before.issues
            ):
                raise ConversationConflictError("Recorded evidence cannot be removed")
            for old, new in zip(before.items, after.items, strict=False):
                if (old.source_id, old.kind) != (new.source_id, new.kind) or (
                    old.complete and old != new
                ):
                    raise ConversationConflictError(
                        "Completed items and item order cannot change"
                    )

    def list(self, offset: int, limit: int) -> tuple[list[ConversationSummary], int]:
        with self.store.connect() as conn:
            conn.execute("begin")
            total = int(
                conn.execute("select count(*) from conversations").fetchone()[0]
            )
            rows = conn.execute(
                "select conversation_id,version,updated_at,"
                "json_extract(document,'$.provider') as provider,"
                "json_extract(document,'$.model') as model,"
                "json_array_length(document,'$.turns') as turns from conversations "
                "order by updated_at desc,conversation_id limit ? offset ?",
                (limit, offset),
            ).fetchall()
        return [ConversationSummary.model_validate(dict(row)) for row in rows], total
