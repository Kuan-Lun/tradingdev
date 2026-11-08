"""Immutable conversation HTML/JSON snapshots, registered for artifact discovery."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from filelock import FileLock

from tradingdev.adapters.reporting.conversation import render_conversation
from tradingdev.adapters.storage.conversations import conversation_bytes
from tradingdev.adapters.storage.filesystem import now_iso
from tradingdev.adapters.storage.workspace_files import WorkspaceFiles
from tradingdev.app.contracts.conversations import ConversationExport

if TYPE_CHECKING:
    from tradingdev.adapters.storage.filesystem import WorkspacePaths
    from tradingdev.adapters.storage.sqlite import SQLiteStore
    from tradingdev.domain.conversations import ConversationRecord

MAX_EXPORT_BYTES = 128 * 1024 * 1024


class ConversationExports:
    def __init__(self, workspace: WorkspacePaths, store: SQLiteStore) -> None:
        self.workspace, self.store = workspace, store
        self.files = WorkspaceFiles(workspace.root)

    def publish(self, record: ConversationRecord) -> ConversationExport:
        files = {
            "conversation.json": conversation_bytes(record),
            "index.html": render_conversation(record).encode("utf-8"),
        }
        hashes = {
            name: hashlib.sha256(content).hexdigest() for name, content in files.items()
        }
        files["manifest.json"] = json.dumps(
            {
                "schema_version": 1,
                "conversation_id": record.conversation_id,
                "version": record.version,
                "files": hashes,
            },
            sort_keys=True,
        ).encode()
        if sum(map(len, files.values())) > MAX_EXPORT_BYTES:
            raise ValueError("Conversation reading snapshot exceeds 128 MiB")
        export_id = hashlib.sha256(files["manifest.json"]).hexdigest()
        root = self.files.safe_path(self.workspace.root / "conversations")
        root.mkdir(exist_ok=True)
        directory = self.files.safe_path(root / export_id)
        artifact_id = f"conversation:{export_id}:html"
        with FileLock(self.files.safe_path(root / f".{export_id}.lock"), timeout=5):
            created = False
            try:
                if directory.exists():
                    for name, content in files.items():
                        if (
                            self.files.read(directory / name, MAX_EXPORT_BYTES)
                            != content
                        ):
                            raise ValueError(
                                "Saved conversation export changed; "
                                "refusing replacement"
                            )
                else:
                    with TemporaryDirectory(
                        prefix=".conversation-", dir=root
                    ) as temporary:
                        staged = Path(temporary) / "snapshot"
                        staged.mkdir()
                        for name, content in files.items():
                            (staged / name).write_bytes(content)
                        self.files.safe_path(directory)
                        staged.rename(directory)
                        created = True
                with self.store.connect() as conn:
                    existing = conn.execute(
                        "select * from artifacts where artifact_id=?", (artifact_id,)
                    ).fetchone()
                    path = str(directory / "index.html")
                    if existing is not None:
                        if (
                            existing["path"],
                            existing["sha256"],
                            existing["artifact_type"],
                        ) != (path, hashes["index.html"], "conversation_html"):
                            raise ValueError("Conversation export registration differs")
                    else:
                        conn.execute(
                            "insert into artifacts values (?,NULL,?,?,?,?,?)",
                            (
                                artifact_id,
                                "conversation_html",
                                path,
                                hashes["index.html"],
                                json.dumps(
                                    {
                                        "conversation_id": record.conversation_id,
                                        "version": record.version,
                                    }
                                ),
                                now_iso(),
                            ),
                        )
            except BaseException:
                if created:
                    shutil.rmtree(directory)
                raise
        return ConversationExport(
            conversation_id=record.conversation_id,
            version=record.version,
            export_id=export_id,
            artifact_id=artifact_id,
            html_path=str(directory / "index.html"),
            json_path=str(directory / "conversation.json"),
            sha256=hashes["index.html"],
        )
