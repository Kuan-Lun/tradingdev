"""Persist immutable confirmation content and serialize approval transitions."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from filelock import FileLock

from tradingdev.domain.execution_plan import ExecutionPlan

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from tradingdev.adapters.storage.filesystem import WorkspacePaths
    from tradingdev.adapters.storage.sqlite import SQLiteStore
    from tradingdev.domain.execution_plan import PlanState


class ExecutionPlanStore:
    """Keep approval state separate from the content the user reviewed."""

    def __init__(self, workspace: WorkspacePaths, store: SQLiteStore) -> None:
        self.workspace = workspace
        self.store = store
        with store.connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS execution_plans (
                plan_id TEXT PRIMARY KEY, payload TEXT NOT NULL,
                state TEXT NOT NULL, confirmation_token TEXT,
                job_id TEXT, error TEXT, html_hash TEXT NOT NULL,
                text_hash TEXT NOT NULL)"""
            )

    def directory(self, plan_id: str) -> Path:
        if re.fullmatch(r"[0-9a-f]{32}", plan_id) is None:
            raise ValueError("Invalid execution plan ID")
        path = self.workspace.root
        for part in ("execution_plans", plan_id):
            path = path / part
            if path.is_symlink() or not path.resolve().is_relative_to(
                self.workspace.root
            ):
                raise ValueError("Execution plan path leaves workspace")
        return path

    def publish(self, plan: ExecutionPlan, text: str, html: str) -> None:
        plan.verify()
        directory = self.directory(plan.plan_id)
        directory.mkdir(parents=True, exist_ok=False)
        try:
            html_path = directory / "confirmation.html"
            html_path.write_text(html, encoding="utf-8")
            (directory / "confirmation.txt").write_text(text, encoding="utf-8")
            html_hash = hashlib.sha256(html.encode()).hexdigest()
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            with self.store.connect() as conn:
                conn.execute(
                    "INSERT INTO execution_plans "
                    "(plan_id,payload,state,html_hash,text_hash) VALUES (?,?,?,?,?)",
                    (
                        plan.plan_id,
                        plan.model_dump_json(),
                        "ready",
                        html_hash,
                        text_hash,
                    ),
                )
                conn.execute(
                    "INSERT INTO artifacts "
                    "(artifact_id,run_id,artifact_type,path,sha256,metadata,"
                    "created_at) "
                    "VALUES (?,NULL,?,?,?,?,?)",
                    (
                        f"{plan.plan_id}:confirmation_html",
                        "execution_confirmation",
                        str(html_path),
                        html_hash,
                        json.dumps(
                            {"plan_id": plan.plan_id, "plan_hash": plan.plan_hash}
                        ),
                        plan.created_at.isoformat(),
                    ),
                )
        except BaseException:
            shutil.rmtree(directory)
            raise

    def get(self, plan_id: str) -> tuple[ExecutionPlan, dict[str, Any]]:
        directory = self.directory(plan_id)
        with self.store.connect() as conn:
            row = conn.execute(
                "SELECT * FROM execution_plans WHERE plan_id=?", (plan_id,)
            ).fetchone()
        if row is None:
            raise LookupError("Execution plan was not found")
        record = dict(row)
        plan = ExecutionPlan.model_validate_json(record["payload"])
        if plan.plan_id != plan_id:
            raise ValueError("Execution plan identity mismatch")
        for name, hash_key in (
            ("confirmation.html", "html_hash"),
            ("confirmation.txt", "text_hash"),
        ):
            path = directory / name
            if (
                path.is_symlink()
                or hashlib.sha256(path.read_bytes()).hexdigest() != record[hash_key]
            ):
                raise ValueError("Saved confirmation document has changed")
        return plan, record

    @contextmanager
    def lock(self, plan_id: str) -> Generator[None]:
        directory = self.directory(plan_id)
        if not directory.is_dir():
            raise LookupError("Execution plan was not found")
        # Retain the lock inode while other processes may have opened it.
        with FileLock(directory / ".approval.lock", timeout=15):
            yield

    def transition(
        self,
        plan_id: str,
        expected: PlanState,
        state: PlanState,
        *,
        token: str | None = None,
        job_id: str | None = None,
        error: str | None = None,
    ) -> None:
        """Call under lock; compare state again before committing the transition."""
        with self.store.connect() as conn:
            changed = conn.execute(
                "UPDATE execution_plans SET state=?,confirmation_token=?,"
                "job_id=?,error=? "
                "WHERE plan_id=? AND state=?",
                (state, token, job_id, error, plan_id, expected),
            ).rowcount
        if changed != 1:
            raise ValueError("Execution plan state changed; inspect the plan again")
