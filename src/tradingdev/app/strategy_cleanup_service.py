"""Explicit cleanup of intact, unreferenced generated strategy drafts."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from tradingdev.adapters.storage.sqlite import SQLiteStore, get_sqlite_store
from tradingdev.adapters.storage.strategy_references import referenced_revisions
from tradingdev.adapters.storage.strategy_revisions import (
    StrategyRevisionError,
    StrategyRevisionStore,
)
from tradingdev.app.contracts.common import ErrorResponse
from tradingdev.app.contracts.strategy_cleanup import (
    StrategyCleanupItem,
    StrategyCleanupResult,
)
from tradingdev.domain.strategies.schemas import StrategyStatus

if TYPE_CHECKING:
    from tradingdev.adapters.storage.filesystem import WorkspacePaths


class StrategyCleanupService:
    """Preview by default; apply only the explicitly enumerated revision IDs."""

    def __init__(
        self, workspace: WorkspacePaths, store: SQLiteStore | None = None
    ) -> None:
        self._workspace = workspace
        self._store = store or get_sqlite_store(workspace)
        self._revisions = StrategyRevisionStore(workspace)

    def cleanup(
        self,
        strategy_id: str,
        revision_ids: list[str] | None = None,
        *,
        apply: bool = False,
    ) -> StrategyCleanupResult | ErrorResponse:
        """Recheck current, lifecycle, references, and file integrity under a lock.

        Validated and runnable sources are always retained, so new execution
        submissions cannot race deletion: only drafts are eligible for removal.
        No migration, automatic retention policy, or removal of legacy files occurs.
        """
        if apply and not revision_ids:
            return ErrorResponse(
                success=False,
                code="cleanup_revision_ids_required",
                error="Apply requires explicit revision_ids from a reviewed preview",
            )
        try:
            with self._revisions.lifecycle_lock(strategy_id):
                ids = (
                    self._revisions.revision_ids(strategy_id)
                    if revision_ids is None
                    else list(dict.fromkeys(revision_ids))
                )
                current = self._revisions.load(strategy_id)
                current_id = current.revision_id if current is not None else None
                references = referenced_revisions(
                    self._workspace, self._store, strategy_id
                )
                items = [
                    self._inspect(strategy_id, revision_id, current_id, references)
                    for revision_id in ids
                ]
                if apply:
                    for item in items:
                        if item.outcome == "eligible":
                            try:
                                self._revisions.delete_draft(
                                    strategy_id, item.revision_id
                                )
                            except (OSError, StrategyRevisionError) as exc:
                                item.outcome = "failed"
                                item.reasons = [str(exc)]
                            else:
                                item.outcome = "deleted"
                                item.reasons = []
                return StrategyCleanupResult(
                    success=not any(
                        item.outcome == "failed"
                        or apply
                        and item.outcome == "protected"
                        for item in items
                    ),
                    strategy_id=strategy_id,
                    applied=apply,
                    revisions=items,
                )
        except (OSError, ValueError, sqlite3.Error) as exc:
            return ErrorResponse(
                success=False, code="strategy_cleanup_blocked", error=str(exc)
            )

    def _inspect(
        self,
        strategy_id: str,
        revision_id: str,
        current_id: str | None,
        references: dict[str, list[str]],
    ) -> StrategyCleanupItem:
        reasons = list(references.get(revision_id, []))
        if revision_id == current_id:
            reasons.append("current_revision")
        try:
            metadata = self._revisions.load(strategy_id, revision_id)
            if metadata is None:
                return StrategyCleanupItem(
                    revision_id=revision_id,
                    outcome="missing",
                    reasons=["revision_not_found", *reasons],
                )
            if metadata.status != StrategyStatus.DRAFT:
                reasons.append(f"status:{metadata.status.value}")
            self._revisions.verify_cleanup_tree(strategy_id, revision_id)
        except (OSError, StrategyRevisionError) as exc:
            reasons.append(f"integrity:{exc}")
        return StrategyCleanupItem(
            revision_id=revision_id,
            outcome="protected" if reasons else "eligible",
            reasons=reasons,
        )
