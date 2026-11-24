"""Select saved results explicitly; experiments never update representatives."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter

from tradingdev.adapters.storage.representatives import (
    RepresentativeSources,
    RepresentativeStore,
)
from tradingdev.app.contracts.common import PageRequest
from tradingdev.app.rolling_trace_summary import query_metadata
from tradingdev.domain.representatives import (
    RepresentativeConflictError,
    RepresentativeId,
    RepresentativeRecord,
    RepresentativeSnapshot,
    RepresentativeVersion,
    ResearchGroup,
    SavedRunId,
    SavedScope,
    SelectionReason,
)
from tradingdev.domain.research_scope import (
    effective_scope_parameters,
    verify_research_scopes,
)

if TYPE_CHECKING:
    from tradingdev.adapters.storage.filesystem import WorkspacePaths
    from tradingdev.adapters.storage.sqlite import SQLiteStore
    from tradingdev.domain.performance.artifacts import PerformanceBundle


@dataclass(frozen=True)
class ResolvedRepresentative:
    record: RepresentativeRecord
    performance: PerformanceBundle


class RepresentativeSourceError(ValueError):
    """An intact historical choice whose original result cannot be verified."""

    def __init__(self, record: RepresentativeRecord, reason: str) -> None:
        super().__init__(reason)
        self.record = record


def _error(exc: Exception) -> dict[str, Any]:
    return {
        "success": False,
        "code": "representative_conflict"
        if isinstance(exc, RepresentativeConflictError)
        else "representative_unavailable",
        "error": str(exc),
    }


class RepresentativeService:
    def __init__(self, workspace: WorkspacePaths, store: SQLiteStore) -> None:
        self.records = RepresentativeStore(store)
        self.sources = RepresentativeSources(workspace, store)

    def capture(self, run_id: str, scope: str) -> RepresentativeSnapshot:
        return self._capture(run_id, scope)[0]

    def _capture(
        self, run_id: str, scope: str
    ) -> tuple[RepresentativeSnapshot, PerformanceBundle]:
        run_id = TypeAdapter(SavedRunId).validate_python(run_id)
        scope = TypeAdapter(SavedScope).validate_python(scope)
        run, manifest, performance, digest, observations, observations_digest = (
            self.sources.load(run_id)
        )
        selected = performance.scopes.get(scope)
        if selected is None:
            raise ValueError(f"Unknown metric scope: {scope}")
        if selected.kind != "backtest":
            raise ValueError(
                "A fold summary is not a complete tradable path; "
                "select a scalar fold scope"
            )
        verify_research_scopes(manifest, performance, observations)
        strategy = manifest.config_copy()["strategy"]
        program = manifest.strategy_execution.program
        if manifest.strategy_execution.kind == "generated" and (
            not strategy.get("revision_id")
            or program is None
            or program.source_hash != strategy["source_hash"]
        ):
            raise ValueError(
                "Generated representative requires its fixed revision/source identity"
            )
        snapshot = RepresentativeSnapshot.model_validate(
            {
                "run_id": run_id,
                "scope": scope,
                "strategy_id": run["strategy_id"],
                "revision_id": run.get("revision_id"),
                "source_hash": strategy["source_hash"],
                "performance_sha256": digest,
                "observations_sha256": observations_digest,
                "manifest": manifest,
                "effective_parameters": effective_scope_parameters(manifest, selected),
                "performance": selected.model_copy(
                    update={
                        "metadata": query_metadata(
                            selected.metadata, run_id=run_id, scope_id=scope
                        )
                    }
                ),
                "definitions": performance.definitions,
            }
        )
        return snapshot, performance

    def create(
        self, group_id: str, run_id: str, scope: str, reason: str = ""
    ) -> dict[str, Any]:
        try:
            group_id = TypeAdapter(ResearchGroup).validate_python(group_id)
            reason = TypeAdapter(SelectionReason).validate_python(reason)
            record = self.records.save(
                self.capture(run_id, scope), group_id=group_id, reason=reason
            )
            return {
                "success": True,
                "representative_id": record.representative_id,
                "version": record.version,
            }
        except (ValueError, OSError, sqlite3.Error) as exc:
            return _error(exc)

    def revise(
        self,
        representative_id: str,
        expected_version: int,
        run_id: str,
        scope: str,
        reason: str = "",
    ) -> dict[str, Any]:
        try:
            representative_id = TypeAdapter(RepresentativeId).validate_python(
                representative_id
            )
            expected_version = TypeAdapter(RepresentativeVersion).validate_python(
                expected_version
            )
            reason = TypeAdapter(SelectionReason).validate_python(reason)
            previous = self.records.load(representative_id)
            if previous.version != expected_version:
                raise RepresentativeConflictError(
                    "Representative changed; reread before revising"
                )
            record = self.records.save(
                self.capture(run_id, scope),
                group_id=previous.group_id,
                reason=reason,
                representative_id=representative_id,
                expected_version=expected_version,
            )
            return {
                "success": True,
                "representative_id": record.representative_id,
                "version": record.version,
            }
        except (ValueError, OSError, sqlite3.Error) as exc:
            return _error(exc)

    def verified(self, representative_id: str, version: int) -> RepresentativeRecord:
        return self.resolve(representative_id, version).record

    def resolve(self, representative_id: str, version: int) -> ResolvedRepresentative:
        representative_id = TypeAdapter(RepresentativeId).validate_python(
            representative_id
        )
        version = TypeAdapter(RepresentativeVersion).validate_python(version)
        record = self.records.load(representative_id, version)
        return self._resolve(record)

    def _resolve(self, record: RepresentativeRecord) -> ResolvedRepresentative:
        try:
            snapshot, performance = self._capture(
                record.snapshot.run_id, record.snapshot.scope
            )
            if snapshot != record.snapshot:
                raise ValueError(
                    "Representative source differs from its saved snapshot"
                )
            return ResolvedRepresentative(record, performance)
        except (ValueError, OSError, sqlite3.Error) as exc:
            raise RepresentativeSourceError(record, str(exc)) from exc

    def _availability(self, record: RepresentativeRecord) -> dict[str, Any]:
        try:
            self._resolve(record)
            return {"source_available": True, "unavailable_reason": None}
        except (ValueError, OSError, sqlite3.Error) as exc:
            return {"source_available": False, "unavailable_reason": str(exc)}

    def get(
        self,
        representative_id: str | None = None,
        version: int | None = None,
        *,
        group_id: str | None = None,
        run_id: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> dict[str, Any]:
        try:
            page = PageRequest(offset=offset, limit=limit)
            if representative_id is not None:
                representative_id = TypeAdapter(RepresentativeId).validate_python(
                    representative_id
                )
                version = TypeAdapter(RepresentativeVersion | None).validate_python(
                    version
                )
                if (
                    group_id is not None
                    or run_id is not None
                    or offset != 0
                    or limit != 20
                ):
                    raise ValueError(
                        "Filters and pagination apply only to the representative list"
                    )
                record = self.records.load(representative_id, version)
                return {
                    "success": True,
                    "representative": record.model_dump(mode="json"),
                    **self._availability(record),
                }
            if version is not None:
                raise ValueError("version requires representative_id")
            group_id = TypeAdapter(ResearchGroup | None).validate_python(group_id)
            run_id = TypeAdapter(SavedRunId | None).validate_python(run_id)
            records, total = self.records.page(
                group_id=group_id, run_id=run_id, offset=page.offset, limit=page.limit
            )
            return {
                "success": True,
                "total": total,
                "offset": offset,
                "limit": limit,
                "next_offset": offset + len(records)
                if offset + len(records) < total
                else None,
                "items": [
                    {
                        "representative_id": r.representative_id,
                        "version": r.version,
                        "group_id": r.group_id,
                        "strategy_id": r.snapshot.strategy_id,
                        "run_id": r.snapshot.run_id,
                        "scope": r.snapshot.scope,
                        "revision_id": r.snapshot.revision_id,
                        **self._availability(r),
                    }
                    for r in records
                ],
            }
        except (ValueError, OSError, sqlite3.Error) as exc:
            return _error(exc)
