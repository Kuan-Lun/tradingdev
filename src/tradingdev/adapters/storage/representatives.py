"""Transactional representative versions and verification of their saved sources."""

from __future__ import annotations

import hashlib
from contextlib import closing
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from tradingdev.adapters.storage.execution_manifests import ExecutionManifestStore
from tradingdev.adapters.storage.filesystem import now_iso, sha256_file
from tradingdev.adapters.storage.performance import PerformanceStore
from tradingdev.domain.representatives import (
    MAX_REPRESENTATIVE_BYTES,
    RepresentativeConflictError,
    RepresentativeRecord,
)
from tradingdev.domain.research_evidence_values import canonical

if TYPE_CHECKING:
    from collections.abc import Iterator
    from sqlite3 import Connection, Row

    from tradingdev.adapters.storage.filesystem import WorkspacePaths
    from tradingdev.adapters.storage.sqlite import SQLiteStore
    from tradingdev.domain.execution import ExecutionManifest
    from tradingdev.domain.performance.artifacts import (
        ObservationsBundle,
        PerformanceBundle,
    )
    from tradingdev.domain.representatives import RepresentativeSnapshot


def _encode(record: RepresentativeRecord) -> bytes:
    content = canonical(record.model_dump(mode="json"))
    if len(content) > MAX_REPRESENTATIVE_BYTES:
        raise ValueError("Representative snapshot exceeds 131072 UTF-8 bytes")
    return content


class RepresentativeStore:
    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    @staticmethod
    def _check_heads(conn: Connection) -> None:
        """A damaged current pointer cannot hide or roll back immutable history."""
        invalid_head = conn.execute(
            "select 1 from representative_heads h left join representative_versions c "
            "on c.representative_id=h.representative_id and c.version=h.version "
            "where c.version is null "
            "or h.version != (select max(v.version) from representative_versions v "
            "where v.representative_id=h.representative_id) "
            "or 1 != (select min(v.version) from representative_versions v "
            "where v.representative_id=h.representative_id) "
            "or h.version != (select count(*) from representative_versions v "
            "where v.representative_id=h.representative_id) "
            "or exists (select 1 from representative_versions v "
            "where v.representative_id=h.representative_id "
            "and typeof(v.version)!='integer') "
            "or json_extract(c.document,'$.group_id') is not h.group_id "
            "or json_extract(c.document,'$.snapshot.strategy_id') is not h.strategy_id "
            "limit 1"
        ).fetchone()
        orphan = conn.execute(
            "select 1 from representative_versions v left join representative_heads h "
            "using(representative_id) where h.representative_id is null limit 1"
        ).fetchone()
        if invalid_head is not None or orphan is not None:
            raise ValueError("Representative current index differs from saved history")

    @staticmethod
    def _decode(row: Row) -> RepresentativeRecord:
        content = row["document"].encode("utf-8")
        if hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError("Representative snapshot checksum differs")
        record = RepresentativeRecord.model_validate_json(content)
        if (
            record.representative_id != row["representative_id"]
            or record.version != row["version"]
            or record.group_id != row["group_id"]
            or record.snapshot.strategy_id != row["strategy_id"]
            or record.snapshot.run_id != row["run_id"]
            or record.snapshot.scope != row["scope"]
            or record.snapshot.revision_id != row["revision_id"]
        ):
            raise ValueError("Representative snapshot differs from its index")
        _encode(record)
        return record

    def load(
        self, representative_id: str, version: int | None = None
    ) -> RepresentativeRecord:
        with closing(self.store.connect()) as conn:
            conn.execute("begin")
            if version is None:
                self._check_heads(conn)
            row = conn.execute(
                "select v.*, h.group_id, h.strategy_id from representative_versions v "
                "join representative_heads h using(representative_id) "
                "where v.representative_id=? and v.version=coalesce(?,h.version)",
                (representative_id, version),
            ).fetchone()
        if row is None:
            raise ValueError("Representative version not found")
        return self._decode(row)

    def save(
        self,
        snapshot: RepresentativeSnapshot,
        *,
        group_id: str,
        reason: str,
        representative_id: str | None = None,
        expected_version: int | None = None,
    ) -> RepresentativeRecord:
        if (representative_id is None) != (expected_version is None):
            raise ValueError("Revision requires representative_id and expected_version")
        with closing(self.store.connect()) as conn, conn:
            conn.execute("begin immediate")
            self._check_heads(conn)
            previous = self.load(representative_id) if representative_id else None
            if previous is not None:
                if previous.version != expected_version:
                    raise RepresentativeConflictError(
                        "Representative changed; reread before revising"
                    )
                if (group_id, snapshot.strategy_id) != (
                    previous.group_id,
                    previous.snapshot.strategy_id,
                ):
                    raise ValueError(
                        "Revision cannot change research group or strategy"
                    )
                if (snapshot.run_id, snapshot.scope) == (
                    previous.snapshot.run_id,
                    previous.snapshot.scope,
                ):
                    raise RepresentativeConflictError(
                        "This run/scope is already the representative"
                    )
            elif conn.execute(
                "select 1 from representative_heads where group_id=? and strategy_id=?",
                (group_id, snapshot.strategy_id),
            ).fetchone():
                raise RepresentativeConflictError(
                    "This group/strategy already has a representative; "
                    "read and revise it"
                )
            now = now_iso()
            record = RepresentativeRecord(
                representative_id=representative_id or uuid4().hex,
                version=previous.version + 1 if previous else 1,
                group_id=group_id,
                created_at=previous.created_at if previous else now,
                selected_at=now,
                reason=reason,
                snapshot=snapshot,
            )
            content = _encode(record)
            conn.execute(
                "insert into representative_versions values (?,?,?,?,?,?,?)",
                (
                    record.representative_id,
                    record.version,
                    snapshot.run_id,
                    snapshot.scope,
                    snapshot.revision_id,
                    content.decode("utf-8"),
                    hashlib.sha256(content).hexdigest(),
                ),
            )
            conn.execute(
                "insert into representative_heads values (?,?,?,?) "
                "on conflict(representative_id) do update set version=excluded.version",
                (
                    record.representative_id,
                    group_id,
                    snapshot.strategy_id,
                    record.version,
                ),
            )
        return RepresentativeRecord.model_validate_json(content)

    def page(
        self, *, group_id: str | None, run_id: str | None, offset: int, limit: int
    ) -> tuple[list[RepresentativeRecord], int]:
        # A run filter includes ALL historical references for retention decisions.
        where = "where (? is null or h.group_id=?) and " + (
            "v.run_id=?" if run_id is not None else "v.version=h.version"
        )
        args: tuple[object, ...] = (group_id, group_id) + (
            (run_id,) if run_id is not None else ()
        )
        query = (
            " from representative_versions v join representative_heads h "
            "using(representative_id) " + where
        )
        with closing(self.store.connect()) as conn:
            conn.execute("begin")
            self._check_heads(conn)
            total = int(conn.execute("select count(*)" + query, args).fetchone()[0])
            rows = conn.execute(
                "select v.*, h.group_id, h.strategy_id"
                + query
                + " order by v.representative_id,v.version desc limit ? offset ?",
                (*args, limit, offset),
            ).fetchall()
        return [self._decode(row) for row in rows], total

    def revisions(self, strategy_id: str) -> Iterator[RepresentativeRecord]:
        """Retain every revision, including superseded or missing sources."""
        with closing(self.store.connect()) as conn:
            conn.execute("begin")
            self._check_heads(conn)
            rows = conn.execute(
                "select v.*, h.group_id, h.strategy_id from representative_versions v "
                "left join representative_heads h using(representative_id) "
                "order by v.representative_id,v.version"
            )
            # Verify before filtering; an index cannot prove that a damaged record
            # belongs elsewhere. Stream snapshots to avoid retaining them all.
            for row in rows:
                record = self._decode(row)
                if record.snapshot.strategy_id == strategy_id:
                    yield record


class RepresentativeSources:
    """Read only fixed workspace artifacts; never consult the strategy registry."""

    def __init__(self, workspace: WorkspacePaths, store: SQLiteStore) -> None:
        self.workspace = workspace
        self.store = store
        self.performance = PerformanceStore(workspace, store)
        self.manifests = ExecutionManifestStore(workspace)

    def load(
        self, run_id: str
    ) -> tuple[
        dict[str, Any],
        ExecutionManifest,
        PerformanceBundle,
        str,
        ObservationsBundle,
        str,
    ]:
        run = self.store.get_run(run_id)
        if run is None:
            raise ValueError(f"Unknown run: {run_id}")
        job = self.store.get_job(str(run["job_id"]))
        if (
            job is not None
            and job.get("job_type") == "optimization"
            and job.get("status") != "done"
        ):
            raise ValueError("Representative requires a completed optimization run")
        if not run.get("manifest_hash"):
            raise ValueError("Representative requires a saved execution manifest")
        manifest = self.manifests.load(run_id, expected_hash=run["manifest_hash"])
        performance, performance_hash = self.performance.load_with_hash(run_id)
        observations, observations_hash = self.performance.load_observations_with_hash(
            run_id
        )
        strategy = manifest.config.get("strategy")
        if not isinstance(strategy, dict) or any(
            run.get(key) != strategy.get(name)
            for key, name in (
                ("strategy_id", "id"),
                ("revision_id", "revision_id"),
                ("source_hash", "source_hash"),
            )
        ):
            raise ValueError("Run strategy/revision/source differs from its manifest")
        source_hash = strategy.get("source_hash")
        if not isinstance(source_hash, str):
            raise ValueError("Historical strategy source identity is unavailable")
        source = self.store.get_artifact(f"{run_id}:strategy_source")
        path = self.manifests.path(run_id).with_name("strategy.py")
        if (
            source is None
            or source.get("run_id") != run_id
            or source.get("artifact_type") != "strategy_source"
            or source.get("path") != str(path)
            or source.get("sha256") != source_hash
            or path.is_symlink()
            or sha256_file(path) != source_hash
        ):
            raise ValueError("Saved strategy source is missing or corrupt")
        return (
            run,
            manifest,
            performance,
            performance_hash,
            observations,
            observations_hash,
        )
