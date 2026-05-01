"""SQLite metadata store for jobs, runs, artifacts, and events."""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING, Any

from tradingdev.adapters.storage.filesystem import WorkspacePaths, now_iso

if TYPE_CHECKING:
    from pathlib import Path

_STORE_CACHE: dict[str, SQLiteStore] = {}


class SQLiteStore:
    """Small SQLite adapter with automatic schema initialization."""

    def __init__(self, workspace: WorkspacePaths | None = None) -> None:
        self._workspace = workspace or WorkspacePaths()
        self._workspace.ensure()
        self._db_path = self._workspace.root / "tradingdev.sqlite"
        self.initialize()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        """Create metadata tables if they do not exist."""
        with self.connect() as conn:
            conn.executescript("""
                create table if not exists jobs (
                    job_id text primary key,
                    job_type text not null,
                    status text not null,
                    strategy_name text not null,
                    symbol text not null,
                    timeframe text not null,
                    start_date text not null,
                    end_date text not null,
                    config_path text not null,
                    pid integer,
                    created_at text not null,
                    started_at text,
                    ended_at text,
                    error text,
                    payload text not null
                );

                create table if not exists runs (
                    run_id text primary key,
                    job_id text not null,
                    strategy_id text not null,
                    config_hash text,
                    source_hash text,
                    random_seed integer,
                    dataset_id text,
                    metrics text,
                    artifact_dir text not null,
                    created_at text not null,
                    foreign key(job_id) references jobs(job_id)
                );

                create table if not exists artifacts (
                    artifact_id text primary key,
                    run_id text,
                    artifact_type text not null,
                    path text not null,
                    sha256 text,
                    metadata text not null,
                    created_at text not null,
                    foreign key(run_id) references runs(run_id)
                );

                create table if not exists events (
                    event_id integer primary key autoincrement,
                    job_id text not null,
                    timestamp text not null,
                    level text not null,
                    message text not null,
                    payload text not null,
                    foreign key(job_id) references jobs(job_id)
                );
                """)
            self._ensure_column(conn, "jobs", "created_at", "text")
            self._ensure_column(conn, "jobs", "started_at", "text")
            self._ensure_column(conn, "jobs", "ended_at", "text")
            self._ensure_column(conn, "runs", "source_hash", "text")
            self._ensure_column(conn, "runs", "random_seed", "integer")

    def upsert_job(self, record: dict[str, Any]) -> None:
        """Insert or replace a job record."""
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self.connect() as conn:
            conn.execute(
                """
                insert into jobs (
                    job_id, job_type, status, strategy_name, symbol, timeframe,
                    start_date, end_date, config_path, pid, created_at, started_at,
                    ended_at,
                    error, payload
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(job_id) do update set
                    job_type=excluded.job_type,
                    status=excluded.status,
                    strategy_name=excluded.strategy_name,
                    symbol=excluded.symbol,
                    timeframe=excluded.timeframe,
                    start_date=excluded.start_date,
                    end_date=excluded.end_date,
                    config_path=excluded.config_path,
                    pid=excluded.pid,
                    created_at=excluded.created_at,
                    started_at=excluded.started_at,
                    ended_at=excluded.ended_at,
                    error=excluded.error,
                    payload=excluded.payload
                """,
                (
                    record["job_id"],
                    record.get("job_type", "backtest"),
                    record["status"],
                    record["strategy_name"],
                    record["symbol"],
                    record["timeframe"],
                    record["start_date"],
                    record["end_date"],
                    record["config_path"],
                    record.get("pid"),
                    record["created_at"],
                    record.get("started_at"),
                    record.get("ended_at"),
                    record.get("error"),
                    payload,
                ),
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Return a job payload by id."""
        with self.connect() as conn:
            row = conn.execute(
                "select payload from jobs where job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(str(row["payload"]))
        return payload if isinstance(payload, dict) else None

    def list_jobs(self) -> list[dict[str, Any]]:
        """Return all job payloads newest-first."""
        with self.connect() as conn:
            rows = conn.execute(
                "select payload from jobs order by created_at desc",
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            payload = json.loads(str(row["payload"]))
            if isinstance(payload, dict):
                result.append(payload)
        return result

    def create_run(
        self,
        *,
        run_id: str,
        job_id: str,
        strategy_id: str,
        artifact_dir: Path,
        metrics: dict[str, Any],
        config_hash: str | None = None,
        source_hash: str | None = None,
        random_seed: int | None = None,
        dataset_id: str | None = None,
    ) -> None:
        """Insert or replace a completed run."""
        with self.connect() as conn:
            conn.execute(
                """
                insert into runs (
                    run_id, job_id, strategy_id, config_hash, source_hash,
                    random_seed, dataset_id, metrics, artifact_dir, created_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(run_id) do update set
                    job_id=excluded.job_id,
                    strategy_id=excluded.strategy_id,
                    config_hash=excluded.config_hash,
                    source_hash=excluded.source_hash,
                    random_seed=excluded.random_seed,
                    dataset_id=excluded.dataset_id,
                    metrics=excluded.metrics,
                    artifact_dir=excluded.artifact_dir
                """,
                (
                    run_id,
                    job_id,
                    strategy_id,
                    config_hash,
                    source_hash,
                    random_seed,
                    dataset_id,
                    json.dumps(metrics, ensure_ascii=False, sort_keys=True),
                    str(artifact_dir),
                    now_iso(),
                ),
            )

    def create_artifact(
        self,
        *,
        artifact_id: str,
        run_id: str | None,
        artifact_type: str,
        path: Path,
        sha256: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert or replace artifact metadata."""
        with self.connect() as conn:
            conn.execute(
                """
                insert into artifacts (
                    artifact_id, run_id, artifact_type, path, sha256,
                    metadata, created_at
                ) values (?, ?, ?, ?, ?, ?, ?)
                on conflict(artifact_id) do update set
                    run_id=excluded.run_id,
                    artifact_type=excluded.artifact_type,
                    path=excluded.path,
                    sha256=excluded.sha256,
                    metadata=excluded.metadata
                """,
                (
                    artifact_id,
                    run_id,
                    artifact_type,
                    str(path),
                    sha256,
                    json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                    now_iso(),
                ),
            )

    def list_artifacts(self, run_id: str | None = None) -> list[dict[str, Any]]:
        """Return artifact metadata, optionally filtered by run."""
        query = "select * from artifacts"
        params: tuple[str, ...] = ()
        if run_id is not None:
            query += " where run_id = ?"
            params = (run_id,)
        query += " order by created_at desc"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        """Return one artifact metadata row."""
        with self.connect() as conn:
            row = conn.execute(
                "select * from artifacts where artifact_id = ?",
                (artifact_id,),
            ).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def add_event(
        self,
        *,
        job_id: str,
        level: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Record a job event."""
        with self.connect() as conn:
            conn.execute(
                """
                insert into events (job_id, timestamp, level, message, payload)
                values (?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    now_iso(),
                    level,
                    message,
                    json.dumps(payload or {}, ensure_ascii=False, sort_keys=True),
                ),
            )

    def list_events(self, job_id: str) -> list[dict[str, Any]]:
        """Return events for a job in chronological order."""
        with self.connect() as conn:
            rows = conn.execute(
                "select * from events where job_id = ? order by timestamp asc",
                (job_id,),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def list_runs(self) -> list[dict[str, Any]]:
        """Return completed runs newest-first."""
        with self.connect() as conn:
            rows = conn.execute(
                "select * from runs order by created_at desc",
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return one run."""
        with self.connect() as conn:
            row = conn.execute(
                "select * from runs where run_id = ?",
                (run_id,),
            ).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        for key in ("metrics", "metadata", "payload"):
            value = result.get(key)
            if isinstance(value, str):
                result[key] = json.loads(value)
        return result

    def _ensure_column(
        self,
        conn: sqlite3.Connection,
        table: str,
        column: str,
        column_type: str,
    ) -> None:
        columns = {
            str(row["name"])
            for row in conn.execute(f"pragma table_info({table})").fetchall()
        }
        if column not in columns:
            conn.execute(f"alter table {table} add column {column} {column_type}")


def get_sqlite_store(workspace: WorkspacePaths | None = None) -> SQLiteStore:
    """Return the process-local SQLiteStore for a workspace."""
    resolved_workspace = workspace or WorkspacePaths()
    db_path = str((resolved_workspace.root / "tradingdev.sqlite").resolve())
    store = _STORE_CACHE.get(db_path)
    if store is None:
        store = SQLiteStore(resolved_workspace)
        _STORE_CACHE[db_path] = store
    return store
