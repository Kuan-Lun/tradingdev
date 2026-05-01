"""Storage adapter tests."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore

if TYPE_CHECKING:
    from pathlib import Path


def test_workspace_paths_create_expected_directories(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")

    workspace.ensure()

    assert workspace.root.exists()
    assert workspace.runs.exists()
    assert workspace.raw_data.exists()
    assert workspace.processed_data.exists()
    assert workspace.generated_strategies.exists()
    assert workspace.configs.exists()


def test_sqlite_store_auto_initializes_metadata_schema(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)

    assert store.db_path.exists()
    with store.connect() as conn:
        tables = {
            str(row["name"])
            for row in conn.execute(
                "select name from sqlite_master where type = 'table'"
            ).fetchall()
        }
        job_columns = {
            str(row["name"])
            for row in conn.execute("pragma table_info(jobs)").fetchall()
        }
        job_column_info = {
            str(row["name"]): row
            for row in conn.execute("pragma table_info(jobs)").fetchall()
        }
        run_columns = {
            str(row["name"])
            for row in conn.execute("pragma table_info(runs)").fetchall()
        }

    assert {"jobs", "runs", "artifacts", "events"}.issubset(tables)
    assert {"created_at", "started_at", "ended_at"}.issubset(job_columns)
    for column in (
        "strategy_name",
        "symbol",
        "timeframe",
        "start_date",
        "end_date",
        "config_path",
    ):
        assert int(job_column_info[column]["notnull"]) == 0
    assert {"source_hash", "random_seed"}.issubset(run_columns)


def test_sqlite_store_accepts_generic_job_payload(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)

    store.upsert_job(
        {
            "job_id": "job_generic",
            "job_type": "feature_request",
            "status": "queued",
            "created_at": "2024-01-01T00:00:00+00:00",
            "result_path": "workspace/runs/job_generic/result.json",
            "request_id": "feature-1",
        }
    )

    job = store.get_job("job_generic")

    assert job is not None
    assert job["job_type"] == "feature_request"
    assert job["request_id"] == "feature-1"
    assert job.get("strategy_name") is None


def test_sqlite_store_relaxes_existing_backtest_job_columns(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    workspace.ensure()
    db_path = workspace.root / "tradingdev.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            create table jobs (
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
            insert into jobs (
                job_id, job_type, status, strategy_name, symbol, timeframe,
                start_date, end_date, config_path, created_at, payload
            ) values (
                'job_old', 'backtest', 'queued', 'fixture', 'BTC/USDT', '1h',
                '2024-01-01', '2024-01-02', 'fixture.yaml',
                '2024-01-01T00:00:00+00:00', '{"job_id": "job_old"}'
            );
        """)

    store = SQLiteStore(workspace)

    with store.connect() as conn:
        info = {
            str(row["name"]): row
            for row in conn.execute("pragma table_info(jobs)").fetchall()
        }
    assert int(info["strategy_name"]["notnull"]) == 0
    assert store.get_job("job_old") == {"job_id": "job_old"}


def test_sqlite_store_persists_job_run_and_artifact_lookup(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    store.upsert_job(
        {
            "job_id": "job_a",
            "job_type": "backtest",
            "status": "done",
            "strategy_name": "fixture",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "config_path": "fixture.yaml",
            "created_at": "2024-01-01T00:00:00+00:00",
            "started_at": "2024-01-01T00:00:01+00:00",
            "ended_at": "2024-01-01T00:00:02+00:00",
            "result_path": "workspace/runs/job_a/result.json",
        }
    )
    run_dir = workspace.runs / "job_a"
    run_dir.mkdir(parents=True)
    artifact_path = run_dir / "result.json"
    artifact_path.write_text("{}", encoding="utf-8")
    store.create_run(
        run_id="run_a",
        job_id="job_a",
        strategy_id="fixture",
        artifact_dir=run_dir,
        metrics={"total_return": 0.1},
        source_hash="source-a",
        random_seed=42,
        dataset_id="dataset-a",
    )
    store.create_artifact(
        artifact_id="run_a:result_json",
        run_id="run_a",
        artifact_type="result_json",
        path=artifact_path,
        metadata={"job_id": "job_a"},
    )

    job = store.get_job("job_a")
    run = store.get_run("run_a")
    artifact = store.get_artifact("run_a:result_json")

    assert job is not None
    assert run is not None
    assert artifact is not None
    assert job["ended_at"] == "2024-01-01T00:00:02+00:00"
    assert run["dataset_id"] == "dataset-a"
    assert run["source_hash"] == "source-a"
    assert run["random_seed"] == 42
    assert artifact["metadata"]["job_id"] == "job_a"
