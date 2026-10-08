"""Offline report presentation, source integrity and write-once publication."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from tradingdev.adapters.storage.execution_manifests import ExecutionManifestStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths, sha256_file
from tradingdev.adapters.storage.performance import PerformanceStore
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.report_service import ReportService
from tradingdev.domain.execution import ExecutionManifest
from tradingdev.domain.performance.artifacts import (
    FoldStats,
    MetricDefinitionSnapshot,
    ObservationsBundle,
    PerformanceArtifacts,
    PerformanceBundle,
    PerformanceScope,
    ScopeObservations,
)
from tradingdev.domain.performance.catalog import METRIC_CATALOG
from tradingdev.domain.strategies.execution import StrategyExecution


@pytest.fixture
def context(tmp_path: Path) -> tuple[WorkspacePaths, SQLiteStore, ReportService]:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    return workspace, store, ReportService(workspace=workspace, store=store)


def _scope(**changes: Any) -> PerformanceScope:
    return PerformanceScope.model_validate(
        {
            "mode": "signal",
            "values": {"total_return": 0.25, "win_rate": None},
            "parameters": {"slow": 26},
            "metadata": {
                "unavailable": {"win_rate": "no_closed_trades"},
                "execution_context": {
                    "symbol": "BTC/USDT",
                    "fees": 0.0004,
                    "slippage": 0.0,
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-31",
                },
                "settings": {"initial_cash": 100.0},
                "providers": {"fixture": "1"},
            },
            **changes,
        }
    )


def _save(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    run_id: str = "run",
    *,
    scopes: dict[str, PerformanceScope] | None = None,
    default_scope: str = "full",
    selected_train: str | None = None,
    modern: bool = False,
    volume: bool = False,
    timestamps: tuple[str, ...] | None = (
        "2024-01-01T00:00:00Z",
        "2024-01-02T00:00:00Z",
        "2024-01-03T00:00:00Z",
    ),
) -> None:
    workspace, store, _ = context
    scopes = scopes or {"full": _scope()}
    manifest = None
    if modern:
        manifest = ExecutionManifest.create(
            kind="backtest",
            config={
                "strategy": {"id": "fixture", "parameters": {"slow": 26}},
                "backtest": {
                    "symbol": "BTC/USDT",
                    "timeframe": "1d",
                    "init_cash": 100.0,
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-31",
                },
            },
            strategy_execution=StrategyExecution(
                kind="generated", constructor_kwargs={"slow": 26, "signal": 9}
            ),
        )
    digest = manifest.manifest_hash if manifest else None
    store.create_run(
        run_id=run_id,
        job_id=run_id,
        strategy_id="fixture",
        revision_id="old-revision",
        manifest_hash=digest,
        artifact_dir=workspace.runs / run_id,
        metrics=scopes[default_scope].model_dump(mode="json")["values"],
        dataset_id="saved-market-hash",
    )
    if manifest is not None:
        path = ExecutionManifestStore(workspace).publish(run_id, manifest)
        store.create_artifact(
            artifact_id=f"{run_id}:execution_manifest",
            run_id=run_id,
            artifact_type="execution_manifest",
            path=path,
            sha256=sha256_file(path),
        )
    definitions = {
        key: MetricDefinitionSnapshot.model_validate(asdict(METRIC_CATALOG[key]))
        for scope in scopes.values()
        for key in scope.values
    }
    observations = ScopeObservations(
        init_cash=None if volume else 100.0,
        equity_curve=[100.0, 90.0, 125.0],
        returns=None if volume else [0.0, -0.1, 0.3888889],
        timestamps=list(timestamps) if timestamps is not None else None,
        trades=[
            {
                "entry_idx": 0,
                "exit_idx": 2,
                "status": "open",
                "direction": -1,
                "entry_price": 50.0,
                "exit_price": 45.0,
                "net_pnl": 25.0,
                "note": "</script><img src=x onerror=alert(1)>",
            }
        ],
    )
    PerformanceStore(workspace, store).publish(
        PerformanceArtifacts(
            performance=PerformanceBundle(
                run_id=run_id,
                manifest_hash=digest,
                default_scope=default_scope,
                scopes=scopes,
                definitions=definitions,
                selected_train_scope=selected_train,
            ),
            observations=ObservationsBundle(
                run_id=run_id,
                manifest_hash=digest,
                scopes={
                    key: observations
                    for key, scope in scopes.items()
                    if scope.kind == "backtest"
                },
            ),
        )
    )


def _payload(path: Path) -> dict[str, Any]:
    text = path.read_text()
    content = text.split("<script id='report-data' type='application/json'>", 1)[1]
    parsed: dict[str, Any] = json.loads(content.split("</script>", 1)[0])
    return parsed


def test_report_uses_immutable_saved_evidence_and_deterministic_retry(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, store, service = context
    _save(context, modern=True)
    before = {p: p.read_bytes() for p in workspace.runs.rglob("*") if p.is_file()}
    monkeypatch.setattr("pickle.load", lambda *_: pytest.fail("must not unpickle"))
    monkeypatch.setattr(
        "tradingdev.app.run_lineage.read_strategy_snapshot",
        lambda *_args, **_kwargs: pytest.fail("must not read current strategy"),
    )
    first = service.generate_report(["run"])
    assert first["success"], first
    path = Path(first["path"])
    original = path.read_bytes()
    assert service.generate_report(["run"]) == first
    assert path.read_bytes() == original
    assert {
        p: p.read_bytes() for p in workspace.runs.rglob("*") if p.is_file()
    } == before
    assert first["sha256"] == hashlib.sha256(original).hexdigest()
    registered = store.get_artifact(first["artifact_id"])
    assert registered is not None
    assert registered["artifact_type"] == "research_report_html"
    payload = _payload(path)
    scope = payload["runs"][0]["scopes"][0]
    assert scope["parameters"] == {"slow": 26, "signal": 9}
    assert scope["values"]["total_return"] == 0.25
    assert scope["observations"]["trades"][0]["exit_price"] == 45.0
    text = path.read_text()
    assert "25.0000%" in text and "no_closed_trades" in text
    assert "mark UTC" in text and "mark price" in text
    assert "不是實際平倉" in text and "暖機" in text
    assert "&lt;/script&gt;&lt;img" in text
    assert "</script><img" not in text and "\\u003c/script\\u003e" in text
    assert "https://" not in text and "<svg" in text
    assert not list((workspace.root / "reports").rglob(".report-*"))
    assert not list((workspace.root / "reports").glob("*.publishing"))


def test_optional_sections_order_and_commentary_are_saved_and_escaped(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    _, _, service = context
    _save(context)
    notes = [{"title": "風險", "text": "<script>alert(1)</script>\n不是計算結果"}]
    response = service.generate_report(["run"], sections=[], commentary=notes)
    assert response["success"], response
    path = Path(response["path"])
    text = path.read_text()
    assert "LLM 評語（非計算結果）" in text
    assert "data-section='metrics'" not in text and "<svg" not in text
    assert "<script>alert(1)</script>" not in text
    manifest = json.loads((path.parent / "manifest.json").read_text())
    assert manifest["sections"] == [] and manifest["commentary"] == notes
    assert manifest["available_scopes"] == {"run": ["full"]}
    assert service.generate_report(["run"], [], notes) == response
    reordered = service.generate_report(["run"], ["trades", "metrics"])
    rendered = Path(reordered["path"]).read_text()
    assert rendered.index("data-section='trades'") < rendered.index(
        "data-section='metrics'"
    )
    assert reordered["report_id"] != response["report_id"]
    catalogue = service.get_report_sections()
    catalogue["templates"]["standard"].clear()
    assert service.get_report_sections()["templates"]["standard"]


def test_catalog_titles_and_recipes_match_the_rendered_document(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    _, _, service = context
    _save(context)
    catalog = service.get_report_sections()
    section_ids = [section["id"] for section in catalog["sections"]]
    assert catalog["templates"]["standard"] == section_ids
    for selected in catalog["templates"].values():
        report = service.generate_report(["run"], sections=selected)
        assert report["success"], report
        content = Path(report["path"]).read_text()
        for section in catalog["sections"]:
            heading = (
                f"<section data-section='{section['id']}'><h2>{section['title']}</h2>"
            )
            assert (heading in content) == (section["id"] in selected)
    catalog["sections"][0]["title"] = "呼叫端修改"
    assert service.get_report_sections()["sections"][0]["title"] == "總覽"


@pytest.mark.parametrize("sections", [["unknown"], ["metrics", "metrics"]])
def test_invalid_section_selection_does_not_publish(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    sections: list[str],
) -> None:
    workspace, _, service = context
    assert (
        service.generate_report(["run"], sections)["code"] == "invalid_report_sections"
    )
    assert not (workspace.root / "reports").exists()


@pytest.mark.parametrize(
    "notes",
    [
        [{"html": "<p>raw</p>"}],
        [{"title": "", "text": "x"}],
        [{"title": "x", "text": "y"}] * 21,
        [{"title": "x", "text": "y" * 20000}],
    ],
)
def test_invalid_commentary_is_rejected(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    notes: list[dict[str, str]],
) -> None:
    assert (
        context[2].generate_report(["run"], commentary=notes)["code"]
        == "invalid_report_commentary"
    )


def test_all_walk_forward_and_optimization_scopes_preserved_with_summary_semantics(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    _, _, service = context
    summary = PerformanceScope(
        kind="fold_summary",
        mode="signal",
        split="test",
        values={
            "total_return": FoldStats(
                mean=0.25, std=0.0, min=0.25, max=0.25, valid_count=2
            )
        },
        metadata={"aggregation": "fold_descriptive"},
    )
    _save(
        context,
        "wf",
        default_scope="test_summary",
        scopes={
            "fold/0/train": _scope(split="train", fold_index=0),
            "fold/0/test": _scope(split="test", fold_index=0),
            "test_summary": summary,
        },
    )
    _save(
        context,
        "opt",
        default_scope="test",
        selected_train="trial/1/train",
        scopes={
            "trial/0/train": _scope(
                split="train", trial_index=0, parameters={"slow": 23}
            ),
            "trial/1/train": _scope(
                split="train", trial_index=1, parameters={"slow": 26}
            ),
            "test": _scope(split="test", parameters={"slow": 26}),
        },
    )
    response = service.generate_report(["wf", "opt"])
    assert response["success"], response
    assert response["scope_count"] == 6
    payload = _payload(Path(response["path"]))
    assert payload["runs"][0]["scopes"][-1]["observations"] is None
    assert payload["runs"][1]["selected_train_scope"] == "trial/1/train"
    assert payload["default_scope_comparison"]["comparable"] is False
    text = Path(response["path"]).read_text()
    assert "fold_descriptive" in text and "不是串接資金" in text
    assert "的觀測日期重疊" in text
    assert "scope-1-2" in text and "trial/0/train" in text


@pytest.mark.parametrize(
    "artifact", ["performance.json", "observations.json", "manifest.json"]
)
def test_corrupt_saved_source_fails_without_writing_reports(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    artifact: str,
) -> None:
    workspace, _, service = context
    _save(context, modern=True)
    path = workspace.runs / "run" / artifact
    path.write_bytes(b"corrupt")
    response = service.generate_report(["run"])
    assert not response["success"]
    assert response["code"] in {
        "performance_artifact_invalid",
        "execution_manifest_invalid",
    }
    assert not (workspace.root / "reports").exists()
    assert path.read_bytes() == b"corrupt"


def test_existing_report_tampering_is_not_overwritten(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    workspace, _, service = context
    _save(context)
    first = service.generate_report(["run"])
    path = Path(first["path"])
    path.write_text("tampered")
    response = service.generate_report(["run"])
    assert response["code"] == "report_artifact_invalid"
    assert path.read_text() == "tampered"
    assert not list((workspace.root / "reports").glob("*.publishing"))


def test_report_path_symlink_is_rejected(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    tmp_path: Path,
) -> None:
    workspace, _, service = context
    _save(context)
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace.root / "reports").symlink_to(outside, target_is_directory=True)
    response = service.generate_report(["run"])
    assert response["code"] == "report_path_invalid"
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("failure", [OSError("write failed"), TimeoutError("deadline")])
def test_publication_failure_cleans_new_files_and_registrations(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    workspace, store, service = context
    _save(context)
    before = store.list_artifacts()
    original = os.link
    calls = 0

    def interrupted(source: str | Path, target: str | Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise failure
        original(source, target)

    monkeypatch.setattr("tradingdev.adapters.reporting.storage.os.link", interrupted)
    response = service.generate_report(["run"])
    assert not response["success"] and response["code"] == "report_generation_failed"
    assert store.list_artifacts() == before
    assert list((workspace.root / "reports").iterdir()) == []


def test_volume_without_timestamps_uses_index_and_amount_drawdown(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    _, _, service = context
    _save(context, volume=True, timestamps=None, scopes={"full": _scope(mode="volume")})
    response = service.generate_report(["run"], ["equity"])
    assert response["success"], response
    text = Path(response["path"]).read_text()
    assert "quote amount" in text and "bar 0" in text and "bar 2" in text


def test_registered_missing_report_is_not_silently_rebuilt(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    _, _, service = context
    _save(context)
    first = service.generate_report(["run"])
    path = Path(first["path"])
    path.unlink()
    failed = service.generate_report(["run"])
    assert failed["code"] == "report_artifact_invalid"
    assert not path.exists()


def test_report_download_returns_the_verified_snapshot_without_a_second_read(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, service = context
    _save(context)
    report = service.generate_report(["run"])
    path = Path(report["path"])
    expected = path.read_bytes()
    original_read = Path.read_bytes
    reads = []

    def changed_after_read(target: Path) -> bytes:
        content = original_read(target)
        if target == path:
            reads.append(target)
            target.write_bytes(b"changed after the verified snapshot was read")
        return content

    monkeypatch.setattr(Path, "read_bytes", changed_after_read)
    download = service.get_report_download(
        report["report_id"], expected_sha256=report["sha256"]
    )

    assert download == {
        "success": True,
        "report_id": report["report_id"],
        "sha256": hashlib.sha256(expected).hexdigest(),
        "content": expected,
    }
    assert reads == [path]
    assert original_read(path) != download["content"]


@pytest.mark.parametrize("change", ["modified", "deleted", "unreadable"])
def test_report_download_rejects_changed_or_unreadable_html(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    _, _, service = context
    _save(context)
    report = service.generate_report(["run"])
    path = Path(report["path"])
    if change == "modified":
        path.write_bytes(b"modified")
    elif change == "deleted":
        path.unlink()
    else:

        def unreadable(target: Path) -> bytes:
            raise PermissionError("Report is unreadable")

        monkeypatch.setattr(Path, "read_bytes", unreadable)

    download = service.get_report_download(
        report["report_id"], expected_sha256=report["sha256"]
    )

    assert not download["success"]
    assert download["code"] == (
        "report_read_failed" if change == "unreadable" else "report_artifact_invalid"
    )
    assert "content" not in download
    if change == "modified":
        assert path.read_bytes() == b"modified"
    elif change == "deleted":
        assert not path.exists()


@pytest.mark.parametrize("field", ["artifact_type", "path", "sha256", "deleted"])
def test_report_download_checks_registration_and_original_expected_hash(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService], field: str
) -> None:
    _, store, service = context
    _save(context)
    report = service.generate_report(["run"])
    with store.connect() as connection:
        if field == "deleted":
            connection.execute(
                "delete from artifacts where artifact_id = ?", (report["artifact_id"],)
            )
        else:
            value = "different"
            if field == "sha256":
                # Even replacing both bytes and the database hash cannot change
                # the report snapshot requested by this caller.
                Path(report["path"]).write_bytes(b"changed")
                value = hashlib.sha256(b"changed").hexdigest()
            connection.execute(
                f"update artifacts set {field} = ? where artifact_id = ?",
                (value, report["artifact_id"]),
            )
    download = service.get_report_download(
        report["report_id"], expected_sha256=report["sha256"]
    )

    assert not download["success"]
    assert download["code"] == (
        "report_not_found" if field == "deleted" else "report_artifact_invalid"
    )
    assert "content" not in download


def test_report_download_rejects_replaced_symlink_even_for_identical_bytes(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService], tmp_path: Path
) -> None:
    _, _, service = context
    _save(context)
    report = service.generate_report(["run"])
    path = Path(report["path"])
    outside = tmp_path / "outside.html"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)

    download = service.get_report_download(
        report["report_id"], expected_sha256=report["sha256"]
    )

    assert download["code"] == "report_path_invalid"
    assert "content" not in download


@pytest.mark.parametrize(
    ("report_id", "expected_sha256", "code"),
    [
        ("../outside", "a" * 64, "report_path_invalid"),
        ("a" * 64, "not-a-hash", "report_artifact_invalid"),
        ("a" * 64, "b" * 64, "report_not_found"),
    ],
)
def test_invalid_report_download_identity_returns_error_without_content(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    report_id: str,
    expected_sha256: str,
    code: str,
) -> None:
    download = context[2].get_report_download(
        report_id, expected_sha256=expected_sha256
    )

    assert not download["success"] and download["code"] == code
    assert "content" not in download


def test_busy_report_does_not_remove_another_publication_marker(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    workspace, _, service = context
    _save(context)
    first = service.generate_report(["run"])
    marker = workspace.root / "reports" / f".{first['report_id']}.publishing"
    marker.mkdir()
    failed = service.generate_report(["run"])
    assert failed["code"] == "report_busy" and marker.is_dir()
    marker.rmdir()


def test_database_failure_rolls_back_pair_and_cleans_files(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    workspace, store, service = context
    _save(context)
    before = store.list_artifacts()
    with store.connect() as connection:
        connection.execute(
            "create trigger reject_report before insert on artifacts "
            "when NEW.artifact_type = 'research_report_manifest' "
            "begin select raise(ABORT, 'database failure'); end"
        )
    failed = service.generate_report(["run"])
    assert failed["code"] == "report_generation_failed"
    assert store.list_artifacts() == before
    assert list((workspace.root / "reports").iterdir()) == []


def test_summary_only_still_checks_observation_bundle_integrity(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    workspace, _, service = context
    summary = PerformanceScope(
        kind="fold_summary",
        mode="signal",
        split="test",
        values={
            "total_return": FoldStats(
                mean=None, std=None, min=None, max=None, valid_count=0
            )
        },
        metadata={"aggregation": "fold_descriptive"},
    )
    _save(context, scopes={"summary": summary}, default_scope="summary")
    (workspace.runs / "run" / "observations.json").write_text("corrupt")
    failed = service.generate_report(
        ["run"], sections=[], commentary=[{"title": "t", "text": "x"}]
    )
    assert failed["code"] == "performance_artifact_invalid"
    assert not (workspace.root / "reports").exists()


def test_utc_labels_convert_offsets_without_changing_original_timestamps(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    _, _, service = context
    stamps = (
        "2024-01-01T08:00:00+08:00",
        "2024-01-02T08:00:00+08:00",
        "2024-01-03T08:00:00+08:00",
    )
    _save(context, timestamps=stamps)
    result = service.generate_report(["run"])
    path = Path(result["path"])
    text = path.read_text()
    assert ">2024-01-01 00:00 UTC</text>" in text
    assert ">2024-01-03 00:00 UTC</text>" in text
    assert ">2024-01-03T08:00:00</text>" not in text
    assert "<td>2024-01-01T00:00:00+00:00</td>" in text
    assert _payload(path)["runs"][0]["scopes"][0]["observations"]["timestamps"] == list(
        stamps
    )


@pytest.mark.parametrize("scope_count", [1, 35])
def test_report_artifact_read_count_does_not_grow_with_scope_count(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    monkeypatch: pytest.MonkeyPatch,
    scope_count: int,
) -> None:
    workspace, _, service = context
    _save(
        context,
        modern=True,
        default_scope="part/0",
        scopes={f"part/{index}": _scope() for index in range(scope_count)},
    )
    original = Path.read_bytes
    reads: dict[str, int] = {}

    def counted(path: Path) -> bytes:
        if path.parent == workspace.runs / "run":
            reads[path.name] = reads.get(path.name, 0) + 1
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", counted)
    response = service.generate_report(["run"])
    assert response["success"], response
    assert response["scope_count"] == scope_count
    # The observation loader also validates its paired performance file.
    assert reads == {"performance.json": 2, "observations.json": 1, "manifest.json": 1}


@pytest.mark.parametrize(
    "corruption", ["sha256", "path", "artifact_type", "run_id", "bytes"]
)
def test_registered_manifest_identity_and_byte_hash_are_checked_before_reporting(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    corruption: str,
) -> None:
    workspace, store, service = context
    _save(context, modern=True)
    path = workspace.runs / "run" / "manifest.json"
    if corruption == "bytes":
        # Same manifest/canonical execution digest, different registered file bytes.
        path.write_text(json.dumps(json.loads(path.read_text()), indent=2))
        run = store.get_run("run")
        assert run is not None
        ExecutionManifestStore(workspace).load(
            "run", expected_hash=run["manifest_hash"]
        )
    else:
        bad_values = {
            "sha256": "0" * 64,
            "path": str(workspace.root / "other.json"),
            "artifact_type": "unrelated",
            "run_id": "other",
        }
        with store.connect() as connection:
            connection.execute(
                f"update artifacts set {corruption} = ? where artifact_id = ?",
                (bad_values[corruption], "run:execution_manifest"),
            )
    response = service.generate_report(["run"])
    assert response["code"] == "execution_manifest_invalid", response
    assert not (workspace.root / "reports").exists()
    from tradingdev.app.trade_history_service import TradeHistoryService

    history = TradeHistoryService(workspace=workspace, store=store).get_run_trades(
        "run"
    )
    assert history["code"] == "execution_manifest_invalid"
