"""Saved ML evidence reaches immutable reports without acquiring new market data."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from html import unescape
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from tests.adapters.test_evaluation_series_store import _bundle
from tests.app.test_comparison_report import _fixture
from tests.app.test_report_service import _payload, _save

import tradingdev.domain.ml.report as report_module
from tradingdev.adapters.storage.evaluation_series import EvaluationSeriesStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.ml_service import MLService
from tradingdev.app.ranking_service import RankingSelectionError, RankingService
from tradingdev.app.report_ml import load_ml_report
from tradingdev.app.report_service import ReportService
from tradingdev.domain.ml.leaderboard import (
    EvaluationContext,
    LeaderboardEntry,
    ProducerSource,
    resolve_method,
)
from tradingdev.domain.ml.registry import LeaderboardStore, ModelArtifactStore

if TYPE_CHECKING:
    from tradingdev.domain.ml.evaluation.series import EvaluationSeriesBundle


@pytest.fixture
def context(tmp_path: Path) -> tuple[WorkspacePaths, SQLiteStore, ReportService]:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    return workspace, store, ReportService(workspace=workspace, store=store)


def _seed(workspace: WorkspacePaths) -> tuple[LeaderboardStore, LeaderboardEntry]:
    store = LeaderboardStore(workspace.ml_leaderboard)
    entry = LeaderboardEntry(
        entry_id="first",
        label="已保存候選",
        context=EvaluationContext(
            "BTC/USDT", "1d", "2024-01-01", "2024-12-31", 1, window=20
        ),
        metrics={"ic_mean": 0.2, "beta": 0.0},
        n_observations=6,
        model_id="model-a",
        dataset_id="dataset-a",
        parameters={"window": 99},
        signal_fingerprint="saved-signal",
    )
    store.record(entry)
    store.record(replace(entry, entry_id="failed", metrics={"ic_mean": None}))
    return store, entry


def _read(result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    assert result["success"], result
    path = Path(result["path"])
    payload, manifest = (
        _payload(path),
        json.loads((path.parent / "manifest.json").read_text(encoding="utf-8")),
    )
    introduction = re.search(
        r"<header>.*?</h1><p>(.*?)</p>", path.read_text(encoding="utf-8"), re.DOTALL
    )
    assert introduction is not None
    assert unescape(introduction[1]) == result["navigation_instructions"]
    assert payload["navigation_instructions"] == manifest["navigation_instructions"]
    assert payload["navigation_instructions"] == result["navigation_instructions"]
    return payload, manifest


def _series_bundle(entry: LeaderboardEntry) -> EvaluationSeriesBundle:
    return _bundle(
        entry_id=entry.entry_id,
        dataset_id=entry.dataset_id,
        signal_fingerprint=entry.signal_fingerprint,
        producer_source=entry.producer_source.value,
        method_family=entry.family,
        training_status="unknown",
        training_dataset_id=entry.training_dataset_id,
        settings={
            "horizon": entry.context.horizon,
            "window": entry.context.window,
            "n_quantiles": entry.context.n_quantiles,
            "candidate": {
                "model_id": entry.model_id,
                "method_name": entry.method_name,
                "parameters": entry.parameters,
            },
        },
    )


@pytest.mark.parametrize("training_dataset_id", ["", "training-dataset"])
def test_report_publication_protects_its_series_reference(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    training_dataset_id: str,
) -> None:
    workspace, _, service = context
    board, entry = _seed(workspace)
    entry = replace(entry, training_dataset_id=training_dataset_id)
    series = EvaluationSeriesStore(workspace.ml_evaluation_series)
    artifact_id = series.save(_series_bundle(entry))
    board.record(replace(entry, series_artifact_id=artifact_id))

    result = service.generate_report(ml_entry_ids=["first", "failed"])

    payload, _ = _read(result)
    scope = payload["ml"]["scopes"][0]
    assert scope["parameters"] == entry.parameters
    assert scope["metadata"]["training_dataset_id"] == training_dataset_id
    board.record(replace(entry, series_artifact_id="b" * 64))
    removed = MLService(workspace=workspace).cleanup_evaluation_series()

    assert artifact_id not in removed["removed"]
    assert series.load(artifact_id).manifest.artifact_id == artifact_id


def test_report_rejects_a_cited_series_from_another_scope(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    workspace, _, service = context
    series = EvaluationSeriesStore(workspace.ml_evaluation_series)
    artifact_id = series.save(_bundle(entry_id="other"))
    board, entry = _seed(workspace)
    board.record(replace(entry, series_artifact_id=artifact_id))

    result = service.generate_report(ml_entry_ids=["first", "failed"])

    assert not result["success"]
    assert result["code"] == "report_generation_failed"
    assert "does not match report scope" in result["error"]


@pytest.mark.parametrize("mismatch", ["parameters", "training_dataset_id"])
def test_report_rejects_a_single_series_candidate_binding_mismatch(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService], mismatch: str
) -> None:
    workspace, database, service = context
    board, entry = _seed(workspace)
    series_entry = (
        replace(entry, parameters={"window": 1234})
        if mismatch == "parameters"
        else replace(entry, training_dataset_id="different-training-dataset")
    )
    series = EvaluationSeriesStore(workspace.ml_evaluation_series)
    artifact_id = series.save(_series_bundle(series_entry))
    # The content-bound artifact is intact; only its link to this scope is wrong.
    assert series.load(artifact_id).manifest.artifact_id == artifact_id
    board.record(replace(entry, series_artifact_id=artifact_id))

    result = service.generate_report(ml_entry_ids=["first"])

    assert result["success"] is False
    assert result["code"] == "report_generation_failed"
    assert result["error"].endswith("does not match report scope: " + mismatch)
    assert not list((workspace.root / "reports").glob("**/*"))
    assert database.list_artifacts() == []
    assert series.prune(protected=frozenset()) == [artifact_id]


@pytest.mark.parametrize("has_run", [False, True])
@pytest.mark.parametrize("show_ml", [False, True])
def test_navigation_matches_selected_ml_and_run_views(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    has_run: bool,
    show_ml: bool,
) -> None:
    workspace, _, service = context
    _seed(workspace)
    sections = ["ml"] if show_ml else []
    if has_run:
        _save(context)
        sections.append("settings")
    result = service.generate_report(
        ["run"] if has_run else None, ml_entry_ids=["first"], sections=sections
    )
    _read(result)
    navigation = result["navigation_instructions"]
    assert ("下載完整 ML 證據 JSON" in navigation) == show_ml
    assert ("未選取 ML 顯示章節" in navigation) != show_ml
    assert ("切換結果" in navigation) == has_run
    assert "離線 HTML" in navigation


def test_ml_only_report_saves_complete_selected_evidence_and_reuses_it(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    workspace, database, service = context
    leaderboard, entry = _seed(workspace)
    before = leaderboard.path.read_bytes()
    result = service.generate_report(ml_entry_ids=["first", "failed"])
    payload, manifest = _read(result)
    assert result["run_ids"] == []
    assert result["ml_entry_ids"] == ["first", "failed"]
    assert result["scope_count"] == 2
    assert payload["sections"] == ["ml"]
    assert payload["ml"] == manifest["ml"]
    assert manifest["ml_entry_ids"] == ["first", "failed"]
    block = payload["ml"]
    assert block["ranking"]["ranked_scope_ids"] == ["first"]
    assert block["ranking"]["skipped"][0]["scope_id"] == "failed"
    assert block["scopes"][1]["values"]["ic_mean"] is None
    assert block["scopes"][0]["values"]["beta"] == 0.0
    assert block["scopes"][0]["metadata"]["window"] == 20
    assert block["scopes"][0]["parameters"] == {"window": 99}
    assert block["scopes"][0]["metadata"]["signal_fingerprint"] == "saved-signal"
    records = [
        entry.to_record(),
        replace(entry, entry_id="failed", metrics={"ic_mean": None}).to_record(),
    ]
    expected = (
        json.dumps(
            records,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    assert block["source_sha256"] == hashlib.sha256(expected.encode()).hexdigest()
    artifact = database.get_artifact(result["artifact_id"])
    assert artifact is not None and artifact["run_id"] is None
    assert artifact["metadata"]["ml_entry_ids"] == ["first", "failed"]
    assert service.generate_report(ml_entry_ids=["first", "failed"]) == result
    assert leaderboard.path.read_bytes() == before


def test_ml_snapshot_survives_rerecording_deletion_and_service_restart(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    workspace, database, service = context
    leaderboard, entry = _seed(workspace)
    first = service.generate_report(ml_entry_ids=["first"])
    original_html = Path(first["path"]).read_bytes()
    leaderboard.record(replace(entry, metrics={"ic_mean": 0.9}))
    second = service.generate_report(ml_entry_ids=["first"])
    assert first["report_id"] != second["report_id"]
    old, _ = _read(first)
    new, _ = _read(second)
    assert old["ml"]["scopes"][0]["values"]["ic_mean"] == 0.2
    assert new["ml"]["scopes"][0]["values"]["ic_mean"] == 0.9
    leaderboard.path.unlink()
    restarted = ReportService(workspace=workspace, store=database)
    download = restarted.get_report_download(
        first["report_id"], expected_sha256=first["sha256"]
    )
    assert download["success"] and download["content"] == original_html
    missing = restarted.generate_report(ml_entry_ids=["first"])
    assert missing["code"] == "ml_entry_not_found"


def test_report_freezes_recorded_sources_without_registry_inference(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, _, service = context
    leaderboard, seed = _seed(workspace)
    entries = [
        replace(
            seed,
            entry_id=source.value,
            producer_source=source,
            training_dataset_id="saved-training-dataset",
        )
        for source in (
            ProducerSource.BACKEND_MODEL,
            ProducerSource.CALLER_ARRAY,
            ProducerSource.UNKNOWN,
        )
    ]
    records = [entry.to_record() for entry in entries]
    records[-1].pop("producer_source")
    leaderboard.path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    def no_current_model(*_args: object, **_kwargs: object) -> None:
        pytest.fail("A report must not infer saved provenance from today's model")

    monkeypatch.setattr(ModelArtifactStore, "load", no_current_model)
    before = leaderboard.path.read_bytes()
    result = service.generate_report(ml_entry_ids=[entry.entry_id for entry in entries])
    payload, manifest = _read(result)
    sources = [
        scope["metadata"]["producer_source"] for scope in payload["ml"]["scopes"]
    ]
    assert sources == ["backend_model", "caller_array", "unknown"]
    assert payload["ml"] == manifest["ml"]
    assert all(
        scope["metadata"]["training_dataset_id"] == "saved-training-dataset"
        for scope in payload["ml"]["scopes"]
    )
    assert leaderboard.path.read_bytes() == before
    leaderboard.record(replace(entries[1], metrics={"ic_mean": 0.9}))
    fixed, _ = _read(result)
    assert fixed["ml"] == payload["ml"]


def test_ml_and_backtest_are_separate_evidence(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    workspace, _, service = context
    _seed(workspace)
    _save(context)
    result = service.generate_report(
        ["run"], ml_entry_ids=["first"], sections=["provenance", "ml"]
    )
    payload, manifest = _read(result)
    assert result["run_ids"] == ["run"] and result["scope_count"] == 2
    assert payload["runs"][0]["scopes"][0]["kind"] == "backtest"
    assert payload["ml"]["scopes"][0]["kind"] == "ml_signal"
    assert manifest["available_scopes"] == {"run": ["full"]}
    assert "各自獨立" in result["navigation_instructions"]


def test_unselected_ml_section_still_embeds_evidence(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    workspace, _, service = context
    _seed(workspace)
    result = service.generate_report(ml_entry_ids=["first"], sections=[])
    payload, manifest = _read(result)
    assert payload["sections"] == [] and payload["ml"]["scopes"]
    assert "ml" in manifest["omitted_sections"]
    assert "未選取 ML" in result["navigation_instructions"]


@pytest.mark.parametrize("successful_run", [True, False])
def test_fixed_comparison_can_include_independent_ml_evidence(
    tmp_path: Path, successful_run: bool
) -> None:
    fixture = _fixture(
        tmp_path,
        statuses=("succeeded", "failed") if successful_run else ("failed", "not_run"),
    )
    _seed(fixture.workspace)
    result = fixture.service.generate_report(
        comparison_job_id="comparison", ml_entry_ids=["first"], sections=["ml"]
    )
    payload, manifest = _read(result)
    assert result["comparison_job_id"] == "comparison"
    assert result["scope_count"] == 1 + int(successful_run)
    assert len(payload["fixed_comparison"]["candidates"]) == 2
    assert manifest["ml"] == payload["ml"]
    assert payload["ml"]["scopes"][0]["metadata"]["model_id"] == "model-a"


@pytest.mark.parametrize(
    ("options", "code"),
    [
        ({"ml_entry_ids": []}, "invalid_report_ml_selection"),
        ({"ml_entry_ids": ["first", "first"]}, "invalid_report_ml_selection"),
        ({"ml_entry_ids": [str(i) for i in range(51)]}, "invalid_report_ml_selection"),
        ({"ml_entry_ids": "first"}, "invalid_report_ml_selection"),
        ({"ml_entry_ids": [""]}, "invalid_report_ml_selection"),
        ({"ml_entry_ids": [1]}, "invalid_report_ml_selection"),
        ({"ml_entry_ids": ["absent"]}, "ml_entry_not_found"),
        ({"ml_allow_parameter_axes": True}, "invalid_report_ml_selection"),
        ({"ml_ranking_metric": "sharpe"}, "invalid_report_ml_selection"),
        (
            {"ml_entry_ids": ["first"], "ml_ranking_metric": "beta"},
            "report_ml_evidence_invalid",
        ),
        (
            {"ml_entry_ids": ["first"], "ml_allow_parameter_axes": 1},
            "invalid_report_ml_selection",
        ),
    ],
)
def test_invalid_ml_selection_does_not_publish_partial_reports(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    options: dict[str, Any],
    code: str,
) -> None:
    workspace, _, service = context
    _seed(workspace)
    result = service.generate_report(**options)
    assert result["success"] is False and result["code"] == code
    assert not list((workspace.root / "reports").glob("**/*"))


def test_surface_requires_explicit_opt_in_and_keeps_the_failed_coordinate(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    workspace, _, service = context
    leaderboard, entry = _seed(workspace)
    leaderboard.record(
        replace(
            entry,
            entry_id="failed",
            context=replace(entry.context, window=30),
            metrics={"ic_mean": None},
        )
    )
    assert service.generate_report(ml_entry_ids=["first", "failed"])["code"] == (
        "report_ml_evidence_invalid"
    )
    result = service.generate_report(
        ml_entry_ids=["first", "failed"], ml_allow_parameter_axes=True
    )
    payload, _ = _read(result)
    assert payload["ml"]["parameter_axes"]["window"] == [20, 30]
    assert any("evaluation setting" in text for text in payload["ml"]["caveats"])


@pytest.mark.parametrize("corruption", ["malformed", "duplicate", "unknown_metric"])
def test_invalid_saved_evidence_is_not_silently_skipped(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService], corruption: str
) -> None:
    workspace, _, service = context
    leaderboard, entry = _seed(workspace)
    if corruption == "malformed":
        leaderboard.path.write_text("{broken", encoding="utf-8")
    elif corruption == "duplicate":
        with leaderboard.path.open("a") as stream:
            stream.write(json.dumps(entry.to_record()) + "\n")
    else:
        leaderboard.record(replace(entry, metrics={"ic_mean": 0.2, "invented": 1.0}))
    result = service.generate_report(ml_entry_ids=["first"])
    assert result["code"] == "report_ml_evidence_invalid"
    assert not list((workspace.root / "reports").glob("**/*"))


def test_no_ml_request_report_bytes_are_pinned(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    # Re-pinned at template 24 when the benchmark section was added; the ML
    # renderer must still be absent from a request that never asks for it.
    _, store, service = context
    _save(context)
    with store.connection() as connection:
        connection.execute("update runs set created_at=?", ("2026-01-01T00:00:00Z",))
    result = service.generate_report(["run"], sections=[])
    payload, manifest = _read(result)
    assert "ml" not in payload and "ml" not in manifest
    assert "ml" not in manifest["omitted_sections"]
    path = Path(result["path"]).parent / "manifest.json"
    # One comparison so a deliberate re-baseline shows every digest at once.
    assert (
        result["report_id"],
        result["sha256"],
        hashlib.sha256(path.read_bytes()).hexdigest(),
    ) == (
        "4ff6c967ef3cf6b9cdbbb9c61a0dccd5dc8def8bb42948cb61a971ef3820a44d",
        "b768b19434280d02a7657d2142de50dadc289aae70926fd7a75dbb13471677f9",
        "8727f7b052fa975f2588e7e13d0aa8bbbaa4bd4d68119e07a83003b2a5951ce4",
    )


def _seed_ranking_set(workspace: WorkspacePaths) -> list[str]:
    """Three saved candidates: two rankable, one unavailable, mixed methods."""
    board = LeaderboardStore(workspace.ml_leaderboard)
    base = LeaderboardEntry(
        entry_id="rule-a",
        label="rule a",
        context=EvaluationContext(
            "BTC/USDT", "1d", "2024-01-01", "2024-12-31", 1, window=20
        ),
        metrics={"ic_mean": 0.4, "beta": 1.0},
        n_observations=6,
        dataset_id="dataset-a",
        parameters={"threshold": 3},
        signal_fingerprint="signal-a",
        method_family="rule",
        method_name="threshold-3",
        producer_source=ProducerSource.CALLER_ARRAY,
    )
    entries = [
        base,
        replace(
            base,
            entry_id="model-b",
            label="model b",
            model_id="model-b",
            method_family="ml_model",
            method_name="",
            parameters={"depth": 2},
            context=replace(base.context, window=30),
            metrics={"ic_mean": 0.2, "beta": 0.5},
            signal_fingerprint="signal-b",
        ),
        replace(
            base,
            entry_id="missing-c",
            label="missing c",
            metrics={"ic_mean": None, "beta": 0.0},
            signal_fingerprint="signal-c",
        ),
    ]
    for entry in entries:
        board.record(entry)
    return [entry.entry_id for entry in entries]


def test_report_and_mcp_share_the_full_candidate_ranking(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    workspace, _, _ = context
    entry_ids = _seed_ranking_set(workspace)
    ml = MLService(workspace=workspace)
    ranking = RankingService(workspace=workspace, ml_service=ml)

    mcp = ml.rank_signal_candidates("ic_mean", allow_parameter_axes=True, limit=100)
    report = load_ml_report(
        workspace,
        entry_ids,
        metric="ic_mean",
        allow_parameter_axes=True,
    )
    page = ranking.rank_candidates(
        "signal.ic_mean", allow_parameter_axes=True, offset=0, limit=1
    )

    assert mcp["n_candidates"] == 3
    assert mcp["n_ranked"] == 2 and mcp["n_skipped"] == 1
    assert report["ranking"]["ranked_scope_ids"] == [
        row["entry_id"] for row in mcp["ranked"]
    ]
    assert [
        (row["scope_id"], row["reason"]) for row in report["ranking"]["skipped"]
    ] == [(row["entry_id"], row["reason"]) for row in mcp["skipped"]]
    assert report["caveats"] == mcp["caveats"]
    assert report["parameter_axes"] == mcp["parameter_axes"]
    values = {row["entry_id"]: row["value"] for row in mcp["ranked"]}
    assert [
        next(
            scope["values"]["ic_mean"]
            for scope in report["scopes"]
            if scope["scope_id"] == scope_id
        )
        for scope_id in report["ranking"]["ranked_scope_ids"]
    ] == [values[scope_id] for scope_id in report["ranking"]["ranked_scope_ids"]]
    # A one-row MCP page must not become the report's candidate set.
    assert page["ranked"] == mcp["ranked"][:1]
    assert len(report["scopes"]) == 3
    assert any("more than one evaluation setting" in text for text in report["caveats"])


def test_report_ranking_cannot_drift_from_the_application_result(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second ranker inside the report must not be able to disagree with MCP."""
    workspace, _, _ = context
    entry_ids = _seed_ranking_set(workspace)
    original = getattr(report_module, "rank_entries", None)
    if original is not None:

        def reversed_ranking(entries: Any, metric: str, **options: Any) -> Any:
            result = original(entries, metric, **options)
            return replace(result, ranked=tuple(reversed(result.ranked)))

        monkeypatch.setattr(
            report_module, "rank_entries", reversed_ranking, raising=False
        )
    mcp = MLService(workspace=workspace).rank_signal_candidates(
        "ic_mean", allow_parameter_axes=True, limit=100
    )
    report = load_ml_report(
        workspace, entry_ids, metric="ic_mean", allow_parameter_axes=True
    )
    assert report["ranking"]["ranked_scope_ids"] == [
        row["entry_id"] for row in mcp["ranked"]
    ]


def test_signal_entry_selection_is_validated_inside_the_service(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    ml = MLService(workspace=workspace)
    with pytest.raises(RankingSelectionError) as duplicate:
        ml.rank_signal_entries(["first", "first"], "ic_mean")
    assert duplicate.value.code == "invalid_signal_selection"


def test_signal_assembly_refuses_duplicate_saved_entries(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    _, entry = _seed(workspace)
    ml = MLService(workspace=workspace)
    with pytest.raises(ValueError, match="unique entry IDs"):
        ml._assemble_signal_ranking(
            (entry, entry),
            "ic_mean",
            allow_parameter_axes=False,
            allow_mixed=False,
        )


def test_leaderboard_page_rejects_invalid_offset_before_reading(
    tmp_path: Path,
) -> None:
    ml = MLService(workspace=WorkspacePaths(tmp_path / "workspace"))
    with pytest.raises(ValueError, match="offset must be a non-negative integer"):
        ml.leaderboard_page("ic_mean", offset=-1)


def test_protect_evaluation_series_counts_distinct_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protected: list[str] = []
    monkeypatch.setattr(
        EvaluationSeriesStore,
        "protect",
        lambda _self, artifact_id: protected.append(artifact_id),
    )
    ml = MLService(workspace=WorkspacePaths(tmp_path / "workspace"))
    assert ml.protect_evaluation_series(["a", "a", "b"]) == 2
    assert protected == ["a", "a", "b"]


def test_panel_symbols_requires_a_nonempty_name() -> None:
    with pytest.raises(ValueError, match="at least one symbol is required"):
        MLService._panel_symbols([])


def test_panel_symbols_rejects_a_trimmed_duplicate() -> None:
    with pytest.raises(ValueError, match="duplicate symbols"):
        MLService._panel_symbols(["BTC/USDT", " BTC/USDT "])


def test_panel_symbols_needs_two_distinct_symbols() -> None:
    with pytest.raises(ValueError, match="at least two distinct symbols"):
        MLService._panel_symbols(["BTC/USDT"])


def _record_arguments(
    identity: Any, *, dataset_id: str, manifest: dict[str, Any] | None
) -> dict[str, Any]:
    return {
        "signal": [1.0],
        "forward_returns": [1.0],
        "pearson_ic": None,
        "spearman_ic": None,
        "evaluation_id": "evaluation",
        "fingerprint": "fingerprint",
        "manifest": manifest,
        "n_observations": 1,
        "horizon": 1,
        "window": 2,
        "n_quantiles": 5,
        "model_id": None,
        "dataset_id": dataset_id,
        "identity": identity,
        "producer_source": ProducerSource.CALLER_ARRAY,
    }


def test_recording_evaluation_requires_a_dataset_id(tmp_path: Path) -> None:
    ml = MLService(workspace=WorkspacePaths(tmp_path / "workspace"))
    identity = resolve_method(
        model_id=None,
        method_family="rule",
        method_name="threshold",
        method_parameters={},
    )
    with pytest.raises(ValueError, match="requires dataset_id"):
        ml._record_evaluation(
            {}, **_record_arguments(identity, dataset_id="", manifest=None)
        )


def test_recording_evaluation_requires_the_dataset_manifest(tmp_path: Path) -> None:
    ml = MLService(workspace=WorkspacePaths(tmp_path / "workspace"))
    identity = resolve_method(
        model_id=None,
        method_family="rule",
        method_name="threshold",
        method_parameters={},
    )
    with pytest.raises(ValueError, match="requires the dataset manifest"):
        ml._record_evaluation(
            {},
            **_record_arguments(identity, dataset_id="dataset", manifest=None),
        )


def test_evaluate_signal_rejects_a_window_below_two(tmp_path: Path) -> None:
    ml = MLService(workspace=WorkspacePaths(tmp_path / "workspace"))
    values = [0.1] * 10
    with pytest.raises(ValueError, match="window must be at least 2"):
        ml.evaluate_signal(values, values, window=1, n_quantiles=5)


def test_evaluate_signal_rejects_a_nonpositive_horizon(tmp_path: Path) -> None:
    ml = MLService(workspace=WorkspacePaths(tmp_path / "workspace"))
    values = [0.1] * 10
    with pytest.raises(ValueError, match="horizon must be at least 1"):
        ml.evaluate_signal(values, values, horizon=0, window=2, n_quantiles=5)
