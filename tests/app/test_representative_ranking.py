"""Representative versions reuse ranking, original search evidence and bounds."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from tests.representative_fixtures import publish_representative_run

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.ranking_service import RankingService
from tradingdev.app.representative_service import RepresentativeService
from tradingdev.domain.representatives import RepresentativeRef

if TYPE_CHECKING:
    from pathlib import Path


def test_manual_trial_choice_preserves_versions_and_original_search_evidence(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    publish_representative_run(workspace, store, "opt", optimization=True, bundled=True)
    representatives = RepresentativeService(workspace, store)
    identity = representatives.create("group", "opt", "trial/0/train")[
        "representative_id"
    ]
    assert representatives.revise(identity, 1, "opt", "test")["version"] == 2
    refs = [RepresentativeRef(representative_id=identity, version=v) for v in (1, 2)]
    result = RankingService(workspace=workspace, store=store).rank_run_scopes(
        "run.total_pnl",
        representatives=refs,
    )
    assert result["success"] is True
    assert [
        (r["scope"], r["value"], r["representative"]["version"])
        for r in result["ranked"]
    ] == [("test", 3.0, 2), ("trial/0/train", 1.0, 1)]
    for row in result["ranked"]:
        assert row["parameters"]["nested"] == {"default": 7}
        assert row["selection_evidence"]["summary"]["planned"] == 2
        assert (
            row["selection_evidence"]["reference"]["performance_sha256"]
            == row["representative"]["performance_sha256"]
        )
    assert result["ranked"][0]["parameters"]["window"] == 20
    assert result["ranked"][1]["parameters"]["window"] == 10
    assert result["ranked"][1]["selection_evidence"]["scope_role"] == "other_train"
    assert result["ranked"][0]["selection_evidence"]["scope_role"] == "selected_test"


def test_unknown_versions_stay_skipped_and_selection_capacity_is_explicit(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    publish_representative_run(workspace, store, "A")
    selected = RepresentativeService(workspace, store).create("group", "A", "full")
    ref = RepresentativeRef(representative_id=selected["representative_id"], version=1)
    unknown = RepresentativeRef(representative_id="f" * 32, version=1)
    ranking = RankingService(workspace=workspace, store=store)
    result = ranking.rank_candidates(
        "run.sharpe_ratio", representatives=[ref, unknown], limit=1
    )
    assert result["n_ranked"] == 1 and result["n_skipped"] == 1
    assert result["ranked"][0]["run_id"] == "A"
    assert result["skipped"][0]["representative"] == unknown.model_dump()
    assert "not found" in result["skipped"][0]["reason"]
    assert result["skipped"][0]["source_available"] is False
    for refs in ([], [ref] * 101):
        with pytest.raises(ValueError):
            ranking.rank_run_scopes("run.sharpe_ratio", representatives=refs)
