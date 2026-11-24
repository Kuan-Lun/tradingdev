"""Saved representative lifecycle, source binding, retention and concurrency."""

from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from threading import Barrier
from typing import TYPE_CHECKING, Any

import pytest
from tests.app.test_strategy_cleanup_service import _draft, _setup
from tests.representative_fixtures import publish_representative_run

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.adapters.storage.strategy_references import referenced_revisions
from tradingdev.app.contracts.common import ErrorResponse
from tradingdev.app.representative_service import RepresentativeService

if TYPE_CHECKING:
    from tradingdev.domain.representatives import RepresentativeSnapshot


@pytest.fixture
def storage(
    tmp_path: Path,
) -> tuple[WorkspacePaths, SQLiteStore, RepresentativeService]:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    return workspace, store, RepresentativeService(workspace, store)


def test_selection_is_explicit_versioned_and_survives_restart(
    storage: tuple[WorkspacePaths, SQLiteStore, RepresentativeService],
) -> None:
    workspace, store, service = storage
    publish_representative_run(workspace, store, "A")
    chosen = service.create("research-1", "A", "full", "selected by user")
    assert chosen["success"] is True and chosen["version"] == 1
    identity = chosen["representative_id"]
    publish_representative_run(
        workspace, store, "B", window=20, revision="revision-2", value=9.0
    )
    restarted = RepresentativeService(workspace, SQLiteStore(workspace))
    assert restarted.get(identity)["representative"]["snapshot"]["run_id"] == "A"
    assert restarted.get(identity)["representative"]["snapshot"][
        "effective_parameters"
    ] == {"window": 10, "nested": {"default": 7}}
    assert restarted.revise(identity, 1, "B", "full")["version"] == 2
    assert restarted.get(identity)["representative"]["snapshot"]["run_id"] == "B"
    assert restarted.get(identity, 1)["representative"]["snapshot"]["run_id"] == "A"
    assert (
        restarted.revise(identity, 1, "A", "full")["code"] == "representative_conflict"
    )
    assert restarted.get(identity)["representative"]["version"] == 2
    refs = restarted.get(run_id="A")
    assert [(r["representative_id"], r["version"]) for r in refs["items"]] == [
        (identity, 1)
    ]


def test_unique_group_strategy_and_explicit_duplicate_refusals(
    storage: tuple[WorkspacePaths, SQLiteStore, RepresentativeService],
) -> None:
    workspace, store, service = storage
    publish_representative_run(workspace, store, "A")
    publish_representative_run(workspace, store, "other", strategy_id="other")
    first = service.create("group", "A", "full")
    assert service.create("group", "A", "full")["code"] == "representative_conflict"
    assert service.create("different-market", "A", "full")["success"] is True
    assert (
        service.revise(first["representative_id"], 1, "A", "full")["code"]
        == "representative_conflict"
    )
    assert (
        service.revise(first["representative_id"], 1, "other", "full")["success"]
        is False
    )
    assert service.get(first["representative_id"])["representative"]["version"] == 1


def test_concurrent_revision_has_one_winner_and_one_conflict(
    storage: tuple[WorkspacePaths, SQLiteStore, RepresentativeService],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, store, service = storage
    for name, window in (("A", 10), ("B", 20), ("C", 30)):
        publish_representative_run(workspace, store, name, window=window)
    identity = service.create("group", "A", "full")["representative_id"]
    # Both writers have read v1 before either can enter the publishing transaction.
    barrier = Barrier(2)
    capture = RepresentativeService.capture

    def synchronized_capture(
        instance: RepresentativeService, run_id: str, scope: str
    ) -> RepresentativeSnapshot:
        snapshot = capture(instance, run_id, scope)
        barrier.wait(timeout=10)
        return snapshot

    monkeypatch.setattr(RepresentativeService, "capture", synchronized_capture)

    def revise(run: str) -> dict[str, Any]:
        return RepresentativeService(workspace, SQLiteStore(workspace)).revise(
            identity, 1, run, "full"
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(revise, ("B", "C")))
    assert sorted(r["success"] for r in outcomes) == [False, True]
    assert (
        next(r for r in outcomes if not r["success"])["code"]
        == "representative_conflict"
    )
    assert service.get(identity)["representative"]["version"] == 2
    assert service.get(identity, 1)["representative"]["snapshot"]["run_id"] == "A"


@pytest.mark.parametrize(
    "damage", ["performance", "observations", "manifest", "source", "missing-run"]
)
def test_lost_or_corrupt_sources_keep_history_and_never_repoint(
    storage: tuple[WorkspacePaths, SQLiteStore, RepresentativeService],
    damage: str,
) -> None:
    workspace, store, service = storage
    publish_representative_run(workspace, store, "A")
    identity = service.create("group", "A", "full")["representative_id"]
    publish_representative_run(workspace, store, "B", value=100.0)
    if damage == "missing-run":
        with closing(store.connect()) as conn, conn:
            conn.execute("delete from runs where run_id='A'")
    else:
        filename = "strategy.py" if damage == "source" else f"{damage}.json"
        (workspace.runs / "A" / filename).write_bytes(b"corrupt")
    result = service.get(identity)
    assert result["success"] is True and result["source_available"] is False
    assert result["unavailable_reason"]
    assert result["representative"]["snapshot"]["run_id"] == "A"
    assert (
        result["representative"]["snapshot"]["performance"]["values"]["sharpe_ratio"]
        == 1.0
    )
    assert service.create("new", "A", "full")["success"] is False


@pytest.mark.parametrize(
    "column,value",
    [("revision_id", "forged"), ("strategy_id", "forged"), ("source_hash", "a" * 64)],
)
def test_forged_run_identity_is_rejected(
    storage: tuple[WorkspacePaths, SQLiteStore, RepresentativeService],
    column: str,
    value: str,
) -> None:
    workspace, store, service = storage
    publish_representative_run(workspace, store, "A")
    with closing(store.connect()) as conn, conn:
        conn.execute(f"update runs set {column}=? where run_id='A'", (value,))
    result = service.create("group", "A", "full")
    assert result["success"] is False
    assert "differs from its manifest" in result["error"]
    assert service.get()["total"] == 0


def test_forged_interior_scope_parameters_fail_even_with_updated_artifact_hash(
    storage: tuple[WorkspacePaths, SQLiteStore, RepresentativeService],
) -> None:
    workspace, store, service = storage
    publish_representative_run(workspace, store, "A")
    path = workspace.runs / "A" / "performance.json"
    data = json.loads(path.read_bytes())
    data["scopes"]["full"]["parameters"]["nested"]["default"] = 999
    content = json.dumps(data).encode()
    path.write_bytes(content)
    with closing(store.connect()) as conn, conn:
        conn.execute(
            "update artifacts set sha256=? where artifact_id='A:performance_json'",
            (hashlib.sha256(content).hexdigest(),),
        )
    result = service.create("group", "A", "full")
    assert result["success"] is False
    assert "approved execution" in result["error"]


def test_optimization_scopes_keep_complete_effective_parameters_and_missing_values(
    storage: tuple[WorkspacePaths, SQLiteStore, RepresentativeService],
) -> None:
    workspace, store, service = storage
    publish_representative_run(
        workspace, store, "opt", optimization=True, bundled=True, value=None
    )
    first = service.create("group", "opt", "trial/0/train")
    assert first["success"] is True
    identity = first["representative_id"]
    snapshot = service.get(identity)["representative"]["snapshot"]
    assert snapshot["effective_parameters"] == {"window": 10, "nested": {"default": 7}}
    assert snapshot["revision_id"] is None
    assert (
        snapshot["performance"]["metadata"]["unavailable"]["sharpe_ratio"]
        == "zero_denominator"
    )
    assert service.revise(identity, 1, "opt", "test")["version"] == 2
    assert (
        service.get(identity)["representative"]["snapshot"]["effective_parameters"][
            "window"
        ]
        == 20
    )


@pytest.mark.parametrize(
    "kwargs",
    [{"version": 1}, {"representative_id": "bogus"}, {"limit": 101}, {"offset": -1}],
)
def test_invalid_queries_are_explicit(
    storage: tuple[WorkspacePaths, SQLiteStore, RepresentativeService],
    kwargs: dict[str, Any],
) -> None:
    assert storage[2].get(**kwargs)["success"] is False


def test_unknown_scope_and_corrupt_snapshot_fail_closed(
    storage: tuple[WorkspacePaths, SQLiteStore, RepresentativeService],
) -> None:
    workspace, store, service = storage
    publish_representative_run(workspace, store, "A")
    assert service.create("group", "missing", "full")["success"] is False
    assert "Unknown metric scope" in service.create("group", "A", "unknown")["error"]
    identity = service.create("group", "A", "full")["representative_id"]
    with closing(store.connect()) as conn, conn:
        conn.execute("update representative_versions set document='{}'")
    result = service.get(identity)
    assert result["success"] is False
    historical = service.get(identity, 1)
    assert historical["success"] is False and "checksum" in historical["error"]


def test_fold_summary_is_refused_but_numbered_fold_scopes_are_selectable(
    storage: tuple[WorkspacePaths, SQLiteStore, RepresentativeService],
) -> None:
    workspace, store, service = storage
    publish_representative_run(workspace, store, "wf", walk_forward=True)
    rejected = service.create("group", "wf", "test_summary")
    assert rejected["success"] is False and "fold summary" in rejected["error"]
    selected = service.create("group", "wf", "fold/0/train")
    assert selected["success"] is True
    revised = service.revise(selected["representative_id"], 1, "wf", "fold/0/test")
    assert revised["version"] == 2


def test_retention_protects_old_and_current_revisions_after_run_sources_are_gone(
    storage: tuple[WorkspacePaths, SQLiteStore, RepresentativeService],
) -> None:
    workspace, store, service = storage
    revisions = ["11111111111141118111111111111111", "22222222222242228222222222222222"]
    for run_id, revision in zip(("A", "B"), revisions, strict=True):
        publish_representative_run(workspace, store, run_id, revision=revision)
    identity = service.create("group", "A", "full")["representative_id"]
    assert service.revise(identity, 1, "B", "full")["version"] == 2
    with closing(store.connect()) as conn, conn:
        conn.execute("delete from runs")
        conn.execute("delete from artifacts")
    for run_id in ("A", "B"):
        shutil.rmtree(workspace.runs / run_id)
    refs = referenced_revisions(workspace, store, "strategy-S")
    assert refs == {
        revision: [f"representative:{identity}@{i}"]
        for i, revision in enumerate(revisions, 1)
    }
    assert service.get(identity)["source_available"] is False


def test_pages_include_historical_references_and_snapshot_capacity_is_atomic(
    storage: tuple[WorkspacePaths, SQLiteStore, RepresentativeService],
) -> None:
    workspace, store, service = storage
    publish_representative_run(workspace, store, "A")
    for group in ("g1", "g2", "g3"):
        assert service.create(group, "A", "full")["success"] is True
    page = service.get(run_id="A", limit=2)
    assert page["total"] == 3 and page["next_offset"] == 2
    tail = service.get(run_id="A", offset=2, limit=2)
    assert len(tail["items"]) == 1 and tail["next_offset"] is None
    assert len({row["representative_id"] for row in page["items"] + tail["items"]}) == 3
    snapshot = service.capture("A", "full")
    snapshot.performance.metadata["oversize"] = "x" * 131072
    with pytest.raises(ValueError, match="131072 UTF-8 bytes"):
        service.records.save(snapshot, group_id="g4", reason="")
    assert service.get()["total"] == 3
    assert service.get(group_id="g4")["items"] == []


@pytest.mark.parametrize("damage", ["strategy-index", "missing-head"])
def test_corrupt_representative_index_blocks_actual_draft_cleanup(
    tmp_path: Path,
    damage: str,
) -> None:
    workspace, store, revisions, cleanup = _setup(tmp_path)
    chosen = _draft(revisions)
    _draft(
        revisions
    )  # Keep a different current draft, so the chosen one is eligible otherwise.
    publish_representative_run(
        workspace,
        store,
        "A",
        strategy_id="fixture",
        revision=chosen.revision_id,
        source=Path(chosen.source_path).read_bytes(),
    )
    service = RepresentativeService(workspace, store)
    assert service.create("group", "A", "full")["success"] is True
    with closing(store.connect()) as conn, conn:
        conn.execute("delete from runs")
        conn.execute("delete from artifacts")
        if damage == "missing-head":
            conn.execute("delete from representative_heads")
        else:
            conn.execute("update representative_heads set strategy_id='forged'")
    shutil.rmtree(workspace.runs / "A")
    result = cleanup.cleanup("fixture", apply=True, revision_ids=[chosen.revision_id])
    assert isinstance(result, ErrorResponse), result
    assert Path(chosen.source_path).is_file(), (
        "A corrupted index cannot erase a historical reference"
    )


def test_current_head_cannot_silently_rollback_to_an_older_saved_version(
    storage: tuple[WorkspacePaths, SQLiteStore, RepresentativeService],
) -> None:
    workspace, store, service = storage
    publish_representative_run(workspace, store, "A")
    publish_representative_run(workspace, store, "B", value=9.0)
    identity = service.create("group", "A", "full")["representative_id"]
    assert service.revise(identity, 1, "B", "full")["version"] == 2
    with closing(store.connect()) as conn, conn:
        conn.execute("update representative_heads set version=1")
    assert service.get(identity)["success"] is False
    assert service.get(group_id="group")["success"] is False
    assert service.revise(identity, 1, "A", "full")["success"] is False
    assert service.get(identity, 2)["representative"]["snapshot"]["run_id"] == "B"


@pytest.mark.parametrize("field", ["group_id", "strategy_id"])
def test_damaged_owner_index_cannot_hide_a_choice_or_allow_duplicate_creation(
    storage: tuple[WorkspacePaths, SQLiteStore, RepresentativeService],
    field: str,
) -> None:
    workspace, store, service = storage
    publish_representative_run(workspace, store, "A")
    identity = service.create("group", "A", "full")["representative_id"]
    with closing(store.connect()) as conn, conn:
        conn.execute(f"update representative_heads set {field}='forged'")
    assert service.get(group_id="group")["success"] is False
    assert service.create("group", "A", "full")["success"] is False
    assert service.get(identity)["success"] is False
    with closing(store.connect()) as conn:
        assert (
            conn.execute("select count(*) from representative_versions").fetchone()[0]
            == 1
        )


def test_missing_interior_history_blocks_current_operations_and_draft_cleanup(
    tmp_path: Path,
) -> None:
    workspace, store, revisions, cleanup = _setup(tmp_path)
    drafts = [_draft(revisions) for _ in range(4)]
    service = RepresentativeService(workspace, store)
    for name, draft in zip(("A", "B", "C"), drafts[:3], strict=True):
        publish_representative_run(
            workspace,
            store,
            name,
            strategy_id="fixture",
            revision=draft.revision_id,
            source=Path(draft.source_path).read_bytes(),
        )
    identity = service.create("group", "A", "full")["representative_id"]
    assert service.revise(identity, 1, "B", "full")["version"] == 2
    assert service.revise(identity, 2, "C", "full")["version"] == 3
    with closing(store.connect()) as conn, conn:
        conn.execute("delete from representative_versions where version=2")
        conn.execute("delete from runs")
        conn.execute("delete from artifacts")
    for name in ("A", "B", "C"):
        shutil.rmtree(workspace.runs / name)
    current = service.get(identity)
    result = cleanup.cleanup("fixture", [drafts[1].revision_id], apply=True)
    assert isinstance(result, ErrorResponse), result
    assert Path(drafts[1].source_path).is_file()
    assert current["success"] is False
    assert service.get(group_id="group")["success"] is False
    assert service.get(identity, 3)["representative"]["snapshot"]["run_id"] == "C"
