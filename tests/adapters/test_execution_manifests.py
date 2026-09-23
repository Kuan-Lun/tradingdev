"""Immutable publication and trusted-hash reads of execution manifests."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest

from tradingdev.adapters.storage.execution_manifests import ExecutionManifestStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.domain.execution import ExecutionManifest, ManifestError

if TYPE_CHECKING:
    from pathlib import Path


def _manifest(*, fees: float = 0.0) -> ExecutionManifest:
    return ExecutionManifest.create(
        kind="backtest",
        config={
            "strategy": {"id": "fixture", "revision_id": None},
            "backtest": {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-02-01",
                "init_cash": 10000,
                "fees": fees,
            },
        },
    )


def test_publish_is_idempotent_but_cannot_replace_an_existing_spec(
    tmp_path: Path,
) -> None:
    store = ExecutionManifestStore(WorkspacePaths(tmp_path))
    manifest = _manifest()
    path = store.publish("job_1", manifest)
    original = path.read_bytes()
    assert store.publish("job_1", manifest) == path
    assert store.load("job_1", expected_hash=manifest.manifest_hash) == manifest
    with pytest.raises(ManifestError):
        store.publish("job_1", _manifest(fees=0.02))
    assert path.read_bytes() == original
    assert list(path.parent.glob(".manifest-*")) == []


def test_replacing_file_with_another_valid_manifest_breaks_trusted_hash(
    tmp_path: Path,
) -> None:
    store = ExecutionManifestStore(WorkspacePaths(tmp_path))
    original = _manifest()
    path = store.publish("job_1", original)
    replacement = _manifest(fees=0.02)
    path.write_text(replacement.model_dump_json(), encoding="utf-8")
    with pytest.raises(ManifestError):
        store.load("job_1", expected_hash=original.manifest_hash)


@pytest.mark.parametrize("content", [None, "{}", "{broken", '{"schema_version": 99}'])
def test_missing_or_invalid_manifest_never_falls_back_to_config_yaml(
    tmp_path: Path, content: str | None
) -> None:
    store = ExecutionManifestStore(WorkspacePaths(tmp_path))
    path = store.path("job_1")
    path.parent.mkdir(parents=True)
    (path.parent / "config.yaml").write_text("strategy: {}", encoding="utf-8")
    if content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(ManifestError):
        store.load("job_1", expected_hash=_manifest().manifest_hash)


def test_failed_publication_removes_only_its_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ExecutionManifestStore(WorkspacePaths(tmp_path))

    def fail_link(source: Path, target: Path) -> None:
        assert source.is_file()
        raise OSError("publication interrupted")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(OSError, match="interrupted"):
        store.publish("job_1", _manifest())
    assert list(store.path("job_1").parent.iterdir()) == []


def test_concurrent_different_publications_cannot_overwrite_each_other(
    tmp_path: Path,
) -> None:
    store = ExecutionManifestStore(WorkspacePaths(tmp_path))
    candidates = [_manifest(), _manifest(fees=0.02)]

    def publish(manifest: ExecutionManifest) -> str | None:
        try:
            store.publish("job_1", manifest)
        except ManifestError:
            return None
        return manifest.manifest_hash

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(publish, candidates))
    hashes = [result for result in results if result is not None]
    assert len(hashes) == 1
    assert store.load("job_1", expected_hash=hashes[0]).manifest_hash == hashes[0]
    assert list(store.path("job_1").parent.glob(".manifest-*")) == []


@pytest.mark.parametrize("run_id", ["../escape", "/absolute", "", "a/b"])
def test_artifact_ids_cannot_escape_workspace(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ManifestError):
        ExecutionManifestStore(WorkspacePaths(tmp_path)).publish(run_id, _manifest())


def test_manifest_store_rejects_symlinked_run_directory(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    workspace.ensure()
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace.runs / "job_1").symlink_to(outside, target_is_directory=True)
    store = ExecutionManifestStore(workspace)
    with pytest.raises(ManifestError, match="path leaves workspace"):
        store.publish("job_1", _manifest())
    assert list(outside.iterdir()) == []
