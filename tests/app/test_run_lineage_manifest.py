"""Executed source digests also protect bundled strategy artifacts."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.run_lineage import read_strategy_snapshot
from tradingdev.app.strategy_service import StrategyNotExecutableError

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("change", ["replace", "remove"])
def test_bundled_source_must_still_match_the_executed_digest(
    tmp_path: Path, change: str
) -> None:
    source = tmp_path / "bundled.py"
    original = b"# original strategy\n"
    source.write_bytes(original)
    config = {
        "strategy": {
            "id": "bundled_fixture",
            "source_path": str(source),
            "source_hash": hashlib.sha256(original).hexdigest(),
        }
    }
    workspace = WorkspacePaths(tmp_path / "workspace")
    verified = read_strategy_snapshot(config, workspace, strategy_id="bundled_fixture")
    assert verified.content == original
    assert verified.revision_id is None
    if change == "replace":
        source.write_bytes(b"# replacement strategy\n")
    else:
        source.unlink()

    with pytest.raises(StrategyNotExecutableError, match="source hash changed"):
        read_strategy_snapshot(config, workspace, strategy_id="bundled_fixture")
