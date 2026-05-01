"""Strategy loader schema contract tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tradingdev.domain.strategies.loader import StrategyLoader

if TYPE_CHECKING:
    from pathlib import Path


def test_loader_rejects_deprecated_strategy_name_field(tmp_path: Path) -> None:
    loader = StrategyLoader(workspace_root=tmp_path / "workspace")

    with pytest.raises(ValueError, match="strategy.id is required"):
        loader.load_class(
            {
                "name": "legacy_strategy",
                "class_name": "LegacyStrategy",
                "source_path": "workspace/generated_strategies/legacy.py",
            }
        )


def test_loader_rejects_deprecated_strategy_class_and_file_fields(
    tmp_path: Path,
) -> None:
    loader = StrategyLoader(workspace_root=tmp_path / "workspace")

    with pytest.raises(ValueError, match="strategy.class_name is required"):
        loader.load_class(
            {
                "id": "legacy_strategy",
                "class": "LegacyStrategy",
                "file": "workspace/generated_strategies/legacy.py",
            }
        )
