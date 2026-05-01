"""Strategy loader schema contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tradingdev.domain.backtest.signal_engine import SignalBacktestEngine
from tradingdev.domain.strategies.loader import StrategyLoader


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


def test_loader_creates_bundled_strategy_without_hard_coded_aliases() -> None:
    config_path = Path(
        "src/tradingdev/domain/strategies/bundled/kd_strategy/config.yaml"
    ).resolve()
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    engine = SignalBacktestEngine(init_cash=10_000.0)

    strategy = StrategyLoader().create_from_config(raw_config, engine)

    assert strategy.__class__.__name__ == "KDStrategy"


def test_loader_rejects_legacy_bundled_strategy_id() -> None:
    config_path = Path(
        "src/tradingdev/domain/strategies/bundled/kd_strategy/config.yaml"
    ).resolve()
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_config["strategy"]["id"] = "kd_strategy"
    engine = SignalBacktestEngine(init_cash=10_000.0)

    with pytest.raises(ValueError, match="Generated strategy must live under"):
        StrategyLoader().create_from_config(raw_config, engine)
