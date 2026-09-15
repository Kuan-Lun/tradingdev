"""Independent generated-strategy semantics check, executed as a subprocess."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.strategy_service import StrategyService
from tradingdev.domain.strategies.loader import StrategyLoader
from tradingdev.shared.utils.config import load_config


def verify(root: Path) -> None:
    """Verify runnable state, configurable periods and independent expected signals."""
    workspace = WorkspacePaths(root / "workspace")
    service = StrategyService(workspace)
    spec = service.resolve_executable("codex_sma_integration")
    assert spec.status.value == "runnable"
    assert Path(spec.source_path).is_relative_to(workspace.generated_strategies)
    assert Path(spec.config_path).is_relative_to(workspace.configs)
    raw_config = load_config(Path(spec.config_path))
    parameters = raw_config["strategy"]["parameters"]
    assert parameters["fast_period"] == 5
    assert parameters["slow_period"] == 20

    close = pd.Series([*range(1, 41), *range(40, 0, -1), *([1.0] * 30)], dtype=float)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(close), freq="h"),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 100.0,
        }
    )
    before = frame.copy(deep=True)
    for fast_period, slow_period in ((5, 20), (3, 8)):
        config = deepcopy(raw_config)
        config["strategy"]["parameters"].update(
            fast_period=fast_period, slow_period=slow_period
        )
        strategy = StrategyLoader(workspace_root=workspace.root).create_from_config(
            config, engine=None
        )
        result = strategy.generate_signals(frame)
        fast = close.rolling(fast_period).mean()
        slow = close.rolling(slow_period).mean()
        expected = pd.Series(0, index=frame.index, name="signal")
        expected.loc[fast > slow] = 1
        expected.loc[fast < slow] = -1
        pd.testing.assert_series_equal(result["signal"], expected, check_dtype=False)
        pd.testing.assert_frame_equal(frame, before)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    (root / "verifier.ready").write_text("ready", encoding="utf-8")
    verify(root)
