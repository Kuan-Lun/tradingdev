"""Backtest service tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import pytest

from tradingdev.app.backtest_service import BacktestService
from tradingdev.app.data_service import DataService, LoadedDataset
from tradingdev.domain.backtest.schemas import BacktestConfig, ParallelConfig
from tradingdev.domain.strategies.base import BaseStrategy

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch

    from tradingdev.domain.backtest.base_engine import BaseBacktestEngine
    from tradingdev.domain.strategies.loader import StrategyLoader


def _frame(rows: int = 24) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=rows,
                freq="h",
                tz="UTC",
            ),
            "close": [100.0 + i for i in range(rows)],
        }
    )


def _raw_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "strategy": {
            "id": "fixture",
            "parameters": {},
        },
        "backtest": {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "init_cash": 10_000.0,
            "fees": 0.0,
            "slippage": 0.0,
        },
    }
    config.update(overrides)
    return config


class _DataServiceStub:
    def __init__(self, dataset: LoadedDataset) -> None:
        self.dataset = dataset
        self.loads: list[BacktestConfig] = []

    def load(
        self,
        raw_config: dict[str, Any],
        backtest_config: BacktestConfig,
    ) -> LoadedDataset:
        self.loads.append(backtest_config)
        return self.dataset


class _SignalStrategy(BaseStrategy):
    def __init__(self) -> None:
        self.fit_lengths: list[int] = []

    def fit(self, df: pd.DataFrame) -> None:
        self.fit_lengths.append(len(df))

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["signal"] = [1 if i % 2 == 0 else -1 for i in range(len(result))]
        return result

    def get_parameters(self) -> dict[str, Any]:
        return {"name": "signal_stub"}


class _StrategyLoaderStub:
    def __init__(self, strategy: BaseStrategy) -> None:
        self.strategy = strategy
        self.engine: BaseBacktestEngine | None = None
        self.parallel_config: ParallelConfig | None = None

    def create_from_config(
        self,
        raw_config: dict[str, Any],
        engine: BaseBacktestEngine,
        parallel_config: ParallelConfig | None = None,
    ) -> BaseStrategy:
        self.engine = engine
        self.parallel_config = parallel_config
        return self.strategy


def _service(
    tmp_path: Path,
    *,
    rows: int = 24,
) -> tuple[BacktestService, _DataServiceStub, _StrategyLoaderStub, _SignalStrategy]:
    dataset = LoadedDataset(
        frame=_frame(rows),
        processed_path=tmp_path / "processed.parquet",
        dataset_id="dataset-fixture",
    )
    data_service = _DataServiceStub(dataset)
    strategy = _SignalStrategy()
    strategy_loader = _StrategyLoaderStub(strategy)
    service = BacktestService(
        data_service=cast("DataService", data_service),
        strategy_loader=cast("StrategyLoader", strategy_loader),
    )
    return service, data_service, strategy_loader, strategy


def test_run_raw_config_simple_backtest_serializes_metrics(tmp_path: Path) -> None:
    service, data_service, strategy_loader, _strategy = _service(tmp_path)

    run = service.run_raw_config(_raw_config())

    assert run.mode == "simple"
    assert run.pipeline.mode == "simple"
    assert run.pipeline.backtest_result is not None
    assert run.processed_path == tmp_path / "processed.parquet"
    assert run.dataset_id == "dataset-fixture"
    assert data_service.loads[0].symbol == "BTC/USDT"
    assert strategy_loader.engine is not None
    assert strategy_loader.parallel_config == ParallelConfig()
    assert "total_return" in run.metrics
    assert "daily_pnl_mean" not in run.metrics


def test_run_raw_config_walk_forward_uses_validation_section(tmp_path: Path) -> None:
    service, _data_service, _strategy_loader, strategy = _service(tmp_path, rows=40)
    raw_config = _raw_config(
        validation={
            "n_splits": 2,
            "train_ratio": 0.5,
            "target_metric": "total_return",
        }
    )

    run = service.run_raw_config(raw_config, walk_forward=True)

    assert run.mode == "walk_forward"
    assert run.pipeline.mode == "walk_forward"
    assert len(run.pipeline.fold_results) == 2
    assert run.metrics["n_folds"] == 2
    assert strategy.fit_lengths == [10, 10]


def test_run_config_rejects_walk_forward_config_without_flag(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "walk_forward.yaml"
    config_path.write_text(
        """\
strategy:
  id: fixture
backtest:
  symbol: BTC/USDT
  timeframe: 1h
  start_date: "2024-01-01"
  end_date: "2024-01-02"
  init_cash: 10000.0
validation:
  n_splits: 1
""",
        encoding="utf-8",
    )
    service, _data_service, _strategy_loader, _strategy = _service(tmp_path)

    with pytest.raises(ValueError, match="start_walk_forward"):
        service.run_config(config_path)
