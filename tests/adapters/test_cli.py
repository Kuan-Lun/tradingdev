"""CLI adapter tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tradingdev.adapters.cli import main as cli_main
from tradingdev.app.backtest_service import BacktestRun
from tradingdev.domain.backtest.pipeline_result import PipelineResult
from tradingdev.domain.backtest.result import BacktestResult

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch


class _LoggerStub:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str, *args: Any) -> None:
        self.messages.append(message % args if args else message)


class _BacktestServiceStub:
    def __init__(self, run: BacktestRun) -> None:
        self.run = run
        self.calls: list[tuple[Path, bool]] = []

    def run_config(
        self, config_path: Path, *, walk_forward: bool = False
    ) -> BacktestRun:
        self.calls.append((config_path, walk_forward))
        return self.run


class _ArtifactServiceStub:
    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self.calls: list[dict[str, Any]] = []

    def cache_pipeline_result(
        self,
        *,
        pipeline: PipelineResult,
        config_path: Path,
        processed_path: Path,
        metrics: dict[str, Any],
        strategy_id: str,
    ) -> Path:
        self.calls.append(
            {
                "pipeline": pipeline,
                "config_path": config_path,
                "processed_path": processed_path,
                "metrics": metrics,
                "strategy_id": strategy_id,
            }
        )
        return self.cache_path


def _metrics() -> dict[str, Any]:
    return {
        "total_pnl": 100.0,
        "total_return": 0.01,
        "annual_return": 0.12,
        "max_drawdown": -0.02,
        "sharpe_ratio": 1.5,
        "win_rate": 0.6,
        "profit_factor": 1.2,
        "total_trades": 4,
        "total_volume": 1_000.0,
        "n_days": 2,
        "n_months": 1,
    }


def _simple_run(tmp_path: Path) -> BacktestRun:
    result = BacktestResult(
        metrics=_metrics(),
        equity_curve=np.array([10_000.0, 10_100.0]),
        mode="signal",
    )
    return BacktestRun(
        mode="simple",
        pipeline=PipelineResult(mode="simple", backtest_result=result),
        metrics=result.metrics,
        processed_path=tmp_path / "processed.parquet",
        dataset_id="dataset-cli",
    )


def test_cli_runs_backtest_and_caches_pipeline(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """\
strategy:
  id: cli_fixture
backtest:
  symbol: BTC/USDT
  timeframe: 1h
  start_date: "2024-01-01"
  end_date: "2024-01-02"
  init_cash: 10000.0
""",
        encoding="utf-8",
    )
    service = _BacktestServiceStub(_simple_run(tmp_path))
    artifacts = _ArtifactServiceStub(tmp_path / "cached.pkl")
    logger = _LoggerStub()
    monkeypatch.setattr(cli_main, "BacktestService", lambda: service)
    monkeypatch.setattr(cli_main, "ArtifactService", lambda: artifacts)
    monkeypatch.setattr(cli_main, "logger", logger)
    monkeypatch.setattr("sys.argv", ["tradingdev", "--config", str(config_path)])

    cli_main.main()

    assert service.calls == [(config_path, False)]
    assert artifacts.calls == [
        {
            "pipeline": service.run.pipeline,
            "config_path": config_path,
            "processed_path": tmp_path / "processed.parquet",
            "metrics": service.run.metrics,
            "strategy_id": "cli_fixture",
        }
    ]
    assert any("Backtest Performance Report" in msg for msg in logger.messages)
    assert any("Result cached" in msg for msg in logger.messages)


def test_cli_forwards_walk_forward_flag(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """\
strategy:
  id: cli_walk_forward
backtest:
  symbol: BTC/USDT
  timeframe: 1h
  start_date: "2024-01-01"
  end_date: "2024-01-02"
  init_cash: 10000.0
""",
        encoding="utf-8",
    )
    run = BacktestRun(
        mode="walk_forward",
        pipeline=PipelineResult(mode="walk_forward", fold_results=[]),
        metrics={"n_folds": 0},
        processed_path=tmp_path / "processed.parquet",
        dataset_id="dataset-cli",
    )
    service = _BacktestServiceStub(run)
    artifacts = _ArtifactServiceStub(tmp_path / "cached.pkl")
    logger = _LoggerStub()
    monkeypatch.setattr(cli_main, "BacktestService", lambda: service)
    monkeypatch.setattr(cli_main, "ArtifactService", lambda: artifacts)
    monkeypatch.setattr(cli_main, "format_walk_forward_report", lambda _folds: "WF")
    monkeypatch.setattr(cli_main, "logger", logger)
    monkeypatch.setattr(
        "sys.argv",
        ["tradingdev", "--config", str(config_path), "--walk-forward"],
    )

    cli_main.main()

    assert service.calls == [(config_path, True)]
    assert artifacts.calls[0]["strategy_id"] == "cli_walk_forward"
    assert any("WF" in msg for msg in logger.messages)
