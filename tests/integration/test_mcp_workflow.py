"""Integration tests for the MCP-facing application workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tradingdev.adapters.execution.process_runner import ProcessRunner
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app import job_store
from tradingdev.app.data_service import DataService
from tradingdev.app.job_service import JobService
from tradingdev.app.strategy_service import StrategyService

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch


class FakeRunner(ProcessRunner):
    """Process runner that records worker calls without spawning processes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def spawn_module(self, module: str, *args: str) -> int:
        self.calls.append((module, args))
        return 4321


_STRATEGY_CODE = """\
from __future__ import annotations

from typing import Any

import pandas as pd

from tradingdev.domain.strategies.base import BaseStrategy


class IntegrationStrategy(BaseStrategy):
    def __init__(self, backtest_engine: object | None = None) -> None:
        self._engine = backtest_engine

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["signal"] = 0
        result.loc[result["close"] > result["open"], "signal"] = 1
        return result

    def get_parameters(self) -> dict[str, Any]:
        return {}
"""

_YAML = """\
strategy:
  id: "integration_strategy"
  version: "0.1.0"
  class_name: "IntegrationStrategy"
  source_path: "workspace/generated_strategies/integration_strategy.py"
  parameters: {}
backtest:
  symbol: "BTC/USDT"
  timeframe: "1h"
  start_date: "2024-01-01"
  end_date: "2024-01-31"
  init_cash: 10000.0
data:
  requirements:
    market:
      source: "binance_api"
      symbol: "BTC/USDT"
      timeframe: "1h"
    features: []
"""


def test_generated_strategy_can_start_backtest_job(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    monkeypatch.setattr(job_store, "_WORKSPACE", workspace)
    monkeypatch.setattr(job_store, "_STORE", store)
    workspace.processed_data.mkdir(parents=True, exist_ok=True)
    (workspace.processed_data / "btcusdt_1h_2024.parquet").write_text(
        "",
        encoding="utf-8",
    )

    strategy_service = StrategyService(workspace)
    strategy_service._quality_gate_diagnostics = lambda _path: []  # type: ignore[assignment,method-assign]
    saved = strategy_service.save_draft(
        "integration_strategy",
        _STRATEGY_CODE,
        _YAML,
        request_summary="integration smoke test",
    )
    assert saved.success is True
    assert strategy_service.validate("integration_strategy")["success"] is True
    assert strategy_service.dry_run("integration_strategy")["success"] is True
    assert strategy_service.promote("integration_strategy")["success"] is True

    runner = FakeRunner()
    service = JobService(
        strategy_service=strategy_service,
        data_service=DataService(workspace),
        process_runner=runner,
        project_root=tmp_path,
    )

    response = service.start_backtest(
        strategy_id="integration_strategy",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-31",
    )

    assert response["job_id"]
    assert response["data_available"] is True
    assert runner.calls == [
        (
            "tradingdev.mcp.workers.backtest",
            (
                response["job_id"],
                str(workspace.configs / "integration_strategy.yaml"),
            ),
        )
    ]
    job = store.get_job(str(response["job_id"]))
    assert job is not None
    assert job["status"] == "queued"
    assert job["pid"] == 4321
    assert job["job_type"] == "backtest"
