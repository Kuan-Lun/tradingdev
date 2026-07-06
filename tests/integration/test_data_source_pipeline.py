"""Integration tests: each data source feeds the existing backtest pipeline.

Both tests stub only the external transport (ccxt client / HTTP client) and
run the real crawler, DataManager yearly cache, DataService feature merge,
strategy loader, and backtest engine end to end.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.backtest_service import BacktestService
from tradingdev.app.data_service import DataService
from tradingdev.app.strategy_service import StrategyService
from tradingdev.domain.strategies.loader import StrategyLoader

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch

_DAY_MS = 86_400_000
_DAY_S = 86_400
_YEAR_2024_START_S = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp())


def _raw_config(source: str, symbol: str) -> dict[str, Any]:
    return {
        "strategy": {
            "id": "kd_crossover",
            "parameters": {
                "k_period": 14,
                "d_period": 3,
                "smooth_k": 3,
                "overbought": 80.0,
                "oversold": 20.0,
            },
        },
        "backtest": {
            "symbol": symbol,
            "timeframe": "1d",
            "start_date": "2024-01-01",
            "end_date": "2024-02-29",
            "init_cash": 10_000.0,
            "fees": 0.0,
            "slippage": 0.0,
        },
        "data": {
            "requirements": {
                "market": {
                    "source": source,
                    "symbol": symbol,
                    "timeframe": "1d",
                },
                "features": [],
            }
        },
    }


def _service(tmp_path: Path) -> tuple[BacktestService, WorkspacePaths]:
    workspace = WorkspacePaths(tmp_path / "workspace")
    service = BacktestService(
        data_service=DataService(workspace),
        strategy_loader=StrategyLoader(workspace_root=workspace.root),
        strategy_gate=StrategyService(workspace),
    )
    return service, workspace


def _daily_close(i: int) -> float:
    return 100.0 + 10.0 * ((i % 20) / 20.0) - 5.0 * ((i % 7) / 7.0)


class _FakeBinance:
    """ccxt.binance stand-in returning deterministic daily candles for 2024."""

    instances: list[_FakeBinance] = []

    def __init__(self, _params: dict[str, Any] | None = None) -> None:
        self.rateLimit = 0
        self.calls = 0
        _FakeBinance.instances.append(self)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: int | None = None,
        limit: int = 1000,
    ) -> list[list[float]]:
        self.calls += 1
        base_ms = _YEAR_2024_START_S * 1000
        candles = []
        for i in range(366):
            ts = base_ms + i * _DAY_MS
            if since is not None and ts < since:
                continue
            close = _daily_close(i)
            candles.append([ts, close - 0.5, close + 1.0, close - 1.0, close, 1000.0])
        return candles[:limit]


class _FakeYahooResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.status_code = 200

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _FakeYahooClient:
    """httpx.Client stand-in serving deterministic daily candles."""

    get_calls = 0

    def __init__(self, **_kwargs: Any) -> None:
        pass

    def get(self, _url: str, params: dict[str, Any]) -> _FakeYahooResponse:
        type(self).get_calls += 1
        period1 = int(params["period1"])
        period2 = int(params["period2"])
        timestamps = list(range(period1, period2 + 1, _DAY_S))
        closes = [
            _daily_close((ts - _YEAR_2024_START_S) // _DAY_S) for ts in timestamps
        ]
        return _FakeYahooResponse(
            {
                "chart": {
                    "result": [
                        {
                            "timestamp": timestamps,
                            "indicators": {
                                "quote": [
                                    {
                                        "open": [c - 0.5 for c in closes],
                                        "high": [c + 1.0 for c in closes],
                                        "low": [c - 1.0 for c in closes],
                                        "close": closes,
                                        "volume": [1000.0] * len(closes),
                                    }
                                ]
                            },
                        }
                    ],
                    "error": None,
                }
            }
        )


def test_ccxt_crypto_source_flows_into_backtest_pipeline(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRADINGDEV_DATA_ROOT", raising=False)
    monkeypatch.setattr(
        "tradingdev.domain.data.crawlers.binance_api.ccxt.binance",
        _FakeBinance,
    )
    _FakeBinance.instances = []
    service, workspace = _service(tmp_path)

    run = service.run_raw_config(_raw_config("binance_api", "BTC/USDT"))

    assert run.mode == "simple"
    assert "total_return" in run.metrics
    assert run.dataset_id.startswith("BTC/USDT:1d:2024-01-01:2024-02-29")
    cached = workspace.processed_data / "btcusdt_1d_2024.parquet"
    assert cached.exists()
    assert sum(inst.calls for inst in _FakeBinance.instances) >= 1

    # Second run must reuse the yearly cache without hitting the exchange.
    calls_before = sum(inst.calls for inst in _FakeBinance.instances)
    service.run_raw_config(_raw_config("binance_api", "BTC/USDT"))
    assert sum(inst.calls for inst in _FakeBinance.instances) == calls_before


def test_yahoo_finance_equity_source_flows_into_backtest_pipeline(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRADINGDEV_DATA_ROOT", raising=False)
    monkeypatch.setattr(
        "tradingdev.domain.data.crawlers.yahoo_finance.httpx.Client",
        _FakeYahooClient,
    )
    _FakeYahooClient.get_calls = 0
    service, workspace = _service(tmp_path)

    run = service.run_raw_config(_raw_config("yahoo_finance", "AAPL"))

    assert run.mode == "simple"
    assert "total_return" in run.metrics
    assert run.dataset_id.startswith("AAPL:1d:2024-01-01:2024-02-29")
    cached = workspace.processed_data / "aapl_1d_2024.parquet"
    assert cached.exists()
    assert _FakeYahooClient.get_calls >= 1

    # Second run must reuse the yearly cache without hitting Yahoo.
    calls_before = _FakeYahooClient.get_calls
    service.run_raw_config(_raw_config("yahoo_finance", "AAPL"))
    assert _FakeYahooClient.get_calls == calls_before
