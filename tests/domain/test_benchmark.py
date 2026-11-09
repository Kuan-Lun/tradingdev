"""Benchmark choice and buy-and-hold arithmetic stay fixed and inspectable."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.benchmark import (
    BENCHMARKS,
    BenchmarkEvidence,
    aligned_closes,
    asset_class_for,
    benchmark_source,
    buy_and_hold,
    is_us_equity_symbol,
    resolve_benchmark,
)


@pytest.mark.parametrize(
    ("source", "symbol", "expected"),
    [
        ("binance_vision", "BTC/USDT", "crypto"),
        ("binance_vision", "ETH/USDT", "crypto"),
        ("binance_api", "SOL/USDT", "crypto"),
        ("yahoo_finance", "AAPL", "us_equity"),
        ("yahoo_finance", "SPY", "us_equity"),
        ("yahoo_finance", "brk.us", "us_equity"),
        ("yahoo_finance", "^GSPC", "us_equity"),
        ("yahoo_finance", "2330.TW", "tw_equity"),
        ("yahoo_finance", "0050.TW", "tw_equity"),
        ("yahoo_finance", "00679B.TW", "tw_equity"),
        ("yahoo_finance", "6488.TWO", "tw_equity"),
        ("yahoo_finance", " 0050.tw ", "tw_equity"),
        ("yahoo_finance", "0700.HK", None),
        ("yahoo_finance", "^TWII", "tw_equity"),
        ("yahoo_finance", "^UNKNOWN.TW", None),
        ("yahoo_finance", ".TW", None),
        ("yahoo_finance", "0050.TW=F", None),
        ("csv", "0050.TW", None),
        ("yahoo_finance", "ES=F", None),
        ("yahoo_finance", "EURUSD=X", None),
        ("yahoo_finance", "BTC-USD", None),
        ("yahoo_finance", "USDT-USD", None),
        ("yahoo_finance", "BRK-B", "us_equity"),
        ("binance_api", "ETH/BTC", None),
        ("binance_vision", "BTC/USD", None),
        ("csv", "AAPL", None),
    ],
)
def test_asset_class_is_derived_only_from_the_saved_market_requirement(
    source: str, symbol: str, expected: str | None
) -> None:
    assert asset_class_for(source, symbol) == expected


def test_unknown_markets_are_undefined_rather_than_borrowing_a_baseline() -> None:
    assert resolve_benchmark("yahoo_finance", "0700.HK") is None
    assert resolve_benchmark("csv", "ANY") is None


def test_each_asset_class_has_exactly_one_fixed_baseline() -> None:
    crypto = resolve_benchmark("binance_vision", "ETH/USDT")
    equities = resolve_benchmark("yahoo_finance", "AAPL")
    assert crypto is BENCHMARKS["crypto"]
    assert equities is BENCHMARKS["us_equity"]
    assert crypto is not None and crypto.symbol == "BTC/USDT"
    assert equities is not None and equities.symbol == "^GSPC"
    # Repeated resolution never depends on the caller, the run or the clock.
    assert resolve_benchmark("binance_vision", "ETH/USDT") is crypto
    assert resolve_benchmark("binance_api", "XRP/USDT") is crypto


@pytest.mark.parametrize("symbol", ["0050.TW", "00679B.TW", "6488.TWO", "^TWII"])
def test_taiwan_listings_share_one_market_baseline(symbol: str) -> None:
    benchmark = resolve_benchmark("yahoo_finance", symbol)
    assert benchmark is not None
    assert benchmark is BENCHMARKS["tw_equity"]
    assert benchmark.symbol == "^TWII"
    assert benchmark.id == "tw_equity_taiex_buy_and_hold"
    assert benchmark.sources == ("yahoo_finance",)


def test_baseline_prefers_the_runs_own_provider_for_the_same_bar_calendar() -> None:
    crypto = BENCHMARKS["crypto"]
    equities = BENCHMARKS["us_equity"]
    assert benchmark_source(crypto, "binance_api") == "binance_api"
    assert benchmark_source(crypto, "binance_vision") == "binance_vision"
    assert benchmark_source(crypto, "csv") == "binance_vision"
    assert benchmark_source(equities, "yahoo_finance") == "yahoo_finance"


def test_us_listing_detection_rejects_other_venues_and_derivatives() -> None:
    assert is_us_equity_symbol("AAPL")
    assert is_us_equity_symbol("^NDX")
    assert not is_us_equity_symbol("^N225")
    assert not is_us_equity_symbol("6758.T")
    assert not is_us_equity_symbol("CL=F")
    assert not is_us_equity_symbol("")


def test_buy_and_hold_charges_both_sides_and_ends_in_cash() -> None:
    # Hand-derived: entry 100 * (1 + 0.01) = 101; quantity = 1000 / (101 * 1.001).
    quantity = 1000 / 101.101
    result = buy_and_hold(
        np.array([100.0, 110.0, 121.0]), 1000.0, fees=0.001, slippage=0.01
    )
    assert result.entry_price == pytest.approx(101.0)
    assert result.quantity == pytest.approx(quantity)
    assert result.entry_notional == pytest.approx(quantity * 101.0)
    assert result.entry_fee == pytest.approx(quantity * 101.0 * 0.001)
    # Notional plus commission spends exactly the initial capital.
    assert result.entry_notional + result.entry_fee == pytest.approx(1000.0)
    assert result.exit_price == pytest.approx(119.79)
    assert result.exit_notional == pytest.approx(quantity * 119.79)
    assert result.exit_fee == pytest.approx(quantity * 119.79 * 0.001)
    assert result.equity == pytest.approx(
        quantity * np.array([100.0, 110.0, 119.79 * 0.999])
    )
    assert result.equity[-1] / 1000.0 - 1 == pytest.approx(
        121 * 0.99 * 0.999 / 101 / 1.001 - 1
    )
    trade = result.trade_record()
    assert trade["status"] == "closed"
    assert trade["entry_idx"] == 0
    assert trade["exit_idx"] == 2
    assert trade["net_pnl"] == pytest.approx(result.equity[-1] - 1000.0)


def test_one_bar_buy_and_hold_pays_a_complete_round_trip() -> None:
    result = buy_and_hold(np.array([99.0]), 100.0, fees=1 / 99, slippage=0.0)
    assert result.quantity == 1.0
    assert result.entry_fee == result.exit_fee == 1.0
    assert result.equity.tolist() == [98.0]
    assert result.trade_record()["net_pnl"] == -2.0


@pytest.mark.parametrize(
    ("prices", "capital", "fee"),
    [
        ([1.7e308], 1000.0, 0.1),
        ([1e100], 1e-300, 0.001),
        ([1e-100, 1e308], 1000.0, 0.001),
        ([1e100, 1e-300], 1000.0, 0.001),
    ],
)
def test_unrepresentable_execution_is_explicitly_unavailable(
    prices: list[float], capital: float, fee: float
) -> None:
    with pytest.raises(ValueError, match="not representable"):
        buy_and_hold(np.asarray(prices), capital, fees=fee, slippage=0.0)


def test_costs_only_ever_reduce_the_baseline() -> None:
    closes = np.array([100.0, 100.0, 100.0])
    free = buy_and_hold(closes, 1000.0, fees=0.0, slippage=0.0)
    paid = buy_and_hold(closes, 1000.0, fees=0.001, slippage=0.0005)
    assert free.equity.tolist() == pytest.approx([1000.0, 1000.0, 1000.0])
    assert paid.equity[0] < free.equity[0]
    assert paid.equity[-1] < free.equity[-1]


@pytest.mark.parametrize(
    ("closes", "init_cash", "fees", "slippage"),
    [
        (np.array([100.0]), 0.0, 0.0, 0.0),
        (np.array([100.0]), -5.0, 0.0, 0.0),
        (np.array([100.0]), 1000.0, 1.0, 0.0),
        (np.array([100.0]), 1000.0, 0.0, 1.0),
        (np.array([100.0, 0.0]), 1000.0, 0.0, 0.0),
        (np.array([100.0, -1.0]), 1000.0, 0.0, 0.0),
        (np.array([]), 1000.0, 0.0, 0.0),
    ],
)
def test_impossible_baselines_are_refused(
    closes: np.ndarray, init_cash: float, fees: float, slippage: float
) -> None:
    with pytest.raises(ValueError):
        buy_and_hold(closes, init_cash, fees=fees, slippage=slippage)


def _evidence(timestamps: pd.DatetimeIndex, closes: list[float]) -> BenchmarkEvidence:
    return BenchmarkEvidence(
        status="available",
        asset_class="crypto",
        definition=BENCHMARKS["crypto"],
        source="binance_vision",
        timestamps=timestamps,
        closes=np.asarray(closes, dtype=np.float64),
    )


def test_alignment_uses_shared_bars_without_filling_gaps() -> None:
    benchmark = pd.DatetimeIndex(
        pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    )
    evidence = _evidence(benchmark, [10.0, 11.0, 12.0])
    scope = pd.DatetimeIndex(
        pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    )
    stamps, closes, missing = aligned_closes(evidence, scope)
    assert stamps.equals(benchmark)
    assert closes.tolist() == [10.0, 11.0, 12.0]
    # The unobserved 2024-01-04 bar is reported, never back-filled.
    assert missing == 1


def test_alignment_reports_a_benchmark_with_no_shared_bars() -> None:
    evidence = _evidence(pd.DatetimeIndex(pd.to_datetime(["2024-01-01"])), [10.0])
    other = pd.DatetimeIndex(pd.to_datetime(["2025-06-01"]))
    stamps, closes, missing = aligned_closes(evidence, other)
    assert len(stamps) == 0 and len(closes) == 0
    assert missing == 1
