"""Canonical buy-and-hold baselines chosen by asset class, never by the caller.

A benchmark answers one question: what would the same capital have earned by
holding the reference asset over the same bars? The mapping from a saved data
requirement to a benchmark is fixed here, so two people running the same market
always receive the same baseline. Nothing in this module fetches data; the
application layer supplies the closes and records how it obtained them.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

# Data sources whose market is a crypto pair quoted in a stablecoin.
CRYPTO_SOURCES = frozenset({"binance_vision", "binance_api"})

# Yahoo lists non-US venues with an exchange suffix and derivates with ``=``.
# Indices are accepted only from this explicit list; an unfamiliar ``^`` symbol
# stays undefined instead of silently borrowing the US large-cap baseline.
US_INDEX_SYMBOLS = frozenset({"^GSPC", "^SPX", "^DJI", "^IXIC", "^NDX", "^RUT"})

BenchmarkStatus = Literal["available", "unavailable", "undefined"]


@dataclass(frozen=True)
class BenchmarkDefinition:
    """The fixed reference asset and the weighting rule applied to it."""

    id: str
    asset_class: str
    symbol: str
    sources: tuple[str, ...]
    label: str
    weighting: str
    rationale: str


BENCHMARKS = MappingProxyType(
    {
        "crypto": BenchmarkDefinition(
            id="crypto_btc_buy_and_hold",
            asset_class="crypto",
            symbol="BTC/USDT",
            sources=("binance_vision", "binance_api"),
            label="BTC/USDT 買進持有",
            weighting="buy_and_hold",
            rationale=(
                "加密貨幣策略的機會成本基準是持有比特幣。使用與回測相同的資料源、"
                "市場類型與 K 線週期，讓比較只反映策略進出場，不混入換交易所或"
                "換週期造成的差異。"
            ),
        ),
        "us_equity": BenchmarkDefinition(
            id="us_equity_sp500_buy_and_hold",
            asset_class="us_equity",
            symbol="^GSPC",
            sources=("yahoo_finance",),
            label="S&P 500（^GSPC）買進持有",
            weighting="buy_and_hold",
            rationale=(
                "美國股票策略的機會成本基準是持有 S&P 500 指數。使用 ^GSPC 的"
                "價格報酬，未計入成分股現金股利，因此不是含息總報酬指數。"
            ),
        ),
        "tw_equity": BenchmarkDefinition(
            id="tw_equity_taiex_buy_and_hold",
            asset_class="tw_equity",
            symbol="^TWII",
            sources=("yahoo_finance",),
            label="台股加權指數（^TWII）買進持有",
            weighting="buy_and_hold",
            rationale=(
                "台灣掛牌標的統一使用台股加權指數作為市場基準，包含上市、上櫃"
                "及 ETF；此政策不代表標的的實際投資範圍都與台股大盤相同。"
                "使用 ^TWII 的價格報酬，未計入現金股利，因此不是含息總報酬指數。"
            ),
        ),
    }
)


def asset_class_for(source: str, symbol: str) -> str | None:
    """Classify a saved market requirement; unknown markets stay unclassified."""
    if source in CRYPTO_SOURCES and re.fullmatch(r"[A-Z0-9]+/?USDT", symbol.upper()):
        return "crypto"
    if source == "yahoo_finance":
        value = symbol.strip().upper()
        # Taiwan's baseline is a listing-market policy, including ETFs with
        # overseas or non-equity holdings, rather than underlying-asset inference.
        if value == "^TWII" or re.fullmatch(r"[A-Z0-9]+\.(?:TW|TWO)", value):
            return "tw_equity"
        return "us_equity" if is_us_equity_symbol(symbol) else None
    return None


def is_us_equity_symbol(symbol: str) -> bool:
    """Accept US listings only, using Yahoo's documented symbol conventions.

    ``AAPL`` and ``SPY`` carry no venue suffix. A ``.XX`` suffix names another
    exchange, ``=F``/``=X`` are futures/FX, and an index must be one of the
    known US indices. Anything else is refused rather than assumed.
    """
    value = symbol.strip().upper()
    if not value or value.endswith(("=X", "=F")):
        return False
    if value.startswith("^"):
        return value in US_INDEX_SYMBOLS
    if "." in value:
        if not value.endswith(".US"):
            return False
        value = value[:-3]
    # Share classes such as BRK-B are supported; currency pairs such as BTC-USD
    # are not evidence of a US equity listing.
    return re.fullmatch(r"[A-Z][A-Z0-9]*(?:-[A-Z])?", value) is not None


def resolve_benchmark(source: str, symbol: str) -> BenchmarkDefinition | None:
    """Return the fixed baseline for a market, or ``None`` when undefined."""
    asset_class = asset_class_for(source, symbol)
    return None if asset_class is None else BENCHMARKS.get(asset_class)


def benchmark_source(definition: BenchmarkDefinition, run_source: str) -> str:
    """Prefer the run's own provider so both series share a bar calendar."""
    return run_source if run_source in definition.sources else definition.sources[0]


@dataclass(frozen=True)
class BenchmarkEvidence:
    """One run's baseline identity and the closes obtained for it.

    ``status`` distinguishes "no baseline is defined for this asset class"
    from "a baseline exists but its data could not be used"; both stay visible
    instead of becoming a zero return.
    """

    status: BenchmarkStatus
    asset_class: str | None = None
    definition: BenchmarkDefinition | None = None
    reason: str | None = None
    detail: str | None = None
    source: str | None = None
    market_type: str | None = None
    dataset_id: str | None = None
    reused_run_frame: bool = False
    timestamps: pd.DatetimeIndex | None = None
    closes: npt.NDArray[np.float64] | None = None


@dataclass(frozen=True)
class BuyAndHold:
    """One entry at the first close and a cash liquidation at the last close."""

    equity: npt.NDArray[np.float64]
    quantity: float
    entry_price: float
    entry_fee: float
    entry_notional: float
    exit_price: float
    exit_fee: float
    exit_notional: float

    def trade_record(self) -> dict[str, float | int | str]:
        """Export the actual round trip using the shared normalized ledger."""
        gross_pnl = self.exit_notional - self.entry_notional
        fee = self.entry_fee + self.exit_fee
        return {
            "direction": 1,
            "entry_idx": 0,
            "exit_idx": len(self.equity) - 1,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "size": self.quantity,
            "size_quote": self.entry_notional,
            "exit_notional": self.exit_notional,
            "entry_fees": self.entry_fee,
            "exit_fees": self.exit_fee,
            "fee": fee,
            "gross_pnl": gross_pnl,
            "net_pnl": gross_pnl - fee,
            "status": "closed",
        }


def buy_and_hold(
    closes: npt.NDArray[np.float64],
    init_cash: float,
    *,
    fees: float,
    slippage: float,
) -> BuyAndHold:
    """Invest at the first close and liquidate at the last observed close.

    Both fills pay the run's per-side commission and slippage. Intermediate
    equity is marked to market; final equity is the exit proceeds after costs.
    A single observed bar therefore includes both entry and exit costs.
    Unrepresentable execution amounts raise ValueError so a supplementary
    benchmark can be reported unavailable without losing the strategy result.
    """
    if not math.isfinite(init_cash) or init_cash <= 0:
        raise ValueError("Benchmark initial capital must be finite and positive")
    prices = np.asarray(closes, dtype=np.float64)
    if prices.ndim != 1 or not len(prices):
        raise ValueError("Benchmark closes must be a non-empty one-dimensional series")
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("Benchmark closes must be finite and positive")
    for name, value in (("fees", fees), ("slippage", slippage)):
        if not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"Benchmark {name} must be a fraction in [0, 1)")
    entry_price = float(prices[0]) * (1.0 + slippage)
    funded_price = entry_price * (1.0 + fees)
    if not math.isfinite(funded_price) or funded_price <= 0:
        raise ValueError("Benchmark funded entry price is not representable")
    quantity = init_cash / funded_price
    exit_price = float(prices[-1]) * (1.0 - slippage)
    entry_notional = quantity * entry_price
    entry_fee = entry_notional * fees
    exit_notional = quantity * exit_price
    exit_fee = exit_notional * fees
    if any(
        not math.isfinite(value) or value <= 0
        for value in (quantity, entry_price, exit_price, entry_notional, exit_notional)
    ) or not math.isfinite(entry_fee + exit_fee):
        raise ValueError("Benchmark execution amounts are not representable")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        equity = quantity * prices
    if not np.isfinite(equity).all() or (equity <= 0).any():
        raise ValueError("Benchmark marked equity is not representable")
    equity[-1] = exit_notional - exit_fee
    return BuyAndHold(
        equity=equity,
        quantity=quantity,
        entry_price=entry_price,
        entry_fee=entry_fee,
        entry_notional=entry_notional,
        exit_price=exit_price,
        exit_fee=exit_fee,
        exit_notional=exit_notional,
    )


def aligned_closes(
    evidence: BenchmarkEvidence, timestamps: pd.DatetimeIndex
) -> tuple[pd.DatetimeIndex, npt.NDArray[np.float64], int]:
    """Restrict a baseline to the observed bars of one result scope.

    Returns the shared timestamps, their closes and how many scope bars the
    baseline is missing. Bars are never filled or shifted: an absent benchmark
    observation is reported, not replaced by the previous close.
    """
    if evidence.timestamps is None or evidence.closes is None:
        raise ValueError("Benchmark evidence has no observations")
    series = pd.Series(evidence.closes, index=evidence.timestamps)
    selected = series.reindex(timestamps)
    present = selected.dropna()
    return (
        pd.DatetimeIndex(present.index),
        np.asarray(present.to_numpy(), dtype=np.float64),
        int(selected.isna().sum()),
    )
