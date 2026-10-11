"""Third-party return/risk and closed-trade statistics with explicit semantics."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import empyrical as ep
import numpy as np
import numpy.typing as npt
import pandas as pd
import vectorbt as vbt
from vectorbt.portfolio.enums import trade_dt

from tradingdev.domain.performance.catalog import METRIC_CATALOG
from tradingdev.domain.performance.sampling import (
    DAILY_EQUITY_METRICS,
    daily_observation_unavailable_reason,
)


@dataclass
class PerformanceAnalysis:
    """Calculated values and the settings required to interpret them."""

    metrics: dict[str, Any]
    metadata: dict[str, Any]
    returns: npt.NDArray[np.float64] | None


def timestamp_index(df: pd.DataFrame) -> pd.DatetimeIndex | None:
    """Use the market timestamps, normalized to UTC, never a synthetic date."""
    values = df.get("timestamp", df.index)
    if "timestamp" not in df and not isinstance(df.index, pd.DatetimeIndex):
        return None
    index = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    if index.hasnans or not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("Market timestamps must be valid, unique, and increasing")
    return index


def normalized_trades(trades: vbt.Trades) -> list[dict[str, Any]]:
    """Export the same vectorbt records used for statistics and reconciliation."""
    result = []
    for record in trades.records_arr:
        entry_notional = float(record["size"] * record["entry_price"])
        is_closed = int(record["status"]) == 1
        fee = float(record["entry_fees"] + record["exit_fees"])
        result.append(
            {
                "direction": 1 if int(record["direction"]) == 0 else -1,
                "entry_idx": int(record["entry_idx"]),
                "exit_idx": int(record["exit_idx"]),
                "entry_price": float(record["entry_price"]),
                "exit_price": float(record["exit_price"]),
                "size": float(record["size"]),
                "size_quote": entry_notional,
                "exit_notional": float(record["size"] * record["exit_price"])
                if is_closed
                else 0.0,
                "entry_fees": float(record["entry_fees"]),
                "exit_fees": float(record["exit_fees"]),
                "fee": fee,
                "gross_pnl": float(record["pnl"]) + fee,
                "net_pnl": float(record["pnl"]),
                "status": "closed" if is_closed else "open",
            }
        )
    return result


def calculate_metrics(
    pf: vbt.Portfolio,
    *,
    periods_per_year: float | None = None,
    risk_free_rate: float = 0.0,
    required_return: float = 0.0,
    frequency: str = "1h",
    timestamps: pd.DatetimeIndex | None = None,
) -> PerformanceAnalysis:
    """Analyze portfolio returns with Empyrical and trades with vectorbt."""
    orders = pf.orders.records_arr
    return _analyze(
        np.asarray(pf.value(), dtype=np.float64),
        pf.trades,
        float(pf.init_cash),
        timestamps,
        float(np.sum(orders["size"] * orders["price"])),
        float(np.sum(orders["fees"])),
        None,
        periods_per_year,
        risk_free_rate,
        required_return,
        frequency,
    )


def calculate_metrics_from_simulation(
    equity_curve: npt.NDArray[np.float64],
    trades: list[dict[str, Any]],
    init_cash: float | None,
    timestamps: pd.DatetimeIndex | None = None,
    *,
    periods_per_year: float | None = None,
    risk_free_rate: float = 0.0,
    required_return: float = 0.0,
    frequency: str = "1h",
) -> PerformanceAnalysis:
    """Normalize an existing ledger into vectorbt records without simulation."""
    records = np.zeros(len(trades), dtype=trade_dt)
    for i, trade in enumerate(trades):
        notional = float(trade["size_quote"])
        records[i] = (
            i,
            0,
            trade["size"],
            trade["entry_idx"],
            trade["entry_price"],
            trade["entry_fees"] + trade.get("entry_slippage", 0.0),
            trade["exit_idx"],
            trade["exit_price"],
            trade["exit_fees"] + trade.get("exit_slippage", 0.0),
            trade["net_pnl"],
            trade["net_pnl"] / notional,
            0 if trade["direction"] == 1 else 1,
            1 if trade["status"] == "closed" else 0,
            i,
        )
    index = timestamps if timestamps is not None else pd.RangeIndex(len(equity_curve))
    wrapper = vbt.ArrayWrapper(index, [0], ndim=1, freq=frequency)
    vbt_trades = vbt.Trades(wrapper, records, close=np.ones(len(equity_curve)))
    return _analyze(
        equity_curve,
        vbt_trades,
        init_cash,
        timestamps,
        sum(float(t["size_quote"] + t["exit_notional"]) for t in trades),
        sum(float(t["fee"]) for t in trades),
        sum(float(t.get("slippage", 0.0)) for t in trades),
        periods_per_year,
        risk_free_rate,
        required_return,
        frequency,
    )


def _analyze(
    equity: npt.NDArray[np.float64],
    trades: vbt.Trades,
    init_cash: float | None,
    timestamps: pd.DatetimeIndex | None,
    total_volume: float,
    total_fees: float,
    total_slippage: float | None,
    periods_per_year: float | None,
    risk_free_rate: float,
    required_return: float,
    frequency: str,
) -> PerformanceAnalysis:
    if not np.all(np.isfinite(equity)):
        raise ValueError("Equity must contain only finite values")
    if timestamps is not None and len(timestamps) != len(equity):
        raise ValueError("Timestamp and equity lengths must match")
    daily_sampling_reason = daily_observation_unavailable_reason(frequency)
    mode = "volume" if init_cash is None else "signal"
    metrics: dict[str, Any] = dict.fromkeys(METRIC_CATALOG)
    unavailable: dict[str, str] = {}
    for metric_id, definition in METRIC_CATALOG.items():
        if mode not in definition.modes:
            unavailable[metric_id] = "not_applicable"
    initial = init_cash if init_cash is not None else 0.0
    metrics.update(
        total_pnl=float(equity[-1]) - initial if len(equity) else 0.0,
        total_volume=total_volume,
        total_fees=total_fees,
        total_slippage=total_slippage,
        total_trades=int(trades.closed.count()),
        open_trades=int(trades.open.count()),
    )
    if total_slippage is None:
        unavailable["total_slippage"] = "embedded_in_fill_prices"
    baseline_equity = np.concatenate(([initial], equity))
    with np.errstate(over="ignore", invalid="ignore"):
        drawdowns = np.maximum.accumulate(baseline_equity) - baseline_equity
        pnl = np.diff(baseline_equity)
    if not np.all(np.isfinite(pnl)) or not np.all(np.isfinite(drawdowns)):
        raise ValueError("Equity differences must be finite")
    metrics["max_drawdown_amount"] = float(np.max(drawdowns))
    _period_metrics(metrics, unavailable, pnl, timestamps, daily_sampling_reason)
    if metrics["total_trades"]:
        closed = trades.closed
        with np.errstate(divide="ignore", invalid="ignore"):
            metrics.update(
                win_rate=float(closed.win_rate()),
                profit_factor=float(closed.profit_factor()),
                trade_expectancy=float(closed.expectancy()),
                avg_holding_bars=float(closed.duration.mean()),
            )
    else:
        for metric_id in (
            "win_rate",
            "profit_factor",
            "trade_expectancy",
            "avg_holding_bars",
        ):
            unavailable[metric_id] = "no_trades"
    returns = None
    if init_cash is not None:
        if len(equity) and np.all(baseline_equity[:-1] > 0) and np.all(equity >= 0):
            with np.errstate(over="ignore", invalid="ignore"):
                returns = np.asarray(pnl / baseline_equity[:-1], dtype=np.float64)
            if not np.all(np.isfinite(returns)):
                raise ValueError("Derived returns must be finite")
            _return_metrics(
                metrics,
                unavailable,
                returns,
                timestamps,
                periods_per_year,
                risk_free_rate,
                required_return,
                daily_sampling_reason,
            )
        else:
            for metric_id, definition in METRIC_CATALOG.items():
                if definition.provider == "empyrical":
                    unavailable[metric_id] = (
                        "insufficient_data"
                        if not len(equity)
                        else "nonpositive_capital"
                    )
    for metric_id, value in metrics.items():
        if value is not None and not np.isfinite(value):
            metrics[metric_id] = None
            unavailable[metric_id] = (
                "unbounded" if np.isinf(value) else "zero_denominator"
            )
        elif value is None and metric_id not in unavailable:
            unavailable[metric_id] = "insufficient_data"
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "providers": {
            "empyrical-reloaded": version("empyrical-reloaded"),
            "vectorbt": version("vectorbt"),
        },
        "mode": mode,
        "settings": {
            "initial_cash": init_cash,
            "periods_per_year": periods_per_year,
            "frequency": frequency,
            "risk_free_rate": risk_free_rate,
            "required_return": required_return,
            "risk_free_rate_per_period": _period_rate(risk_free_rate, periods_per_year)
            if daily_sampling_reason is None
            else None,
            "required_return_per_period": _period_rate(
                required_return, periods_per_year
            )
            if daily_sampling_reason is None
            else None,
            "returns_basis": "simple" if init_cash is not None else "not_applicable",
            "return_sampling": "observed_daily"
            if daily_sampling_reason is None
            else "unavailable",
            "stored_returns_sampling": "bar",
            "drawdown_sampling": "bar",
            "calmar_drawdown_sampling": "observed_daily"
            if daily_sampling_reason is None
            else "unavailable",
            "missing_dates": "not_filled",
            "trade_scope": "closed",
            "calendar_timezone": "UTC",
            "cost_model": "commission_and_separate_notional_slippage_charges"
            if mode == "volume"
            else "commission_and_slippage_adjusted_fill_prices",
            "return_std_ddof": 1,
            "period_pnl_std_ddof": 0,
        },
        "unavailable": unavailable,
    }
    return PerformanceAnalysis(metrics, metadata, returns)


def _period_rate(annual_rate: float, periods_per_year: float | None) -> float | None:
    if periods_per_year is None:
        return None
    return float(np.expm1(np.log1p(annual_rate) / periods_per_year))


def _return_metrics(
    metrics: dict[str, Any],
    unavailable: dict[str, str],
    returns: npt.NDArray[np.float64],
    timestamps: pd.DatetimeIndex | None,
    periods_per_year: float | None,
    risk_free_rate: float,
    required_return: float,
    daily_sampling_reason: str | None,
) -> None:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        metrics["total_return"] = float(ep.cum_returns_final(returns))
        metrics["max_drawdown"] = abs(float(ep.max_drawdown(returns)))
        if daily_sampling_reason is not None:
            for metric_id in DAILY_EQUITY_METRICS:
                if METRIC_CATALOG[metric_id].provider == "empyrical":
                    unavailable[metric_id] = daily_sampling_reason
            return
        if timestamps is not None:
            daily_returns = np.asarray(
                (
                    pd.Series(1 + returns, index=timestamps)
                    .groupby(timestamps.date)
                    .prod()
                    - 1
                ),
                dtype=np.float64,
            )
            metrics["daily_max_drawdown"] = abs(float(ep.max_drawdown(daily_returns)))
        else:
            unavailable["daily_max_drawdown"] = "missing_timestamps"
        if periods_per_year is None:
            for metric_id, definition in METRIC_CATALOG.items():
                if definition.requires_annualization:
                    unavailable[metric_id] = "missing_annualization"
            return
        if timestamps is None:
            for metric_id, definition in METRIC_CATALOG.items():
                if (
                    definition.requires_annualization
                    or metric_id == "daily_max_drawdown"
                ):
                    unavailable[metric_id] = "missing_timestamps"
            return
        returns = daily_returns
        kwargs = {"annualization": periods_per_year}
        metrics["annual_return"] = float(ep.annual_return(returns, **kwargs))
        metrics["calmar_ratio"] = float(ep.calmar_ratio(returns, **kwargs))
        if len(returns) < 2:
            for metric_id in ("sharpe_ratio", "sortino_ratio", "annual_volatility"):
                unavailable[metric_id] = "insufficient_data"
            return
        metrics["sharpe_ratio"] = float(
            ep.sharpe_ratio(
                returns,
                risk_free=_period_rate(risk_free_rate, periods_per_year),
                **kwargs,
            )
        )
        metrics["sortino_ratio"] = float(
            ep.sortino_ratio(
                returns,
                required_return=_period_rate(required_return, periods_per_year),
                **kwargs,
            )
        )
        metrics["annual_volatility"] = float(ep.annual_volatility(returns, **kwargs))


def _period_metrics(
    metrics: dict[str, Any],
    unavailable: dict[str, str],
    pnl: npt.NDArray[np.float64],
    timestamps: pd.DatetimeIndex | None,
    daily_sampling_reason: str | None,
) -> None:
    if timestamps is None or not len(pnl):
        reason = "missing_timestamps" if timestamps is None else "insufficient_data"
        for metric_id, definition in METRIC_CATALOG.items():
            if definition.category == "period":
                unavailable[metric_id] = reason
        return
    if len(timestamps) != len(pnl):
        raise ValueError("Timestamp and equity lengths must match")
    series = pd.Series(pnl, index=timestamps)
    for prefix, groups in (
        ("daily", timestamps.date),
        ("monthly", timestamps.tz_localize(None).to_period("M")),
    ):
        metrics["n_days" if prefix == "daily" else "n_months"] = len(
            pd.Index(groups).unique()
        )
        values = (
            np.asarray(series.groupby(groups).sum(), dtype=np.float64)
            if daily_sampling_reason is None
            else None
        )
        for stat, func in (
            ("mean", np.mean),
            ("std", np.std),
            ("min", np.min),
            ("max", np.max),
            ("median", np.median),
        ):
            metric_id = f"{prefix}_pnl_{stat}"
            if values is None:
                assert daily_sampling_reason is not None
                unavailable[metric_id] = daily_sampling_reason
            else:
                metrics[metric_id] = float(func(values))
    metrics["monthly_trades_mean"] = metrics["total_trades"] / metrics["n_months"]
    metrics["monthly_volume_mean"] = metrics["total_volume"] / metrics["n_months"]
