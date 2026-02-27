"""Performance metrics for vectorbt Portfolio and custom simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:
    import vectorbt as vbt

from quant_backtest.utils.logger import setup_logger

logger = setup_logger(__name__)


def calculate_metrics(pf: vbt.Portfolio) -> dict[str, Any]:
    """Extract key performance metrics from a vectorbt Portfolio."""
    trades = pf.trades
    records = trades.records_readable
    has_trades = len(records) > 0

    total_volume = 0.0
    if has_trades and "Size" in records.columns:
        entry_prices = records.get(
            "Avg Entry Price",
            records.get("Entry Price"),
        )
        if entry_prices is not None:
            total_volume = float((records["Size"].abs() * entry_prices).sum())

    daily_pnl = _daily_pnl_from_portfolio(pf)
    monthly_pnl = _monthly_pnl_from_portfolio(pf)

    total_ret = float(pf.total_return())
    value_series = pf.value()
    total_pnl = (
        float(value_series.iloc[-1] - value_series.iloc[0])
        if len(value_series) > 1
        else 0.0
    )

    total_trades_count = int(trades.count())
    n_months = monthly_pnl.get("n_months", 0)

    metrics: dict[str, Any] = {
        "total_return": total_ret,
        "total_pnl": total_pnl,
        "sharpe_ratio": float(pf.sharpe_ratio()),
        "max_drawdown": float(pf.max_drawdown()),
        "win_rate": (float(trades.win_rate()) if has_trades else 0.0),
        "profit_factor": (float(trades.profit_factor()) if has_trades else 0.0),
        "total_trades": total_trades_count,
        "total_volume": total_volume,
        "annual_return": float(pf.annualized_return()),
    }
    metrics.update(daily_pnl)
    metrics.update(monthly_pnl)
    metrics["monthly_trades_mean"] = (
        total_trades_count / n_months if n_months > 0 else 0.0
    )
    metrics["monthly_volume_mean"] = (
        total_volume / n_months if n_months > 0 else 0.0
    )
    return metrics


def _daily_pnl_from_portfolio(
    pf: vbt.Portfolio,
) -> dict[str, Any]:
    """Compute daily P&L stats from a vectorbt Portfolio."""
    try:
        value = pf.value()
        if len(value) < 2:
            return _empty_daily_pnl()

        pnl_series = value.diff().dropna()
        if hasattr(pnl_series.index, "date"):
            daily = pnl_series.groupby(pnl_series.index.date).sum()
        else:
            daily = pnl_series

        return _compute_daily_stats(np.array(daily, dtype=float))
    except Exception:
        logger.debug(
            "Could not compute daily P&L stats",
            exc_info=True,
        )
        return _empty_daily_pnl()


def calculate_metrics_from_simulation(
    equity_curve: npt.NDArray[np.float64],
    trades: list[dict[str, Any]],
    init_cash: float | None,
    timestamps: Any | None = None,
) -> dict[str, Any]:
    """Compute metrics from a custom bar-by-bar simulation.

    When ``init_cash`` is ``None`` (volume / fixed-notional mode),
    the equity curve is treated as cumulative P&L starting from zero.
    Percentage-based metrics (``total_return``, ``annual_return``) are
    set to ``0.0``, and ``max_drawdown`` is reported in absolute terms
    (USDT).
    """
    n = len(equity_curve)
    is_volume_mode = init_cash is None
    effective_init = init_cash if init_cash is not None else 0.0
    final_equity = float(equity_curve[-1]) if n > 0 else effective_init

    total_pnl = final_equity - effective_init
    if is_volume_mode:
        total_return = 0.0
    else:
        total_return = total_pnl / effective_init if effective_init != 0 else 0.0

    total_trades = len(trades)
    total_volume = sum(t["size_quote"] for t in trades) * 2

    wins = [t for t in trades if t["net_pnl"] > 0]
    losses = [t for t in trades if t["net_pnl"] <= 0]
    win_rate = len(wins) / total_trades if total_trades else 0.0
    sum_wins = sum(t["net_pnl"] for t in wins)
    sum_losses = abs(sum(t["net_pnl"] for t in losses))
    profit_factor = sum_wins / sum_losses if sum_losses > 0 else 0.0

    daily_pnl = _daily_pnl_from_equity(equity_curve, timestamps)
    monthly_pnl = _monthly_pnl_from_equity(equity_curve, timestamps)

    if n >= 2:
        n_days = daily_pnl.get("n_days", 1)
        bars_per_day = max(n / max(n_days, 1), 1)
        ann_factor = np.sqrt(365 * bars_per_day)

        if is_volume_mode:
            # Absolute P&L-based Sharpe (dollar Sharpe)
            pnl_diffs = np.diff(equity_curve)
            std = float(np.std(pnl_diffs))
            sharpe = float(np.mean(pnl_diffs)) / std * ann_factor if std > 0 else 0.0

            # Max drawdown in absolute USDT
            cummax = np.maximum.accumulate(equity_curve)
            max_drawdown = float(np.max(cummax - equity_curve))

            annual_return = 0.0
        else:
            # Percentage-based metrics (signal mode)
            returns = np.diff(equity_curve) / equity_curve[:-1]
            std = float(np.std(returns))
            sharpe = float(np.mean(returns)) / std * ann_factor if std > 0 else 0.0

            cummax = np.maximum.accumulate(equity_curve)
            drawdowns = (cummax - equity_curve) / cummax
            max_drawdown = float(np.max(drawdowns))

            total_days = max(n_days, 1)
            annual_return = (1 + total_return) ** (365 / total_days) - 1
    else:
        sharpe = 0.0
        max_drawdown = 0.0
        annual_return = 0.0

    n_months = monthly_pnl.get("n_months", 0)

    metrics: dict[str, Any] = {
        "total_return": total_return,
        "total_pnl": total_pnl,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_trades": total_trades,
        "total_volume": total_volume,
        "annual_return": annual_return,
    }
    metrics.update(daily_pnl)
    metrics.update(monthly_pnl)
    metrics["monthly_trades_mean"] = (
        total_trades / n_months if n_months > 0 else 0.0
    )
    metrics["monthly_volume_mean"] = (
        total_volume / n_months if n_months > 0 else 0.0
    )
    return metrics


def _daily_pnl_from_equity(
    equity_curve: npt.NDArray[np.float64],
    timestamps: Any | None = None,
) -> dict[str, Any]:
    """Compute daily P&L stats from an equity curve."""
    if len(equity_curve) < 2:
        return _empty_daily_pnl()

    pnl_arr: npt.NDArray[np.float64] = np.diff(equity_curve)

    if timestamps is not None:
        try:
            ts = pd.Series(timestamps[1:])
            dates = ts.dt.date if hasattr(ts.dt, "date") else ts
            daily = pd.Series(pnl_arr).groupby(dates.values).sum()  # type: ignore[arg-type]
            pnl_arr = np.asarray(daily, dtype=np.float64)
        except Exception:
            pass

    return _compute_daily_stats(pnl_arr)


def _compute_daily_stats(
    arr: npt.NDArray[np.float64],
) -> dict[str, Any]:
    """Shared daily P&L statistics computation."""
    if len(arr) == 0:
        return _empty_daily_pnl()
    return {
        "daily_pnl_mean": float(np.mean(arr)),
        "daily_pnl_std": float(np.std(arr)),
        "daily_pnl_min": float(np.min(arr)),
        "daily_pnl_max": float(np.max(arr)),
        "daily_pnl_median": float(np.median(arr)),
        "n_days": len(arr),
    }


def _empty_daily_pnl() -> dict[str, Any]:
    return {
        "daily_pnl_mean": 0.0,
        "daily_pnl_std": 0.0,
        "daily_pnl_min": 0.0,
        "daily_pnl_max": 0.0,
        "daily_pnl_median": 0.0,
        "n_days": 0,
    }


# ---------------------------------------------------------------------------
# Monthly P&L statistics
# ---------------------------------------------------------------------------


def _monthly_pnl_from_portfolio(
    pf: vbt.Portfolio,
) -> dict[str, Any]:
    """Compute monthly P&L stats from a vectorbt Portfolio."""
    try:
        value = pf.value()
        if len(value) < 2:
            return _empty_monthly_pnl()

        pnl_series = value.diff().dropna()
        if hasattr(pnl_series.index, "to_period"):
            monthly = pnl_series.groupby(pnl_series.index.to_period("M")).sum()
        else:
            return _empty_monthly_pnl()

        return _compute_monthly_stats(np.array(monthly, dtype=float))
    except Exception:
        logger.debug(
            "Could not compute monthly P&L stats",
            exc_info=True,
        )
        return _empty_monthly_pnl()


def _monthly_pnl_from_equity(
    equity_curve: npt.NDArray[np.float64],
    timestamps: Any | None = None,
) -> dict[str, Any]:
    """Compute monthly P&L stats from an equity curve."""
    if len(equity_curve) < 2:
        return _empty_monthly_pnl()

    pnl_arr: npt.NDArray[np.float64] = np.diff(equity_curve)

    if timestamps is not None:
        try:
            ts = pd.Series(timestamps[1:])
            if hasattr(ts.dt, "tz") and ts.dt.tz is not None:
                ts = ts.dt.tz_localize(None)
            periods = ts.dt.to_period("M") if hasattr(ts.dt, "to_period") else None
            if periods is not None:
                monthly = pd.Series(pnl_arr).groupby(periods.values).sum()  # type: ignore[arg-type]
                return _compute_monthly_stats(
                    np.asarray(monthly, dtype=np.float64),
                )
        except Exception:
            pass

    return _empty_monthly_pnl()


def _compute_monthly_stats(
    arr: npt.NDArray[np.float64],
) -> dict[str, Any]:
    """Shared monthly P&L statistics computation."""
    if len(arr) == 0:
        return _empty_monthly_pnl()
    return {
        "monthly_pnl_mean": float(np.mean(arr)),
        "monthly_pnl_std": float(np.std(arr)),
        "monthly_pnl_min": float(np.min(arr)),
        "monthly_pnl_max": float(np.max(arr)),
        "monthly_pnl_median": float(np.median(arr)),
        "n_months": len(arr),
    }


def _empty_monthly_pnl() -> dict[str, Any]:
    return {
        "monthly_pnl_mean": 0.0,
        "monthly_pnl_std": 0.0,
        "monthly_pnl_min": 0.0,
        "monthly_pnl_max": 0.0,
        "monthly_pnl_median": 0.0,
        "n_months": 0,
    }
