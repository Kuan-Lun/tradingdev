"""Performance metrics for vectorbt Portfolio and custom simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:
    import vectorbt as vbt

from btc_strategy.utils.logger import setup_logger

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

    metrics: dict[str, Any] = {
        "total_return": float(pf.total_return()),
        "sharpe_ratio": float(pf.sharpe_ratio()),
        "max_drawdown": float(pf.max_drawdown()),
        "win_rate": (float(trades.win_rate()) if has_trades else 0.0),
        "profit_factor": (float(trades.profit_factor()) if has_trades else 0.0),
        "total_trades": int(trades.count()),
        "total_volume_usdt": total_volume,
        "annual_return": float(pf.annualized_return()),
    }
    metrics.update(daily_pnl)
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
    init_cash: float,
    timestamps: Any | None = None,
) -> dict[str, Any]:
    """Compute metrics from a custom bar-by-bar simulation."""
    n = len(equity_curve)
    final_equity = float(equity_curve[-1]) if n > 0 else init_cash

    total_return = (final_equity - init_cash) / init_cash
    total_trades = len(trades)
    total_volume = sum(t["size_usdt"] for t in trades) * 2

    wins = [t for t in trades if t["net_pnl"] > 0]
    losses = [t for t in trades if t["net_pnl"] <= 0]
    win_rate = len(wins) / total_trades if total_trades else 0.0
    sum_wins = sum(t["net_pnl"] for t in wins)
    sum_losses = abs(sum(t["net_pnl"] for t in losses))
    profit_factor = sum_wins / sum_losses if sum_losses > 0 else 0.0

    daily_pnl = _daily_pnl_from_equity(equity_curve, timestamps)

    if n >= 2:
        returns = np.diff(equity_curve) / equity_curve[:-1]
        n_days = daily_pnl.get("n_days", 1)
        bars_per_day = max(n / max(n_days, 1), 1)
        ann_factor = np.sqrt(365 * bars_per_day)
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

    metrics: dict[str, Any] = {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_trades": total_trades,
        "total_volume_usdt": total_volume,
        "annual_return": annual_return,
    }
    metrics.update(daily_pnl)
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
