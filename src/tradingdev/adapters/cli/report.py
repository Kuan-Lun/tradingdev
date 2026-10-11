"""Human-readable formatting for backtest metrics."""

from __future__ import annotations

import math
from typing import Any


def _format_number(value: object, spec: str) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "N/A"
    return format(value, spec) if math.isfinite(value) else "N/A"


def format_metrics_report(
    metrics: dict[str, Any],
    mode: str = "signal",
) -> str:
    """Display explicit metric units and unavailable values without conversion."""
    rows = [
        ("Total P&L", "total_pnl", "+,.2f"),
        ("Total Return", "total_return", ".2%"),
        ("Annual Return", "annual_return", ".2%"),
        ("Max Drawdown", "max_drawdown", ".2%"),
        ("Max Drawdown (amount)", "max_drawdown_amount", ",.2f"),
        ("Sharpe Ratio", "sharpe_ratio", ".4f"),
        ("Sortino Ratio", "sortino_ratio", ".4f"),
        ("Calmar Ratio", "calmar_ratio", ".4f"),
        ("Annual Volatility", "annual_volatility", ".2%"),
        ("Win Rate", "win_rate", ".2%"),
        ("Profit Factor", "profit_factor", ".4f"),
        ("Trade Expectancy", "trade_expectancy", "+,.2f"),
        ("Total Fees", "total_fees", ",.2f"),
        ("Total Trades", "total_trades", ",.0f"),
        ("Total Volume", "total_volume", ",.2f"),
        ("Daily P&L mean", "daily_pnl_mean", "+,.2f"),
        ("Daily P&L std", "daily_pnl_std", ",.2f"),
        ("Daily P&L min", "daily_pnl_min", "+,.2f"),
        ("Daily P&L max", "daily_pnl_max", "+,.2f"),
        ("Daily P&L median", "daily_pnl_median", "+,.2f"),
        ("Days", "n_days", ",.0f"),
        ("Monthly P&L mean", "monthly_pnl_mean", "+,.2f"),
        ("Monthly P&L std", "monthly_pnl_std", ",.2f"),
        ("Monthly P&L min", "monthly_pnl_min", "+,.2f"),
        ("Monthly P&L max", "monthly_pnl_max", "+,.2f"),
        ("Monthly P&L median", "monthly_pnl_median", "+,.2f"),
        ("Monthly Trades", "monthly_trades_mean", ",.2f"),
        ("Monthly Volume", "monthly_volume_mean", ",.2f"),
        ("Months", "n_months", ",.0f"),
    ]
    lines = [
        "=" * 60,
        f"  Backtest Performance Report ({mode})",
        "  Monetary values are in the market's quote currency.",
        "=" * 60,
    ]
    lines.extend(
        f"  {label + ':':<24} {_format_number(metrics.get(key), spec):>14}"
        for label, key, spec in rows
    )
    lines.append("=" * 60)
    return "\n".join(lines)
