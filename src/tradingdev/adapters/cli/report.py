"""Human-readable formatting for backtest metrics."""

from __future__ import annotations

import math
from typing import Any


def _finite_number(value: object) -> int | float | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    return None


def _format_number(value: object, spec: str, width: int = 10) -> str:
    number = _finite_number(value)
    text = format(number, spec) if number is not None else "N/A"
    return f"{text:>{width}}"


def format_metrics_report(
    metrics: dict[str, Any],
    mode: str = "signal",
) -> str:
    """Format metrics, displaying missing or non-finite values as N/A."""
    vol = _finite_number(metrics.get("total_volume"))
    n_days = _finite_number(metrics.get("n_days"))
    monthly_vol = (
        vol / max(n_days, 1) * 30 if vol is not None and n_days is not None else None
    )
    drawdown = _finite_number(metrics.get("max_drawdown"))
    negative_drawdown = -drawdown if drawdown is not None else None

    is_volume = mode == "volume"

    lines = [
        "=" * 55,
        "  Backtest Performance Report",
        "=" * 55,
        f"  Total P&L:         {_format_number(metrics.get('total_pnl'), '+,.0f')}",
    ]

    if is_volume:
        lines.append(
            f"  Max Drawdown:      {_format_number(negative_drawdown, ',.0f')} USDT",
        )
    else:
        lines.extend(
            [
                "  Total Return:      "
                f"{_format_number(metrics.get('total_return'), '.2%')}",
                "  Annual Return:     "
                f"{_format_number(metrics.get('annual_return'), '.2%')}",
                f"  Max Drawdown:      {_format_number(negative_drawdown, '.2%')}",
            ]
        )

    lines.extend(
        [
            "  Sharpe Ratio:      "
            f"{_format_number(metrics.get('sharpe_ratio'), '.4f')}",
            f"  Win Rate:          {_format_number(metrics.get('win_rate'), '.2%')}",
            "  Profit Factor:     "
            f"{_format_number(metrics.get('profit_factor'), '.4f')}",
            "-" * 55,
            f"  Total Trades:      {_format_number(metrics.get('total_trades'), 'd')}",
            f"  Total Volume:      {_format_number(vol, ',.0f', 13)}",
            f"  Est. Monthly Vol:  {_format_number(monthly_vol, ',.0f', 13)}",
            "-" * 55,
            "  Daily P&L  mean:   "
            f"{_format_number(metrics.get('daily_pnl_mean'), '+.2f')}",
            "  Daily P&L  std:    "
            f"{_format_number(metrics.get('daily_pnl_std'), '.2f')}",
            "  Daily P&L  min:    "
            f"{_format_number(metrics.get('daily_pnl_min'), '+.2f')}",
            "  Daily P&L  max:    "
            f"{_format_number(metrics.get('daily_pnl_max'), '+.2f')}",
            "  Daily P&L  median: "
            f"{_format_number(metrics.get('daily_pnl_median'), '+.2f')}",
            f"  Period:            {_format_number(n_days, 'd', 7)} days",
            "-" * 55,
            "  Monthly P&L  mean:   "
            f"{_format_number(metrics.get('monthly_pnl_mean'), '+.2f')}",
            "  Monthly P&L  std:    "
            f"{_format_number(metrics.get('monthly_pnl_std'), '.2f')}",
            "  Monthly P&L  min:    "
            f"{_format_number(metrics.get('monthly_pnl_min'), '+.2f')}",
            "  Monthly P&L  max:    "
            f"{_format_number(metrics.get('monthly_pnl_max'), '+.2f')}",
            "  Monthly P&L  median: "
            f"{_format_number(metrics.get('monthly_pnl_median'), '+.2f')}",
            "  Monthly Trades:      "
            f"{_format_number(metrics.get('monthly_trades_mean'), ',.0f')}",
            "  Monthly Volume:      "
            f"{_format_number(metrics.get('monthly_volume_mean'), ',.0f')}",
            "  Period:              "
            f"{_format_number(metrics.get('n_months'), 'd', 7)} months",
            "=" * 55,
        ]
    )
    return "\n".join(lines)
