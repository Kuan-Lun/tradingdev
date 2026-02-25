"""Human-readable formatting for backtest metrics."""

from __future__ import annotations

from typing import Any


def format_metrics_report(
    metrics: dict[str, Any],
    mode: str = "signal",
) -> str:
    """Format metrics dictionary into a human-readable report."""
    vol = metrics.get("total_volume", 0)
    n_days = metrics.get("n_days", 0)
    monthly_vol = vol / max(n_days, 1) * 30

    is_volume = mode == "volume"

    lines = [
        "=" * 55,
        "  Backtest Performance Report",
        "=" * 55,
        f"  Total P&L:         {metrics.get('total_pnl', 0):>+10,.0f}",
    ]

    if is_volume:
        lines.append(
            f"  Max Drawdown:      {metrics['max_drawdown']:>10,.0f} USDT",
        )
    else:
        lines.extend(
            [
                f"  Total Return:      {metrics['total_return']:>10.2%}",
                f"  Annual Return:     {metrics['annual_return']:>10.2%}",
                f"  Max Drawdown:      {metrics['max_drawdown']:>10.2%}",
            ]
        )

    lines.extend(
        [
            f"  Sharpe Ratio:      {metrics['sharpe_ratio']:>10.4f}",
            f"  Win Rate:          {metrics['win_rate']:>10.2%}",
            f"  Profit Factor:     {metrics['profit_factor']:>10.4f}",
            "-" * 55,
            f"  Total Trades:      {metrics['total_trades']:>10d}",
            f"  Total Volume:      {vol:>13,.0f}",
            f"  Est. Monthly Vol:  {monthly_vol:>13,.0f}",
            "-" * 55,
            f"  Daily P&L  mean:   {metrics.get('daily_pnl_mean', 0):>+10.2f}",
            f"  Daily P&L  std:    {metrics.get('daily_pnl_std', 0):>10.2f}",
            f"  Daily P&L  min:    {metrics.get('daily_pnl_min', 0):>+10.2f}",
            f"  Daily P&L  max:    {metrics.get('daily_pnl_max', 0):>+10.2f}",
            f"  Daily P&L  median: {metrics.get('daily_pnl_median', 0):>+10.2f}",
            f"  Period:            {metrics.get('n_days', 0):>7d} days",
            "=" * 55,
        ]
    )
    return "\n".join(lines)
