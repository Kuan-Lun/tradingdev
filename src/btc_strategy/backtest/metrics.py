"""Performance metrics extraction from vectorbt Portfolio."""

from typing import Any

import vectorbt as vbt

from btc_strategy.utils.logger import setup_logger

logger = setup_logger(__name__)


def calculate_metrics(pf: vbt.Portfolio) -> dict[str, Any]:
    """Extract key performance metrics from a vectorbt Portfolio.

    Args:
        pf: A vectorbt Portfolio object after backtest execution.

    Returns:
        Dictionary containing:
            - total_return (float): Total return as a ratio.
            - sharpe_ratio (float): Annualized Sharpe ratio.
            - max_drawdown (float): Maximum drawdown as a ratio.
            - win_rate (float): Percentage of winning trades.
            - profit_factor (float): Ratio of gross profit to gross loss.
            - total_trades (int): Total number of trades.
            - annual_return (float): Annualized return.
    """
    trades = pf.trades
    has_trades = len(trades.records_readable) > 0

    return {
        "total_return": float(pf.total_return()),
        "sharpe_ratio": float(pf.sharpe_ratio()),
        "max_drawdown": float(pf.max_drawdown()),
        "win_rate": float(trades.win_rate()) if has_trades else 0.0,
        "profit_factor": float(trades.profit_factor()) if has_trades else 0.0,
        "total_trades": int(trades.count()),
        "annual_return": float(pf.annualized_return()),
    }


def format_metrics_report(metrics: dict[str, Any]) -> str:
    """Format metrics dictionary into a human-readable report.

    Args:
        metrics: Dictionary returned by :func:`calculate_metrics`.

    Returns:
        Formatted multi-line string.
    """
    lines = [
        "=" * 50,
        "  Backtest Performance Report",
        "=" * 50,
        f"  Total Return:    {metrics['total_return']:>10.2%}",
        f"  Annual Return:   {metrics['annual_return']:>10.2%}",
        f"  Sharpe Ratio:    {metrics['sharpe_ratio']:>10.4f}",
        f"  Max Drawdown:    {metrics['max_drawdown']:>10.2%}",
        f"  Win Rate:        {metrics['win_rate']:>10.2%}",
        f"  Profit Factor:   {metrics['profit_factor']:>10.4f}",
        f"  Total Trades:    {metrics['total_trades']:>10d}",
        "=" * 50,
    ]
    return "\n".join(lines)
