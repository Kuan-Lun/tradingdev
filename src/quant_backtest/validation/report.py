"""Formatting and summary for walk-forward validation results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from quant_backtest.validation.walk_forward import (
        WalkForwardResult,
    )


_INTEGER_METRIC_KEYS = frozenset(
    {
        "total_volume",
        "total_pnl",
        "total_trades",
        "n_days",
        "n_months",
        "monthly_trades_mean",
        "monthly_volume_mean",
    },
)

_PERCENT_METRIC_KEYS = frozenset(
    {"total_return", "annual_return", "max_drawdown", "win_rate"},
)

# In volume mode, max_drawdown is absolute USDT, not a percentage.
_VOLUME_MODE_OVERRIDE_TO_INTEGER = frozenset({"max_drawdown"})

# Percentage-based metrics that are meaningless in volume mode (no init_cash).
_VOLUME_MODE_HIDDEN_KEYS = frozenset({"total_return", "annual_return"})


def _fmt_metric(key: str, value: float, *, mode: str = "signal") -> str:
    """Format a metric value based on its key."""
    # max_drawdown is stored as a positive number; display as negative.
    if key == "max_drawdown":
        if mode == "volume":
            return f"{-value:>14,.0f}"
        return f"{-value:>14.2%}"
    if key in _INTEGER_METRIC_KEYS:
        return f"{value:>+14,.0f}"
    if key in _PERCENT_METRIC_KEYS:
        return f"{value:>14.2%}"
    return f"{value:>14.4f}"


def summarize_results(
    results: list[WalkForwardResult],
) -> dict[str, Any]:
    """Aggregate metrics across all folds.

    Returns:
        Dictionary with mean, std, min, max for each metric.
    """
    if not results:
        return {}

    metric_keys = list(results[0].test_metrics.keys())
    summary: dict[str, Any] = {"n_folds": len(results)}

    for key in metric_keys:
        values = [r.test_metrics[key] for r in results]
        arr = np.array(values, dtype=float)
        # Replace inf/-inf with NaN so nanmean/nanstd ignore them cleanly
        arr = np.where(np.isinf(arr), np.nan, arr)
        with np.errstate(invalid="ignore"):
            summary[key] = {
                "mean": float(np.nanmean(arr)),
                "std": float(np.nanstd(arr)),
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
            }

    return summary


def _infer_mode(results: list[WalkForwardResult]) -> str:
    """Infer backtest mode from the first fold's BacktestResult."""
    for r in results:
        if r.test_backtest is not None:
            return r.test_backtest.mode
        if r.train_backtest is not None:
            return r.train_backtest.mode
    return "signal"


def format_walk_forward_report(
    results: list[WalkForwardResult],
) -> str:
    """Format walk-forward results into a readable report."""
    mode = _infer_mode(results)

    lines: list[str] = [
        "=" * 60,
        "  Walk-Forward Validation Report",
        "=" * 60,
    ]

    for r in results:
        lines.append(f"\n  Fold {r.fold_index}:")
        lines.append(f"    Train: {r.train_start:%Y-%m-%d} ~ {r.train_end:%Y-%m-%d}")
        lines.append(f"    Test:  {r.test_start:%Y-%m-%d} ~ {r.test_end:%Y-%m-%d}")
        lines.append(f"    Params: {r.strategy_params}")

        lines.append("    Train metrics:")
        for k, v in r.train_metrics.items():
            if mode == "volume" and k in _VOLUME_MODE_HIDDEN_KEYS:
                continue
            if isinstance(v, float):
                lines.append(f"      {k:>20s}: {_fmt_metric(k, v, mode=mode)}")
            else:
                lines.append(f"      {k:>20s}: {v!s:>14s}")

        lines.append("    Test metrics:")
        for k, v in r.test_metrics.items():
            if mode == "volume" and k in _VOLUME_MODE_HIDDEN_KEYS:
                continue
            if isinstance(v, float):
                lines.append(f"      {k:>20s}: {_fmt_metric(k, v, mode=mode)}")
            else:
                lines.append(f"      {k:>20s}: {v!s:>14s}")

    summary = summarize_results(results)
    if summary.get("n_folds", 0) > 1:
        lines.append("\n" + "-" * 60)
        lines.append("  Summary across folds:")
        for key, stats in summary.items():
            if mode == "volume" and key in _VOLUME_MODE_HIDDEN_KEYS:
                continue
            if isinstance(stats, dict):
                if key == "max_drawdown":
                    fmt = ",.0f" if mode == "volume" else ".2%"
                    # Negate: drawdown stored positive, display negative.
                    s = stats
                    lines.append(
                        f"    {key}: mean={-s['mean']:{fmt}} "
                        f"std={s['std']:{fmt}} "
                        f"[{-s['max']:{fmt}}, "
                        f"{-s['min']:{fmt}}]"
                    )
                else:
                    if key in _INTEGER_METRIC_KEYS:
                        fmt = ",.0f"
                    elif key in _PERCENT_METRIC_KEYS:
                        fmt = ".2%"
                    else:
                        fmt = ".4f"
                    lines.append(
                        f"    {key}: mean={stats['mean']:{fmt}} "
                        f"std={stats['std']:{fmt}} "
                        f"[{stats['min']:{fmt}}, "
                        f"{stats['max']:{fmt}}]"
                    )

    lines.append("=" * 60)
    return "\n".join(lines)
