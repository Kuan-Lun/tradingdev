"""Formatting and descriptive summaries of walk-forward fold metrics."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from tradingdev.domain.performance.catalog import METRIC_CATALOG

if TYPE_CHECKING:
    from tradingdev.domain.validation.walk_forward import WalkForwardResult


def _finite_number(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return value if math.isfinite(value) else None


def _fmt_metric(key: str, value: object) -> str:
    """Keep a metric's unit and drawdown sign independent of execution mode."""
    number = _finite_number(value)
    if number is None:
        return "N/A"
    definition = METRIC_CATALOG.get(key)
    unit = definition.unit if definition is not None else "ratio"
    if unit == "fraction":
        return f"{number:.2%}"
    if unit == "amount":
        return f"{number:,.2f}"
    if unit in {"count", "bars"}:
        return f"{number:,.2f}"
    return f"{number:.4f}"


def summarize_results(
    results: list[WalkForwardResult],
) -> dict[str, Any]:
    """Describe valid test-fold values, without treating means as pooled metrics.

    Each metric includes the number of finite fold values. Missing, null and
    non-finite values are excluded; no valid values yields null statistics.
    """
    if not results:
        return {}

    metric_keys = dict.fromkeys(
        key for result in results for key in result.test_metrics
    )
    summary: dict[str, Any] = {"n_folds": len(results)}
    for key in metric_keys:
        values = [
            number
            for result in results
            if (number := _finite_number(result.test_metrics.get(key))) is not None
        ]
        stats: dict[str, float | int | None] = {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "valid_count": len(values),
        }
        if values:
            arr = np.array(values, dtype=float)
            stats.update(
                mean=float(np.mean(arr)),
                std=float(np.std(arr)),
                min=float(np.min(arr)),
                max=float(np.max(arr)),
            )
        summary[key] = stats
    return summary


def format_walk_forward_report(results: list[WalkForwardResult]) -> str:
    """Render per-fold values and explicitly labeled test-fold summaries."""
    lines = [
        "=" * 60,
        "  Walk-Forward Validation Report",
        "  Monetary values are in the market's quote currency.",
        "=" * 60,
    ]
    for result in results:
        lines.extend(
            [
                f"\n  Fold {result.fold_index}:",
                f"    Train: {result.train_start:%Y-%m-%d} ~ "
                f"{result.train_end:%Y-%m-%d}",
                f"    Test:  {result.test_start:%Y-%m-%d} ~ {result.test_end:%Y-%m-%d}",
                f"    Params: {result.strategy_params}",
            ]
        )
        for label, metrics in (
            ("Train", result.train_metrics),
            ("Test", result.test_metrics),
        ):
            lines.append(f"    {label} metrics:")
            lines.extend(
                f"      {key:>24s}: {_fmt_metric(key, value):>14s}"
                for key, value in metrics.items()
            )

    summary = summarize_results(results)
    if len(results) > 1:
        lines.extend(
            [
                "\n" + "-" * 60,
                "  Descriptive test-fold statistics:",
                "  Fold means are not full-period portfolio performance.",
            ]
        )
        for key, stats in summary.items():
            if isinstance(stats, dict):
                lines.append(
                    f"    {key}: mean={_fmt_metric(key, stats['mean'])} "
                    f"std={_fmt_metric(key, stats['std'])} "
                    f"[{_fmt_metric(key, stats['min'])}, "
                    f"{_fmt_metric(key, stats['max'])}] "
                    f"valid={stats['valid_count']}/{len(results)}"
                )
    lines.append("=" * 60)
    return "\n".join(lines)
