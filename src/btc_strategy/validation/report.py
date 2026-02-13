"""Formatting and summary for walk-forward validation results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from btc_strategy.validation.walk_forward import (
        WalkForwardResult,
    )


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
        summary[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    return summary


def format_walk_forward_report(
    results: list[WalkForwardResult],
) -> str:
    """Format walk-forward results into a readable report."""
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
            if isinstance(v, float):
                lines.append(f"      {k:>20s}: {v:>10.4f}")
            else:
                lines.append(f"      {k:>20s}: {v!s:>10s}")

        lines.append("    Test metrics:")
        for k, v in r.test_metrics.items():
            if isinstance(v, float):
                lines.append(f"      {k:>20s}: {v:>10.4f}")
            else:
                lines.append(f"      {k:>20s}: {v!s:>10s}")

    summary = summarize_results(results)
    if summary.get("n_folds", 0) > 1:
        lines.append("\n" + "-" * 60)
        lines.append("  Summary across folds:")
        for key, stats in summary.items():
            if isinstance(stats, dict):
                lines.append(
                    f"    {key}: mean={stats['mean']:.4f} "
                    f"std={stats['std']:.4f} "
                    f"[{stats['min']:.4f}, "
                    f"{stats['max']:.4f}]"
                )

    lines.append("=" * 60)
    return "\n".join(lines)
