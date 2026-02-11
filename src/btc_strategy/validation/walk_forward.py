"""Walk-forward validation for trading strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from btc_strategy.utils.logger import setup_logger

if TYPE_CHECKING:
    from datetime import datetime

    from btc_strategy.backtest.engine import BacktestEngine
    from btc_strategy.data.schemas import WalkForwardConfig
    from btc_strategy.strategies.base import BaseStrategy

logger = setup_logger(__name__)


@dataclass
class WalkForwardResult:
    """Result from a single walk-forward fold."""

    fold_index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_metrics: dict[str, Any] = field(default_factory=dict)
    test_metrics: dict[str, Any] = field(default_factory=dict)
    strategy_params: dict[str, Any] = field(default_factory=dict)


class WalkForwardValidator:
    """Orchestrate walk-forward validation cycles.

    Core operation per fold:
        1. Split data into train/test by date.
        2. ``strategy.fit(train_data)``
        3. ``strategy.generate_signals(test_data)``
        4. ``engine.run(signals)`` → metrics
        5. Collect :class:`WalkForwardResult`.

    Supports:
        - Explicit date splits (single fold).
        - Rolling windows (``n_splits`` with fixed window size).
        - Expanding windows (``n_splits`` with growing train window).
    """

    def __init__(
        self,
        config: WalkForwardConfig,
        engine: BacktestEngine,
    ) -> None:
        self._config = config
        self._engine = engine

    def validate(
        self,
        strategy: BaseStrategy,
        df: pd.DataFrame,
    ) -> list[WalkForwardResult]:
        """Run walk-forward validation.

        Args:
            strategy: Strategy instance (mutated by ``fit()`` each fold).
            df: Full OHLCV DataFrame covering all periods.
                Must have a ``timestamp`` column or a DatetimeIndex.

        Returns:
            List of :class:`WalkForwardResult`, one per fold.
        """
        splits = self._split_data(df)
        results: list[WalkForwardResult] = []

        for i, (train_df, test_df) in enumerate(splits):
            train_start = train_df["timestamp"].iloc[0]
            train_end = train_df["timestamp"].iloc[-1]
            test_start = test_df["timestamp"].iloc[0]
            test_end = test_df["timestamp"].iloc[-1]

            logger.info(
                "Fold %d: train [%s ~ %s] → test [%s ~ %s]",
                i,
                train_start,
                train_end,
                test_start,
                test_end,
            )

            # 1. Fit strategy on training data
            strategy.fit(train_df)

            # 2. Evaluate on training data (for overfitting inspection)
            train_signals = strategy.generate_signals(train_df)
            train_metrics = self._engine.run(train_signals)

            # 3. Evaluate on test data
            test_signals = strategy.generate_signals(test_df)
            test_metrics = self._engine.run(test_signals)

            result = WalkForwardResult(
                fold_index=i,
                train_start=train_start.to_pydatetime(),
                train_end=train_end.to_pydatetime(),
                test_start=test_start.to_pydatetime(),
                test_end=test_end.to_pydatetime(),
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                strategy_params=strategy.get_parameters(),
            )
            results.append(result)

            logger.info(
                "Fold %d results: train %s=%.4f, test %s=%.4f",
                i,
                self._config.target_metric,
                train_metrics.get(self._config.target_metric, float("nan")),
                self._config.target_metric,
                test_metrics.get(self._config.target_metric, float("nan")),
            )

        return results

    def _split_data(
        self,
        df: pd.DataFrame,
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate (train, test) DataFrame pairs.

        If explicit dates are provided in config, returns a single split.
        Otherwise, generates ``n_splits`` using rolling/expanding windows.
        """
        cfg = self._config

        # Explicit date split mode
        if (
            cfg.train_start is not None
            and cfg.train_end is not None
            and cfg.test_start is not None
            and cfg.test_end is not None
        ):
            ts = df["timestamp"]

            def _to_utc_ts(dt: datetime) -> pd.Timestamp:
                t = pd.Timestamp(dt)
                if t.tzinfo is None:
                    t = t.tz_localize("UTC")
                return t

            train_mask = (ts >= _to_utc_ts(cfg.train_start)) & (
                ts < _to_utc_ts(cfg.train_end) + pd.Timedelta(days=1)
            )
            test_mask = (ts >= _to_utc_ts(cfg.test_start)) & (
                ts < _to_utc_ts(cfg.test_end) + pd.Timedelta(days=1)
            )
            return [(df[train_mask].copy(), df[test_mask].copy())]

        # Auto-split mode: rolling or expanding windows
        n = len(df)
        total_per_split = n // cfg.n_splits
        train_size = int(total_per_split * cfg.train_ratio)
        test_size = total_per_split - train_size

        splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
        for i in range(cfg.n_splits):
            if cfg.expanding:
                # Expanding: train starts from beginning, grows each fold
                train_end_idx = train_size + i * test_size
                test_start_idx = train_end_idx
                test_end_idx = test_start_idx + test_size
            else:
                # Rolling: fixed-size train window moves forward
                train_start_idx = i * test_size
                train_end_idx = train_start_idx + train_size
                test_start_idx = train_end_idx
                test_end_idx = test_start_idx + test_size

            if test_end_idx > n:
                break

            if cfg.expanding:
                train_df = df.iloc[:train_end_idx].copy()
            else:
                train_df = df.iloc[train_start_idx:train_end_idx].copy()
            test_df = df.iloc[test_start_idx:test_end_idx].copy()

            splits.append((train_df, test_df))

        return splits

    @staticmethod
    def summary(results: list[WalkForwardResult]) -> dict[str, Any]:
        """Aggregate metrics across all folds.

        Returns:
            Dictionary with mean, std, min, max for each test metric.
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
    """Format walk-forward results into a human-readable report.

    Args:
        results: List of :class:`WalkForwardResult`.

    Returns:
        Formatted multi-line string.
    """
    lines: list[str] = [
        "=" * 60,
        "  Walk-Forward Validation Report",
        "=" * 60,
    ]

    for r in results:
        lines.append(f"\n  Fold {r.fold_index}:")
        lines.append(
            f"    Train: {r.train_start:%Y-%m-%d} ~ "
            f"{r.train_end:%Y-%m-%d}"
        )
        lines.append(
            f"    Test:  {r.test_start:%Y-%m-%d} ~ "
            f"{r.test_end:%Y-%m-%d}"
        )
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

    # Summary across folds
    summary = WalkForwardValidator.summary(results)
    if summary.get("n_folds", 0) > 1:
        lines.append("\n" + "-" * 60)
        lines.append("  Summary across folds:")
        for key, stats in summary.items():
            if isinstance(stats, dict):
                lines.append(
                    f"    {key}: mean={stats['mean']:.4f} "
                    f"std={stats['std']:.4f} "
                    f"[{stats['min']:.4f}, {stats['max']:.4f}]"
                )

    lines.append("=" * 60)
    return "\n".join(lines)
