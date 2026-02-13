"""XGBoost-based ML trading strategy with rolling retraining.

Supports two optimisation objectives configured via
``signal_threshold_candidates``:

- **Profit maximisation** (no candidates): uses a fixed
  ``signal_threshold`` and optimises lookback by accuracy.
- **Volume maximisation** (with candidates): after finding the best
  lookback, searches for the *lowest* threshold where the backtest
  P&L >= 0 on the validation set, maximising trade count.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from btc_strategy.ml.features import FeatureEngineer
from btc_strategy.ml.xgboost_model import XGBoostDirectionModel
from btc_strategy.strategies.base import BaseStrategy
from btc_strategy.utils.logger import setup_logger

if TYPE_CHECKING:
    from btc_strategy.backtest.engine import BacktestEngine
    from btc_strategy.data.schemas import XGBoostStrategyConfig

logger = setup_logger(__name__)


class XGBoostStrategy(BaseStrategy):
    """XGBoost direction prediction strategy.

    ``fit()`` phase (train + validation data):
        1. Split input by ``validation_ratio`` into train / val.
        2. For each candidate lookback window, train an XGBoost model
           with early stopping on validation, then evaluate accuracy.
        3. Select the best lookback.
        4. If ``signal_threshold_candidates`` is set, search for the
           lowest threshold where P&L >= 0 on the validation set
           (volume-maximisation mode).
        5. Retrain on the full fit data.

    ``generate_signals()`` phase (test data):
        - Every ``retrain_interval`` bars, retrain the model on the
          most recent history.
        - Predict probabilities for each bar, apply threshold and
          cooldown → emit signal.
    """

    def __init__(
        self,
        config: XGBoostStrategyConfig,
        backtest_engine: BacktestEngine | None = None,
    ) -> None:
        self._config = config
        self._backtest_engine = backtest_engine
        self._model: XGBoostDirectionModel | None = None
        self._feature_engineer: FeatureEngineer | None = None
        self._best_lookback: int | None = None
        self._best_threshold: float = config.signal_threshold
        self._train_data: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> None:
        """Optimise lookback window and train the XGBoost model.

        Args:
            df: OHLCV DataFrame covering the full fit period.
        """
        val_ratio = self._config.validation_ratio
        split_idx = int(len(df) * (1 - val_ratio))
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        logger.info(
            "XGBoost fit: %d train rows, %d val rows, "
            "%d lookback candidates",
            len(train_df),
            len(val_df),
            len(self._config.lookback_candidates),
        )

        # Phase 1 — find best lookback by accuracy
        best_score = -1.0
        best_lookback = self._config.lookback_candidates[0]
        best_val_model: XGBoostDirectionModel | None = None
        best_val_fe: FeatureEngineer | None = None

        n_candidates = len(self._config.lookback_candidates)
        for idx, lb in enumerate(
            self._config.lookback_candidates, 1
        ):
            logger.info(
                "[fit] Lookback candidate %d/%d (lb=%d)",
                idx,
                n_candidates,
                lb,
            )
            fe = FeatureEngineer(
                lookback=lb,
                target_horizon=self._config.target_horizon,
            )

            feat_train = fe.transform(
                train_df, include_target=True
            )
            feat_val = fe.transform(val_df, include_target=True)

            if len(feat_train) < 10 or len(feat_val) < 5:
                logger.warning(
                    "Lookback %d: insufficient rows after "
                    "transform (train=%d, val=%d), skipping",
                    lb,
                    len(feat_train),
                    len(feat_val),
                )
                continue

            model = XGBoostDirectionModel(
                config=self._config.model
            )
            model.train(feat_train, eval_df=feat_val)

            preds = model.predict(feat_val)
            score = float(
                accuracy_score(feat_val["target"], preds)
            )

            logger.info(
                "Lookback %d: val accuracy=%.4f", lb, score
            )

            if score > best_score:
                best_score = score
                best_lookback = lb
                best_val_model = model
                best_val_fe = fe

        self._best_lookback = best_lookback
        self._feature_engineer = FeatureEngineer(
            lookback=best_lookback,
            target_horizon=self._config.target_horizon,
        )

        # Phase 2 — threshold search (volume-maximisation mode)
        if (
            self._config.signal_threshold_candidates is not None
            and best_val_model is not None
            and best_val_fe is not None
            and self._backtest_engine is not None
        ):
            self._search_threshold(
                val_df, best_val_model, best_val_fe
            )

        # Phase 3 — retrain final model on full fit data
        feat_full = self._feature_engineer.transform(
            df, include_target=True
        )
        self._model = XGBoostDirectionModel(
            config=self._config.model
        )
        self._model.train(feat_full)

        # Store fit data for rolling retraining context
        self._train_data = df.copy()

        logger.info(
            "XGBoost fit complete: best_lookback=%d, "
            "val_accuracy=%.4f, threshold=%.2f",
            best_lookback,
            best_score,
            self._best_threshold,
        )

    # ------------------------------------------------------------------
    # threshold search
    # ------------------------------------------------------------------

    def _search_threshold(
        self,
        val_df: pd.DataFrame,
        model: XGBoostDirectionModel,
        fe: FeatureEngineer,
    ) -> None:
        """Find the lowest threshold with P&L >= 0 on *val_df*."""
        engine = self._backtest_engine
        if engine is None:
            return

        candidates = sorted(
            self._config.signal_threshold_candidates or []
        )
        feat_val = fe.transform(val_df, include_target=False)
        if len(feat_val) == 0:
            return

        proba = model.predict_proba(feat_val)

        for threshold in candidates:
            signals = self._apply_threshold(proba, threshold)

            # Build a signal DataFrame for engine.run()
            signal_df = val_df.iloc[-len(signals) :].copy()
            signal_df = signal_df.reset_index(drop=True)
            signal_df["signal"] = signals.values

            metrics = engine.run(signal_df)
            total_return = metrics.get("total_return", -1.0)
            total_trades = metrics.get("total_trades", 0)

            logger.info(
                "Threshold %.2f: return=%.4f, trades=%d",
                threshold,
                total_return,
                total_trades,
            )

            if (
                isinstance(total_return, float)
                and total_return >= 0
            ):
                self._best_threshold = threshold
                logger.info(
                    "Selected threshold=%.2f "
                    "(lowest with P&L>=0, trades=%d)",
                    threshold,
                    total_trades,
                )
                return

        # None worked — use the highest candidate
        self._best_threshold = (
            candidates[-1]
            if candidates
            else self._config.signal_threshold
        )
        logger.warning(
            "No threshold achieved P&L>=0; "
            "using highest=%.2f",
            self._best_threshold,
        )

    def _apply_threshold(
        self,
        proba: pd.DataFrame,
        threshold: float,
    ) -> pd.Series:
        """Convert probability DataFrame to signals."""
        p_long = (
            proba[1]
            if 1 in proba.columns
            else pd.Series(0.0, index=proba.index)
        )
        p_short = (
            proba[-1]
            if -1 in proba.columns
            else pd.Series(0.0, index=proba.index)
        )

        signals = pd.Series(0, index=proba.index, dtype=int)
        long_mask = (p_long >= threshold) & (p_long >= p_short)
        short_mask = (p_short >= threshold) & (~long_mask)
        signals = signals.where(~long_mask, 1)
        signals = signals.where(~short_mask, -1)

        # Apply cooldown between trades
        cooldown = self._config.min_bars_between_trades
        if cooldown > 1:
            last_bar = -cooldown
            for i in range(len(signals)):
                if signals.iloc[i] != 0:
                    if i - last_bar < cooldown:
                        signals.iloc[i] = 0
                    else:
                        last_bar = i

        return signals

    # ------------------------------------------------------------------
    # generate_signals
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals via rolling prediction and retraining.

        Args:
            df: Test-period OHLCV DataFrame.

        Returns:
            *df* with a ``signal`` column appended.

        Raises:
            RuntimeError: If ``fit()`` has not been called.
        """
        if (
            self._model is None
            or self._feature_engineer is None
            or self._best_lookback is None
            or self._train_data is None
        ):
            msg = (
                "Strategy has not been fitted. "
                "Call fit() first."
            )
            raise RuntimeError(msg)

        fe = self._feature_engineer
        retrain_interval = self._config.retrain_interval
        threshold = self._best_threshold
        cooldown = self._config.min_bars_between_trades
        # Window for retraining: use enough history
        window_size = max(
            self._best_lookback * 10, len(self._train_data)
        )

        # Combine historical context + test data
        combined = pd.concat(
            [self._train_data, df], ignore_index=True
        )
        # Index where test data starts in combined
        test_offset = len(self._train_data)

        signals = np.zeros(len(df), dtype=int)
        model = self._model
        total_bars = len(df)
        total_retrains = (
            (total_bars - 1) // retrain_interval + 1
        )
        retrain_count = 0
        last_trade_bar = -cooldown  # allow first trade
        t_start = time.monotonic()
        log_interval = max(total_bars // 20, 1)

        logger.info(
            "[signals] Starting: %d bars, retrain every %d "
            "bars (~%d retrains), threshold=%.2f",
            total_bars,
            retrain_interval,
            total_retrains,
            threshold,
        )

        for i in range(total_bars):
            combined_idx = test_offset + i

            # Retrain periodically
            if i % retrain_interval == 0:
                retrain_count += 1
                logger.info(
                    "[signals] Retrain %d/%d at bar %d/%d",
                    retrain_count,
                    total_retrains,
                    i,
                    total_bars,
                )
                start = max(0, combined_idx - window_size)
                window = combined.iloc[
                    start:combined_idx
                ].copy()

                if len(window) > self._best_lookback + 10:
                    feat = fe.transform(
                        window, include_target=True
                    )
                    if len(feat) >= 10:
                        model = XGBoostDirectionModel(
                            config=self._config.model,
                        )
                        model.train(feat)

            # Progress log
            if i > 0 and i % log_interval == 0:
                elapsed = time.monotonic() - t_start
                bars_per_sec = i / elapsed
                eta = (total_bars - i) / bars_per_sec
                logger.info(
                    "[signals] Progress: %d/%d (%.1f%%) "
                    "| %.0f bars/s | ETA %.0fs",
                    i,
                    total_bars,
                    i / total_bars * 100,
                    bars_per_sec,
                    eta,
                )

            # Cooldown check
            if i - last_trade_bar < cooldown:
                continue

            # Predict current bar via probabilities
            ctx_start = max(
                0, combined_idx - self._best_lookback * 2
            )
            ctx = combined.iloc[
                ctx_start : combined_idx + 1
            ].copy()
            feat_pred = fe.transform(
                ctx, include_target=False
            )

            if len(feat_pred) > 0:
                proba = model.predict_proba(
                    feat_pred.iloc[[-1]]
                )
                p_long = (
                    float(proba[1].iloc[0])
                    if 1 in proba.columns
                    else 0.0
                )
                p_short = (
                    float(proba[-1].iloc[0])
                    if -1 in proba.columns
                    else 0.0
                )

                if (
                    p_long >= threshold
                    and p_long >= p_short
                ):
                    signals[i] = 1
                    last_trade_bar = i
                elif p_short >= threshold:
                    signals[i] = -1
                    last_trade_bar = i

        elapsed_total = time.monotonic() - t_start
        n_signals = int(np.count_nonzero(signals))
        logger.info(
            "[signals] Complete: %d bars in %.1fs "
            "(%.0f bars/s), %d signals emitted",
            total_bars,
            elapsed_total,
            (
                total_bars / elapsed_total
                if elapsed_total > 0
                else 0
            ),
            n_signals,
        )

        result = df.copy()
        result["signal"] = signals
        return result

    def get_parameters(self) -> dict[str, Any]:
        """Return strategy parameters including best lookback."""
        params = self._config.model_dump()
        params["best_lookback"] = self._best_lookback
        params["best_threshold"] = self._best_threshold
        return params
