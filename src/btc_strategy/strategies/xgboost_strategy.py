"""XGBoost-based ML trading strategy with rolling retraining."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from btc_strategy.ml.features import FeatureEngineer
from btc_strategy.ml.xgboost_model import XGBoostDirectionModel
from btc_strategy.strategies.base import BaseStrategy
from btc_strategy.utils.logger import setup_logger

if TYPE_CHECKING:
    from btc_strategy.data.schemas import XGBoostStrategyConfig

logger = setup_logger(__name__)


class XGBoostStrategy(BaseStrategy):
    """XGBoost direction prediction strategy.

    ``fit()`` phase (train + validation data):
        1. Split input by ``validation_ratio`` into train / val.
        2. For each candidate lookback window, train an XGBoost model
           with early stopping on validation, then evaluate accuracy.
        3. Select the best lookback, retrain on the full fit data.
        4. Store training data for use during rolling retraining.

    ``generate_signals()`` phase (test data):
        - Every ``retrain_interval`` bars, retrain the model on the
          most recent history.
        - Predict direction for each bar → emit signal.
    """

    def __init__(self, config: XGBoostStrategyConfig) -> None:
        self._config = config
        self._model: XGBoostDirectionModel | None = None
        self._feature_engineer: FeatureEngineer | None = None
        self._best_lookback: int | None = None
        self._train_data: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame) -> None:
        """Optimise lookback window and train the XGBoost model.

        The input *df* is split into train and validation subsets
        according to ``config.validation_ratio``.  For each candidate
        lookback window the model is trained with early stopping and
        evaluated on the validation set.

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

        best_score = -1.0
        best_lookback = self._config.lookback_candidates[0]

        for lb in self._config.lookback_candidates:
            fe = FeatureEngineer(lookback=lb)

            feat_train = fe.transform(train_df, include_target=True)
            feat_val = fe.transform(val_df, include_target=True)

            if len(feat_train) < 10 or len(feat_val) < 5:
                logger.warning(
                    "Lookback %d: insufficient rows after transform "
                    "(train=%d, val=%d), skipping",
                    lb,
                    len(feat_train),
                    len(feat_val),
                )
                continue

            model = XGBoostDirectionModel(config=self._config.model)
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

        self._best_lookback = best_lookback
        self._feature_engineer = FeatureEngineer(lookback=best_lookback)

        # Retrain final model on full fit data
        feat_full = self._feature_engineer.transform(
            df, include_target=True
        )
        self._model = XGBoostDirectionModel(config=self._config.model)
        self._model.train(feat_full)

        # Store fit data for rolling retraining context
        self._train_data = df.copy()

        logger.info(
            "XGBoost fit complete: best_lookback=%d, "
            "val_accuracy=%.4f",
            best_lookback,
            best_score,
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals via rolling prediction and retraining.

        The stored training data is prepended to *df* so the model
        always has sufficient history for feature engineering.  Every
        ``retrain_interval`` bars the model is retrained on the most
        recent window of data.

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
                "Strategy has not been fitted. Call fit() first."
            )
            raise RuntimeError(msg)

        fe = self._feature_engineer
        retrain_interval = self._config.retrain_interval
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

        for i in range(len(df)):
            combined_idx = test_offset + i

            # Retrain periodically
            if i % retrain_interval == 0:
                start = max(0, combined_idx - window_size)
                window = combined.iloc[start:combined_idx].copy()

                if len(window) > self._best_lookback + 10:
                    feat = fe.transform(
                        window, include_target=True
                    )
                    if len(feat) >= 10:
                        model = XGBoostDirectionModel(
                            config=self._config.model,
                        )
                        model.train(feat)

            # Predict current bar
            # Need enough context for feature engineering
            ctx_start = max(
                0, combined_idx - self._best_lookback * 2
            )
            ctx = combined.iloc[ctx_start : combined_idx + 1].copy()
            feat_pred = fe.transform(
                ctx, include_target=False
            )

            if len(feat_pred) > 0:
                pred = model.predict(feat_pred.iloc[[-1]])
                signals[i] = int(pred.iloc[0])

        result = df.copy()
        result["signal"] = signals
        return result

    def get_parameters(self) -> dict[str, Any]:
        """Return strategy parameters including best lookback."""
        params = self._config.model_dump()
        params["best_lookback"] = self._best_lookback
        return params
