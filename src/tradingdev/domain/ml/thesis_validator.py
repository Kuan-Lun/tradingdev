"""Thesis Validator — prediction validity model for quantile strategy exit.

Trains a lightweight XGBoost binary classifier that, given the entry-time
quantile predictions and the observed price path so far, estimates the
probability that the original prediction is still valid.

When P(valid) drops below ``exit_threshold``, the strategy exits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xgboost as xgb

from tradingdev.shared.utils.logger import setup_logger

if TYPE_CHECKING:
    import numpy.typing as npt

logger = setup_logger(__name__)


class ThesisValidator:
    """Binary classifier for thesis (prediction) validity.

    Trained on historical entry-to-exit paths, it learns to predict
    whether the quantile model's prediction will ultimately prove
    accurate (label=1) or not (label=0) at each intermediate time
    step within the holding period.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._random_state = random_state
        self._model: xgb.XGBClassifier | None = None
        self._feature_cols: list[str] = []

    @property
    def is_trained(self) -> bool:
        """Whether the model has been trained."""
        return self._model is not None

    def train(
        self,
        close: npt.NDArray[np.floating[Any]],
        quantile_preds: pd.DataFrame,
        horizon: int,
        quantiles: list[float],
    ) -> None:
        """Train the validator on historical quantile predictions.

        For each bar where a prediction exists, simulate holding for
        1..horizon bars and generate training samples:
        - Features: entry predictions, observed return, deviation rank,
          realized vol, time progress.
        - Label: 1 if final return falls within Q10-Q90, else 0.

        Args:
            close: Array of close prices.
            quantile_preds: DataFrame with columns like ``q05``..``q95``.
            horizon: Prediction horizon (max holding bars).
            quantiles: List of quantile levels (e.g. [0.05, ..., 0.95]).
        """
        n = len(close)
        log_close = np.log(np.maximum(close, 1e-10))

        q_cols = [f"q{int(q * 100):02d}" for q in quantiles]
        q_vals = quantile_preds[q_cols].values

        # Find Q10 and Q90 columns for label definition
        q10_idx = _find_nearest_idx(quantiles, 0.10)
        q90_idx = _find_nearest_idx(quantiles, 0.90)

        samples: list[dict[str, float]] = []
        # Sub-sample to keep training size manageable
        step = max(1, horizon // 3)

        for i in range(0, n - horizon, step):
            final_ret = log_close[i + horizon] - log_close[i]
            q10_val = float(q_vals[i, q10_idx])
            q90_val = float(q_vals[i, q90_idx])
            label = 1.0 if q10_val <= final_ret <= q90_val else 0.0

            # Generate intermediate observation points
            for t in range(1, horizon + 1, max(1, horizon // 5)):
                if i + t >= n:
                    break
                obs_ret = log_close[i + t] - log_close[i]
                sample = _build_sample(
                    entry_preds=q_vals[i],
                    q_cols=q_cols,
                    observed_return=obs_ret,
                    time_progress=t / horizon,
                    close_slice=close[i : i + t + 1],
                    label=label,
                )
                samples.append(sample)

        if len(samples) < 50:
            logger.warning(
                "ThesisValidator: only %d samples, skipping training",
                len(samples),
            )
            return

        train_df = pd.DataFrame(samples)
        feature_cols = [c for c in train_df.columns if c != "label"]
        x = train_df[feature_cols].values
        y = train_df["label"].values

        self._model = xgb.XGBClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            random_state=self._random_state,
            eval_metric="logloss",
            verbosity=0,
        )
        self._model.fit(x, y, verbose=False)
        self._feature_cols = feature_cols

        pos_rate = float(np.asarray(y, dtype=np.float64).mean()) * 100
        logger.info(
            "ThesisValidator trained: %d samples, %.1f%% positive",
            len(samples),
            pos_rate,
        )

    def predict_validity(
        self,
        entry_preds: npt.NDArray[np.floating[Any]],
        q_cols: list[str],
        observed_return: float,
        time_progress: float,
        close_slice: npt.NDArray[np.floating[Any]],
    ) -> float:
        """Predict P(thesis still valid).

        Args:
            entry_preds: Quantile predictions at entry time.
            q_cols: Column names for the quantile predictions.
            observed_return: Log return from entry to now.
            time_progress: Fraction of horizon elapsed (0-1).
            close_slice: Close prices from entry to now.

        Returns:
            Probability that the prediction is still valid (0-1).
        """
        if self._model is None:
            return 1.0  # No model → always valid

        sample = _build_sample(
            entry_preds=entry_preds,
            q_cols=q_cols,
            observed_return=observed_return,
            time_progress=time_progress,
            close_slice=close_slice,
            label=None,
        )
        x = np.array([[sample[c] for c in self._feature_cols]])
        proba = self._model.predict_proba(x)
        # Return P(label=1)
        return float(proba[0, 1])


def _build_sample(
    entry_preds: npt.NDArray[np.floating[Any]],
    q_cols: list[str],
    observed_return: float,
    time_progress: float,
    close_slice: npt.NDArray[np.floating[Any]],
    label: float | None,
) -> dict[str, float]:
    """Build a single training/inference sample."""
    sample: dict[str, float] = {}

    # Entry predictions
    for j, col in enumerate(q_cols):
        sample[f"entry_{col}"] = float(entry_preds[j])

    # Trimmed mean of entry predictions
    sample["entry_trimmed_mean"] = float(np.mean(entry_preds[1:-1]))

    # Observed return
    sample["observed_return"] = observed_return

    # Deviation: where does observed_return fall in the distribution
    n_below = int(np.sum(entry_preds < observed_return))
    sample["deviation_rank"] = n_below / max(len(entry_preds), 1)

    # Time progress
    sample["time_progress"] = time_progress

    # Path features from close_slice
    if len(close_slice) > 1:
        log_rets = np.diff(np.log(np.maximum(close_slice, 1e-10)))
        sample["path_realized_vol"] = (
            float(np.std(log_rets)) if len(log_rets) > 1 else 0.0
        )
        cum_rets = np.cumsum(log_rets)
        peak = np.maximum.accumulate(cum_rets)
        sample["path_max_drawdown"] = float(
            np.min(cum_rets - peak),
        )
        if observed_return != 0:
            direction = np.sign(observed_return)
            sample["path_consistency"] = float(
                np.mean(np.sign(log_rets) == direction),
            )
        else:
            sample["path_consistency"] = 0.5
    else:
        sample["path_realized_vol"] = 0.0
        sample["path_max_drawdown"] = 0.0
        sample["path_consistency"] = 0.5

    if label is not None:
        sample["label"] = label

    return sample


def _find_nearest_idx(quantiles: list[float], target: float) -> int:
    """Find index of the quantile nearest to target."""
    return int(np.argmin([abs(q - target) for q in quantiles]))
