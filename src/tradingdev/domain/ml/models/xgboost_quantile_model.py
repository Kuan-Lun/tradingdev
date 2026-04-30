"""XGBoost quantile regression model for return distribution prediction.

Trains one XGBRegressor per quantile using ``reg:quantileerror`` objective
to predict conditional quantiles of future log returns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import xgboost as xgb

from tradingdev.domain.ml.base import BaseModel
from tradingdev.utils.logger import setup_logger

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

logger = setup_logger(__name__)


class XGBoostQuantileModel(BaseModel):
    """Multi-quantile XGBoost regression model.

    For each quantile in ``quantiles``, trains a separate
    ``XGBRegressor`` with ``objective="reg:quantileerror"``
    and ``quantile_alpha=q``.
    """

    def __init__(
        self,
        quantiles: list[float] | None = None,
        horizon: int = 30,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        feature_names: list[str] | None = None,
    ) -> None:
        self._quantiles = quantiles or [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        self._horizon = horizon
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._subsample = subsample
        self._colsample_bytree = colsample_bytree
        self._random_state = random_state
        self._feature_names = feature_names or []
        self._models: dict[float, xgb.XGBRegressor] = {}

    @property
    def quantiles(self) -> list[float]:
        """Quantile levels being predicted."""
        return list(self._quantiles)

    @property
    def is_trained(self) -> bool:
        """Whether all quantile models have been trained."""
        return len(self._models) == len(self._quantiles)

    def train(
        self,
        df: pd.DataFrame,
        eval_df: pd.DataFrame | None = None,
        subsample_step: int = 1,
    ) -> None:
        """Train one XGBRegressor per quantile.

        Args:
            df: Training DataFrame with feature columns and ``target``.
            eval_df: Optional evaluation set for early stopping.
            subsample_step: Take every Nth row for training to reduce
                overlap between consecutive targets.  1 = use all rows.
        """
        if not self._feature_names:
            msg = "feature_names must be set before training"
            raise ValueError(msg)

        train_slice = df.iloc[::subsample_step] if subsample_step > 1 else df
        x_train = train_slice[self._feature_names].values
        y_train = train_slice["target"].values

        eval_set: list[tuple[Any, Any]] | None = None
        if eval_df is not None:
            x_eval = eval_df[self._feature_names].values
            y_eval = eval_df["target"].values
            eval_set = [(x_eval, y_eval)]

        for q in self._quantiles:
            model = xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=q,
                n_estimators=self._n_estimators,
                max_depth=self._max_depth,
                learning_rate=self._learning_rate,
                subsample=self._subsample,
                colsample_bytree=self._colsample_bytree,
                random_state=self._random_state,
                verbosity=0,
            )
            if eval_set is not None:
                model.fit(
                    x_train,
                    y_train,
                    eval_set=eval_set,
                    verbose=False,
                )
            else:
                model.fit(x_train, y_train, verbose=False)
            self._models[q] = model

        logger.info(
            "Trained %d quantile models (horizon=%d, n=%d)",
            len(self._quantiles),
            self._horizon,
            len(x_train),
        )

    def predict(self, df: pd.DataFrame) -> pd.Series[float]:
        """Return median (Q50) prediction.

        Args:
            df: DataFrame with feature columns.

        Returns:
            Series of median predictions.
        """
        if 0.50 not in self._models:
            msg = "Model not trained yet"
            raise RuntimeError(msg)
        x = df[self._feature_names].values
        preds = self._models[0.50].predict(x)
        return pd.Series(preds, index=df.index, name="q50")

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Not applicable for regression; returns predict_quantiles instead."""
        return self.predict_quantiles(df)

    def predict_quantiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return all quantile predictions.

        Args:
            df: DataFrame with feature columns.

        Returns:
            DataFrame with columns ``q05``, ``q10``, ..., ``q95``.
        """
        if not self.is_trained:
            msg = "Model not trained yet"
            raise RuntimeError(msg)

        x = df[self._feature_names].values
        result: dict[str, npt.NDArray[np.floating[Any]]] = {}
        for q in self._quantiles:
            col_name = f"q{int(q * 100):02d}"
            result[col_name] = self._models[q].predict(x)

        return pd.DataFrame(result, index=df.index)
