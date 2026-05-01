"""AutoGluon-based direction prediction model."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from tradingdev.domain.ml.base import BaseModel
from tradingdev.shared.utils.logger import setup_logger

logger = setup_logger(__name__)

_EXCLUDE_COLS = frozenset(
    {
        "target",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dvol",
        "dvol_open",
        "dvol_high",
        "dvol_low",
        "dvol_close",
    }
)


class AutoGluonDirectionModel(BaseModel):
    """AutoGluon TabularPredictor wrapper for direction prediction.

    Automatically selects and ensembles models (LightGBM, CatBoost,
    XGBoost, neural networks, etc.) for binary direction prediction.
    """

    def __init__(
        self,
        time_limit: int = 300,
        presets: str = "medium_quality",
        eval_metric: str = "accuracy",
        verbosity: int = 0,
        num_cpus: int | None = None,
        random_seed: int | None = 42,
    ) -> None:
        self._time_limit = time_limit
        self._presets = presets
        self._eval_metric = eval_metric
        self._verbosity = verbosity
        self._num_cpus = num_cpus
        self._random_seed = random_seed
        self._predictor: Any = None  # TabularPredictor
        self._feature_names: list[str] = []
        self._model_path: Path | None = None

    def train(
        self,
        df: pd.DataFrame,
        eval_df: pd.DataFrame | None = None,
    ) -> None:
        """Train AutoGluon on feature DataFrame.

        Args:
            df: Training DataFrame with feature columns and ``target``.
            eval_df: Optional tuning DataFrame (AutoGluon handles
                internal validation automatically, so this is optional).
        """
        if self._random_seed is not None:
            from autogluon.common.utils.utils import seed_everything

            seed_everything(self._random_seed)
        from autogluon.tabular import TabularPredictor

        feature_cols = [c for c in df.columns if c not in _EXCLUDE_COLS]
        self._feature_names = feature_cols

        train_data = df[feature_cols + ["target"]].copy()

        # AutoGluon writes model artifacts to disk
        self._model_path = Path(
            tempfile.mkdtemp(prefix="autogluon_direction_"),
        )

        self._predictor = TabularPredictor(
            label="target",
            eval_metric=self._eval_metric,
            path=str(self._model_path),
            verbosity=self._verbosity,
        )

        fit_kwargs: dict[str, Any] = {
            "time_limit": self._time_limit,
            "presets": self._presets,
        }
        if self._num_cpus is not None:
            fit_kwargs["num_cpus"] = self._num_cpus
        if eval_df is not None:
            tuning_data = eval_df[feature_cols + ["target"]].copy()
            fit_kwargs["tuning_data"] = tuning_data

        self._predictor.fit(train_data, **fit_kwargs)

        leaderboard = self._predictor.leaderboard(silent=True)
        best_model = leaderboard.iloc[0]["model"]
        best_score = leaderboard.iloc[0]["score_val"]
        n_models = len(leaderboard)
        logger.info(
            "AutoGluon trained: %d models, best=%s (val_score=%.4f), "
            "%d features, %d samples",
            n_models,
            best_model,
            best_score,
            len(feature_cols),
            len(train_data),
        )

    def predict(self, df: pd.DataFrame) -> pd.Series[float]:
        """Predict direction class (0=down, 1=up).

        Returns:
            Series of predicted classes.
        """
        if self._predictor is None:
            msg = "Model has not been trained. Call train() first."
            raise RuntimeError(msg)

        x = df[self._feature_names]
        preds = self._predictor.predict(x)
        return pd.Series(preds.values, index=df.index, dtype=float)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return class probabilities.

        Returns:
            DataFrame with columns for each class probability.
        """
        if self._predictor is None:
            msg = "Model has not been trained. Call train() first."
            raise RuntimeError(msg)

        x = df[self._feature_names]
        result: pd.DataFrame = self._predictor.predict_proba(x)
        return result

    def cleanup(self) -> None:
        """Remove temporary model artifacts from disk."""
        if self._model_path is not None and self._model_path.exists():
            shutil.rmtree(self._model_path, ignore_errors=True)
            self._model_path = None

    def get_parameters(self) -> dict[str, Any]:
        """Return model configuration."""
        params: dict[str, Any] = {
            "time_limit": self._time_limit,
            "presets": self._presets,
            "eval_metric": self._eval_metric,
        }
        if self._num_cpus is not None:
            params["num_cpus"] = self._num_cpus
        if self._random_seed is not None:
            params["random_seed"] = self._random_seed
        return params
