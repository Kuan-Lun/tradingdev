"""XGBoost-based ML trading strategy with rolling retraining."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd  # noqa: TC002
from sklearn.metrics import accuracy_score

from quant_backtest.ml.features import FeatureEngineer
from quant_backtest.ml.xgboost_model import XGBoostDirectionModel
from quant_backtest.strategies.base import BaseStrategy
from strategies.rolling_retrainer import (
    RollingRetrainer,
)
from strategies.threshold_optimizer import (
    ThresholdOptimizer,
)
from quant_backtest.utils.logger import setup_logger

if TYPE_CHECKING:
    from quant_backtest.backtest.base_engine import (
        BaseBacktestEngine,
    )
    from quant_backtest.data.schemas import (
        XGBoostStrategyConfig,
    )

logger = setup_logger(__name__)


class XGBoostStrategy(BaseStrategy):
    """XGBoost direction prediction strategy."""

    def __init__(
        self,
        config: XGBoostStrategyConfig,
        backtest_engine: BaseBacktestEngine | None = None,
    ) -> None:
        self._config = config
        self._backtest_engine = backtest_engine
        self._model: XGBoostDirectionModel | None = None
        self._feature_engineer: FeatureEngineer | None = None
        self._best_lookback: int | None = None
        self._best_threshold: float = config.signal_threshold
        self._train_data: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame) -> None:
        """Optimise lookback and train the XGBoost model."""
        val_ratio = self._config.validation_ratio
        split_idx = int(len(df) * (1 - val_ratio))
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        best_lookback, best_model, best_fe = self._search_lookback(train_df, val_df)

        self._best_lookback = best_lookback
        self._feature_engineer = FeatureEngineer(
            lookback=best_lookback,
            target_horizon=self._config.target_horizon,
        )

        # Threshold search (volume-maximisation mode)
        if (
            self._config.signal_threshold_candidates
            and best_model is not None
            and best_fe is not None
            and self._backtest_engine is not None
        ):
            optimizer = ThresholdOptimizer(
                engine=self._backtest_engine,
                min_bars_between_trades=(self._config.min_bars_between_trades),
            )
            self._best_threshold = optimizer.search(
                val_df,
                best_model,
                self._config.signal_threshold_candidates,
                self._config.signal_threshold,
                feature_engineer=best_fe,
            )

        # Retrain final model on full data
        feat_full = self._feature_engineer.transform(df, include_target=True)
        self._model = XGBoostDirectionModel(config=self._config.model)
        self._model.train(feat_full)
        self._train_data = df.copy()

        logger.info(
            "XGBoost fit complete: best_lookback=%d, threshold=%.2f",
            best_lookback,
            self._best_threshold,
        )

    def _search_lookback(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> tuple[
        int,
        XGBoostDirectionModel | None,
        FeatureEngineer | None,
    ]:
        """Find the best lookback by validation accuracy."""
        best_score = -1.0
        best_lb = self._config.lookback_candidates[0]
        best_model: XGBoostDirectionModel | None = None
        best_fe: FeatureEngineer | None = None

        for lb in self._config.lookback_candidates:
            fe = FeatureEngineer(
                lookback=lb,
                target_horizon=self._config.target_horizon,
            )
            feat_train = fe.transform(train_df, include_target=True)
            feat_val = fe.transform(val_df, include_target=True)

            if len(feat_train) < 10 or len(feat_val) < 5:
                continue

            model = XGBoostDirectionModel(config=self._config.model)
            model.train(feat_train, eval_df=feat_val)

            preds = model.predict(feat_val)
            score = float(accuracy_score(feat_val["target"], preds))
            logger.info(
                "Lookback %d: val accuracy=%.4f",
                lb,
                score,
            )

            if score > best_score:
                best_score = score
                best_lb = lb
                best_model = model
                best_fe = fe

        return best_lb, best_model, best_fe

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals via rolling prediction."""
        if (
            self._model is None
            or self._feature_engineer is None
            or self._best_lookback is None
            or self._train_data is None
        ):
            msg = "Strategy not fitted. Call fit() first."
            raise RuntimeError(msg)

        retrainer = RollingRetrainer(
            model_config=self._config.model,
            retrain_interval=self._config.retrain_interval,
            threshold=self._best_threshold,
            cooldown=self._config.min_bars_between_trades,
            lookback=self._best_lookback,
        )
        signals = retrainer.run(
            df,
            self._train_data,
            self._model,
            self._feature_engineer,
        )

        result = df.copy()
        result["signal"] = signals
        return result

    def get_parameters(self) -> dict[str, Any]:
        """Return strategy parameters."""
        params = self._config.model_dump()
        params["best_lookback"] = self._best_lookback
        params["best_threshold"] = self._best_threshold
        return params
