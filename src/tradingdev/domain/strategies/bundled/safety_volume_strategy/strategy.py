"""Safety-first volume strategy with risk-gated entry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics import roc_auc_score

from tradingdev.domain import indicators
from tradingdev.domain.ml.features.context import FeatureContext
from tradingdev.domain.ml.features.risk_features import RiskFeatureEngineer
from tradingdev.domain.ml.models.xgboost_model import XGBoostDirectionModel
from tradingdev.domain.ml.probabilities import class_probability, validate_probabilities
from tradingdev.domain.ml.rolling_gen import (
    resolve_geometry,
    rolling_outcome,
    trace_payload,
)
from tradingdev.domain.optimization.grid_search import finite_metric_value
from tradingdev.domain.strategies.base import BaseStrategy
from tradingdev.domain.strategies.bundled.safety_volume_strategy.risk_gate import (
    RiskDecision,
    evaluate_risk,
)
from tradingdev.shared.utils.logger import setup_logger

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd

    from tradingdev.domain.backtest.base_engine import (
        BaseBacktestEngine,
    )
    from tradingdev.domain.ml.features.features import FeatureEngineer
    from tradingdev.domain.strategies.bundled.safety_volume_strategy.config import (
        SafetyVolumeStrategyConfig,
    )

logger = setup_logger(__name__)

_DIRECTION_INPUT_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


def _direction_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Project a frame onto the OHLCV columns direction features are built from.

    Direction features read prices and volume only, so any other column is
    neither an input nor allowed to collide with an engineered name.
    """
    columns = [name for name in _DIRECTION_INPUT_COLUMNS if name in df.columns]
    return df.loc[:, columns]


@dataclass(frozen=True)
class _MLDirection:
    """A direction feature pipeline together with the model fitted on it.

    Keeping the pair together ties a model to the exact feature contract it was
    trained on, so no call site can score it through another pipeline.
    """

    features: FeatureEngineer
    model: XGBoostDirectionModel


class SafetyVolumeStrategy(BaseStrategy):
    """Risk-gated volume strategy that minimises loss.

    Three-layer architecture:

    1. **Risk Gate** (XGBoost): predicts whether it is safe to
       open a position now.  Only enters when ``P(safe) >= threshold``.
    2. **Direction** (SMA crossover or optional ML): determines
       long vs short once the Risk Gate approves entry.
    3. **Holding Manager** (state machine): enforces minimum /
       maximum holding time, risk-based exits, and position reversals.
    """

    def __init__(
        self,
        config: SafetyVolumeStrategyConfig,
        backtest_engine: BaseBacktestEngine | None = None,
    ) -> None:
        self._config = config
        self._backtest_engine = backtest_engine

        # Populated by fit()
        self._risk_model: XGBoostDirectionModel | None = None
        self._risk_fe: RiskFeatureEngineer | None = None
        self._best_lookback: int | None = None
        self._best_threshold: float = config.risk_threshold
        self._train_data: pd.DataFrame | None = None

        # Optional ML direction model
        self._dir_model: XGBoostDirectionModel | None = None
        self._dir_fe: FeatureEngineer | None = None

        self._rolling_trace: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> None:
        """Train the Risk Gate model and optionally the direction model.

        Raises:
            ValueError: If a threshold search runs on a frame whose index
                labels are not unique, because the scored validation rows could
                not be mapped back to the bars their directions come from.
        """
        # Validate every candidate's geometry before any model is trained.
        scheme = (
            self._config.rolling_gen.resolve()
            if self._config.rolling_gen.enabled
            else None
        )
        for lookback in self._config.lookback_candidates:
            resolve_geometry(
                scheme,
                lookback=lookback,
                retrain_interval=self._config.retrain_interval,
                context=FeatureContext.for_features(
                    RiskFeatureEngineer(lookback=lookback)
                ),
                history_bars=len(df),
                label_horizon=self._config.target_holding_bars,
            )
        val_ratio = self._config.validation_ratio
        split_idx = int(len(df) * (1 - val_ratio))
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        best_lb, best_model, best_fe = self._search_lookback(
            train_df,
            val_df,
        )
        self._best_lookback = best_lb

        # --- Threshold search ---
        # Each fit decides its own threshold: a fit whose lookback search
        # qualifies nothing must not trade on the previous fold's value.
        self._best_threshold = self._config.risk_threshold
        if self._config.risk_threshold_candidates and best_model:
            direction_source = self._search_direction_source(train_df, best_lb)
            self._best_threshold = self._search_threshold(
                val_df,
                df,
                best_model,
                best_fe,
                direction_source,
            )

        # --- Retrain final risk model on full data ---
        self._risk_fe = RiskFeatureEngineer(
            lookback=best_lb,
            target_holding_bars=self._config.target_holding_bars,
            fee_rate=self._config.fee_rate,
            max_acceptable_loss_pct=self._config.max_acceptable_loss_pct,
        )
        feat_full = self._risk_fe.transform(df, include_target=True)
        self._risk_model = XGBoostDirectionModel(
            config=self._config.risk_model,
        )
        self._risk_model.train(feat_full)
        self._train_data = df.copy()

        # --- Optional ML direction model ---
        if self._config.use_ml_direction:
            self._fit_direction_model(df)

        logger.info(
            "SafetyVolume fit complete: lookback=%d, threshold=%.2f",
            best_lb,
            self._best_threshold,
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate position-state signals via the 3-layer architecture."""
        if (
            self._risk_model is None
            or self._risk_fe is None
            or self._best_lookback is None
            or self._train_data is None
        ):
            msg = "Strategy not fitted. Call fit() first."
            raise RuntimeError(msg)

        self._rolling_trace = None
        try:
            risk_scores = self._predict_risk_rolling(df)
            directions = self._compute_directions(df)
            signals = self._run_state_machine(risk_scores, directions)
        except Exception:
            # A successful risk pass alone is not evidence of a successful run.
            self._rolling_trace = None
            raise

        result = df.copy()
        result["signal"] = signals
        return result

    def get_parameters(self) -> dict[str, Any]:
        """Return strategy parameters."""
        params = self._config.model_dump()
        params["best_lookback"] = self._best_lookback
        params["best_threshold"] = self._best_threshold
        return params

    # ------------------------------------------------------------------
    # Lookback search
    # ------------------------------------------------------------------

    def _search_lookback(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> tuple[
        int,
        XGBoostDirectionModel | None,
        RiskFeatureEngineer | None,
    ]:
        """Find the best lookback by validation AUC-ROC."""
        best_score = -1.0
        best_lb = self._config.lookback_candidates[0]
        best_model: XGBoostDirectionModel | None = None
        best_fe: RiskFeatureEngineer | None = None

        for lb in self._config.lookback_candidates:
            fe = RiskFeatureEngineer(
                lookback=lb,
                target_holding_bars=self._config.target_holding_bars,
                fee_rate=self._config.fee_rate,
                max_acceptable_loss_pct=(self._config.max_acceptable_loss_pct),
            )
            feat_train = fe.transform(train_df, include_target=True)
            feat_val = fe.transform(val_df, include_target=True)

            if len(feat_train) < 10 or len(feat_val) < 5:
                continue

            # Skip if only one class present in training data
            train_classes = feat_train["target"].nunique()
            if train_classes < 2:
                logger.warning(
                    "Lookback %d: only %d class in train",
                    lb,
                    train_classes,
                )
                continue

            model = XGBoostDirectionModel(
                config=self._config.risk_model,
            )
            model.train(feat_train, eval_df=feat_val)

            # Use AUC-ROC (handles class imbalance better)
            proba = validate_probabilities(model.predict_proba(feat_val), feat_val)
            y_true = np.asarray(feat_val["target"].values)
            p_safe = class_probability(proba, 1).to_numpy()

            unique_classes = np.unique(y_true)
            if len(unique_classes) < 2:
                logger.warning(
                    "Lookback %d: only one class in val set",
                    lb,
                )
                continue

            score = float(roc_auc_score(y_true, p_safe))
            logger.info(
                "Lookback %d: val AUC-ROC=%.4f",
                lb,
                score,
            )

            if score > best_score:
                best_score = score
                best_lb = lb
                best_model = model
                best_fe = fe

        return best_lb, best_model, best_fe

    # ------------------------------------------------------------------
    # Threshold search
    # ------------------------------------------------------------------

    def _search_direction_source(
        self,
        train_df: pd.DataFrame,
        lookback: int,
    ) -> _MLDirection | None:
        """Decide the direction mode this fit's threshold search scores with.

        The declared configuration decides the mode, never the model an earlier
        fit left behind.  With ML direction enabled the model is fitted on this
        fit's training split alone, so the validation rows the threshold is
        chosen on stay unseen and cannot be rewarded by a model that already
        memorised them.  ``fit`` then refits on the full frame for live signals,
        the same way the risk gate is retrained after its own search.
        """
        if not self._config.use_ml_direction:
            return None
        return self._train_ml_direction(train_df, lookback)

    def _validation_directions(
        self,
        feat_val: pd.DataFrame,
        context_df: pd.DataFrame,
        source: _MLDirection | None,
    ) -> npt.NDArray[np.floating[Any]]:
        """Directions for the scored validation bars, computed with their history.

        Scoring the validation slice on its own spends the whole window on
        feature warm-up: the risk gate already dropped its first ``lookback``
        bars, and the direction pipeline needs as many again.  The search would
        then compare candidates over directions that are all zero and call the
        most permissive one best.  ``context_df`` supplies the bars that precede
        the scored rows; only those rows are returned.
        """
        if not context_df.index.is_unique:
            msg = (
                "fit() needs unique index labels to align the threshold search "
                "with the validation rows"
            )
            raise ValueError(msg)
        positions = context_df.index.get_indexer(feat_val.index)
        if (positions < 0).any():
            msg = "Validation rows must come from the frame passed to fit()"
            raise ValueError(msg)
        directions = (
            self._compute_sma_directions(context_df)
            if source is None
            else self._ml_directions(context_df, source)
        )
        return directions[positions]

    def _search_threshold(
        self,
        val_df: pd.DataFrame,
        context_df: pd.DataFrame,
        model: XGBoostDirectionModel,
        fe: RiskFeatureEngineer | None,
        direction_source: _MLDirection | None,
    ) -> float:
        """Find the threshold that minimises loss while meeting volume.

        ``direction_source`` states the direction mode the candidates are scored
        under: ``None`` is the SMA crossover, a trained source is ML direction.
        It is resolved for this fit rather than read from the strategy, so a
        leftover model cannot silently change how the candidates are compared.
        """
        if fe is None or self._backtest_engine is None:
            return self._config.risk_threshold

        candidates = sorted(
            self._config.risk_threshold_candidates or [],
        )
        feat_val = fe.transform(val_df, include_target=False)
        proba = validate_probabilities(model.predict_proba(feat_val), feat_val)
        p_safe = class_probability(proba, 1).to_numpy()

        directions = self._validation_directions(
            feat_val,
            context_df,
            direction_source,
        )
        if not np.any(directions):
            logger.warning(
                "Threshold search skipped: no validation bar has a direction",
            )
            return self._config.risk_threshold

        best_threshold = self._config.risk_threshold
        best_pnl: float | None = None

        for thr in candidates:
            signals_for_thr = self._run_state_machine_with_threshold(
                p_safe,
                directions,
                thr,
                self._config.min_holding_bars,
                self._config.max_holding_bars,
            )
            sim_df = feat_val.copy()
            sim_df["signal"] = signals_for_thr
            result = self._backtest_engine.run(sim_df)

            total_pnl = finite_metric_value(result.metrics.get("total_pnl"))
            volume = result.metrics.get("total_volume", 0.0)

            logger.info(
                "Threshold %.2f: net_pnl=%s, volume=%s",
                thr,
                total_pnl,
                volume,
            )

            if total_pnl is not None and (best_pnl is None or total_pnl > best_pnl):
                best_pnl = total_pnl
                best_threshold = thr

        logger.info(
            "Best threshold: %.2f (net_pnl=%s)",
            best_threshold,
            best_pnl,
        )
        return best_threshold

    # ------------------------------------------------------------------
    # Risk prediction (rolling retrain)
    # ------------------------------------------------------------------

    def _predict_risk_rolling(
        self,
        df: pd.DataFrame,
    ) -> npt.NDArray[np.floating[Any]]:
        """Predict P(safe) for each bar with periodic retraining.

        Raises:
            ValueError: If the frames have no usable time identity, because the
                overlap with the training history cannot be detected without it.
        """
        assert self._risk_fe is not None
        assert self._risk_model is not None
        assert self._train_data is not None

        self._rolling_trace = None
        outcome = rolling_outcome(
            self._train_data,
            df,
            self._config.rolling_gen.resolve()
            if self._config.rolling_gen.enabled
            else None,
            model=self._risk_model,
            model_config=self._config.risk_model,
            fe=self._risk_fe,
            lookback=self._risk_fe.lookback,
            retrain_interval=self._config.retrain_interval,
            label_horizon=self._config.target_holding_bars,
        )
        self._rolling_trace = trace_payload(outcome)
        return class_probability(outcome.probabilities, 1).to_numpy(dtype=np.float64)

    def rolling_trace(self) -> dict[str, Any] | None:
        """Return the resolved geometry and provenance of the last successful run."""
        return self._rolling_trace

    # ------------------------------------------------------------------
    # Direction signal
    # ------------------------------------------------------------------

    def _compute_directions(
        self,
        df: pd.DataFrame,
    ) -> npt.NDArray[np.floating[Any]]:
        """Compute direction signal (1=long, -1=short) per bar."""
        if self._config.use_ml_direction:
            return self._compute_ml_directions(df)
        return self._compute_sma_directions(df)

    def _compute_sma_directions(
        self,
        df: pd.DataFrame,
    ) -> npt.NDArray[np.floating[Any]]:
        """SMA crossover direction: fast > slow → long."""
        close = df["close"].astype(float)
        sma_fast = indicators.sma(close, self._config.sma_fast)
        sma_slow = indicators.sma(close, self._config.sma_slow)
        direction = np.where(sma_fast > sma_slow, 1, -1)
        # NaN region at start → no direction
        nan_mask = np.asarray(sma_slow.isna().values)
        direction[nan_mask] = 0
        return np.asarray(direction, dtype=np.float64)

    def _compute_ml_directions(
        self,
        df: pd.DataFrame,
    ) -> npt.NDArray[np.floating[Any]]:
        """Select the most probable direction from causal rolling predictions."""
        if self._dir_model is None or self._dir_fe is None or self._train_data is None:
            raise RuntimeError("ML direction is not fitted. Call fit() first.")
        outcome = rolling_outcome(
            _direction_inputs(self._train_data),
            _direction_inputs(df),
            self._config.rolling_gen.resolve()
            if self._config.rolling_gen.enabled
            else None,
            model=self._dir_model,
            model_config=self._config.direction_model,
            fe=self._dir_fe,
            lookback=self._dir_fe.lookback,
            retrain_interval=self._config.retrain_interval,
            label_horizon=self._dir_fe.target_horizon,
        )
        if self._rolling_trace is not None:
            self._rolling_trace["direction"] = trace_payload(outcome)
        available = ~outcome.probabilities.isna().all(axis=1)
        directions = np.zeros(len(df), dtype=np.float64)
        directions[available.to_numpy()] = (
            outcome.probabilities.loc[available].idxmax(axis=1).to_numpy(dtype=float)
        )
        return directions

    def _ml_directions(
        self,
        df: pd.DataFrame,
        source: _MLDirection,
    ) -> npt.NDArray[np.floating[Any]]:
        """Score one direction per row with an explicit direction source.

        Only the OHLCV columns the pipeline was fitted on reach it.  A frame
        that already carries engineered columns - the risk gate's validation
        matrix, for instance - would otherwise contribute a second copy of every
        shared name, and XGBoost rejects duplicate columns.
        """
        out = np.zeros(len(df), dtype=np.float64)
        if df.empty:
            return out
        # Preserve source positions through dropna, including interior gaps.
        # The copied RangeIndex is metadata, never an input feature for the model.
        feat = source.features.transform(
            _direction_inputs(df).reset_index(drop=True),
            include_target=False,
        )
        if feat.empty:
            return out
        preds = source.model.predict(feat)
        if len(preds) != len(feat) or not preds.index.equals(feat.index):
            msg = (
                "Direction predictions do not match the feature rows: "
                f"{len(preds)} predictions for {len(feat)} rows; "
                "row count and source indices must match"
            )
            raise RuntimeError(msg)
        out[feat.index.to_numpy(dtype=np.intp)] = preds.to_numpy(dtype=float)
        return out

    def _fit_direction_model(
        self,
        df: pd.DataFrame,
    ) -> None:
        """Train the optional ML direction model that scores live signals."""
        source = self._train_ml_direction(df, self._best_lookback or 720)
        self._dir_fe = source.features
        self._dir_model = source.model

    def _train_ml_direction(
        self,
        df: pd.DataFrame,
        lookback: int,
    ) -> _MLDirection:
        """Fit a direction pipeline on exactly the rows it is given."""
        from tradingdev.domain.ml.features.features import FeatureEngineer

        features = FeatureEngineer(
            lookback=lookback,
            target_horizon=1,
        )
        feat = features.transform(
            _direction_inputs(df),
            include_target=True,
        )
        model = XGBoostDirectionModel(
            config=self._config.direction_model,
        )
        model.train(feat)
        return _MLDirection(features=features, model=model)

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    @staticmethod
    def _run_state_machine_with_threshold(
        risk_scores: npt.NDArray[np.floating[Any]],
        directions: npt.NDArray[np.floating[Any]],
        threshold: float,
        min_hold: int = 5,
        max_hold: int = 30,
    ) -> npt.NDArray[np.int_]:
        """State machine that produces position-state signals.

        Returns an array of (1, -1, 0) where:
        - 1 = long position
        - -1 = short position
        - 0 = flat (no position)
        """
        decisions = evaluate_risk(risk_scores, threshold)
        n = len(decisions)
        signals: npt.NDArray[np.int_] = np.zeros(n, dtype=int)
        state = 0
        bars_in_pos = 0

        for i in range(n):
            decision = decisions[i]
            d = int(directions[i])

            if state == 0:
                # FLAT: enter if safe and have direction
                if decision == RiskDecision.ALLOW and d != 0:
                    state = d
                    bars_in_pos = 0
            else:
                bars_in_pos += 1

                if bars_in_pos < min_hold:
                    pass  # hold through min period
                elif bars_in_pos >= max_hold:
                    state = 0  # forced exit
                elif decision != RiskDecision.ALLOW:
                    state = 0  # blocked or unavailable after the minimum hold
                elif d != 0 and d != state:
                    # direction reversal → generates 2x volume
                    state = d
                    bars_in_pos = 0

            signals[i] = state

        return signals

    def _run_state_machine(
        self,
        risk_scores: npt.NDArray[np.floating[Any]],
        directions: npt.NDArray[np.floating[Any]],
    ) -> npt.NDArray[np.int_]:
        """Run the state machine with configured parameters."""
        return self._run_state_machine_with_threshold(
            risk_scores,
            directions,
            self._best_threshold,
            self._config.min_holding_bars,
            self._config.max_holding_bars,
        )
