"""One rolling execution core for risk and direction prediction.

Feature requirements and resolved geometry determine every window. Strategy
consumers decide what class probabilities mean and what unavailable bars do.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd

from tradingdev.domain.ml.features.context import FeatureContext, FeatureTransformer
from tradingdev.domain.ml.models.xgboost_model import XGBoostDirectionModel
from tradingdev.domain.ml.probabilities import class_probability, validate_probabilities
from tradingdev.domain.ml.rolling_gen.schedule import RollWindow, enumerate_rolls
from tradingdev.domain.ml.rolling_gen.scheme import (
    RollingGeometry,
    RollingScheme,
    resolve_geometry,
)
from tradingdev.domain.ml.scoring_history import align_scoring_history
from tradingdev.shared.utils.logger import setup_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from tradingdev.domain.ml.schemas import XGBoostModelConfig

logger = setup_logger(__name__)


class UnavailableReason(StrEnum):
    HISTORY = "insufficient_history"
    FEATURES = "incomplete_features"
    MODEL = "no_causal_model"


@dataclass(frozen=True)
class RollTrace:
    """The scheduled window and what its refit and scoring actually did."""

    window: RollWindow
    n_train_rows: int
    trained: bool
    bars_scored: int


@dataclass(frozen=True)
class RollingOutcome:
    """Class-labelled probabilities aligned to the original scored frame.

    Missing rows are NaN in every class, with an explicit unavailable reason.
    A class absent from a fitted model has zero probability on a scored row.
    """

    probabilities: pd.DataFrame
    unavailable: tuple[UnavailableReason | None, ...]
    bar_roll: tuple[int, ...]
    rolls: tuple[RollTrace, ...]
    geometry: RollingGeometry
    label_horizon: int
    initial_model_available: bool


def trace_payload(outcome: RollingOutcome) -> dict[str, object]:
    """Persist effective geometry, roll attribution and unavailable counts."""
    return {
        "scheme": outcome.geometry.name,
        "rtype": outcome.geometry.rtype.value,
        "step": outcome.geometry.step,
        "window_size": outcome.geometry.window_size,
        "min_train_bars": outcome.geometry.min_train_bars,
        "predict_lookback": outcome.geometry.predict_lookback,
        "label_horizon": outcome.label_horizon,
        "initial_model_available": outcome.initial_model_available,
        "bars": len(outcome.bar_roll),
        "bars_scored": sum(reason is None for reason in outcome.unavailable),
        "unscored_bars": [
            bar for bar, reason in enumerate(outcome.unavailable) if reason is not None
        ],
        "unavailable_counts": dict(
            Counter(
                reason.value for reason in outcome.unavailable if reason is not None
            )
        ),
        "rolls": [
            {
                "index": trace.window.index,
                "retrain_bar": trace.window.retrain_bar,
                "train_start": trace.window.train_start,
                "train_end": trace.window.train_end,
                "test_start": trace.window.test_start,
                "test_end": trace.window.test_end,
                "n_train_rows": trace.n_train_rows,
                "trained": trace.trained,
                "bars_scored": trace.bars_scored,
            }
            for trace in outcome.rolls
        ],
    }


@dataclass(frozen=True)
class _Retrainer:
    combined: pd.DataFrame
    geometry: RollingGeometry
    fe: FeatureTransformer
    context: FeatureContext
    model_config: XGBoostModelConfig
    model_factory: Callable[[XGBoostModelConfig], XGBoostDirectionModel]
    label_horizon: int

    def retrain(
        self, window: RollWindow, fallback: XGBoostDirectionModel | None
    ) -> tuple[XGBoostDirectionModel | None, int, bool]:
        """Use only mature labels preceding this roll; never stretch its window."""
        source = self.combined.iloc[window.train_start : window.train_end].copy()
        if len(source) < self.context.minimum_rows + self.label_horizon:
            return fallback, 0, False
        labelled = self.fe.transform(source, include_target=True)
        if not labelled.index.is_unique or not labelled.index.isin(source.index).all():
            raise RuntimeError("Training features lost their source row identity")
        if "timestamp" not in labelled.columns or not labelled["timestamp"].equals(
            source.loc[labelled.index, "timestamp"]
        ):
            raise RuntimeError("Training features lost their source time identity")
        # Enforce maturity even if a transformer incorrectly fills its label tail.
        labelled = labelled.loc[labelled.index + self.label_horizon < window.train_end]
        count = len(labelled)
        if count < self.geometry.min_train_bars:
            return fallback, count, False
        model = self.model_factory(self.model_config)
        model.train(labelled)
        return model, count, True

    def score(
        self,
        position: int,
        model: XGBoostDirectionModel,
    ) -> tuple[pd.DataFrame | None, UnavailableReason | None]:
        start, stop = self.geometry.predict_context(position)
        source = self.combined.iloc[start:stop].copy()
        if len(source) < self.context.minimum_rows:
            return None, UnavailableReason.HISTORY
        features = self.fe.transform(source, include_target=False)
        if not features.index.is_unique or not features.index.isin(source.index).all():
            raise RuntimeError("Prediction features lost their source row identity")
        if position not in features.index:
            return None, UnavailableReason.FEATURES
        row = features.loc[[position]]
        if "timestamp" not in row.columns or not row["timestamp"].equals(
            source.loc[[position], "timestamp"]
        ):
            raise RuntimeError("Prediction features lost their source time identity")
        proba = validate_probabilities(model.predict_proba(row), row)
        if not proba.columns.isin(self.fe.target_classes).all():
            raise RuntimeError(
                "Model probabilities contain classes outside the feature target"
            )
        return proba, None


def rolling_outcome(
    train_data: pd.DataFrame,
    test_df: pd.DataFrame,
    scheme: RollingScheme | None,
    *,
    model: XGBoostDirectionModel,
    model_config: XGBoostModelConfig,
    fe: FeatureTransformer,
    lookback: int,
    retrain_interval: int,
    label_horizon: int,
    model_factory: Callable[[XGBoostModelConfig], XGBoostDirectionModel] | None = None,
) -> RollingOutcome:
    """Score using one resolved schedule; None selects automatic geometry.

    An overlapping replay discards the initial model and waits for a causal
    refit. This does not undo full-fold parameter selection: training replay
    remains an in-sample diagnostic. Entirely unscored non-empty runs fail.
    """
    history, scored = align_scoring_history(train_data, test_df)
    context = FeatureContext.for_features(fe)
    geometry = resolve_geometry(
        scheme,
        lookback=lookback,
        retrain_interval=retrain_interval,
        context=context,
        history_bars=len(history),
        label_horizon=label_horizon,
    )
    schedule = enumerate_rolls(
        geometry, n_scored_bars=len(scored), history_bars=len(history)
    )
    combined = pd.concat([history, scored], ignore_index=True)
    initial_available = bool(len(history) and len(history) == len(train_data))
    current_model = model if initial_available else None
    model_was_available = current_model is not None
    retrainer = _Retrainer(
        combined,
        geometry,
        fe,
        context,
        model_config,
        XGBoostDirectionModel if model_factory is None else model_factory,
        label_horizon,
    )
    classes = fe.target_classes
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("Feature target classes must be nonempty and unique")
    values = np.full((len(scored), len(classes)), np.nan, dtype=np.float64)
    unavailable: list[UnavailableReason | None] = [UnavailableReason.MODEL] * len(
        scored
    )
    bar_roll = [-1] * len(scored)
    traces: list[RollTrace] = []
    for window in schedule:
        current_model, n_train_rows, trained = retrainer.retrain(window, current_model)
        scored_count = 0
        for bar in range(window.test_start, window.test_end):
            if current_model is None:
                continue
            model_was_available = True
            proba, reason = retrainer.score(len(history) + bar, current_model)
            unavailable[bar] = reason
            if proba is not None:
                values[bar] = (
                    proba.reindex(columns=list(classes), fill_value=0.0)
                    .iloc[0]
                    .to_numpy(dtype=float)
                )
                bar_roll[bar] = window.index
                scored_count += 1
        traces.append(RollTrace(window, n_train_rows, trained, scored_count))
        logger.info(
            "[rolling:%s] roll %d: %d mature rows, trained=%s, %d/%d bars scored",
            geometry.name,
            window.index,
            n_train_rows,
            trained,
            scored_count,
            window.test_bars,
        )
    if len(scored) and all(reason is not None for reason in unavailable):
        if not model_was_available:
            raise ValueError(
                "rolling replay has no causal model: no scheduled window could fit "
                "a model; provide earlier training history or revise the geometry"
            )
        diagnostic = (
            "provide more history before the scored window"
            if all(reason is UnavailableReason.HISTORY for reason in unavailable)
            else "features are incomplete for another reason; "
            "check missing or degenerate data"
        )
        raise ValueError(
            f"rolling prediction scored none of {len(scored)} bars; features need "
            f"at least {context.minimum_rows} rows; {diagnostic}"
        )
    return RollingOutcome(
        pd.DataFrame(values, index=test_df.index, columns=list(classes)),
        tuple(unavailable),
        tuple(bar_roll),
        tuple(traces),
        geometry,
        label_horizon,
        initial_available,
    )


def signals_from_probabilities(
    probabilities: pd.DataFrame,
    *,
    threshold: float,
    cooldown: int,
) -> npt.NDArray[np.intp]:
    """Direction policy over class labels; cooldown never changes model scoring."""
    if not np.isfinite(threshold) or not 0 <= threshold <= 1 or cooldown < 0:
        raise ValueError(
            "Direction threshold must be in [0, 1] and cooldown nonnegative"
        )
    available = ~probabilities.isna().all(axis=1)
    validate_probabilities(probabilities.loc[available], probabilities.loc[available])
    p_long = class_probability(probabilities, 1).to_numpy(dtype=float)
    p_short = class_probability(probabilities, -1).to_numpy(dtype=float)
    signals = np.zeros(len(probabilities), dtype=int)
    last_trade_bar = -cooldown
    for bar in range(len(probabilities)):
        if bar - last_trade_bar < cooldown or not available.iloc[bar]:
            continue
        if p_long[bar] >= threshold and p_long[bar] >= p_short[bar]:
            signals[bar] = 1
        elif p_short[bar] >= threshold:
            signals[bar] = -1
        if signals[bar]:
            last_trade_bar = bar
    return signals


def rolling_probabilities(
    train_data: pd.DataFrame,
    test_df: pd.DataFrame,
    scheme: RollingScheme | None,
    *,
    model: XGBoostDirectionModel,
    model_config: XGBoostModelConfig,
    fe: FeatureTransformer,
    lookback: int,
    retrain_interval: int,
    label_horizon: int,
    model_factory: Callable[[XGBoostModelConfig], XGBoostDirectionModel] | None = None,
) -> pd.DataFrame:
    """The probabilities alone, for callers that do not need the provenance."""
    return rolling_outcome(
        train_data,
        test_df,
        scheme,
        model=model,
        model_config=model_config,
        fe=fe,
        lookback=lookback,
        retrain_interval=retrain_interval,
        label_horizon=label_horizon,
        model_factory=model_factory,
    ).probabilities


def run_rolling(
    train_data: pd.DataFrame,
    test_df: pd.DataFrame,
    scheme: RollingScheme | None,
    *,
    model: XGBoostDirectionModel,
    model_config: XGBoostModelConfig,
    fe: FeatureTransformer,
    threshold: float,
    cooldown: int,
    lookback: int,
    retrain_interval: int,
    label_horizon: int,
    model_factory: Callable[[XGBoostModelConfig], XGBoostDirectionModel] | None = None,
) -> npt.NDArray[np.intp]:
    """Score ``test_df`` with the scheme and map probabilities to signals.

    Same arguments as :func:`rolling_probabilities`, plus the decision rule the
    direction strategy applies to each bar's probabilities.
    """
    return signals_from_probabilities(
        rolling_probabilities(
            train_data,
            test_df,
            scheme,
            model=model,
            model_config=model_config,
            fe=fe,
            lookback=lookback,
            retrain_interval=retrain_interval,
            label_horizon=label_horizon,
            model_factory=model_factory,
        ),
        threshold=threshold,
        cooldown=cooldown,
    )
