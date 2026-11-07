"""Public SafetyVolume scoring preserves prefixes with optional ML direction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.ml.models.xgboost_model import XGBoostDirectionModel
from tradingdev.domain.ml.schemas import XGBoostModelConfig
from tradingdev.domain.strategies.bundled.safety_volume_strategy.config import (
    SafetyVolumeStrategyConfig,
)
from tradingdev.domain.strategies.bundled.safety_volume_strategy.strategy import (
    SafetyVolumeStrategy,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class _DirectionProbe:
    """A deterministic predictor that consumes the actual engineered features."""

    def __init__(self) -> None:
        self.frames: list[pd.DataFrame] = []

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        self.frames.append(df.copy(deep=True))
        directions = np.sign(df["log_return"])
        return pd.DataFrame(
            {label: (directions == label).astype(float) for label in (-1, 0, 1)},
            index=df.index,
        )


@pytest.mark.parametrize("use_ml_direction", [False, True], ids=["sma", "ml"])
def test_public_signals_are_invariant_to_future_rows(
    large_ohlcv_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    use_ml_direction: bool,
) -> None:
    config = SafetyVolumeStrategyConfig(
        risk_model=XGBoostModelConfig(n_estimators=2, max_depth=2, n_jobs=1),
        direction_model=XGBoostModelConfig(n_estimators=2, max_depth=2, n_jobs=1),
        lookback_candidates=[24],
        use_ml_direction=use_ml_direction,
        target_holding_bars=3,
        min_holding_bars=1,
        max_holding_bars=7,
        sma_fast=3,
        sma_slow=10,
    )
    strategy = SafetyVolumeStrategy(config)
    history = large_ohlcv_df.iloc[:400].copy()
    history_original = history.copy(deep=True)
    scored = large_ohlcv_df.iloc[400:580].copy()
    scored.index = pd.Index(np.arange(len(scored)) * 3 + 7)
    # A later flat interval removes feature rows without changing earlier prices.
    gap_index = scored.index[110:150]
    scored.loc[gap_index, ["open", "close"]] = 100.0
    scored.loc[gap_index, "high"] = 101.0
    scored.loc[gap_index, "low"] = 99.0
    scored.loc[gap_index, "volume"] = 1000.0
    original = scored.copy(deep=True)
    strategy.fit(history)
    assert history["timestamp"].iloc[-1] < scored["timestamp"].iloc[0]
    predictor = _DirectionProbe()
    monkeypatch.setattr(
        XGBoostDirectionModel,
        "predict_proba",
        lambda _model, frame: predictor.predict_proba(frame),
    )
    risk_calls: list[int] = []

    def allow_entries(
        _strategy: SafetyVolumeStrategy, frame: pd.DataFrame
    ) -> NDArray[np.float64]:
        risk_calls.append(len(frame))
        return np.ones(len(frame), dtype=np.float64)

    monkeypatch.setattr(SafetyVolumeStrategy, "_predict_risk_rolling", allow_entries)
    full = strategy.generate_signals(scored)
    prefix = strategy.generate_signals(scored.iloc[:90])
    changed_future = scored.copy(deep=True)
    for column in ("open", "high", "low", "close", "volume"):
        changed_future.loc[changed_future.index[90:], column] *= np.linspace(
            1.5, 3.0, len(scored) - 90
        )
    perturbed = strategy.generate_signals(changed_future)

    assert risk_calls == [180, 90, 180]
    assert bool(predictor.frames) is use_ml_direction
    assert prefix["signal"].ne(0).any(), "the causal comparison must exercise entries"
    pd.testing.assert_series_equal(full["signal"].iloc[:90], prefix["signal"])
    pd.testing.assert_series_equal(
        full["signal"].iloc[:90], perturbed["signal"].iloc[:90]
    )
    pd.testing.assert_frame_equal(full[scored.columns], original)
    pd.testing.assert_frame_equal(scored, original)
    pd.testing.assert_frame_equal(history, history_original)
    if use_ml_direction:
        assert strategy._dir_fe is not None
        assert all(not frame.empty for frame in predictor.frames)
        assert all("log_return" in frame.columns for frame in predictor.frames)


@pytest.mark.parametrize("declared", [False, True], ids=["automatic", "declared"])
@pytest.mark.parametrize("overlap", [False, True], ids=["held_out", "replay"])
def test_both_models_only_score_after_their_training_labels_mature(
    large_ohlcv_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    declared: bool,
    overlap: bool,
) -> None:
    """A causal risk gate must not hide a direction model trained on the future."""
    trained: dict[XGBoostDirectionModel, pd.Timestamp] = {}
    predictions: list[tuple[int, pd.Timestamp, pd.Timestamp]] = []

    def train(
        model: XGBoostDirectionModel,
        frame: pd.DataFrame,
        eval_df: pd.DataFrame | None = None,
    ) -> None:
        trained[model] = frame["timestamp"].max()

    def predict(model: XGBoostDirectionModel, frame: pd.DataFrame) -> pd.DataFrame:
        horizon = 3 if model._config.n_estimators == 2 else 1
        predictions.append((horizon, trained[model], frame["timestamp"].min()))
        classes = (0, 1) if horizon == 3 else (-1, 0, 1)
        return pd.DataFrame(
            {label: float(label == 1) for label in classes}, index=frame.index
        )

    monkeypatch.setattr(XGBoostDirectionModel, "train", train)
    monkeypatch.setattr(XGBoostDirectionModel, "predict_proba", predict)
    strategy = SafetyVolumeStrategy(
        SafetyVolumeStrategyConfig.model_validate(
            {
                "risk_model": {"n_estimators": 2},
                "direction_model": {"n_estimators": 3},
                "lookback_candidates": [24],
                "use_ml_direction": True,
                "target_holding_bars": 3,
                "retrain_interval": 100,
                "rolling_gen": {"enabled": declared},
            }
        )
    )
    history = large_ohlcv_df.iloc[:300]
    scored = history if overlap else large_ohlcv_df.iloc[300:450]
    strategy.fit(history)
    predictions.clear()
    result = strategy.generate_signals(scored)

    assert {horizon for horizon, _, _ in predictions} == {1, 3}
    for horizon, trained_until, scored_from in predictions:
        assert trained_until + pd.Timedelta(hours=horizon) < scored_from
    assert result["signal"].ne(0).any()
    if overlap:
        assert result["signal"].iloc[:100].eq(0).all()
        assert result["signal"].iloc[100] == 1
    trace = strategy.rolling_trace()
    assert trace is not None
    for model_trace in (trace, trace["direction"]):
        assert model_trace["initial_model_available"] is not overlap
        assert model_trace["bars_scored"] > 0
        assert model_trace["scheme"] == ("roll_sd" if declared else "automatic")

    # A later direction failure cannot leave either the previous trace or a
    # partial risk-only trace looking like evidence of a successful execution.
    def fail_direction(
        _strategy: SafetyVolumeStrategy, _frame: pd.DataFrame
    ) -> NDArray[np.float64]:
        raise RuntimeError("direction failed")

    monkeypatch.setattr(SafetyVolumeStrategy, "_compute_ml_directions", fail_direction)
    with pytest.raises(RuntimeError, match="direction failed"):
        strategy.generate_signals(scored)
    assert strategy.rolling_trace() is None
