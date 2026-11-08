"""Optional ML training must not disable otherwise valid GLFT entries."""

from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import pytest

from tradingdev.domain.strategies.bundled.glft_ml_strategy.config import (
    GLFTMLStrategyConfig,
)
from tradingdev.domain.strategies.bundled.glft_ml_strategy.strategy import (
    GLFTMLStrategy,
)
from tradingdev.domain.strategies.bundled.glft_strategy.config import GLFTStrategyConfig
from tradingdev.domain.strategies.bundled.glft_strategy.strategy import GLFTStrategy
from tradingdev.domain.strategies.signals import generate_checked_signals


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=8, freq="min", tz="UTC"),
            "open": 100.0,
            "close": [100.0, 100.0, 98.0, 100.0, 102.0, 100.0, 98.0, 100.0],
            "high": 110.0,
            "low": 90.0,
            "volume": 10.0,
        }
    )


def _config() -> GLFTMLStrategyConfig:
    return GLFTMLStrategyConfig(
        vol_type="realized",
        vol_window=2,
        gamma=0.0,
        gamma_candidates=[0.0],
        kappa=1000.0,
        kappa_candidates=[1000.0],
        ema_window=2,
        ema_window_candidates=[2],
        min_holding_bars=1,
        max_holding_bars=3,
        max_holding_bars_candidates=[3],
        min_entry_edge_candidates=[0.0012],
        profit_target_ratio=1.0,
        profit_target_ratio_candidates=[1.0],
        strategy_sl=0.0,
        confidence_threshold_candidates=[0.55],
    )


@pytest.mark.parametrize("call_fit", [False, True], ids=["untrained", "no_engine_fit"])
def test_untrained_strategy_enters_and_exits_without_accessing_ml(
    call_fit: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _prices()
    before = frame.copy(deep=True)
    strategy = GLFTMLStrategy(_config())
    ml = Mock()
    ml.train.side_effect = AssertionError("Untrained path must not invoke AutoML")
    ml.predict_proba.side_effect = AssertionError("No model is ready to predict")
    features = Mock()
    features.transform.side_effect = AssertionError("No ML features are required")
    monkeypatch.setattr(strategy, "_ml_model", ml)
    monkeypatch.setattr(strategy, "_feature_eng", features)
    if call_fit:
        strategy.fit(frame)

    result = generate_checked_signals(strategy, frame)
    assert not strategy.get_parameters()["ml_trained"]
    # EMA(2) warms up at index 1, then price excursions enter both directions;
    # the next return through fair value closes each position.
    assert result["signal"].tolist() == [0, 0, 1, 0, -1, 0, 1, 0]
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(result[frame.columns], before)

    pure = GLFTStrategy(
        GLFTStrategyConfig(
            vol_type="realized",
            vol_window=2,
            gamma=0.0,
            kappa=1000.0,
            ema_window=2,
            min_holding_bars=1,
            max_holding_bars=3,
            min_entry_edge=0.0012,
            profit_target_ratio=1.0,
            strategy_sl=0.0,
            momentum_guard=False,
        )
    )
    pd.testing.assert_series_equal(
        result["signal"], generate_checked_signals(pure, frame)["signal"]
    )


@pytest.mark.parametrize(
    "up_probability, expected",
    [
        (0.5, [0, 0, 0, 0, 0, 0, 0, 0]),
        (0.9, [0, 0, 1, 0, 0, 1, 1, 0]),
        (0.1, [0, 0, 0, -1, -1, 0, 0, -1]),
    ],
    ids=["uncertain", "up", "down"],
)
def test_successfully_trained_path_still_requires_ml_direction_agreement(
    up_probability: float,
    expected: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _prices()
    engine = Mock()
    engine.run.return_value.metrics = {"total_volume": 1.0}
    strategy = GLFTMLStrategy(_config(), backtest_engine=engine)
    # Deterministic model probabilities isolate gating from the external learner.
    monkeypatch.setattr(
        strategy._feature_eng, "transform", lambda df, **kwargs: df.copy()
    )
    train = Mock()
    monkeypatch.setattr(strategy._ml_model, "train", train)
    monkeypatch.setattr(
        strategy._ml_model,
        "predict_proba",
        lambda df: pd.DataFrame(
            {0: 1 - up_probability, 1: up_probability}, index=df.index
        ),
    )
    strategy.fit(frame)
    train.assert_called_once()
    assert strategy.get_parameters()["ml_trained"]
    result = generate_checked_signals(strategy, frame)
    # Blocking one direction changes which later bars see a flat position:
    # up-only enters again at 5; down-only can enter at 3 and again at 7.
    assert result["signal"].tolist() == expected
    engine.run.assert_called_once()
    pd.testing.assert_frame_equal(engine.run.call_args.args[0], result)
