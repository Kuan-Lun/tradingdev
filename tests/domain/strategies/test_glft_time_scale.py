"""GLFT uses the actual input-bar duration throughout its public entry points."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
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
from tradingdev.domain.strategies.bundled.glft_time import glft_bar_seconds
from tradingdev.domain.strategies.signals import generate_checked_signals


def _frame(frequency: str = "min") -> pd.DataFrame:
    close = np.arange(100.0, 120.0)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01 00:02", periods=len(close), freq=frequency, tz="UTC"
            ),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 100.0,
            "dvol": 50.0,
        }
    )


@pytest.mark.parametrize("mode", ["pure", "ml_fallback", "ml_trained"])
@pytest.mark.parametrize("frequency,seconds", [("s", 1), ("min", 60), ("h", 3600)])
def test_public_signals_pass_correct_per_bar_sigma_to_the_state_machine(
    mode: str, frequency: str, seconds: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    strategy: GLFTStrategy | GLFTMLStrategy
    if mode == "pure":
        strategy = GLFTStrategy(
            GLFTStrategyConfig(vol_type="implied", ema_window=2, signal_agg_bars=5)
        )
        method = "_run_glft_state_machine"
    else:
        strategy = GLFTMLStrategy(GLFTMLStrategyConfig(ema_window=2))
        method = "_run_ml_glft_state_machine"
        if mode == "ml_trained":
            strategy._ml_trained = True
            monkeypatch.setattr(
                strategy, "_predict_ml_directions", lambda df, _: np.ones(len(df))
            )
    observed = Mock(wraps=getattr(strategy, method))
    monkeypatch.setattr(strategy, method, observed)
    frame = _frame(frequency)
    before = frame.copy(deep=True)
    result = generate_checked_signals(strategy, frame)
    expected = 0.5 * np.sqrt(seconds / (365.25 * 24 * 3600))
    np.testing.assert_allclose(observed.call_args.kwargs["sigma"], expected)
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(result[frame.columns], before)


@pytest.mark.parametrize("ml", [False, True])
@pytest.mark.parametrize("entry_point", ["fit", "generate_signals"])
def test_irregular_timeline_is_rejected_before_training_or_signals(
    ml: bool, entry_point: str
) -> None:
    engine = Mock()
    strategy = (
        GLFTMLStrategy(GLFTMLStrategyConfig(), backtest_engine=engine)
        if ml
        else GLFTStrategy(GLFTStrategyConfig(), backtest_engine=engine)
    )
    frame = _frame().drop(index=5)
    with pytest.raises(ValueError, match="uniform"):
        getattr(strategy, entry_point)(frame)
    engine.run.assert_not_called()


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "reverse", "numeric", "object_numeric", "short"]
)
def test_time_scale_requires_observable_valid_timestamps(fault: str) -> None:
    frame = _frame()
    if fault == "missing":
        frame.loc[3, "timestamp"] = pd.NaT
    elif fault == "duplicate":
        frame.loc[3, "timestamp"] = frame.loc[2, "timestamp"]
    elif fault == "reverse":
        frame = frame.iloc[::-1]
    elif fault == "numeric":
        frame["timestamp"] = np.arange(len(frame))
    elif fault == "object_numeric":
        frame["timestamp"] = pd.Series(
            1704067200 + 60 * np.arange(len(frame)), dtype=object
        )
    else:
        frame = frame.head(1)
    with pytest.raises(ValueError, match="GLFT"):
        glft_bar_seconds(frame)


def test_time_scale_is_elapsed_time_across_timezone_offset_changes() -> None:
    frame = _frame("h")
    frame["timestamp"] = pd.date_range(
        "2024-03-10", periods=len(frame), freq="h", tz="America/New_York"
    )
    assert glft_bar_seconds(frame) == 3600


@pytest.mark.parametrize("ml", [False, True])
@pytest.mark.parametrize("frequency,seconds", [("min", 60), ("h", 3600)])
def test_fit_and_execution_share_scale_and_require_refit_when_frequency_changes(
    ml: bool, frequency: str, seconds: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradingdev.domain.strategies.bundled.glft_strategy import strategy as module

    monkeypatch.setattr(module, "estimate_n_jobs", lambda *a, **kw: 1)
    engine = Mock()
    engine.run.return_value.metrics = {"total_pnl": 1.0, "total_volume": 1.0}
    parameters: dict[str, object] = {
        "gamma_candidates": [5000.0],
        "kappa_candidates": [1000.0],
        "ema_window_candidates": [2],
        "max_holding_bars_candidates": [2],
        "min_holding_bars": 1,
        "max_holding_bars": 2,
        "min_entry_edge_candidates": [0.0012],
        "profit_target_ratio_candidates": [0.0],
        "strategy_sl": 0.0,
        "vol_type": "implied",
    }
    strategy: GLFTStrategy | GLFTMLStrategy
    cls: type[GLFTStrategy] | type[GLFTMLStrategy]
    if ml:
        parameters["confidence_threshold_candidates"] = [0.55]
        strategy = GLFTMLStrategy(
            GLFTMLStrategyConfig.model_validate(parameters), backtest_engine=engine
        )
        monkeypatch.setattr(strategy._feature_eng, "transform", lambda df, **kw: df)
        monkeypatch.setattr(strategy._ml_model, "train", Mock())
        monkeypatch.setattr(
            strategy._ml_model,
            "predict_proba",
            lambda df: pd.DataFrame({0: 0.0, 1: 1.0}, index=df.index),
        )
        method = "_run_ml_glft_state_machine"
        cls = GLFTMLStrategy
    else:
        parameters["vol_window_candidates"] = [2]
        parameters["momentum_guard"] = False
        strategy = GLFTStrategy(
            GLFTStrategyConfig.model_validate(parameters), backtest_engine=engine
        )
        method = "_run_glft_state_machine"
        cls = GLFTStrategy
    observed = Mock(wraps=getattr(cls, method))
    monkeypatch.setattr(cls, method, staticmethod(observed))
    frame = _frame(frequency)
    frame["close"] = np.tile([100.0, 98.0], len(frame) // 2)
    strategy.fit(frame)
    engine.run.assert_called_once()
    result = generate_checked_signals(strategy, frame)
    pd.testing.assert_frame_equal(engine.run.call_args.args[0], result)
    assert observed.call_count == 2
    for call in observed.call_args_list:
        np.testing.assert_allclose(
            call.kwargs["sigma"], 0.5 * np.sqrt(seconds / (365.25 * 24 * 3600))
        )
    # With identical prices, the correctly larger hourly risk term blocks entry.
    assert result["signal"].ne(0).any() == (frequency == "min")
    different = frame.copy()
    different["timestamp"] = _frame("s")["timestamp"]
    with pytest.raises(ValueError, match="refit"):
        strategy.generate_signals(different)
    if ml:
        # A new model with an unsuccessful parameter search must not be usable
        # at either frequency, or silently fall back to untrained behavior.
        engine.run.side_effect = RuntimeError("parameter search failed")
        with pytest.raises(RuntimeError, match="parameter search failed"):
            strategy.fit(different)
        for candidate in (frame, different):
            with pytest.raises(ValueError, match="fit did not complete"):
                strategy.generate_signals(candidate)
        assert strategy.get_parameters()["ml_trained"] is False
        engine.run.side_effect = None
        strategy.fit(different)
        generate_checked_signals(strategy, different)
        assert strategy.get_parameters()["ml_trained"] is True


@pytest.mark.parametrize("ml", [False, True])
def test_object_unix_seconds_cannot_silently_change_the_public_signal_scale(
    ml: bool,
) -> None:
    strategy = (
        GLFTMLStrategy(GLFTMLStrategyConfig())
        if ml
        else GLFTStrategy(GLFTStrategyConfig(vol_type="implied"))
    )
    frame = _frame()
    frame["timestamp"] = pd.Series(
        1704067200 + 60 * np.arange(len(frame)), dtype=object
    )
    with pytest.raises(ValueError, match="not numeric offsets"):
        strategy.generate_signals(frame)
