"""Automatic and declared geometry share one risk/direction execution core."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.ml.features.risk_features import RiskFeatureEngineer
from tradingdev.domain.ml.models.xgboost_model import XGBoostDirectionModel
from tradingdev.domain.ml.schemas import XGBoostModelConfig
from tradingdev.domain.strategies.base import BaseStrategy
from tradingdev.domain.strategies.bundled.safety_volume_strategy.config import (
    SafetyVolumeStrategyConfig,
)
from tradingdev.domain.strategies.bundled.safety_volume_strategy.strategy import (
    SafetyVolumeStrategy,
)
from tradingdev.domain.strategies.bundled.xgboost_strategy.config import (
    XGBoostStrategyConfig,
)
from tradingdev.domain.strategies.bundled.xgboost_strategy.strategy import (
    XGBoostStrategy,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_FIT_BARS = 800


class _PlainStrategy(BaseStrategy):
    """A strategy that neither fits nor retrains."""

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(signal=0)

    def get_parameters(self) -> dict[str, Any]:
        return {}


def _fitted_xgboost(df: pd.DataFrame, *, enabled: bool) -> XGBoostStrategy:
    strategy = XGBoostStrategy(
        XGBoostStrategyConfig(
            model=XGBoostModelConfig(n_estimators=5, max_depth=2, n_jobs=1),
            lookback_candidates=[24, 48],
            retrain_interval=50,
            target_horizon=1,
            rolling_gen={"enabled": enabled, "scheme": "roll_ex"},
        )
    )
    strategy.fit(df.iloc[:_FIT_BARS])
    return strategy


@pytest.mark.parametrize("enabled", [False, True])
def test_the_switch_selects_geometry_in_the_shared_core(
    large_ohlcv_df: pd.DataFrame,
    enabled: bool,
) -> None:
    strategy = _fitted_xgboost(large_ohlcv_df, enabled=enabled)
    strategy.generate_signals(large_ohlcv_df.iloc[_FIT_BARS:])
    trace = strategy.rolling_trace()
    assert trace is not None
    assert trace["scheme"] == ("roll_ex" if enabled else "automatic")
    assert trace["rtype"] == ("ROLL_EX" if enabled else "ROLL_SD")


def test_enabled_run_keeps_the_input_frame_and_emits_valid_signals(
    large_ohlcv_df: pd.DataFrame,
) -> None:
    strategy = _fitted_xgboost(large_ohlcv_df, enabled=True)
    scored = large_ohlcv_df.iloc[_FIT_BARS:].copy()
    original = scored.copy(deep=True)

    result = strategy.generate_signals(scored)

    assert set(result["signal"].unique()).issubset({-1, 0, 1})
    pd.testing.assert_frame_equal(result[scored.columns], original)
    pd.testing.assert_frame_equal(scored, original)


def test_enabled_run_has_no_look_ahead(
    large_ohlcv_df: pd.DataFrame,
    assert_no_look_ahead: Callable[..., None],
) -> None:
    """Truncating the future must not change an earlier bar's signal."""
    strategy = _fitted_xgboost(large_ohlcv_df, enabled=True)
    scored = large_ohlcv_df.iloc[_FIT_BARS:].reset_index(drop=True)

    assert_no_look_ahead(strategy, scored, check_points=[60, 120, 199])


def test_a_strategy_that_never_rolls_has_no_trace() -> None:
    """The default keeps run metadata unchanged for every other strategy."""
    assert _PlainStrategy().rolling_trace() is None


def test_the_trace_accounts_for_every_scored_bar(
    large_ohlcv_df: pd.DataFrame,
) -> None:
    strategy = _fitted_xgboost(large_ohlcv_df, enabled=True)
    scored = large_ohlcv_df.iloc[_FIT_BARS:]

    strategy.generate_signals(scored)

    trace = strategy.rolling_trace()
    assert trace is not None
    assert trace["scheme"] == "roll_ex"
    assert trace["bars"] == len(scored)
    rolls = trace["rolls"]
    assert isinstance(rolls, list)
    assert len(rolls) >= 1
    # The rolls partition the scored bars, so their counts must add up.
    assert sum(entry["bars_scored"] for entry in rolls) == trace["bars_scored"]


class _AlwaysSafe(XGBoostDirectionModel):
    """Deterministic risk-gate stand-in: every bar looks safe."""

    def __init__(self, config: XGBoostModelConfig | None = None) -> None:
        super().__init__(config or XGBoostModelConfig())

    def train(self, df: pd.DataFrame, eval_df: pd.DataFrame | None = None) -> None:
        return None

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({0: 0.0, 1: 1.0}, index=df.index)


def _fitted_safety(
    train: pd.DataFrame,
    *,
    enabled: bool,
    lookback: int = 96,
) -> tuple[SafetyVolumeStrategy, RiskFeatureEngineer]:
    strategy = SafetyVolumeStrategy(
        SafetyVolumeStrategyConfig(
            risk_model=XGBoostModelConfig(n_estimators=2, max_depth=2, n_jobs=1),
            lookback_candidates=[lookback],
            target_holding_bars=3,
            min_holding_bars=1,
            max_holding_bars=7,
            rolling_gen={"enabled": enabled, "scheme": "roll_ex"},
        )
    )
    fe = RiskFeatureEngineer(lookback=lookback, target_holding_bars=3)
    strategy._risk_fe = fe
    strategy._risk_model = _AlwaysSafe()
    strategy._train_data = train.copy()
    strategy._best_lookback = lookback
    return strategy, fe


def test_safety_risk_scores_one_value_per_bar_when_enabled(
    large_ohlcv_df: pd.DataFrame,
) -> None:
    train = large_ohlcv_df.iloc[:300].copy()
    scored = large_ohlcv_df.iloc[300:500].copy()
    strategy, _ = _fitted_safety(train, enabled=True)

    scores = strategy._predict_risk_rolling(scored)

    assert len(scores) == len(scored)
    assert np.isfinite(scores).all()
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_safety_risk_records_the_rolls_it_ran(
    large_ohlcv_df: pd.DataFrame,
) -> None:
    train = large_ohlcv_df.iloc[:300].copy()
    scored = large_ohlcv_df.iloc[300:500].copy()
    strategy, _ = _fitted_safety(train, enabled=True)

    strategy._predict_risk_rolling(scored)

    trace = strategy.rolling_trace()
    assert trace is not None
    assert trace["scheme"] == "roll_ex"
    assert trace["bars"] == len(scored)
    assert trace["bars_scored"] == len(scored)


@pytest.mark.parametrize("enabled", [False, True])
def test_safety_risk_selects_geometry_in_the_shared_core(
    large_ohlcv_df: pd.DataFrame,
    enabled: bool,
) -> None:
    strategy, _ = _fitted_safety(large_ohlcv_df.iloc[:300], enabled=enabled)
    strategy._predict_risk_rolling(large_ohlcv_df.iloc[300:500])
    trace = strategy.rolling_trace()
    assert trace is not None
    assert trace["scheme"] == ("roll_ex" if enabled else "automatic")
    assert trace["bars_scored"] == 200
