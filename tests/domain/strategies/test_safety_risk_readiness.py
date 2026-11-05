"""Issue #105: an unavailable risk estimate is never a model probability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.ml.features.risk_features import RiskFeatureEngineer
from tradingdev.domain.ml.models.xgboost_model import XGBoostDirectionModel
from tradingdev.domain.ml.schemas import XGBoostModelConfig
from tradingdev.domain.strategies.bundled.safety_volume_strategy.config import (
    SafetyVolumeStrategyConfig,
)
from tradingdev.domain.strategies.bundled.safety_volume_strategy.strategy import (
    SafetyVolumeStrategy,
)


def _market(rows: int = 1850) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, rows)))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01", periods=rows, freq="min", tz="UTC"
            ),
            "open": close,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": rng.uniform(100, 200, rows),
        }
    )


@pytest.mark.parametrize("lookback", [24, 60])
def test_fitted_risk_model_really_scores_every_bar(
    lookback: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _market()
    strategy = SafetyVolumeStrategy(
        SafetyVolumeStrategyConfig(
            lookback_candidates=[lookback],
            max_acceptable_loss_pct=0.001,
            risk_model=XGBoostModelConfig(n_estimators=3, n_jobs=1, max_depth=2),
        )
    )
    strategy.fit(frame.iloc[:1800])
    original = XGBoostDirectionModel.predict_proba
    predictions: list[pd.DataFrame] = []

    def observed(self: XGBoostDirectionModel, data: pd.DataFrame) -> pd.DataFrame:
        result = original(self, data)
        predictions.append(result)
        return result

    monkeypatch.setattr(XGBoostDirectionModel, "predict_proba", observed)
    scores = strategy._predict_risk_rolling(frame.iloc[1800:])

    assert len(predictions) == 50
    expected = pd.concat(predictions)
    assert expected.index.tolist() == list(range(1800, 1850))
    np.testing.assert_array_equal(scores, expected[1].to_numpy())


@pytest.mark.parametrize("lookback,rows", [(24, 61), (60, 62), (168, 170)])
def test_risk_feature_minimum_is_exact(lookback: int, rows: int) -> None:
    fe = RiskFeatureEngineer(lookback)
    frame = _market(rows)
    assert fe.minimum_context_rows == rows
    assert fe.transform(frame.iloc[:-1], include_target=False).empty
    assert fe.transform(frame, include_target=False).index.tolist() == [rows - 1]


@pytest.mark.parametrize("threshold", [0.0, 0.5, 0.8])
def test_unavailable_risk_blocks_entry_and_reversal_until_exit(
    threshold: float,
) -> None:
    scores = np.array([np.nan, 0.9, np.nan, np.nan, np.nan, 0.9])
    directions = np.array([1, 1, -1, -1, -1, -1], dtype=float)
    signals = SafetyVolumeStrategy._run_state_machine_with_threshold(
        scores, directions, threshold, min_hold=3, max_hold=10
    )
    assert signals.tolist() == [0, 1, 1, 1, 0, -1]


def test_a_real_half_probability_still_passes_its_threshold() -> None:
    signals = SafetyVolumeStrategy._run_state_machine_with_threshold(
        np.array([0.5, np.nan]), np.ones(2), 0.5, min_hold=1
    )
    assert signals.tolist() == [1, 0]


class _SafeModel(XGBoostDirectionModel):
    def __init__(self) -> None:
        super().__init__(XGBoostModelConfig())
        self.requests: list[pd.DataFrame] = []

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        self.requests.append(df.copy())
        return pd.DataFrame({0: 0.0, 1: 1.0}, index=df.index)


def _ready_strategy(history: pd.DataFrame, enabled: bool) -> SafetyVolumeStrategy:
    strategy = SafetyVolumeStrategy(
        SafetyVolumeStrategyConfig(
            lookback_candidates=[24],
            rolling_gen={
                "enabled": enabled,
                "scheme": "roll_ex",
                "predict_lookback": 215,
            },
        )
    )
    strategy._risk_model = _SafeModel()
    # No mature labels: this isolates feature availability from retraining.
    strategy._risk_fe = RiskFeatureEngineer(24, target_holding_bars=1000)
    strategy._best_lookback = 24
    strategy._train_data = history.copy()
    return strategy


@pytest.mark.parametrize("enabled", [False, True])
def test_warmup_and_interior_gap_remain_unavailable(enabled: bool) -> None:
    frame = _market(80)
    frame.loc[70, "open"] = np.nan
    strategy = _ready_strategy(frame.iloc[:40], enabled)
    scores = strategy._predict_risk_rolling(frame.iloc[40:])
    expected = np.ones(40)
    expected[:20] = np.nan
    expected[30] = np.nan
    np.testing.assert_array_equal(scores, expected)
    assert isinstance(strategy._risk_model, _SafeModel)
    assert [int(row.index[0]) for row in strategy._risk_model.requests] == [
        bar for bar in range(60, 80) if bar != 70
    ]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("rows,missing", [(60, False), (300, True)])
def test_entirely_unavailable_window_fails(
    enabled: bool, rows: int, missing: bool
) -> None:
    frame = _market(rows)
    if missing:
        frame["open"] = np.nan
    strategy = _ready_strategy(frame.iloc[:-10], enabled)
    with pytest.raises(ValueError, match="scored none"):
        strategy._predict_risk_rolling(frame.iloc[-10:])


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.1, 1.1])
def test_malformed_model_output_is_an_error(
    value: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _market(300)
    strategy = _ready_strategy(frame.iloc[:-10], False)
    monkeypatch.setattr(
        _SafeModel,
        "predict_proba",
        lambda self, data: pd.DataFrame({0: 1 - value, 1: value}, index=data.index),
    )
    with pytest.raises(RuntimeError, match="finite normalized"):
        strategy._predict_risk_rolling(frame.iloc[-10:])


def test_preferred_context_reduces_seed_error() -> None:
    frame = _market(500)
    fe = RiskFeatureEngineer(24)
    full = fe.transform(frame, include_target=False)
    preferred = fe.transform(frame.iloc[-fe.context_rows :], include_target=False)
    bare = fe.transform(frame.iloc[-61:], include_target=False)
    spread = full["adx_14"].std()
    reference = full["adx_14"].iloc[-1]
    assert abs(preferred["adx_14"].iloc[-1] - reference) < 0.01 * spread
    assert abs(bare["adx_14"].iloc[-1] - reference) > 0.01 * spread
