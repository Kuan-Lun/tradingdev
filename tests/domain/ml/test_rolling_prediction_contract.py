"""Shared prediction identity, availability and model-output failure boundaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.ml.features.risk_features import RiskFeatureEngineer
from tradingdev.domain.ml.models.xgboost_model import XGBoostDirectionModel
from tradingdev.domain.ml.rolling_gen import UnavailableReason, rolling_outcome
from tradingdev.domain.ml.schemas import XGBoostModelConfig
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


class _Features(RiskFeatureEngineer):
    @property
    def minimum_context_rows(self) -> int:
        return 2

    @property
    def context_rows(self) -> int:
        return 4

    def transform(
        self, df: pd.DataFrame, *, include_target: bool = True
    ) -> pd.DataFrame:
        return df.drop(index=6, errors="ignore") if not include_target else df


class _Model(XGBoostDirectionModel):
    def __init__(self, defect: str = "") -> None:
        super().__init__(XGBoostModelConfig())
        self.defect = defect

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame({0: 0.5, 1: 0.5}, index=df.index)
        if self.defect == "wrong_index":
            result.index = pd.Index([-1])
        elif self.defect == "wrong_count":
            result = pd.concat([result, result])
        elif self.defect == "duplicate_class":
            result.columns = pd.Index([1, 1])
        elif self.defect == "empty_classes":
            result = result.iloc[:, :0]
        elif self.defect == "unknown_class":
            result.columns = pd.Index([0, 2])
        elif self.defect == "unnormalized":
            result[0] = 0.4
        elif self.defect == "nonfinite":
            result[0] = np.nan
        elif self.defect == "range":
            result[0], result[1] = -0.1, 1.1
        return result


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC"),
            "close": np.arange(8) + 100.0,
        },
        index=np.arange(8) * 7,
    )


def test_shared_result_keeps_source_identity_classes_and_gap_reason() -> None:
    frame = _frame()
    outcome = rolling_outcome(
        frame.iloc[:5],
        frame.iloc[5:],
        None,
        model=_Model(),
        model_config=XGBoostModelConfig(),
        fe=_Features(),
        lookback=24,
        retrain_interval=10,
        label_horizon=1,
    )
    pd.testing.assert_index_equal(outcome.probabilities.index, frame.index[5:])
    pd.testing.assert_index_equal(outcome.probabilities.columns, pd.Index([0, 1]))
    np.testing.assert_array_equal(outcome.probabilities[1], [0.5, np.nan, 0.5])
    assert outcome.unavailable == (None, UnavailableReason.FEATURES, None)
    assert outcome.bar_roll == (0, -1, 0)


@pytest.mark.parametrize(
    "defect",
    [
        "wrong_index",
        "wrong_count",
        "duplicate_class",
        "empty_classes",
        "unknown_class",
        "unnormalized",
        "nonfinite",
        "range",
    ],
)
def test_malformed_probabilities_fail_instead_of_becoming_unavailable(
    defect: str,
) -> None:
    frame = _frame()
    with pytest.raises(RuntimeError, match="probabilities"):
        rolling_outcome(
            frame.iloc[:5],
            frame.iloc[5:],
            None,
            model=_Model(defect),
            model_config=XGBoostModelConfig(),
            fe=_Features(),
            lookback=24,
            retrain_interval=10,
            label_horizon=1,
        )


@pytest.mark.parametrize(
    ("strategy_name", "lookbacks", "prior_bars"),
    [
        ("risk", [24], 48),
        ("direction", [12], 20),
        ("risk", [12, 72], 60),
        ("direction", [12, 72], 40),
    ],
)
@pytest.mark.parametrize("phase", ["configuration", "mutated_fit"])
def test_explicit_short_context_is_rejected_before_any_training(
    strategy_name: str,
    lookbacks: list[int],
    prior_bars: int,
    phase: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("training must not start for impossible geometry")

    monkeypatch.setattr(XGBoostDirectionModel, "train", forbidden)
    values = {
        "lookback_candidates": lookbacks,
        "rolling_gen": {
            "enabled": True,
            "predict_lookback": prior_bars if phase == "configuration" else None,
        },
    }
    with pytest.raises(ValueError, match="predict_lookback must be at least"):
        if strategy_name == "risk":
            risk_config = SafetyVolumeStrategyConfig.model_validate(values)
            risk_config.rolling_gen.predict_lookback = prior_bars
            SafetyVolumeStrategy(risk_config).fit(_frame())
        else:
            direction_config = XGBoostStrategyConfig.model_validate(values)
            direction_config.rolling_gen.predict_lookback = prior_bars
            XGBoostStrategy(direction_config).fit(_frame())
