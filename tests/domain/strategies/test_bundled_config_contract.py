"""Bundled strategy config contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from tradingdev.domain.strategies.bundled.glft_ml_strategy.config import (
    GLFTMLStrategyConfig,
)
from tradingdev.domain.strategies.bundled.glft_strategy.config import (
    GLFTStrategyConfig,
)
from tradingdev.domain.strategies.bundled.quantile_strategy.config import (
    QuantileStrategyConfig,
)
from tradingdev.domain.strategies.bundled.safety_volume_strategy.config import (
    SafetyVolumeStrategyConfig,
)

_BUNDLED_ROOT = Path("src/tradingdev/domain/strategies/bundled")
_DATA_PATH_PARAMETERS = {
    "dvol_raw_path",
    "dvol_processed_path",
    "funding_rate_path",
}
_MOVING_AVERAGE_FIELDS = [
    pytest.param(GLFTStrategyConfig, "ema_window", False, id="glft-ema"),
    pytest.param(GLFTStrategyConfig, "ema_window_candidates", True, id="glft-grid"),
    pytest.param(GLFTMLStrategyConfig, "ema_window", False, id="glft-ml-ema"),
    pytest.param(
        GLFTMLStrategyConfig, "ema_window_candidates", True, id="glft-ml-grid"
    ),
    pytest.param(SafetyVolumeStrategyConfig, "sma_fast", False, id="safety-fast"),
    pytest.param(SafetyVolumeStrategyConfig, "sma_slow", False, id="safety-slow"),
]


def test_bundled_strategy_parameters_do_not_declare_feature_paths() -> None:
    for config_path in sorted(_BUNDLED_ROOT.glob("*/config.yaml")):
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        strategy = raw["strategy"]
        parameters = strategy.get("parameters", {})

        assert not _DATA_PATH_PARAMETERS.intersection(parameters), config_path


def test_bundled_config_models_do_not_expose_feature_path_parameters() -> None:
    for config_model in (
        GLFTStrategyConfig,
        GLFTMLStrategyConfig,
        QuantileStrategyConfig,
    ):
        assert not _DATA_PATH_PARAMETERS.intersection(config_model.model_fields)


@pytest.mark.parametrize(
    ("config_model", "field", "is_candidates"), _MOVING_AVERAGE_FIELDS
)
@pytest.mark.parametrize("period", [2, 100_000])
def test_moving_average_period_boundaries_are_valid(
    config_model: type[BaseModel], field: str, is_candidates: bool, period: int
) -> None:
    value = [period] if is_candidates else period
    config = config_model.model_validate({field: value})

    assert config.model_dump()[field] == value


@pytest.mark.parametrize(
    ("config_model", "field", "is_candidates"), _MOVING_AVERAGE_FIELDS
)
@pytest.mark.parametrize("period", [-1, 0, 1, 100_001, True, False, 2.0, "2"])
def test_invalid_moving_average_periods_fail_config_validation(
    config_model: type[BaseModel], field: str, is_candidates: bool, period: object
) -> None:
    value = [2, period] if is_candidates else period
    with pytest.raises(ValidationError) as exc_info:
        config_model.model_validate({field: value})

    expected_location = (field, 1) if is_candidates else (field,)
    assert exc_info.value.errors()[0]["loc"] == expected_location


@pytest.mark.parametrize("field", ["trend_ema_window", "trend_ema_candidates"])
@pytest.mark.parametrize("period", [0, 2, 100_000])
def test_trend_ema_accepts_disabled_or_valid_periods(field: str, period: int) -> None:
    value = [period] if field.endswith("candidates") else period
    config = GLFTStrategyConfig.model_validate({field: value})

    assert config.model_dump()[field] == value


@pytest.mark.parametrize("field", ["trend_ema_window", "trend_ema_candidates"])
@pytest.mark.parametrize("period", [-1, 1, 100_001, True, False, 2.0, "2"])
def test_trend_ema_rejects_invalid_periods(field: str, period: object) -> None:
    is_candidates = field.endswith("candidates")
    value = [0, period] if is_candidates else period
    with pytest.raises(ValidationError) as exc_info:
        GLFTStrategyConfig.model_validate({field: value})

    expected_location = (field, 1) if is_candidates else (field,)
    assert exc_info.value.errors()[0]["loc"] == expected_location
