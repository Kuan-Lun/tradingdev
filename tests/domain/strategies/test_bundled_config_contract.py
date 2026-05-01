"""Bundled strategy config contract tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from tradingdev.domain.strategies.bundled.glft_ml_strategy.config import (
    GLFTMLStrategyConfig,
)
from tradingdev.domain.strategies.bundled.glft_strategy.config import (
    GLFTStrategyConfig,
)
from tradingdev.domain.strategies.bundled.quantile_strategy.config import (
    QuantileStrategyConfig,
)

_BUNDLED_ROOT = Path("src/tradingdev/domain/strategies/bundled")
_DATA_PATH_PARAMETERS = {
    "dvol_raw_path",
    "dvol_processed_path",
    "funding_rate_path",
}


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
