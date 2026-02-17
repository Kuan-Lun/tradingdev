"""Strategy registry for dynamic strategy creation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from btc_strategy.data.schemas import (
    GLFTMLStrategyConfig,
    GLFTStrategyConfig,
    KDFitConfig,
    KDStrategyConfig,
    SafetyVolumeStrategyConfig,
    XGBoostStrategyConfig,
)
from btc_strategy.strategies.glft_ml_strategy import GLFTMLStrategy
from btc_strategy.strategies.glft_strategy import GLFTStrategy
from btc_strategy.strategies.kd_strategy import KDStrategy
from btc_strategy.strategies.safety_volume_strategy import (
    SafetyVolumeStrategy,
)
from btc_strategy.strategies.xgboost_strategy import (
    XGBoostStrategy,
)

if TYPE_CHECKING:
    from btc_strategy.backtest.base_engine import (
        BaseBacktestEngine,
    )
    from btc_strategy.strategies.base import BaseStrategy


def create_strategy(
    raw_config: dict[str, Any],
    engine: BaseBacktestEngine,
) -> BaseStrategy:
    """Build a strategy instance from YAML configuration.

    Raises:
        ValueError: If the strategy name is not recognised.
    """
    name: str = raw_config["strategy"]["name"]
    params = raw_config["strategy"]["parameters"]

    if name == "kd_crossover":
        strategy_config = KDStrategyConfig(**params)
        fit_config: KDFitConfig | None = None
        if "fit" in raw_config["strategy"]:
            fit_config = KDFitConfig(**raw_config["strategy"]["fit"])
        return KDStrategy(
            config=strategy_config,
            fit_config=fit_config,
            backtest_engine=engine,
        )

    if name == "xgboost_direction":
        strategy_config_xgb = XGBoostStrategyConfig(**params)
        return XGBoostStrategy(
            config=strategy_config_xgb,
            backtest_engine=engine,
        )

    if name == "safety_first_volume":
        sfv_config = SafetyVolumeStrategyConfig(**params)
        return SafetyVolumeStrategy(
            config=sfv_config,
            backtest_engine=engine,
        )

    if name == "glft_market_making":
        glft_config = GLFTStrategyConfig(**params)
        return GLFTStrategy(
            config=glft_config,
            backtest_engine=engine,
        )

    if name == "glft_ml":
        glft_ml_config = GLFTMLStrategyConfig(**params)
        return GLFTMLStrategy(
            config=glft_ml_config,
            backtest_engine=engine,
        )

    msg = f"Unknown strategy: {name}"
    raise ValueError(msg)
