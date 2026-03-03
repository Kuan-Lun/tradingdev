"""Strategy registry for dynamic strategy creation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quant_backtest.data.schemas import (
    GLFTMLStrategyConfig,
    GLFTStrategyConfig,
    KDFitConfig,
    KDStrategyConfig,
    ParallelConfig,
    SafetyVolumeStrategyConfig,
    XGBoostStrategyConfig,
)
from strategies.glft_ml_strategy import GLFTMLStrategy
from strategies.glft_strategy import GLFTStrategy
from strategies.kd_strategy import KDStrategy
from strategies.safety_volume_strategy import (
    SafetyVolumeStrategy,
)
from strategies.xgboost_strategy import (
    XGBoostStrategy,
)

if TYPE_CHECKING:
    from quant_backtest.backtest.base_engine import (
        BaseBacktestEngine,
    )
    from quant_backtest.strategies.base import BaseStrategy


def create_strategy(
    raw_config: dict[str, Any],
    engine: BaseBacktestEngine,
    parallel_config: ParallelConfig | None = None,
) -> BaseStrategy:
    """Build a strategy instance from YAML configuration.

    Args:
        raw_config: Raw YAML configuration dictionary.
        engine: Backtest engine instance.
        parallel_config: Parallel execution settings.  When ``None``,
            default values are used.

    Raises:
        ValueError: If the strategy name is not recognised.
    """
    name: str = raw_config["strategy"]["name"]
    params = raw_config["strategy"]["parameters"]
    p_cfg = parallel_config or ParallelConfig()

    if name == "kd_crossover":
        strategy_config = KDStrategyConfig(**params)
        fit_config: KDFitConfig | None = None
        if "fit" in raw_config["strategy"]:
            fit_config = KDFitConfig(**raw_config["strategy"]["fit"])
        return KDStrategy(
            config=strategy_config,
            fit_config=fit_config,
            backtest_engine=engine,
            parallel_config=p_cfg,
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
            parallel_config=p_cfg,
        )

    if name == "glft_ml":
        glft_ml_config = GLFTMLStrategyConfig(**params)
        return GLFTMLStrategy(
            config=glft_ml_config,
            backtest_engine=engine,
            parallel_config=p_cfg,
        )

    msg = f"Unknown strategy: {name}"
    raise ValueError(msg)
