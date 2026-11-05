"""Configuration models for the bundled XGBoost strategy."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, model_validator

from tradingdev.domain.ml.features.context import FeatureContext
from tradingdev.domain.ml.features.features import FeatureEngineer
from tradingdev.domain.ml.rolling_gen import RollingGenConfig, resolve_geometry
from tradingdev.domain.ml.schemas import XGBoostModelConfig


class XGBoostStrategyConfig(BaseModel):
    """XGBoost direction prediction strategy configuration."""

    model: XGBoostModelConfig = XGBoostModelConfig()
    lookback_candidates: list[int] = [12, 24, 48, 96, 168]
    retrain_interval: int = 24
    validation_ratio: float = 0.2
    signal_threshold: float = 0.55
    signal_threshold_candidates: list[float] | None = None
    target_horizon: int = 1
    min_bars_between_trades: int = 1
    monthly_volume_target: float | None = None
    rolling_gen: RollingGenConfig = RollingGenConfig()
    """Optional declarative rolling geometry; disabled unless enabled is set."""

    @model_validator(mode="after")
    def _validate_rolling_configuration(self) -> Self:
        self.validate_rolling_geometry()
        return self

    def validate_rolling_geometry(self) -> None:
        """Reject impossible feature contexts without loading market data."""
        if not self.lookback_candidates:
            raise ValueError("lookback_candidates must not be empty")
        scheme = self.rolling_gen.resolve() if self.rolling_gen.enabled else None
        for lookback in self.lookback_candidates:
            resolve_geometry(
                scheme,
                lookback=lookback,
                retrain_interval=self.retrain_interval,
                context=FeatureContext.for_features(FeatureEngineer(lookback=lookback)),
                label_horizon=self.target_horizon,
            )
