"""Configuration models for the bundled safety-volume strategy."""

from typing import Self

from pydantic import BaseModel, model_validator

from tradingdev.domain.ml.features.context import FeatureContext
from tradingdev.domain.ml.features.features import FeatureEngineer
from tradingdev.domain.ml.features.risk_features import RiskFeatureEngineer
from tradingdev.domain.ml.rolling_gen import RollingGenConfig, resolve_geometry
from tradingdev.domain.ml.schemas import XGBoostModelConfig
from tradingdev.domain.strategies.bundled.indicator_parameters import (
    MovingAveragePeriod,
)


class SafetyVolumeStrategyConfig(BaseModel):
    """Safety-first volume strategy configuration."""

    risk_model: XGBoostModelConfig = XGBoostModelConfig()
    risk_threshold: float = 0.5
    risk_threshold_candidates: list[float] | None = None
    target_holding_bars: int = 5
    max_acceptable_loss_pct: float = 0.003
    fee_rate: float = 0.0011
    use_ml_direction: bool = False
    direction_model: XGBoostModelConfig = XGBoostModelConfig()
    sma_fast: MovingAveragePeriod = 5
    sma_slow: MovingAveragePeriod = 20
    min_holding_bars: int = 5
    max_holding_bars: int = 30
    lookback_candidates: list[int] = [360, 720, 1440]
    retrain_interval: int = 720
    validation_ratio: float = 0.2
    monthly_volume_target: float | None = None
    position_size: float = 3000.0
    rolling_gen: RollingGenConfig = RollingGenConfig()
    """Optional declarative rolling geometry; disabled unless enabled is set."""

    @model_validator(mode="after")
    def _validate_rolling_configuration(self) -> Self:
        self.validate_rolling_geometry()
        return self

    def validate_rolling_geometry(self) -> None:
        """Validate every candidate and enabled model before market-data loading."""
        if not self.lookback_candidates:
            raise ValueError("lookback_candidates must not be empty")
        scheme = self.rolling_gen.resolve() if self.rolling_gen.enabled else None
        for lookback in self.lookback_candidates:
            resolve_geometry(
                scheme,
                lookback=lookback,
                retrain_interval=self.retrain_interval,
                context=FeatureContext.for_features(
                    RiskFeatureEngineer(lookback=lookback)
                ),
                label_horizon=self.target_holding_bars,
            )
            if self.use_ml_direction:
                resolve_geometry(
                    scheme,
                    lookback=lookback,
                    retrain_interval=self.retrain_interval,
                    context=FeatureContext.for_features(
                        FeatureEngineer(lookback=lookback)
                    ),
                    label_horizon=1,
                )
