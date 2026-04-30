"""Configuration models for the bundled quantile strategy."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, field_validator, model_validator


class QuantileStrategyConfig(BaseModel):
    """XGBoost quantile regression volume strategy configuration."""

    quantiles: list[float] = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    horizon: int = 30
    horizon_candidates: list[int] = []
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    min_holding_bars: int = 5
    profit_target: float = 0.003
    strategy_sl: float = 0.003
    max_p_both: float = 0.15
    max_p_neither: float = 0.20
    min_ratio: float = 1.5
    min_ratio_candidates: list[float] = []
    min_entry_edge: float = 0.0015
    min_entry_edge_candidates: list[float] = []
    dynamic_sizing: bool = True
    min_position_size: float = 50000.0
    edge_for_full_size: float = 0.005
    edge_for_full_size_candidates: list[float] = []
    position_size: float = 50000.0
    monthly_volume_target: float = 12500000.0
    fee_rate: float = 0.0005
    exit_threshold: float = 0.5
    exit_threshold_candidates: list[float] = []
    validator_n_estimators: int = 100
    validator_max_depth: int = 4
    target_metric: str = "total_volume"
    min_monthly_pnl: float = -1500.0
    train_subsample_step: int = 5
    retrain_interval: int = 1440
    train_window: int = 20160
    dvol_raw_path: str = ""
    dvol_processed_path: str = ""
    funding_rate_path: str = ""

    @field_validator("min_entry_edge")
    @classmethod
    def min_entry_edge_positive(cls, v: float) -> float:
        """Validate min_entry_edge > 0."""
        if v <= 0:
            msg = "min_entry_edge must be positive"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def quantile_min_position_le_max(self) -> Self:
        """Validate min_position_size <= position_size."""
        if self.dynamic_sizing and self.min_position_size > self.position_size:
            msg = "min_position_size must be <= position_size"
            raise ValueError(msg)
        return self
