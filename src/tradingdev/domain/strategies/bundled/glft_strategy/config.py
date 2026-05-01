"""Configuration models for the bundled GLFT strategy."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, field_validator, model_validator


class GLFTStrategyConfig(BaseModel):
    """GLFT market-making strategy configuration."""

    gamma: float = 500.0
    kappa: float = 1000.0
    ema_window: int = 21
    vol_window: int = 30
    vol_type: Literal["realized", "parkinson", "implied"] = "realized"
    min_holding_bars: int = 5
    max_holding_bars: int = 30
    gamma_candidates: list[float] = [0.0, 200.0, 500.0, 1000.0]
    kappa_candidates: list[float] = [500.0, 1000.0, 5000.0]
    ema_window_candidates: list[int] = [10, 21, 50]
    max_holding_bars_candidates: list[int] = [30]
    vol_window_candidates: list[int] = [30]
    target_metric: str = "total_return"
    position_size: float = 3000.0
    monthly_volume_target: float | None = None
    fee_rate: float = 0.0006
    min_entry_edge: float = 0.0012
    min_entry_edge_candidates: list[float] = [0.0012, 0.0015, 0.002, 0.003]
    trend_ema_window: int = 0
    trend_ema_candidates: list[int] = [0]
    profit_target_ratio: float = 1.0
    profit_target_ratio_candidates: list[float] = [0.5, 0.75, 1.0]
    strategy_sl: float = 0.005
    momentum_guard: bool = True
    signal_agg_minutes: int = 1
    signal_agg_minutes_candidates: list[int] = [1]
    dynamic_sizing: bool = False
    min_position_size: float = 500.0
    edge_for_full_size: float = 0.005
    edge_for_full_size_candidates: list[float] = [0.003, 0.005, 0.008]
    min_monthly_pnl: float | None = None

    @field_validator("gamma")
    @classmethod
    def gamma_non_negative(cls, v: float) -> float:
        """Validate gamma >= 0."""
        if v < 0:
            msg = "gamma must be non-negative"
            raise ValueError(msg)
        return v

    @field_validator("kappa")
    @classmethod
    def kappa_positive(cls, v: float) -> float:
        """Validate kappa > 0."""
        if v <= 0:
            msg = "kappa must be positive"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def max_gt_min_holding(self) -> Self:
        """Validate max_holding_bars > min_holding_bars."""
        if self.max_holding_bars <= self.min_holding_bars:
            msg = "max_holding_bars must be greater than min_holding_bars"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def min_position_le_max(self) -> Self:
        """Validate min_position_size <= position_size."""
        if self.dynamic_sizing and self.min_position_size > self.position_size:
            msg = "min_position_size must be <= position_size"
            raise ValueError(msg)
        return self
