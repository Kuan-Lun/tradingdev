"""Configuration models for the bundled KD strategy."""

from __future__ import annotations

from pydantic import BaseModel, field_validator


class KDStrategyConfig(BaseModel):
    """KD Stochastic Oscillator strategy parameters."""

    k_period: int = 14
    d_period: int = 3
    smooth_k: int = 3
    overbought: float = 80.0
    oversold: float = 20.0

    @field_validator("overbought")
    @classmethod
    def overbought_range(cls, v: float) -> float:
        """Validate overbought is between 0 and 100."""
        if not 0 <= v <= 100:
            msg = "overbought must be between 0 and 100"
            raise ValueError(msg)
        return v

    @field_validator("oversold")
    @classmethod
    def oversold_range(cls, v: float) -> float:
        """Validate oversold is between 0 and 100."""
        if not 0 <= v <= 100:
            msg = "oversold must be between 0 and 100"
            raise ValueError(msg)
        return v


class KDFitConfig(BaseModel):
    """Grid search configuration for KDStrategy.fit()."""

    k_period_range: list[int] = [9, 14, 21]
    d_period_range: list[int] = [3, 5]
    smooth_k_range: list[int] = [3, 5]
    overbought_range: list[float] = [70.0, 80.0, 90.0]
    oversold_range: list[float] = [10.0, 20.0, 30.0]
    target_metric: str = "sharpe_ratio"
