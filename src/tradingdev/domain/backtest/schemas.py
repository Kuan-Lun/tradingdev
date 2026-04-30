"""Domain schemas for backtest execution."""

from datetime import datetime
from typing import Self

from pydantic import BaseModel, field_validator, model_validator


class BacktestConfig(BaseModel):
    """Backtest execution configuration."""

    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    init_cash: float | None = None
    fees: float = 0.0006
    slippage: float = 0.0005
    position_size: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    signal_as_position: bool = False
    re_entry_after_sl: bool = True
    mode: str = "signal"
    monthly_max_loss: float = 1500.0

    @field_validator("end_date")
    @classmethod
    def end_after_start(cls, v: datetime, info: object) -> datetime:
        """Validate that end_date is after start_date."""
        return v

    @model_validator(mode="after")
    def signal_mode_requires_init_cash(self) -> Self:
        """Signal mode requires init_cash to be set."""
        if self.mode == "signal" and self.init_cash is None:
            msg = "init_cash is required when mode is 'signal'"
            raise ValueError(msg)
        return self


class ParallelConfig(BaseModel):
    """Parallel execution configuration."""

    reserve_cores: int = 2
    safety_factor: float = 0.6
    overhead_multiplier: float = 3.0

    @field_validator("reserve_cores")
    @classmethod
    def reserve_cores_non_negative(cls, v: int) -> int:
        """Validate reserve_cores >= 0."""
        if v < 0:
            msg = "reserve_cores must be non-negative"
            raise ValueError(msg)
        return v

    @field_validator("safety_factor")
    @classmethod
    def safety_factor_range(cls, v: float) -> float:
        """Validate 0 < safety_factor <= 1."""
        if not 0 < v <= 1:
            msg = "safety_factor must be between 0 (exclusive) and 1 (inclusive)"
            raise ValueError(msg)
        return v


class WalkForwardConfig(BaseModel):
    """Walk-forward validation configuration."""

    train_start: datetime | None = None
    train_end: datetime | None = None
    test_start: datetime | None = None
    test_end: datetime | None = None
    n_splits: int = 1
    train_ratio: float = 0.8
    expanding: bool = False
    target_metric: str = "sharpe_ratio"
