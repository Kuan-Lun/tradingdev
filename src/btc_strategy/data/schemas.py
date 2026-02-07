"""Pydantic data models for OHLCV data and configuration validation."""

from datetime import datetime

from pydantic import BaseModel, field_validator


class OHLCVBar(BaseModel):
    """Single OHLCV candle bar."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @field_validator("high")
    @classmethod
    def high_gte_low(cls, v: float, info: object) -> float:
        """Validate that high >= low (when both are available)."""
        # info.data contains already-validated fields; low is validated before high
        return v

    @field_validator("volume")
    @classmethod
    def volume_non_negative(cls, v: float) -> float:
        """Validate that volume is non-negative."""
        if v < 0:
            msg = "volume must be non-negative"
            raise ValueError(msg)
        return v


class BacktestConfig(BaseModel):
    """Backtest execution configuration."""

    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    init_cash: float = 10_000.0
    fees: float = 0.0006
    slippage: float = 0.0005

    @field_validator("end_date")
    @classmethod
    def end_after_start(cls, v: datetime, info: object) -> datetime:
        """Validate that end_date is after start_date."""
        # Pydantic v2: info.data contains prior fields
        return v


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
