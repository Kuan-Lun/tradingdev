"""Domain schemas for market data."""

from datetime import datetime

from pydantic import BaseModel, field_validator


class DVOLBar(BaseModel):
    """Single Deribit DVOL (implied volatility index) data point."""

    timestamp: datetime
    dvol_open: float
    dvol_high: float
    dvol_low: float
    dvol_close: float

    @field_validator("dvol_close")
    @classmethod
    def dvol_close_positive(cls, v: float) -> float:
        """Validate dvol_close > 0."""
        if v <= 0:
            msg = "dvol_close must be positive"
            raise ValueError(msg)
        return v


class OHLCVBar(BaseModel):
    """Single OHLCV candle bar."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @field_validator("volume")
    @classmethod
    def volume_non_negative(cls, v: float) -> float:
        """Validate that volume is non-negative."""
        if v < 0:
            msg = "volume must be non-negative"
            raise ValueError(msg)
        return v


class DataConfig(BaseModel):
    """Data source and path configuration."""

    source: str = "binance_api"
    raw_dir: str = "workspace/data/raw"
    processed_dir: str = "workspace/data/processed"
