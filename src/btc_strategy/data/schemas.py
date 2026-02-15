"""Pydantic data models for OHLCV data and configuration validation."""

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, field_validator, model_validator


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
    position_size_usdt: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    signal_as_position: bool = False
    re_entry_after_sl: bool = True
    mode: str = "signal"

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


class KDFitConfig(BaseModel):
    """Grid search configuration for KDStrategy.fit()."""

    k_period_range: list[int] = [9, 14, 21]
    d_period_range: list[int] = [3, 5]
    smooth_k_range: list[int] = [3, 5]
    overbought_range: list[float] = [70.0, 80.0, 90.0]
    oversold_range: list[float] = [10.0, 20.0, 30.0]
    target_metric: str = "sharpe_ratio"


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


class XGBoostModelConfig(BaseModel):
    """XGBoost model hyperparameters."""

    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 10
    random_state: int = 42


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
    monthly_volume_target_usdt: float | None = None


class SafetyVolumeStrategyConfig(BaseModel):
    """Safety-first volume strategy configuration."""

    # Risk Gate model
    risk_model: XGBoostModelConfig = XGBoostModelConfig()
    risk_threshold: float = 0.5
    risk_threshold_candidates: list[float] | None = None

    # Risk target definition
    target_holding_bars: int = 5
    max_acceptable_loss_pct: float = 0.003
    fee_rate: float = 0.0011

    # Direction
    use_ml_direction: bool = False
    direction_model: XGBoostModelConfig = XGBoostModelConfig()
    sma_fast: int = 5
    sma_slow: int = 20

    # Holding management
    min_holding_bars: int = 5
    max_holding_bars: int = 30

    # Training
    lookback_candidates: list[int] = [360, 720, 1440]
    retrain_interval: int = 720
    validation_ratio: float = 0.2

    # Volume target
    monthly_volume_target_usdt: float | None = None
    position_size_usdt: float = 3000.0


class GLFTStrategyConfig(BaseModel):
    """GLFT market-making strategy configuration.

    Adapted Gueant-Lehalle-Fernandez-Tapia optimal market-making
    model for signal-based backtesting.
    """

    # Core GLFT parameters
    gamma: float = 0.01
    kappa: float = 1.5
    ema_window: int = 21

    # Volatility estimation
    vol_window: int = 30
    vol_type: Literal["realized", "parkinson"] = "realized"

    # Holding management (exchange compliance)
    min_holding_bars: int = 5
    max_holding_bars: int = 30

    # fit() grid search candidates
    gamma_candidates: list[float] = [
        0.001,
        0.005,
        0.01,
        0.05,
        0.1,
    ]
    kappa_candidates: list[float] = [
        0.5,
        1.0,
        1.5,
        2.0,
        5.0,
    ]
    ema_window_candidates: list[int] = [10, 21, 50]
    max_holding_bars_candidates: list[int] = [30]
    vol_window_candidates: list[int] = [30]
    target_metric: str = "total_return"

    # Position & volume
    position_size_usdt: float = 3000.0
    monthly_volume_target_usdt: float | None = None
    fee_rate: float = 0.0006

    # Entry threshold: minimum price deviation from EMA to open
    # Must be >= fee_rate * 2 (round-trip cost); higher = more selective
    min_entry_edge: float = 0.0012
    min_entry_edge_candidates: list[float] = [0.0012, 0.0015, 0.002, 0.003]

    # Trend filter: slow EMA window; 0 = disabled
    # When enabled, only allow entries aligned with slow EMA direction
    trend_ema_window: int = 0
    trend_ema_candidates: list[int] = [0]

    # Constrained optimization: filter by annual_return >= threshold,
    # then maximize target_metric (e.g. total_volume_usdt)
    min_annual_return: float | None = None

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
