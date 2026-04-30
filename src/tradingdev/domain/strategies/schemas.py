"""Domain schemas for strategy configuration and lifecycle state."""

from typing import Literal, Self

from pydantic import BaseModel, field_validator, model_validator

from tradingdev.domain.ml.schemas import XGBoostModelConfig


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
    monthly_volume_target: float | None = None
    position_size: float = 3000.0


class GLFTStrategyConfig(BaseModel):
    """GLFT market-making strategy configuration.

    Adapted Gueant-Lehalle-Fernandez-Tapia optimal market-making
    model for signal-based backtesting.
    """

    # Core GLFT parameters (dimensionless %-space)
    # gamma and kappa operate in pure percentage space,
    # independent of the asset's price level.
    gamma: float = 500.0
    kappa: float = 1000.0
    ema_window: int = 21

    # Volatility estimation
    vol_window: int = 30
    vol_type: Literal["realized", "parkinson", "implied"] = "realized"

    # DVOL data paths (required when vol_type == "implied")
    dvol_raw_path: str | None = None
    dvol_processed_path: str | None = None

    # Holding management (exchange compliance)
    min_holding_bars: int = 5
    max_holding_bars: int = 30

    # fit() grid search candidates
    gamma_candidates: list[float] = [
        0.0,
        200.0,
        500.0,
        1000.0,
    ]
    kappa_candidates: list[float] = [
        500.0,
        1000.0,
        5000.0,
    ]
    ema_window_candidates: list[int] = [10, 21, 50]
    max_holding_bars_candidates: list[int] = [30]
    vol_window_candidates: list[int] = [30]
    target_metric: str = "total_return"

    # Position & volume
    position_size: float = 3000.0
    monthly_volume_target: float | None = None
    fee_rate: float = 0.0006

    # Entry threshold: minimum price deviation from EMA to open
    # Must be >= fee_rate * 2 (round-trip cost); higher = more selective
    min_entry_edge: float = 0.0012
    min_entry_edge_candidates: list[float] = [0.0012, 0.0015, 0.002, 0.003]

    # Trend filter: slow EMA window; 0 = disabled
    # When enabled, only allow entries aligned with slow EMA direction
    trend_ema_window: int = 0
    trend_ema_candidates: list[int] = [0]

    # Exit: profit target — fraction of entry deviation to capture
    # before exiting.  1.0 = wait for full mean-reversion to EMA;
    # >1.0 = wait for overshoot beyond EMA.
    profit_target_ratio: float = 1.0
    profit_target_ratio_candidates: list[float] = [0.5, 0.75, 1.0]

    # Exit: strategy-level stop-loss — max additional adverse
    # deviation (in normalised units) beyond entry deviation.
    # 0 = disabled.
    strategy_sl: float = 0.005

    # Entry: momentum guard — only enter when deviation is
    # narrowing (price moving back toward EMA), not widening.
    momentum_guard: bool = True

    # Multi-timeframe: aggregate N 1-min bars into one for EMA
    # computation.  1 = no aggregation (default); 5 = 5-min EMA.
    # Execution remains at 1-min resolution.
    signal_agg_minutes: int = 1
    signal_agg_minutes_candidates: list[int] = [1]

    # Dynamic position sizing: scale position by |deviation| / edge_for_full_size.
    # When disabled, all positions use position_size (legacy behaviour).
    dynamic_sizing: bool = False
    min_position_size: float = 500.0
    edge_for_full_size: float = 0.005
    edge_for_full_size_candidates: list[float] = [0.003, 0.005, 0.008]

    # Constrained optimization: filter by estimated monthly PnL >= threshold,
    # then maximize target_metric (e.g. total_volume)
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
    def implied_requires_dvol_path(self) -> Self:
        """Validate dvol_processed_path is set when vol_type is 'implied'."""
        if self.vol_type == "implied" and self.dvol_processed_path is None:
            msg = "dvol_processed_path is required when vol_type is 'implied'"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def min_position_le_max(self) -> Self:
        """Validate min_position_size <= position_size."""
        if self.dynamic_sizing and self.min_position_size > self.position_size:
            msg = "min_position_size must be <= position_size"
            raise ValueError(msg)
        return self


class QuantileStrategyConfig(BaseModel):
    """XGBoost quantile regression volume strategy configuration."""

    # Quantile model
    quantiles: list[float] = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    horizon: int = 30
    horizon_candidates: list[int] = []

    # XGBoost hyperparameters
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8

    # Holding management (max_holding = horizon)
    min_holding_bars: int = 5

    # Profit target for label definition (what counts as "opportunity")
    profit_target: float = 0.003

    # Strategy-level stop-loss (tighter than engine hard SL)
    strategy_sl: float = 0.003

    # Regime filters: skip entry when P(both) or P(neither) too high
    max_p_both: float = 0.15
    max_p_neither: float = 0.20

    # Risk/reward entry: minimum P(win)/P(lose) ratio
    min_ratio: float = 1.5
    min_ratio_candidates: list[float] = []

    # Entry threshold (minimum P(opportunity) to consider)
    min_entry_edge: float = 0.0015
    min_entry_edge_candidates: list[float] = []

    # Dynamic sizing
    dynamic_sizing: bool = True
    min_position_size: float = 50000.0
    edge_for_full_size: float = 0.005
    edge_for_full_size_candidates: list[float] = []

    # Position & volume
    position_size: float = 50000.0
    monthly_volume_target: float = 12500000.0
    fee_rate: float = 0.0005

    # Thesis validator
    exit_threshold: float = 0.5
    exit_threshold_candidates: list[float] = []
    validator_n_estimators: int = 100
    validator_max_depth: int = 4

    # fit() objective
    target_metric: str = "total_volume"
    min_monthly_pnl: float = -1500.0

    # Training subsample: take every Nth bar to reduce target overlap
    train_subsample_step: int = 5

    # Rolling retrain
    retrain_interval: int = 1440
    train_window: int = 20160

    # DVOL (optional)
    dvol_raw_path: str = ""
    dvol_processed_path: str = ""

    # Funding rate (optional)
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


class GLFTMLStrategyConfig(BaseModel):
    """GLFT + ML direction prediction strategy configuration.

    Combines the GLFT analytical spread model with an AutoGluon ML
    direction predictor for limit-order market making.
    """

    # --- ML parameters ---
    prediction_horizon: int = 5
    feature_lookback: int = 60
    ml_time_limit: int = 300
    ml_presets: str = "medium_quality"
    confidence_threshold: float = 0.55
    confidence_threshold_candidates: list[float] = [0.52, 0.55, 0.60]

    # --- GLFT core parameters (%-space) ---
    gamma: float = 0.0
    kappa: float = 1000.0
    ema_window: int = 15

    # --- Volatility ---
    vol_window: int = 30
    vol_type: Literal["realized", "parkinson", "implied"] = "implied"
    dvol_raw_path: str | None = None
    dvol_processed_path: str | None = None

    # --- Holding management ---
    min_holding_bars: int = 5
    max_holding_bars: int = 13

    # --- Grid search candidates ---
    gamma_candidates: list[float] = [0.0, 200.0, 500.0]
    kappa_candidates: list[float] = [500.0, 1000.0]
    ema_window_candidates: list[int] = [5, 15, 30, 75]
    max_holding_bars_candidates: list[int] = [8, 13, 30]
    min_entry_edge_candidates: list[float] = [0.0008, 0.0012, 0.002]
    profit_target_ratio_candidates: list[float] = [0.5, 0.75, 1.0]
    target_metric: str = "total_volume"

    # --- Entry/Exit ---
    min_entry_edge: float = 0.0012
    profit_target_ratio: float = 0.75
    strategy_sl: float = 0.003

    # --- Position & volume ---
    position_size: float = 3000.0
    monthly_volume_target: float | None = None
    fee_rate: float = 0.0002  # Maker fee

    # --- Constrained optimisation ---
    min_monthly_pnl: float | None = None

    @model_validator(mode="after")
    def glft_ml_max_gt_min_holding(self) -> Self:
        """Validate max_holding_bars > min_holding_bars."""
        if self.max_holding_bars <= self.min_holding_bars:
            msg = "max_holding_bars must be greater than min_holding_bars"
            raise ValueError(msg)
        return self
