# XGBoost Direction Strategy

Source:
`src/tradingdev/domain/strategies/bundled/xgboost_strategy/strategy.py`

Config:
`src/tradingdev/domain/strategies/bundled/xgboost_strategy/config.yaml`

## Config Schema

```yaml
strategy:
  id: "xgboost_direction"
  class_name: "XGBoostStrategy"
  source_path: "src/tradingdev/domain/strategies/bundled/xgboost_strategy/strategy.py"
  parameters:
    lookback_candidates: [360, 720, 1440]
    retrain_interval: 720
    target_horizon: 5
    signal_threshold_candidates: [0.0, 0.35, 0.40, 0.45, 0.50]

data:
  requirements:
    market:
      source: "binance_api"
      symbol: "BTC/USDT"
      timeframe: "1m"
    features: []
```

## Intent

This bundled strategy trains an XGBoost direction model and uses rolling
prediction to produce high-frequency long/short signals in volume mode. It is
designed for research on market-making volume targets, not live execution.

## Signal Logic

- `fit()` searches candidate lookback windows by validation accuracy.
- Optional threshold search uses the backtest engine to choose a signal
  threshold from `signal_threshold_candidates`.
- `generate_signals()` requires a fitted model and rolling retrains according to
  `retrain_interval`.
- Signals follow the project convention: `1` long, `-1` short, `0` flat.

## Data Requirements

The strategy declares market OHLCV only:

```yaml
data:
  requirements:
    market:
      source: "binance_api"
      symbol: "BTC/USDT"
      timeframe: "1m"
    features: []
```

Use `inspect_dataset(config_path)` before running long validations to verify
the workspace or `TRADINGDEV_DATA_ROOT` cache.

## Key Parameters

- `lookback_candidates`: candidate training windows.
- `retrain_interval`: bars between rolling retrains.
- `target_horizon`: direction prediction horizon.
- `signal_threshold_candidates`: thresholds explored during `fit()`.
- `monthly_volume_target`: research target used by volume-mode analysis.
