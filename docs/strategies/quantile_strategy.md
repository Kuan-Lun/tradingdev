# Quantile Regime Volume Strategy

Source:
`src/tradingdev/domain/strategies/bundled/quantile_strategy/strategy.py`

Config:
`src/tradingdev/domain/strategies/bundled/quantile_strategy/config.yaml`

## Config Schema

```yaml
strategy:
  id: "quantile_volume"
  class_name: "QuantileStrategy"
  source_path: "src/tradingdev/domain/strategies/bundled/quantile_strategy/strategy.py"
  parameters:
    horizon: 30
    horizon_candidates: [15, 30]
    min_entry_edge_candidates: [0.25, 0.30, 0.35, 0.40]
    dvol_processed_path: "workspace/data/processed/btc_dvol_1m_2024_2025.parquet"
    funding_rate_path: "workspace/data/processed/btc_funding_rate_2025.parquet"
```

## Intent

This bundled strategy trains an XGBoost regime classifier. It classifies each
bar into `long_only`, `short_only`, `both`, or `neither`, then trades only when
one direction has a clear edge.

## Signal Logic

- `fit()` trains regime classifiers for candidate horizons and searches entry
  edge plus sizing parameters.
- The state machine skips regimes where `P(both)` or `P(neither)` is too high.
- Dynamic sizing scales exposure by directional edge when enabled.
- Signals follow the project convention: `1` long, `-1` short, `0` flat.

## Data Requirements

The config declares OHLCV plus two feature sources:

```yaml
data:
  requirements:
    market:
      source: "binance_api"
      symbol: "BTC/USDT"
      timeframe: "1m"
    features:
      - type: "dvol"
        source: "deribit"
        column: "dvol"
        path: "workspace/data/processed/btc_dvol_1m_2024_2025.parquet"
        raw_path: "workspace/data/raw/btc_dvol_1m_2024_2025.csv"
      - type: "funding_rate"
        source: "binance"
        column: "funding_rate"
        path: "workspace/data/processed/btc_funding_rate_2025.parquet"
```

`DataService.inspect_dataset(config_path)` reports whether each feature path
exists, how many rows it has, and how many missing values are present in the
declared column.

## Key Parameters

- `horizon_candidates`: prediction horizons explored in `fit()`.
- `min_entry_edge_candidates`: directional edge thresholds.
- `edge_for_full_size_candidates`: dynamic sizing scale candidates.
- `retrain_interval` and `train_window`: rolling retrain cadence and context.
- `target_metric` and `min_monthly_pnl`: optimization objective and guardrail.
