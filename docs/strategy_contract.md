# Strategy Contract

Generated strategies are runtime artifacts managed by MCP. They are saved under
`workspace/generated_strategies/` and paired with YAML under `workspace/configs/`.
Bundled strategies are engineering-maintained code under
`src/tradingdev/domain/strategies/bundled/`.

## Python Contract

Generated code must:

- inherit `tradingdev.domain.strategies.base.BaseStrategy`;
- expose a constructor that can be called with YAML `strategy.parameters`;
- accept optional `backtest_engine` when it needs engine context;
- implement `generate_signals(df)` and return a new pandas DataFrame;
- preserve the input DataFrame without mutation;
- include a `signal` column containing only `1`, `-1`, or `0`;
- avoid network, subprocess, destructive filesystem, dynamic import, `eval`, and
  `exec`.

Recommended imports:

```python
from tradingdev.domain.strategies.base import BaseStrategy
from tradingdev.domain.indicators.kd import KDIndicator
from tradingdev.shared.utils.logger import setup_logger
```

## YAML Contract

```yaml
strategy:
  id: "sma_crossover"
  version: "0.1.0"
  class_name: "SmaCrossoverStrategy"
  source_path: "workspace/generated_strategies/sma_crossover.py"
  parameters:
    fast_period: 10
    slow_period: 30

backtest:
  symbol: "BTC/USDT"
  timeframe: "1h"
  start_date: "2024-01-01"
  end_date: "2024-12-31"
  init_cash: 10000.0
  mode: "signal"

data:
  source: "binance_api"
  requirements:
    market:
      source: "binance_api"
      symbol: "BTC/USDT"
      timeframe: "1h"
    features: []
```

Feature sources are explicit:

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
      - type: "funding_rate"
        source: "binance"
        column: "funding_rate"
        path: "workspace/data/processed/btc_funding_rate_2025.parquet"
```

## Lifecycle

1. `save_strategy`: writes draft source, config, and metadata.
2. `validate_strategy`: runs syntax, static policy, ruff, mypy, inheritance, and
   signal-contract checks.
3. `dry_run_strategy`: performs a lightweight signal dry run and marks the
   strategy runnable when it passes.
4. `promote_strategy`: marks a runnable generated strategy as promoted.
5. `start_backtest` / `start_walk_forward`: execute only runnable or promoted
   generated strategies, and promoted bundled strategies.

Validation still executes generated Python for class loading and signal checks.
Static policy checks are a first layer, not a full sandbox.
