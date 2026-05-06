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

Allowed import roots for generated strategies are intentionally small:

- Python standard library: `__future__`, `collections`, `dataclasses`,
  `datetime`, `enum`, `math`, `statistics`, `typing`, `typing_extensions`.
- Runtime libraries: `numpy`, `pandas`.
- Project APIs: `tradingdev`.

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
  # source defaults to "binance_vision"; set "binance_api" to use ccxt instead
  # market_type defaults to "futures/um"; set "spot" for spot markets
  requirements:
    market:
      symbol: "BTC/USDT"
      timeframe: "1h"
    features: []
```

Feature sources are explicit:

```yaml
data:
  requirements:
    market:
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
2. `validate_strategy`: runs syntax, static policy, restricted import checks,
   ruff, mypy, inheritance, constructor, and signal-contract checks. It returns
   structured diagnostics with `level`, `code`, `phase`, `message`, and optional
   `fix`.
3. `dry_run_strategy`: accepts only `validated` strategies, performs a
   lightweight signal dry run, returns `signal_analysis`, and marks the
   strategy runnable when it passes.
4. `promote_strategy`: artifact tool that marks a runnable generated strategy as
   promoted.
5. `start_backtest` / `start_walk_forward`: execute only runnable or promoted
   generated strategies, and promoted bundled strategies.

## Security Model

Validation currently uses static policy checks plus restricted imports as a
first layer. It rejects known unsafe imports (`os`, `sys`, `subprocess`,
`socket`, `requests`, `httpx`, `ccxt`, `shutil`, `pathlib`), dynamic execution
calls (`eval`, `exec`, `__import__`), raw `open`, and common destructive file
operations such as `unlink`, `remove`, `rmtree`, and `write_text`.

This is not a full process sandbox. `validate_strategy` and `dry_run_strategy`
still import and execute generated Python to check class loading, constructor
behavior, BaseStrategy inheritance, input immutability, signal values, and smoke
dataframe output. Generated strategies must therefore be reviewed as runtime
code until a dedicated sandboxed execution layer is added.

`signal_analysis` includes row count, signal distribution, missing-signal count,
transition count, active-signal ratio, and timestamp bounds. Use it to debug
strategies that technically pass static checks but produce unusable signals.
