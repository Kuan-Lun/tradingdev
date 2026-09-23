# Strategy Contract

Generated strategies are runtime artifacts managed by MCP. They are saved under
`workspace/generated_strategies/<strategy_id>/revisions/<revision_id>/` with
`strategy.py`, `config.yaml`, and `metadata.json`. Source and base config are
immutable through the service; metadata records the selected revision's lifecycle
and validation evidence. `current.json` in the strategy directory points to the
most recently saved revision.
Bundled strategies are engineering-maintained code under
`src/tradingdev/domain/strategies/bundled/`.

Generated code must pass the repository's Ruff and strict Mypy rules from
`pyproject.toml`, including the Pydantic plugin and explicit dependency overrides.
The installed wheel carries this same policy. Caller-local configuration cannot
relax it. Mypy follows imports silently so diagnostics target the generated file;
unknown imports and missing annotations are still rejected.

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

Validation, dry-run, backtest and walk-forward use the same constructor binding:
YAML parameters become keyword arguments. Missing required parameters and names
the constructor cannot accept fail validation. `backtest_engine` is injected by
the application and cannot be overridden in YAML. Saving a revised draft
creates a new revision and loads that source, including
same-size edits made within one filesystem timestamp interval. An existing
revision is never replaced by a later save.

Allowed import roots for generated strategies are intentionally small:

- Python standard library: `__future__`, `collections`, `dataclasses`,
  `datetime`, `enum`, `math`, `statistics`, `typing`.
- Runtime libraries: `numpy`, `pandas`, `talib`, `typing_extensions`.
- Project APIs: `tradingdev`.

Prefer `tradingdev.domain.indicators` (`sma`, `ema`, `rsi`, `macd`,
`bollinger_bands`, `atr`, `adx`, `stochastic`) for standard indicators. This
facade calls the official TA-Lib Python wrapper, preserves the input index,
and exposes named result fields for indicators with multiple outputs. Bundled
strategies and feature engineering must use this facade rather than call
TA-Lib directly. Bollinger Bands use population standard deviation.

The facade preserves TA-Lib's native warm-up NaNs and NaN propagation, including
NaNs that can extend to the end of the output after an interior missing value.
Short inputs therefore remain unready for indicators whose full lookback has
not elapsed. Keep signals flat until every value required by the signal is
finite; do not backfill indicators from future rows or substitute zero for
unavailable indicator values. OHLC inputs must have identical indexes and
lengths, and infinite input values are rejected. Empty inputs produce empty
float64 outputs with the same index.

Periods must be integers other than booleans and at most 100,000. SMA, EMA,
RSI, ADX, Bollinger Bands, and MACD fast/slow periods require at least 2 bars;
ATR, MACD signal, and Stochastic periods allow 1. Bollinger Bands expose only
population standard deviation: the previous `ddof` parameter is removed.
The `std` argument to `bollinger_bands` must be finite and in the inclusive
range `0 <= std <= 3e37`. Negative values, values above `3e37`, NaN, and
positive or negative infinity raise `ValueError`.

Generated strategies may also import `talib` directly. Use its Function API
with float64 NumPy arrays, unpack multi-output tuples, and restore the input
index when constructing pandas Series. The output order is `line, signal,
histogram` for `MACD`, `upper, middle, lower` for `BBANDS`, and `k, d` for
`STOCH`; these are not pandas-ta DataFrames with parameter-encoded column names.
The same warm-up and missing-value rules apply. The import allowlist rejects
`pandas_ta`; existing generated strategies using it must be revised and
validated again. Static validation does not enforce the numerical rules above.

The public `indicator_column` helper has also been removed. Remove its imports
and calls, replacing pandas-ta column-name lookups with the facade's named
result fields, such as `macd(close).histogram` or `bollinger_bands(close).upper`.
When calling `talib` directly, unpack its output tuples in the order described
above instead. The facade's result objects expose named fields and are not
tuples to unpack.

Ruff treats `tradingdev` as first-party code. Separate its imports from
third-party `numpy`, `pandas`, and `talib` imports with a blank line.

Recommended imports:

```python
from tradingdev.domain import indicators
from tradingdev.domain.indicators.kd import KDIndicator
from tradingdev.domain.strategies.base import BaseStrategy
from tradingdev.shared.utils.logger import setup_logger
```

## YAML Contract

```yaml
strategy:
  id: "sma_crossover"
  version: "0.1.0"
  class_name: "SmaCrossoverStrategy"
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
  # market_type defaults to "futures/um" (binance_vision only); "spot" for spot
  requirements:
    market:
      # source selects the market data crawler registered in
      # domain/data/crawlers/registry.py:
      #   "binance_vision" (default) - crypto, data.binance.vision
      #   "binance_api"              - crypto, ccxt
      #   "yahoo_finance"            - stocks/ETFs/futures/indices/FX,
      #                                Yahoo symbols ("AAPL", "ES=F", "^GSPC")
      # omitted -> inherits legacy top-level data.source
      source: "binance_vision"
      symbol: "BTC/USDT"
      timeframe: "1h"
    features: []
```

`save_strategy` assigns `strategy.id`, `strategy.revision_id`, and
`strategy.source_path` to the saved revision. These identity fields are managed
by the service; callers need not supply them. `strategy.version` remains optional
descriptive configuration, not the execution revision identifier.

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

1. `save_strategy`: writes a new draft source/config revision and metadata,
   returns its `revision_id`, and updates the current pointer. Even an identical
   save creates a new revision; it does not inherit validation evidence.
2. `validate_strategy`: runs syntax, static policy, restricted import checks,
   ruff, mypy, inheritance, constructor, and the shared signal-contract gate on
   a short fixture for the selected revision. It returns that `revision_id` and
   structured diagnostics with `level`, `code`, `phase`, `message`, and optional
   `fix`.
3. `dry_run_strategy`: accepts only `validated` strategies, re-runs the same
   signal-contract gate on a longer fixture, returns `signal_analysis`, and
   marks the strategy runnable when it passes.
4. `promote_strategy`: strategy tool that marks a runnable generated strategy
   as promoted (owned by `StrategyService`).
5. `start_backtest` / `start_walk_forward` / `start_optimization`: execute only
   runnable or promoted generated strategies, and promoted bundled strategies.
   The gate is enforced
   again at execution time inside `BacktestService`, so the CLI and subprocess
   workers cannot bypass the lifecycle, and the config's `source_path` must
   match the selected revision's source.

Pass the `revision_id` returned by saving to get, validate, dry-run, promote,
and execution tools. Omitting it selects current once at the start of that
operation. Validation and dry-run persist evidence to that selected revision,
even if another save changes current while checks are running. A new draft B
does not revoke revision A's evidence, and B cannot inherit it.

Generated execution requires matching source/config hashes and successful
validation and dry-run records for the selected revision. Editing revision files
directly invalidates them; repair through a new save. An unknown explicit
revision never falls back to current. Submitted jobs, worker configs, completed
runs and strategy source artifacts remain tied to the selected revision when
current changes, including during optimization confirmation.

Runtime symbol, timeframe, and dates are separate from the revision's base
config. For generated strategies, ordinary backtest and walk-forward execution
configs must preserve the saved `strategy` mapping in full, including parameters,
identity, and any descriptive or constructor settings such as `description`,
`version`, or `fit`. Adding, removing, or changing those fields requires saving
and checking a new revision, even if identity and parameters are unchanged.
The comparison excludes the execution-managed `source_hash` and separately
verifies that `source_path` resolves to the selected revision's source. Copy the
saved config and apply market/date/cost changes outside the `strategy` section.
Optimization may override only its search parameters, retaining all other base
parameters and all other saved strategy settings. Validation evidence covers
the base parameters; it does not certify
every possible optimization candidate. This revision identity does not freeze
imported Python dependencies, the engine environment,
or market data, and is not a full execution manifest. Bundled strategies remain
Git-managed and promoted with `revision_id: null`.

Legacy flat source/metadata/config files are not migrated or overwritten. They
cannot execute using their old lifecycle state. `list_strategies` discovers
root-level `<id>.json` names separately from current revisions, without parsing
legacy metadata or blocking bundled/current strategies. These entries have
`kind: legacy`, `status: revision_required`, `revision_id: null`, and
`code: strategy_revision_required`, with explicit resaving instructions.
`get_strategy(id)` reads the fixed `generated_strategies/<id>.py` and
`configs/<id>.yaml` locations as source/YAML recovery material; it does not follow
paths or trust lifecycle evidence from legacy metadata. Missing, unreadable,
non-UTF-8, or symlinked source/config files return `strategy_revision_invalid`;
their legacy entry remains discoverable. Restore the files or provide replacement
content to `save_strategy`, then validate/dry-run the newly returned revision.
Validate, dry-run, and promote reject legacy entries with
`strategy_revision_required`; execution still rejects them as non-executable.
A saved current revision supersedes its legacy discovery entry, leaving the old
files untouched. Explicit revision lookups never fall back to legacy files.
If a legacy entry shares a bundled ID, discovery lists both kinds and default
source lookup/execution selects bundled. Read the legacy source explicitly with
`get_strategy(strategy_id, legacy=true)` and save it under a different,
non-reserved ID; saving over bundled IDs returns `reserved_strategy_id`.
The explicit legacy query never falls back to bundled/current, and combining it
with `revision_id` returns `strategy_revision_invalid`.
Historical run records with no revision retain `revision_id: null`.

## Security Model

Validation currently uses static policy checks plus restricted imports as a
first layer. It rejects known unsafe imports (`os`, `sys`, `subprocess`,
`socket`, `requests`, `httpx`, `ccxt`, `shutil`, `pathlib`), dynamic execution
calls (`eval`, `exec`, `__import__`), raw `open`, and common destructive file
operations such as `unlink`, `remove`, `rmtree`, and `write_text`.
Static-policy or quality-gate errors stop validation before generated code is
imported or executed.

This is not a full process sandbox. `validate_strategy` and `dry_run_strategy`
still import and execute generated Python to check class loading, constructor
behavior, BaseStrategy inheritance, input immutability, signal values, and smoke
dataframe output. Generated strategies must therefore be reviewed as runtime
code until a dedicated sandboxed execution layer is added.

`signal_analysis` includes row count, signal distribution, missing-signal count,
transition count, active-signal ratio, and timestamp bounds. Use it to debug
strategies that technically pass static checks but produce unusable signals.
