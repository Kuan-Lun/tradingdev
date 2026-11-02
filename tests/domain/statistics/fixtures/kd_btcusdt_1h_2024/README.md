# KD(14,3,3) BTC/USDT 1h 2024 fixture

The smallest data that reproduces three KD backtest figures (annualised daily
Sharpe, traded notional, cost over net P&L) through the statistics modules, so
tests can pin them although the workspace cache is not versioned.

## Source

| Pickle | SHA-256 | Used for |
| --- | --- | --- |
| `workspace/data/processed/cache/cd03bf19459b48af.pkl` | `b9493b6bf5f1f171fee472812a9c6f7aab7a58f5d93fc6644386211d5b6febd0` | equity curve and timestamps; the reference the shared fields are checked against |
| `workspace/data/processed/cache/cli_5ed5dc0b0579cc80_95a98854a8984f82be488c4bff48902d.pkl` | `af4fc3905570f44f318dc900baa2b754af0892b32829da8f45b80cf1466aacb7` | every trade field and the engine metrics |

`cd03bf19459b48af.pkl` is the input the KD statistics demo report cites. It was written by
an older engine whose trades lack `exit_notional`, `status` and recorded fees,
and whose `metrics` use superseded conventions (bars counted as days, negative
`max_drawdown`). The second pickle is the current engine's run of the same
configuration: its equity curve and timestamps are bitwise identical and its
412 trades have identical `size_quote`, `entry_price`, `exit_price` and
`net_pnl`. `extract.py` checks both hashes and those equalities before
writing, then writes the trade table from the second pickle, so the shared
fields equal those of `cd03bf19459b48af.pkl`.

## Contents

- `equity.parquet`: 8,761 hourly bars, `timestamp` (UTC, ns) and `equity`
  (float64), 2024-01-01 00:00 to 2024-12-31 00:00. About 11 KB.
- `trades.parquet`: 412 trades with `size_quote`, `entry_price`, `exit_price`,
  `net_pnl`, `fee`, `exit_notional` and `status` (all `closed`). About 26 KB.
- `run.json`: provenance, strategy parameters, `init_cash`, `fees`,
  `slippage`, `mode` and the current engine's 33 `engine_metrics`. About 2 KB.

## Why this is enough

- Annualised daily Sharpe: hourly equity and timestamps go through
  `period_returns(..., "daily")`, so the UTC daily resample is exercised, not
  skipped as it would be with stored daily returns.
- Traded notional: `summarize_breakeven` reads `net_pnl`, `size_quote`,
  `entry_price`, `exit_price`, `fee` and `status`, plus the first and last
  timestamps. `exit_notional` lets tests cross-check against the engine's
  `size_quote + exit_notional`.
- Cost / abs(net P&L): traded notional times `fees + slippage`, over the
  equity change.
- Run validation tools outside this package also need the engine's metrics,
  `mode` and `init_cash`, which `run.json` carries.

## Format

Parquet with zstd keeps float64 and nanosecond timestamps bit-exact on a round
trip, which a text format only does with careful float formatting. Delta
encoding for the regular hourly timestamps and dictionary encoding for the
equity column (825 distinct values, flat while out of the market) bring the
bars from about 86 KB to 11 KB.

To rebuild, with both pickles present:
`uv run python tests/domain/statistics/fixtures/kd_btcusdt_1h_2024/extract.py`.
