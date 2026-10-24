# KD strategy: inference statistics

Source: `workspace/data/processed/cache/cd03bf19459b48af.pkl` — strategy `kd_crossover` {'k_period': 14, 'd_period': 3, 'smooth_k': 3, 'overbought': 80.0, 'oversold': 20.0}, BTC/USDT 1h, 2024-01-01 to 2024-12-31, init cash 10,000.

## Sample size

| Statistic | Value |
| --- | ---: |
| Real days (UTC days with data) | 366 |
| Months | 12 |
| Share of months with negative P&L | 100.0% (12/12) |

## Sharpe inference (daily returns, annualised x sqrt(365))

| Statistic | Value |
| --- | ---: |
| Annualised Sharpe (daily) | -7.57 |
| Engine-reported Sharpe (hourly) | -8.70 |
| Stationary bootstrap 95% CI (seed 0, 1000 reps) | [-9.18, -5.87] |
| Bootstrap block size (days) | 2 |
| PSR (P[true SR > 0]) | 0.0000 |
| t-statistic of daily P&L | -7.33 |

## Break-even win rate and turnover

| Statistic | Value |
| --- | ---: |
| Break-even win rate p* | 62.5% |
| Realised win rate (decisive trades) | 29.4% |
| Gap (realised - p*) | -33.1% |
| Payoff ratio W/L | 0.60 |
| Trades (win / loss) | 412 (121 / 291) |
| Annual turnover (x capital) | 501.8 |

## Cost breakdown

Estimated cost = traded notional (entry + exit) x (fees 0.0006 + slippage 0.0005). Trade records carry `fee = 0` because the engine folds costs into fill prices, so this is an estimate. The engine's `total_volume` metric changed upstream from entry-only to entry plus exit notional, so the cost is always derived from the trade records, never from that metric.

| Statistic | Value |
| --- | ---: |
| Traded notional (entry + exit) | 5,014,736 |
| Estimated cost | 5,516.21 |
| Gross P&L (net + cost) | -872.13 |
| Net P&L | -6,388.34 |
| Cost / abs(net P&L) | 86.3% |
| Cost / initial capital | 55.2% |
