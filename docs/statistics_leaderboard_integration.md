# Statistics and ML Leaderboard Integration

This is a handover contract for the owner of the ML signal leaderboard. It
lists what the statistics modules in `src/tradingdev/domain/statistics/` need
before they can analyse leaderboard candidates, and why.

Every statement about the leaderboard comes from reading `upstream/main` at
commit `3fa0852` (2026-10-23), which this branch has merged, so the file paths
below refer to this branch.

The leaderboard already does several things right that this contract relies
on: it refuses to rank incomparable contexts, it reports the candidate count
with a selection caveat, and it labels its normal approximation as optimistic
for short series. Nothing here asks to weaken any of that.

## Two Selection Corrections

The project has two multiple-comparison corrections for two metric families:

| Correction | Where | Metric family | Question answered |
| --- | --- | --- | --- |
| Bonferroni t-threshold, `Phi^-1(1 - alpha / (2 N))` | `required_t_stat`, `domain/ml/leaderboard.py:272` | IC t-statistics | Would any of `N` IC tests reject at family-wise `alpha`? |
| Deflated Sharpe ratio (Bailey and Lopez de Prado) | `deflated_sharpe_ratio`, `domain/statistics/sharpe.py` | Sharpe ratios of return series | How likely does the selected Sharpe beat the best of `N` skill-less trials? |

They are not interchangeable and must never both be applied to the same
number. Reports that show both must label each figure with its correction.
Both count `N` as independent candidates, while neighbouring candidates are
correlated, so both over-correct. The `deflated_sharpe_ratio` docstring
records the same rule from the statistics side.

## 1. Parameter Plateau Analysis

### What Plateau Analysis Needs

`trials_from_metrics` (`domain/statistics/plateau.py:136`) builds the trials
for `plateau_summary` and `plateau_grid` from rows. The minimum input is:

| Input | Requirement |
| --- | --- |
| Numeric parameter coordinates per candidate | A non-empty mapping of parameter name to a finite number, with the same names in every row (`plateau.py:177-186`). |
| One metric value per candidate | A number, or `None` for a failed candidate. The key must be present: a missing key raises `ValueError` (`plateau.py:173-175`). |
| The metric's optimisation direction | Passed as `direction`, `"maximize"` or `"minimize"`, with no default; `"minimize"` metrics are negated (`plateau.py:141`, `:165-167`). |

A parameter set must not repeat (`plateau.py:187-189`).

Failed candidates must stay in the input with a `None` metric. Since commit
`1b51608` ("keep failed slices on the plateau grid axes"),
`trials_from_metrics` drops their trials but keeps their parameter values on
the axes. A failed value is then a missing grid point between its neighbours.
If failed candidates were left out, a value whose candidates all failed would
disappear from its axis, and its neighbours would look adjacent.

The direction is already available on the leaderboard side:
`metric_direction` (`leaderboard.py:249`) reads the catalog's
`optimization_direction` from `SIGNAL_METRIC_CATALOG`.

### What the Leaderboard Records for Plateaus

- `LeaderboardEntry` (`leaderboard.py:112-121`) has no parameter field. Its
  fields are `entry_id`, `label`, `context`, `metrics`, `n_observations`,
  `model_id` and `dataset_id`. Candidates differ by `model_id` and `label`;
  model hyperparameters are not recorded.
- The only numeric settings on the record are `horizon`, `window` and
  `n_quantiles` in `EvaluationContext` (`leaderboard.py:45-59`). All three are
  part of `EvaluationContext.key` (`leaderboard.py:68-83`).
- `comparability_error` (`leaderboard.py:285-295`) rejects entries whose
  `context.key` values differ, so entries at different `window` or `horizon`
  cannot be ranked together. `MLService.leaderboard` skips that check only when
  `allow_mixed` is set (`app/ml_service.py:442`, `:486-489`).
- `_record_evaluation` (`app/ml_service.py:381-385`) persists only finite
  metrics, so a failed metric is dropped from the mapping rather than stored
  as `None`. `LeaderboardEntry.metrics` is typed `Mapping[str, float]`, and
  `to_record` and `from_record` convert every value with `float(...)`
  (`leaderboard.py:139`, `:183`), so a `None` cannot be stored either.

### Two Purposes Behind One Gate

`comparability_error` is the right protection for picking a winner: a ranking
across different horizons or windows compares numbers measured on different
questions.

Sensitivity analysis has the opposite purpose. It compares across `window`
and `horizon` on purpose, to ask whether the best setting is an isolated spike
or part of a plateau. Today both purposes pass through the same gate, so the
gate that correctly protects the ranking also blocks plateau analysis.
`allow_mixed` is not a substitute: it lifts the check for every part of the
context at once, including instrument, dates and data, and the result is
still a ranking.

A plateau view needs the candidates that share every other part of the
context, with `window`, `horizon` or `n_quantiles` as the axes. The rest of
the context (instrument, dates, feature set, code and data fingerprints) must
still match. The plateau result must not feed the winner selection.

### Missing for Plateau Analysis

1. Parameter coordinates per entry: model hyperparameters, plus `window`,
   `horizon` and `n_quantiles` when they are searched.
2. A way to group entries that match on everything except the searched
   settings, separate from the `comparability_error` check for ranking.
3. Failed candidates kept as entries, with the metric recorded as missing in
   a form an adapter can turn into `None`, rather than absent without trace.

## 2. Paired Bootstrap of IC Series

### What the Leaderboard Records for IC Series

- `LeaderboardEntry.metrics` is `Mapping[str, float]` (`leaderboard.py:118`).
  It holds scalar summaries only.
- `rolling_ic` (`domain/ml/evaluation/metrics.py:111`) returns one IC per
  aligned bar. `signal_metrics` (`metrics.py:247-283`) reduces those series to
  scalars through `ic_statistics`; the series is not kept.
- Predictions are not stored, and the evaluated signal cannot be
  recovered with certainty. An ad-hoc entry keeps only
  `signal_fingerprint(signal)` (`ml_service.py:387`), a hash. A
  model-backed entry keeps no fingerprint at all, and its `model_id` is "a
  caller label, not proof of signal origin" (`evaluate_signal` docstring,
  `ml_service.py:193`). `MLService.predict(model_id=..., dataset_id=...)`
  (`ml_service.py:755-797`) regenerates what the stored model produces on
  the immutable dataset, but nothing recorded confirms that this is the
  signal that was scored and ranked.
- Forward returns are not on the entry, but they can be recovered:
  `dataset_id` names an immutable dataset whose frame has a `forward_return`
  column (`ml_service.py:303`, `:331`), and the context carries its content
  hash.

### Why the Series Are Worth Keeping

Neighbouring candidates are scored on the same bars, so their IC series share
most of their noise. A paired bootstrap resamples the same bars for both
candidates, and the shared noise cancels in the difference. The difference
interval is then much narrower than comparing each candidate with a threshold
on its own.

The effect has been measured on daily strategy returns, not yet on IC series.
For KD(9, 3, 3) and KD(14, 3, 3), whose daily returns correlate at 0.68, the
paired Sharpe difference interval is 3.53 wide, which is 0.69 times the 5.15
expected from the two individual intervals if the strategies were
independent (the square root of the sum of the squared widths 3.89 and 3.38).
See `demo_output/METHODS.md`, sections 2.12 and 2.13. IC series of
neighbouring candidates are expected to correlate at least as strongly, but
that has not been measured.

`difference_interval` (`domain/statistics/comparison.py`) accepts any
statistic, so it works on IC series as it does on returns once both series
are available on a shared index.

### Effective Sample Size: An Option

`ic_statistics` (`metrics.py:147-175`) corrects for overlapping windows with
`n_effective = max(1, n_bars // window)` (`metrics.py:151`). This is a
reasonable correction, and the module docstring states that it is an
approximation.

An improvement worth considering: the stationary bootstrap used by
`difference_interval` chooses its mean block length with the Politis and White
(2004) estimate, which measures the dependence in the data itself. It would
reflect the actual overlap, which may be shorter or longer than `window` bars.

### Missing for the Paired Bootstrap

1. The per-bar IC series of each candidate, or its predictions. For
   model-backed entries, `MLService.predict` plus `rolling_ic` is a
   possible substitute only if the entry also records a fingerprint of the
   evaluated signal, so an adapter can check that the regenerated signal is
   the one that was ranked.
2. A shared bar index for those series, so `align_returns` can pair them.
3. Nothing more for forward returns: they are recoverable through
   `dataset_id`.
