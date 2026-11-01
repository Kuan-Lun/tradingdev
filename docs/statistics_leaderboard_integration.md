# Statistics, Ranking and ML Leaderboard Integration

## Purpose and Status

This document describes how ranking, the ML leaderboard and the statistics
package (`src/tradingdev/domain/statistics/`) divide the work of comparing
research results: what exists today, what the statistics package can consume,
what fails along the way, and what is still to be built or decided.

This branch has merged upstream main at `a24594c` (2026-11-01). Every
statement about existing code was checked against this tree, including the
method families (`method_family` and `method_name` in `domain/ml/leaderboard.py`),
the threshold rules (`domain/ml/rules.py`), `rank_entries`, and the run
queries `get_run_metrics` and `compare_runs` (`app/run_service.py`,
`mcp/tools/runs.py`). Upstream PR #136 is described at its head `d22e380`;
it is open and not part of this tree. Code is cited by module, function and
constant names.

The resampling and selection methods behind the statistics package, with
their literature and the steps this project changed, are in
[statistics_bootstrap_methods.md](statistics_bootstrap_methods.md).

## Scope

Kuan-Lun confirmed the scope on PR #38 (2026-11-01):

- **Ranking covers both signal quality and complete strategy performance.**
  During development, signals, parameters and backtests are compared. Across
  strategies, each strategy contributes one chosen set of parameters and its
  backtest result as its representative, and the comparison uses
  performance figures such as the Sharpe ratio and the maximum drawdown.
- **Statistical comparison is a further analysis after ranking, not a
  condition for being ranked.** Two backtests without a common observation
  period are still shown with their own figures and ranks. A paired analysis
  that needs aligned observations reports why it cannot run.
- **No holdout requirement for ranking.** The project does not restrict
  development to older data, and neither a shared holdout test period nor
  out-of-sample evidence is a condition for being ranked. Reserving final
  evaluation data is a recommendation to users (README, "使用樣本外驗證與最佳化"),
  not something the system checks.

This replaces the earlier proposal that the leaderboard ranks only signal
constructions on one dataset.

## Three Responsibilities

Comparing results splits into three responsibilities, owned by different
layers. This is the agreed division of work. Today only the signal source and
its ranking exist (see [What Exists Today](#what-exists-today)); the rest is
listed under [Follow-up Work](#follow-up-work).

| Responsibility | Owner | What it does |
| --- | --- | --- |
| Candidate selection | Application service | Finds the candidates, keeps their identity, conditions and representative choice, and passes them to ranking. |
| Ranking | Shared ranking core (to be extracted), fed by each candidate source | Orders candidates by one metric in its catalog direction, assigns tied ranks and keeps missing values with their reason. |
| Statistical analysis | `domain/statistics` | Optional analysis of ranked candidates: intervals, paired differences, plateau, deflated Sharpe. |

The target is that MCP tools and reports both use the application service's
results and neither ranks on its own. Today the ML report block
(`domain/ml/report.py`, `ml_report_block`) still calls `rank_entries`
itself; moving it onto the service is listed under
[Follow-up Work](#follow-up-work).

### Candidate Selection

There are two candidate sources, kept apart:

- **Signal candidates** reference a recorded evaluation by `entry_id`
  (`LeaderboardEntry`). This exists today.
- **Backtest candidates** reference a run by `run_id` plus a `scope` (for
  example `trial/0/train` and `trial/1/train` of one optimisation run), and
  keep the strategy version and parameters. This does not exist yet; see
  [Follow-up Work](#follow-up-work).

Within one strategy, several parameter sets of the same run may be selected
side by side. A formal cross-strategy ranking uses one representative result
per strategy: the user explicitly marks a strategy version and parameter set
as finished for development and selects one result as its representative.
That choice is to be saved and not replaced automatically when new
experiments are added. The mark means the result is included in the comparison; it does
not mean its out-of-sample performance has been demonstrated.

Each candidate keeps its trading conditions: instrument, bar frequency,
period and costs. Ranking is to show them with the figures.

Fast ranking reads the saved metrics; `get_run_metrics` already returns them
without rerunning anything. When a result has to be re-evaluated, a new
backtest is run and saved first, and the new run then goes through the same
ranking.

### Ranking

Ranking is one procedure for both sources: sort by one metric in the
direction its catalog defines, give equal values one rank, and set aside
candidates whose value is missing, with the reason. Each source supplies its
own metric definitions and directions:

- Signal metrics come from `SIGNAL_METRIC_CATALOG` through `metric_direction`.
  IC statistics and their selection threshold (`required_t_stat`) stay with
  the signal layer.
- Backtest metrics come from the performance metric catalog
  (`domain/performance/catalog.py`), whose `optimization_direction` the grid
  search already uses.
- A signal simulation's `sharpe` (a cost-free long/short book over
  non-overlapping holding intervals; `docs/ml_research.md` states it is not a
  backtest result) and a backtest's Sharpe ratio keep their separate
  definitions. They are different metrics, not one metric from two sources.

Rules that hold for every ranking:

- Metric definitions and units are the same for every candidate in one
  ranking.
- A missing value keeps its reason and is never filled with zero.
- Walk-forward fold statistics (the `test_summary` scope of kind
  `fold_summary`) summarise folds; they are not the performance of one
  continuous backtest and are not ranked as if they were.

Single-strategy and cross-strategy rankings use the same procedure; they
differ only in which candidates were selected.

### Statistical Analysis

The statistics package analyses candidates after ranking. Nothing it reports
changes a rank or decides whether a candidate is ranked. See
[Statistical Comparison After Ranking](#statistical-comparison-after-ranking).

## What Exists Today

### Signal Leaderboard

The full field contract of a recorded candidate is the `LeaderboardEntry`
definition and its docstrings in `domain/ml/leaderboard.py`; the user-facing
contract, including method families and threshold rules, is
`docs/ml_research.md` ("Recorded Signal Evaluations" and "Method families").
The flows below read these fields:

- `entry_id`: a 16-hex-digit hash from `build_entry_id`. Named methods hash
  family, name, parameters, `dataset_id` and the context key; other signals
  hash `model_id`, `dataset_id`, the context key and, without a model,
  `signal_fingerprint`.
- `context` (`EvaluationContext`): the three tunable settings `horizon`,
  `window` and `n_quantiles` (`EvaluationContext.tunable_parameters`), and
  the identity that `comparability_error` checks.
- `metrics`: `Mapping[str, float | None]` with three states. A key that is
  absent was never recorded; a key holding `None` was measured but is
  unavailable or non-finite; a finite value is an ordinary measurement.
  `MLService._record_evaluation` stores NaN and infinite metrics as `None`.
  Older records dropped them, so a missing key on an old record cannot tell
  the two cases apart.
- `model_id`, `method_name` and `LeaderboardEntry.family`: what built the
  signal. `family` is `ml_model`, `rule` or `other`; on rows written before
  families existed it is inferred from the presence of a model ID. The family
  is a declaration validated by `resolve_method`, not proof of provenance.
  Nothing records whether the backend produced the signal from the model
  (`MLService.evaluate_prediction`) or the caller supplied it
  (`evaluate_signal` with a `model_id`); both produce the same entry.
- `parameters`: for `ml_model`, copied from the model artifact; for a rule,
  declared by the caller. `ThresholdRule.parameters` stores the exact level
  as `{"threshold": level}`, while `ThresholdRule.name`, which becomes
  `method_name`, shows it rounded with the `g` format.
- `dataset_id`: the dataset the signal was **evaluated** on.
- `signal_fingerprint`: a hash of the full evaluated signal, empty on older
  rows.

Over MCP, `evaluate_signal` takes either a `model_id` or a `column` plus
`threshold` rule; signals of family `other` come only through the Python
`MLService.evaluate_signal` API or from older rows.

`MLService.leaderboard` (and `leaderboard_page`, the MCP `leaderboard` tool)
filters by `dataset_id`, `symbol`, `timeframe`, `horizon`, `feature_set` and
`family`, then:

1. computes `parameter_axes(entries)` over the filtered entries, including
   the ones that will be skipped;
2. unless `allow_mixed` is set, calls `comparability_error`:
   - by default every field of `EvaluationContext.key` must match;
   - with `allow_parameter_axes=True` it calls `_surface_error` instead, which
     lets `horizon`, `window` and `n_quantiles` differ but requires one
     `EvaluationContext.identity_key` (symbol, timeframe, dates, feature set,
     feature code) and one `(dataset_id, data_fingerprint)` pair per horizon;
3. ranks with `rank_entries(entries, metric, *, top_k=None, alpha=0.05)`.

`rank_entries` takes `LeaderboardEntry` objects only. It reads the direction
from `metric_direction`, which raises for a metric without one; the rankable
metrics are listed by `rankable_metrics`. Entries without a finite value go to
`skipped` with the reason `no '<metric>' recorded` (key absent) or
`'<metric>' is unavailable or not finite` (key present). The rest are sorted
by `entry_id` and then, stably, by value, so ties keep `entry_id` order.
Ranks use competition ranking (1, 2, 2, 4): a value within `math.isclose` of
the previous one shares its rank, and `tied` counts the entries at that rank.
The sort, tie and missing-value logic is written inside `rank_entries`; it
is not a separate function another source could call.

The response carries `direction`, `parameter_axes`, `ranked` rows (`value` is
the ranking metric, with `horizon`, `window`, `n_quantiles`, `parameters` and
`signal_fingerprint`), `skipped` rows (with `reason` and the three
coordinates), and `caveats`, including `surface_caveats` when an axis has more
than one value. Ranking and caveats always cover the full candidate set, even
when only one page is returned.

`parameter_axes` returns `dict[str, list[int]]` with exactly the keys
`horizon`, `window` and `n_quantiles`, each sorted and deduplicated.

### Saved Backtest Results

- `RunService.get_run_metrics(run_id, metric_ids=None, scope=None)` (MCP
  `get_run_metrics`) returns the saved values of one scope of a run: its kind,
  mode, split, fold or trial index, parameters and metadata (settings),
  metrics, display metrics and definitions. It reruns nothing. The scopes are
  `full` for a backtest; `fold/{i}/train`, `fold/{i}/test` and `test_summary`
  for walk-forward; `trial/{i}/train` and `test` for optimisation
  (`domain/research_scope.py`).
- `RunService.compare_runs(run_ids, metric_ids=None, scope=None)` (MCP
  `compare_runs`) lists recorded scopes side by side with per-metric
  compatibility and context differences. It does not rank. It requires at
  least two run IDs and refuses repeated ones (`duplicate_runs`), and one
  `scope` applies to every run, so it cannot compare several trials of one
  optimisation run.
- Walk-forward saves per-fold backtest scopes and a `test_summary` scope with
  fold statistics. It saves no per-fold signals or predictions: the saved
  observations hold equity, returns, timestamps, trades, execution records
  and account history.

### Training Provenance on the Record

Nothing on the current record distinguishes an in-sample evaluation from a
held-out one:

- `LeaderboardEntry` stores only the evaluation `dataset_id`.
- `ModelArtifact` (`domain/ml/registry/model_registry.py`) carries
  `n_train_samples` but no training-dataset identity.
- `compute_model_id` hashes the model name, parameters, ordered feature names
  and the content hash of the pickled model. No data identity is part of it,
  so joining an entry's `model_id` to its artifact does not answer the
  question either.
- `MLTrainingService.train_model` publishes an artifact fitted on every row of
  its dataset; its validation returns scores, not out-of-fold predictions.
  `MLService.predict` checks only that the dataset supplies the artifact's
  feature names, and its docstring says prediction "does not establish
  out-of-sample provenance".

Upstream PR #136 adds part of this; see
[Open Items](#open-items).

## Statistical Comparison After Ranking

Statistical comparison is optional analysis of candidates that are already
ranked:

- `comparison.difference_interval_outcome` gives a paired interval and
  p-value for the difference of a statistic between two return series on
  their common periods.
- `comparison.interval_table_outcome` gives individual intervals on the
  common periods.
- `sharpe.sharpe_interval_outcome`, `probabilistic_sharpe_ratio` and
  `deflated_sharpe_ratio` analyse one series; the deflated Sharpe ratio uses
  the trials of the search that selected it.
- `plateau` describes the stability of a metric across a parameter search.

It needs the return series, which ranking does not: ranking reads saved
metrics. For backtests the saved observations of a scope include its returns
and timestamps when they are available; they are `None` when, for example,
the initial capital is unknown or equity is not positive, and the paired
analysis is then unavailable with that reason. For signals the series is not saved (see
[Integration Flow 2](#integration-flow-2-paired-bootstrap-of-ic-series)).

When two series share no period, nothing about the ranking changes; only the
paired analysis is unavailable. Today
`difference_interval_outcome` then returns
`IntervalUnavailable(reason="insufficient_observations", observations=0)`,
with `estimate=None` and `reps=0`, the same reason as a short overlap. A
caller can tell the two cases apart only by `observations == 0`. Mixing a
timezone-naive and a timezone-aware index raises `ValueError` in
`align_returns` instead. A separate `"no_overlap"` reason is a proposal for a
later change (see [Open Items](#open-items)); this branch does not change the
code.

Comparing results that differ in instrument, period or costs is a question
of design, not statistics: the paired analysis only guarantees a common
period. A paired difference of in-sample results measures which candidate
fits its sample more closely, not which one is better.

## Integration Flow 1: Parameter Plateau

The plateau functions (`trials_from_metrics`, `plateau_summary`,
`plateau_grid`) can consume a leaderboard parameter surface over `horizon`,
`window` and `n_quantiles` today, and the trial scopes of an optimisation run
(`trial/{i}/train`) in the same way.

### Minimal Input

`trials_from_metrics(rows, params_key=..., metric_key=..., direction=...)`
needs one row per candidate:

| Input | Source on the leaderboard |
| --- | --- |
| Numeric coordinates, the same names in every row | `entry.context.tunable_parameters`, restricted to the swept axes. |
| One metric value, or `None` for a failed candidate; the key must be present | `entry.metrics[metric]` for entries where the key is present. |
| `direction`, `"maximize"` or `"minimize"`, no default | `metric_direction(metric)`, or the response's `direction`. |

`plateau_summary(result.trials, axes=...)` then needs the axes. Rows whose
metric is `None` are dropped from `result.trials` but keep their coordinates in
`result.axes`, so a failed setting stays a missing grid point between its
neighbours. Surface `result.dropped_count` with the result.

### Using `parameter_axes`

`parameter_axes` already is the axes mapping in the form `plateau_summary`
accepts, and it is computed over the same filtered entries, failed ones
included, so it equals `result.axes` when the coordinates are the three
context settings. Conditions:

- Use it only from a call with `allow_parameter_axes=True` and without
  `allow_mixed`. It is computed before the comparability check, so under
  `allow_mixed` it can mix entries of different identities.
- Keep only the axes with more than one value, and give the coordinates the
  same names. An axis with a single value adds a dimension: the neighbourhoods
  are unaffected, but `dimension_warning` fires spuriously.
- It covers only the three context settings. If model hyperparameters are
  also coordinates, pass `result.axes` from `trials_from_metrics` instead;
  passing `parameter_axes` raises because the parameter names differ from the
  axes. Do not use `entry.parameters` wholesale as coordinates: it can hold
  non-numeric settings, which raise. Select the swept numeric settings
  explicitly.
- A surface across horizons cannot use the `dataset_id` filter, because each
  horizon is its own dataset and `leaderboard` checks every entry against the
  filtered dataset's manifest horizon. Filter by `symbol`, `timeframe` and
  `feature_set` instead. A surface over `window` and `n_quantiles` alone can
  use one `dataset_id`.

The recommended adapter works on `LeaderboardEntry` objects from
`LeaderboardStore.entries` with the same filters, calling `comparability_error`
with `allow_parameter_axes=True` and then `parameter_axes`. Entries keep the
three metric states directly. The MCP response marks a key-absent metric only
through the text of a skipped row's `reason`.

### Failure Paths

| Situation | What happens | What the adapter does |
| --- | --- | --- |
| Entries differ in `window` or `horizon` and `allow_parameter_axes` is off | `comparability_error` returns "entries were not measured on the same conditions" and `leaderboard` raises `ValueError` | Opt in to `allow_parameter_axes`; never use `allow_mixed` for a surface. |
| Different instrument, dates, feature set or feature code | `_surface_error` refuses with "not measured on the same instrument, dates, feature set or feature code" | Build one surface per identity. |
| One horizon measured on two dataset versions | `_surface_error` refuses with "horizon N was measured on M different dataset versions" | Re-record on one `dataset_id`, or build one surface per version. |
| Two candidates at the same coordinates | `trials_from_metrics` raises because the parameter set repeats | Select exactly one candidate per coordinate (next section). |
| Metric key absent on an entry (an older record) | `trials_from_metrics` raises; it does not treat absence as failure | Do not map absence to `None`. Re-evaluate and record, or report the gap. |
| Hyperparameter coordinates with `parameter_axes` as axes | `plateau_summary` raises because parameter names differ from the axes | Use `result.axes`. |

### One Candidate per Coordinate

Every candidate at a coordinate is a separate row, and the leaderboard has no
field saying that two rows at different coordinates are "the same candidate".
A surface therefore needs an explicit grouping key chosen by the adapter:

- Within one horizon, a model is one `model_id`. A rule is one
  `method_name` together with its exact `parameters`, as `build_entry_id`
  identifies it; the name alone rounds the level, so two close thresholds
  can share a name.
- Across horizons a model is necessarily retrained on each horizon's dataset,
  so its `model_id` changes. The grouping key is then a declared
  configuration, such as model name plus selected hyperparameters, and each
  point is a different fitted model.
- Filtering by `family` alone does not select one candidate.

### Threshold Sweeps Are Not a Surface

**Do not build a plateau over rule thresholds from the current records.** A
threshold sweep is stored as one candidate per level: each level is its own
`rule` entry, with the exact level in `parameters` and a rounded form in
`method_name`, and all of them sit at the same `horizon`, `window` and
`n_quantiles`. `parameter_axes` has no threshold axis, and
`ThresholdRule.parameters` states that threshold surfaces are not supported.

The leaderboard maintainer has said a threshold axis will not be added, and
this side agrees not to reconstruct a grid from the stored levels. A grid
inferred that way would not be a grid:

- The stored levels are whatever was tried, not a declared grid, so missing
  levels cannot be told apart from levels nobody tried.
- A level that makes the signal constant is refused by `ThresholdRule.signal`
  before anything is recorded. The failed point leaves no row, which breaks
  the plateau's rule that a failed setting keeps its coordinate.
- Choosing levels after viewing the dataset is in-sample selection, which the
  ML contract says the backend does not verify.

### What a Plateau Does Not Establish

A plateau result describes the stability of a score across settings; it must
not feed winner selection. It also inherits the provenance of its inputs. A
model evaluated on the dataset it was trained on gives in-sample scores, and a
broad plateau of in-sample scores is still in-sample; so is a plateau over
optimisation trial scopes, which are training-period results. The leaderboard
record cannot currently say which case applies (see
[Training Provenance on the Record](#training-provenance-on-the-record)).

## Integration Flow 2: Paired Bootstrap of IC Series

The intended use is `comparison.difference_interval_outcome` on two
candidates' IC series over a shared bar index: neighbouring candidates scored
on the same bars share most of their noise, and a paired bootstrap cancels it.
This flow is **not possible today**, for two separate reasons. Solving the
first does not solve the second.

### Layer 1: The Series Are Not Recorded

- `LeaderboardEntry.metrics` holds scalar summaries only. `rolling_ic`
  (`domain/ml/evaluation/metrics.py`) produces one IC per bar, and
  `signal_metrics` reduces it to scalars through `ic_statistics`; the series
  is not kept.
- The entry keeps `signal_fingerprint`, a hash of the evaluated signal, but
  not the signal, its timestamps or the IC observations.
- Regenerating is possible for some models but is not a recorded artifact.
  Rerunning `MLService.predict` on the entry's `dataset_id` and comparing
  `signal_fingerprint` confirms that the regenerated signal is the ranked one;
  an end-to-end test on 2026-10-31 matched the fingerprint and reproduced the
  stored `ic_mean` to ten decimal places. This needs the original model,
  feature code and dataset, and fails once any of them is gone. It is a
  diagnostic, not a substitute for the saved series: a later analysis must
  use the series that was actually evaluated.
- `predict` returns no timestamps, so the adapter must attach the dataset's
  `timestamp` column in saved row order before `comparison.align_returns` can
  pair two series.

Issue #62 tracks a versioned artifact with aligned timestamps, signals,
realised returns and IC observations. #62 remains necessary: later analysis
needs the series as they were evaluated, and saving them comes first. Its
acceptance criterion below makes the saved series carry the identities
needed to judge them.

### Layer 2: Holdout Status Is Unknown

Even with the series in hand, a paired comparison would not by itself answer
the question it is usually asked:

- `train_model`'s saved artifact is fitted on every row of its dataset, so
  `predict` on that dataset returns predictions on the training data.
  Walk-forward and purged validation report scores, not out-of-fold
  predictions; they estimate the training procedure, not the saved model.
- The record cannot tell an in-sample evaluation from a held-out one. See
  [Training Provenance on the Record](#training-provenance-on-the-record).
- A rule has no fitting step, but its threshold may have been chosen after
  viewing the same dataset, and nothing records that either.

On in-sample series a paired difference measures which candidate overfits
more thoroughly. In the 2026-10-31 test, the deeper of two LightGBM models
(`max_depth` 6 with 200 trees against `max_depth` 2 with 50 trees) "won" the
paired IC comparison at `p = 0`. That is this artifact, not a finding about
the models. Under the criterion for #62 below, saved series must
therefore carry their training and evaluation identities, and an analysis
reports the relationship with its result; holdout status is not a condition
for saving them or for ranking.

### Mechanics Once Series Are Saved

The steps the 2026-10-31 test validated:

1. Take two candidates on the same dataset (the same rows). Across horizons
   the datasets differ and the label intervals differ; pairing them needs a
   separate decision.
2. Use the IC method that matches the compared metric: Pearson for `ic_*`,
   Spearman for `rank_ic_*`.
3. Drop the first `window - 1` values by position, not with a global
   `dropna()`, so the structural warm-up stays apart from genuine failures.
4. Remove remaining non-finite values with a counted mask per series. Never
   fill with zero: zero is a plausible IC, and filling it fabricates
   observations and shrinks both the mean and the variance.
5. Take one value every `window` bars. Adjacent rolling ICs share
   `window - 1` bars; in the test the overlapping series pushed the estimated
   block length to 92, longer than the window, and its `n = 2782` overstated
   the sample. One value per window matches the `n_effective` that
   `ic_statistics` assumes.
6. Check that at least `MIN_BOOTSTRAP_OBSERVATIONS` values remain, roughly
   `21 * window` bars. Otherwise `difference_interval_outcome` returns
   `IntervalUnavailable` with the reason.

## Follow-up Work

Documented here, not implemented on this branch:

1. **Backtest candidates.** A candidate source that references `run_id` plus
   `scope` and carries the strategy version, parameters and trading
   conditions (instrument, frequency, period, costs), read from the saved
   metrics without rerunning. It must allow several scopes of one run, which
   `compare_runs` does not.
2. **Shared ranking core.** Extract the sort, tie and missing-value handling
   from `rank_entries` into a module both sources call, each supplying its
   metric definitions and directions. IC caveats and `required_t_stat` stay
   in the signal layer.
3. **Representative results.** Save the user's explicit mark that a strategy
   version and parameter set is finished for development, and the chosen
   representative result. New experiments never replace it automatically. The
   mark means inclusion in the comparison, not demonstrated out-of-sample
   performance.
4. **Walk-forward fold statistics.** Present `test_summary` fold statistics
   as a fold summary; never rank them as the performance of a whole backtest.
5. **Reports on the service.** Have the ML report take its ranking from the
   application service instead of calling `rank_entries` in
   `ml_report_block`, so MCP and reports share one result.
6. **Missing-value reasons.** Carry the reason a metric is missing from the
   saved record through ranking to MCP and reports, for both sources, instead
   of a bare `None` or a dropped candidate.

## Open Items

1. **`"no_overlap"` reason.** Proposed: report zero common periods as its own
   `UnavailableReason`, `"no_overlap"`, instead of
   `"insufficient_observations"` with `observations=0`, so tools can say "these
   two results never overlap" rather than "too short". Not changed on this
   branch.
2. **PR #136 (training dataset binding).** At head `d22e380` the PR adds
   `training_dataset_id` to `ModelArtifact` (schema 2; schema 1 stays
   readable), requires it when a model is built and refuses to overwrite a
   recorded value. When an evaluation is recorded, the value is copied to
   `LeaderboardEntry`, `RecordedEvaluation`, `LeaderboardRow` and
   `LeaderboardSkipped`. It records no training date range, label horizon,
   embargo or holdout status. It uses `""` both for "no model" (rules and
   other signals) and for "unknown" (a missing or unreadable artifact, or an
   older record); only the HTML report separates the two by method family.
   Nor does it record whether the backend produced the signal from the model
   or the caller supplied it, so for a caller-supplied signal the value says
   only where the named artifact was fitted. Both points fall short of the
   criterion for #62 below.
3. **Holdout eligibility rule.** Which training/evaluation pairs count as
   held out for a given analysis (for example non-overlapping date ranges
   after label horizons and embargo) is undecided. It is a property an
   analysis reports, not a condition for ranking.
4. **Cross-horizon surfaces.** `surface_caveats` states that cross-horizon
   source and raw-data equivalence is not verified, and each horizon is
   scored on a different set of rows. A plateau across horizons should carry
   that caveat until shared source bars and common evaluation rows are
   defined.
5. **Skipped rows lack settings.** Skipped rows in the response carry the
   three coordinates but not `parameters` or `signal_fingerprint`. A surface
   over hyperparameters must therefore read entries from the store; this was
   suggested to the maintainer and is not on `main`.

### Acceptance Criterion for #62

The criterion below was posted to issue #62 on 2026-11-01, and Kuan-Lun agreed
with its direction on PR #38. The text is the posted version, followed by three
additions made after Kuan-Lun's reply on PR #38 of 2026-11-01.

> **Why.** The paired comparison in `statistics/comparison.py` assumes it is
> comparing out-of-sample results. On in-sample series, a paired difference
> measures which candidate overfits more thoroughly, not which one is better.
> In an end-to-end test on the current pipeline, the deeper of two LightGBM
> models (`max_depth` 6 with 200 trees against `max_depth` 2 with 50 trees)
> "won" the paired IC comparison at p = 0, because both were scored on the rows
> they had been fitted on. That is the overfitting artifact, not a result.
>
> Nothing on the current record distinguishes the two cases: `LeaderboardEntry`
> stores only the evaluation `dataset_id`, `ModelArtifact` carries
> `n_train_samples` but no training-dataset identity, and `compute_model_id`
> hashes the model name, parameters, ordered feature names and the pickled
> blob's content hash; no data identity is part of it, so joining an entry's
> `model_id` to its artifact does not answer the question either.
>
> A saved series that cannot say whether it was held out would let a downstream
> comparison reproduce that artifact *with full provenance attached*. That is
> worse than today's gap, not better: the number becomes citable.
>
> **Proposed acceptance criterion.** Each saved evaluation series either
> declares its holdout status, or at minimum records the pairing of
> training-data identity and evaluation-data identity, so that a downstream
> consumer can judge eligibility itself. Concretely, a consumer holding two
> saved series can tell from the record alone, for each one, whether it was
> evaluated on the dataset its model was trained on, on a different dataset, or
> under an unknown relationship.
>
> Candidates without a fitting step (rules and other declared signals) record
> that they have no training data, distinctly from "unknown".
>
> Older entries, and any entry whose training identity is missing, report the
> relationship as unknown and are never treated as eligible by default, the
> same pattern the existing Acceptance already applies to missing series.
>
> **The pairing has to travel with the series artifact, not only sit on the
> leaderboard entry.** A consumer holding two saved series files should not
> need to re-join them to the leaderboard to learn whether either was held out.
>
> **Limitation, stated as part of the criterion.** Recording the
> training/evaluation pair makes the relationship auditable; it does not
> establish a holdout. Two different dataset IDs can still cover overlapping
> periods, and a rule's threshold may have been chosen after viewing the
> evaluation data. A v1 that meets this criterion therefore records what
> happened and does not assert `out_of_sample`. The rule for which pairs are
> eligible (for example non-overlapping date ranges after label horizons and
> embargo) remains open, and the presence of the fields should not be read as
> settling it.

Added after Kuan-Lun's reply on PR #38 (2026-11-01):

1. **Eligible means usable as input to a paired comparison.** It is not a
   condition for being ranked; ranking does not require a holdout.
2. **No fitting step means no training data, nothing more.** It does not mean
   a rule's form or threshold was never tuned on the evaluation data.
3. **Walk-forward series and historical records.** A walk-forward series
   saves, from the original evaluation run, each fold's actual predictions
   with that fold's training and evaluation ranges; predictions of a model
   retrained at the end on the same data do not substitute for them. A
   historical record without a saved series reports the series as
   unavailable; it is not regenerated and not presumed eligible.

## Statistical Basis

The literature, implementation and project-specific choices of each
procedure are in
[statistics_bootstrap_methods.md](statistics_bootstrap_methods.md). The
points below concern the leaderboard specifically.

### Two Selection Corrections

The project corrects for multiple comparisons in two ways, one per metric
family:

| Correction | Where | Metric family | Question answered |
| --- | --- | --- | --- |
| Two-sided Bonferroni threshold `Phi^-1(1 - alpha / (2 N))` | `required_t_stat`, `domain/ml/leaderboard.py` | IC t-statistics | Would any of `N` IC tests reject at family-wise `alpha`? |
| Deflated Sharpe ratio (Bailey and Lopez de Prado) | `deflated_sharpe_ratio`, `domain/statistics/sharpe.py` | Sharpe ratios of return series | How likely does the selected Sharpe beat the best of `N` skill-less trials? |

"Metric family" here groups the corrected numbers; it is unrelated to the
leaderboard's `method_family`. For the leaderboard, `N` is the number of
rankable candidates that the call's filters selected (`rank_entries` passes
that count to `required_t_stat`), so filtering by `family`, like any other
filter, lowers `N` to the candidates of that family. The correction then
covers the selection within the filtered set, not every candidate tried on
the dataset.

Their assumptions about dependence, and why they are never stacked, are in
[the methods page](statistics_bootstrap_methods.md#not-stacked-with-bonferroni).
For the leaderboard specifically, the Bonferroni guarantee is only as good as
each individual test, and the per-test normal approximation is
anti-conservative for short IC series, which `required_t_stat` and
`_approximation_note` already disclose.

### Why Paired

The variance of a difference is `var(A) + var(B) - 2 cov(A, B)`. Two candidates
scored on the same bars share their shocks, so the covariance term is large
and the paired interval is much narrower than two individual intervals
suggest. For KD(9, 3, 3) and KD(14, 3, 3), whose daily returns correlate at
0.68, the paired Sharpe difference interval was 3.53 wide, 0.69 times the 5.15
expected if the strategies were independent. See `METHODS.md`, sections 2.12
and 2.13, in the
[demo evidence](https://github.com/cklivse/tradingdev/tree/b74fd7e/evidence/2026-10-27-statistics-demo)
kept outside this repository's history. Those figures predate the current
block length rule for paired differences and have not been regenerated. IC
series of neighbouring candidates are expected to correlate at least as
strongly; the 2026-10-31 test measured 0.718 for two models, on in-sample
series.

### Effective Sample Size

`ic_statistics` corrects for overlapping windows with
`n_effective = max(1, n_bars // window)` and states that this is an
approximation. The stationary bootstrap in `comparison` instead chooses its
block length from the data with the Politis and White (2004) estimate, which
reflects the actual dependence and may be shorter or longer than `window`
bars. Taking one IC per window (step 5 above) keeps the two conventions
consistent.
