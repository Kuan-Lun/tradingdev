# Statistics and ML Leaderboard Integration

## Purpose and Status

This document describes how the statistics package
(`src/tradingdev/domain/statistics/`) can consume what the ML leaderboard
records, what fails along the way, and what is still undecided.

This branch has merged upstream main at `a24594c` (2026-11-01). Every
statement about the leaderboard was checked against that code in this tree,
including the method families (`method_family` and `method_name` in
`domain/ml/leaderboard.py`) and the threshold rules (`domain/ml/rules.py`).
Code is cited by module, function and constant names.

### Scope

> The ML leaderboard ranks signal constructions, including non-ML rules, on
> one saved dataset. It does not rank strategies or backtests; strategy-level
> comparison remains in the statistics package.

This wording was proposed by the leaderboard maintainer on PR #38 and is
adopted here. It is **pending confirmation from Kuan-Lun**.

Under this scope the two sides do not overlap, and the interface between them
is a return series:

- A leaderboard candidate is a signal: a model's prediction series, a rule's
  `+1`/`-1` series, or another declared or unnamed signal, scored against
  forward returns. `method_family` says what built the signal, not what
  produced a strategy return.
- The leaderboard's `sharpe` and `annual_return` describe a cost-free
  simulated long/short book over non-overlapping holding intervals.
  `docs/ml_research.md` states that they are not backtest results.
- `comparison.difference_interval` takes two realised, cost-bearing return
  series of strategies run on the same data. The leaderboard produces no such
  series, so its figures are not inputs to `comparison` or `benchmark`.

## What the Leaderboard Provides Today

### Recorded Entry

The full field contract of a recorded candidate is the `LeaderboardEntry`
definition and its docstrings in `domain/ml/leaderboard.py`; the user-facing
contract, including method families and threshold rules, is
`docs/ml_research.md` ("Recorded Signal Evaluations" and "Method families").
The integration flows below read these fields:

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

### Ranking and Parameter Surfaces

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
3. ranks with `rank_entries` by the catalog direction from `metric_direction`.

The response carries `direction`, `parameter_axes`, `ranked` rows (`value` is
the ranking metric, with `horizon`, `window`, `n_quantiles`, `parameters` and
`signal_fingerprint`), `skipped` rows (with `reason` and the three
coordinates), and `caveats`, including `surface_caveats` when an axis has more
than one value. Ranking and caveats always cover the full candidate set, even
when only one page is returned.

`parameter_axes` returns `dict[str, list[int]]` with exactly the keys
`horizon`, `window` and `n_quantiles`, each sorted and deduplicated.

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
  its dataset. `MLService.predict` checks only that the dataset supplies the
  artifact's feature names, and its docstring says prediction "does not
  establish out-of-sample provenance".

## Integration Flow 1: Parameter Plateau

The plateau functions (`trials_from_metrics`, `plateau_summary`,
`plateau_grid`) can consume a leaderboard parameter surface over `horizon`,
`window` and `n_quantiles` today.

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
`method_name`, and all of them sit at the same `horizon`, `window` and `n_quantiles`. `parameter_axes`
has no threshold axis, and `ThresholdRule.parameters` states that threshold
surfaces are not supported.

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
broad plateau of in-sample scores is still in-sample. The record cannot
currently say which case applies (see
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
- Regenerating is possible but is not a recorded artifact. For a model,
  rerunning `MLService.predict` on the entry's `dataset_id` and comparing
  `signal_fingerprint` confirms that the regenerated signal is the ranked one.
  An end-to-end test on 2026-10-31 matched the fingerprint and reproduced the
  stored `ic_mean` to ten decimal places. This needs the original model,
  feature code and dataset, and it fails once any of them is gone.
- `predict` returns no timestamps, so the adapter must attach the dataset's
  `timestamp` column in saved row order before `comparison.align_returns` can
  pair two series.

Issue #62 tracks a versioned artifact with aligned timestamps, signals,
realised returns and IC observations. It would solve this layer.

### Layer 2: Holdout Status Is Unknown

Even with the series in hand, the comparison would not answer the question it
is meant to answer:

- `train_model`'s saved artifact is fitted on every row of its dataset, so
  `predict` on that dataset returns predictions on the training data.
  Walk-forward and purged folds report scores, not out-of-fold predictions;
  they estimate the training procedure, not the saved model.
- The record cannot tell an in-sample evaluation from a held-out one. See
  [Training Provenance on the Record](#training-provenance-on-the-record).
- A rule has no fitting step, but its threshold may have been chosen after
  viewing the same dataset, and nothing records that either.

The `comparison` module compares out-of-sample results. On in-sample series a
paired difference measures which candidate overfits more thoroughly. In the
2026-10-31 test, the deeper of two LightGBM models (`max_depth` 6 with 200
trees against `max_depth` 2 with 50 trees) "won" the paired IC comparison at
`p = 0`. That is this artifact, not a finding about the models.

### Mechanics Once Both Layers Are Solved

For when a held-out series exists, the steps the 2026-10-31 test validated:

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

## Open Items

1. **Scope confirmation.** The scope wording above is pending Kuan-Lun's
   confirmation on PR #38. If the team meant strategy-level ranking as well,
   ownership of `comparison` and `benchmark` must be settled before either
   side builds it.
2. **Training/evaluation pair.** The leaderboard maintainer has announced
   recording the training-dataset identity on the model artifact and the
   training/evaluation pair on the entry, with older records reading as
   unknown (upstream branch `feat/ml-training-dataset-binding`, not on
   `main` at `a24594c`). This makes the relationship auditable but does not
   prove a holdout; see the proposed criterion below.
3. **Holdout eligibility rule.** Which training/evaluation pairs count as
   eligible for a paired comparison (for example non-overlapping date ranges
   after label horizons and embargo) is undecided. It belongs to the
   statistics side and Kuan-Lun.
4. **Cross-horizon surfaces.** `surface_caveats` states that cross-horizon
   source and raw-data equivalence is not verified, and each horizon is
   scored on a different set of rows. A plateau across horizons should carry
   that caveat until shared source bars and common evaluation rows are
   defined.
5. **Skipped rows lack settings.** Skipped rows in the response carry the
   three coordinates but not `parameters` or `signal_fingerprint`. A surface
   over hyperparameters must therefore read entries from the store; this was
   suggested to the maintainer and is not on `main`.

### Proposed Acceptance Criterion for #62

The following text is proposed for issue #62:

> **Proposed acceptance criterion: holdout eligibility.** *Why.* The paired comparison in `statistics/comparison.py` assumes it is
> comparing out-of-sample results. On in-sample series, a paired difference
> measures which candidate overfits more thoroughly, not which one is better.
> In an end-to-end test on the current pipeline, the deeper of two LightGBM
> models (`max_depth` 6 with 200 trees against `max_depth` 2 with 50 trees)
> "won" the paired IC comparison at p = 0, because both were scored on the
> rows they had been fitted on. That is the overfitting artifact, not a
> result. A saved series that cannot say whether it was held out would let a
> downstream comparison reproduce that artifact with full provenance attached,
> so we would suggest that the artifact not ship without the following.
>
> *Criterion.* Each saved evaluation series either declares its holdout
> status, or at minimum records the pairing of training-data identity and
> evaluation-data identity, so that a downstream consumer can judge
> eligibility itself. Concretely, a consumer holding two saved series can
> tell from the record alone, for each one, whether it was evaluated on the
> dataset its model was trained on, on a different dataset, or under an
> unknown relationship. Candidates without a fitting step (rules and other
> signals) record that they have no training data, distinctly from "unknown".
> Older entries, and any entry whose training identity is missing, report the
> relationship as unknown and are never treated as eligible by default.
>
> *Limitation, stated as part of the criterion.* Recording the
> training/evaluation pair makes the relationship auditable; it does not
> establish a holdout. Two different dataset IDs can still cover overlapping
> periods, and a rule's threshold may have been chosen after viewing the
> evaluation data. A v1 that meets this criterion therefore records what
> happened and does not assert `out_of_sample`. The rule for which pairs are
> eligible (for example, non-overlapping date ranges after label horizons and
> embargo) remains open, and the presence of the fields should not be read as
> settling it.

## Statistical Basis

### Two Selection Corrections

The project corrects for multiple comparisons in two ways, one per metric
family:

| Correction | Where | Metric family | Question answered |
| --- | --- | --- | --- |
| Two-sided Bonferroni threshold `Phi^-1(1 - alpha / (2 N))` | `required_t_stat`, `domain/ml/leaderboard.py` | IC t-statistics | Would any of `N` IC tests reject at family-wise `alpha`? |
| Deflated Sharpe ratio (Bailey and Lopez de Prado) | `deflated_sharpe_ratio`, `domain/statistics/sharpe.py` | Sharpe ratios of return series | How likely does the selected Sharpe beat the best of `N` skill-less trials? |

They are not interchangeable and must never both be applied to the same
number: each already accounts for the whole selection. Reports that show both
must label each figure with its correction.

"Metric family" here groups the corrected numbers; it is unrelated to the
leaderboard's `method_family`. For the leaderboard, `N` is the number of
rankable candidates that the call's filters selected (`rank_entries` passes
that count to `required_t_stat`), so filtering by `family`, like any other
filter, lowers `N` to the candidates of that family. The correction then
covers the selection within the filtered set, not every candidate tried on
the dataset.

They differ in what they assume about dependence between candidates:

- **Bonferroni needs no independence.** It rests on the union bound: the
  probability that any of `N` tests rejects is at most the sum of their
  individual probabilities, whatever the dependence between them. Correlated
  candidates only make it conservative, so the actual family-wise error rate
  is below `alpha`. Its guarantee is only as good as each individual test,
  and the leaderboard's per-test normal approximation is anti-conservative
  for short IC series, which `required_t_stat` and `_approximation_note`
  already disclose.
- **The deflated Sharpe ratio assumes independent trials.** Its hurdle is the
  expected maximum Sharpe of `N` independent skill-less trials. Correlated
  grid trials have a smaller effective `N`, so using the raw `N` raises the
  hurdle; but they also have a smaller measured Sharpe dispersion, which
  lowers it. Leaning towards over-correction is therefore the documented
  intent, not a measured result. The `deflated_sharpe_ratio` docstring and
  the statistics package docstring record the same.

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
