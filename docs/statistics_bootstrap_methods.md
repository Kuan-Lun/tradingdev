# Statistics Bootstrap and Selection Methods

## Purpose

This document lists, for each resampling and selection procedure in
`src/tradingdev/domain/statistics/`, the literature it follows, the
implementation it relies on, the steps taken as published, and the steps that
were omitted, replaced or combined in this project, with the reasons. The
function docstrings remain the contract for arguments, return values and
failure paths; this page records where each method comes from and where it
departs from its source.

Implementation source: the resampling and block-length estimation come from
the `arch` package, version 8.0.0 as locked in `uv.lock`
(`arch.bootstrap.StationaryBootstrap` and
`arch.bootstrap.optimal_block_length`). Everything else is implemented in this
package with NumPy and the standard library.

## References

- **[PR94]** Politis, D. N., and Romano, J. P. (1994). The stationary
  bootstrap. *Journal of the American Statistical Association*, 89(428),
  1303–1313.
- **[PW04]** Politis, D. N., and White, H. (2004). Automatic block-length
  selection for the dependent bootstrap. *Econometric Reviews*, 23(1), 53–70.
- **[PPW09]** Patton, A., Politis, D. N., and White, H. (2009). Correction to
  "Automatic block-length selection for the dependent bootstrap" by D. Politis
  and H. White. *Econometric Reviews*, 28(4), 372–375.
- **[ET93]** Efron, B., and Tibshirani, R. J. (1993). *An Introduction to the
  Bootstrap*. Chapman & Hall. (Percentile interval: chapter 13, "Confidence
  intervals based on bootstrap percentiles", section 13.3, "The percentile
  interval".)
- **[LW08]** Ledoit, O., and Wolf, M. (2008). Robust performance hypothesis
  testing with the Sharpe ratio. *Journal of Empirical Finance*, 15(5),
  850–859. The time-series test is in section 3.2.2.
- **[BL12]** Bailey, D. H., and López de Prado, M. (2012). The Sharpe ratio
  efficient frontier. *Journal of Risk*, 15(2), 3–44 (December 2012).
  doi:10.21314/JOR.2012.255. The appendix gives the authors' Python code
  ([preprint](https://www.davidhbailey.com/dhbpapers/sharpe-frontier.pdf)).
- **[BL14]** Bailey, D. H., and López de Prado, M. (2014). The deflated Sharpe
  ratio: correcting for selection bias, backtest overfitting, and
  non-normality. *Journal of Portfolio Management*, 40(5), 94–107.
- **[K89]** Künsch, H. R. (1989). The jackknife and the bootstrap for general
  stationary observations. *The Annals of Statistics*, 17(3), 1217–1241.
  Cited only as background for linearising a statistic through its
  influence values; it is not the source of the rule in
  [Block length of a paired difference](#block-length-of-a-paired-difference).

## Shared Building Blocks

### Resampling

All intervals use the stationary bootstrap [PR94] through
`arch.bootstrap.StationaryBootstrap`: blocks start at uniformly drawn periods,
have geometrically distributed lengths with the chosen mean, and wrap around
the end of the sample. Paired and joint resampling pass every aligned series to
one `StationaryBootstrap`, so a resample draws the same period indices for all
of them. The generator is `numpy.random.default_rng(seed)`.

As published. No step is omitted.

### Automatic block length

`arch.bootstrap.optimal_block_length` implements the automatic estimate of
[PW04] with the correction of [PPW09]; the package reads its `stationary`
column. Applied as published, with these additions in `sharpe._block_length`:

- **Rounding.** The estimate is a real number; it is rounded up to a whole
  number of periods, with a minimum of one, because the bootstrap needs an
  integer mean block length and rounding down shortens blocks, which narrows
  the interval.
- **Several columns.** When the estimate is taken over several columns, the
  largest finite estimate is used. A block length too short for any column
  understates that column's dependence; one too long costs only precision.
- **Constant columns** are dropped before estimation, because they have no
  dependence to preserve and make the estimator fail; with no finite estimate
  the block length is one.
- **Rescaling.** A column whose spread is below `2**-60` or whose largest
  magnitude is above `2**60` is rescaled by an exact power of two before
  estimation (`_scaling.rescaled_columns`); other columns are used as they
  are. The estimate does not depend on scale, but its numerics fall back to
  one far from unit spread.

### Minimum sample: `MIN_BOOTSTRAP_OBSERVATIONS = 20`

Fewer than 20 returns (after alignment, for comparisons) give no interval;
the `*_outcome` functions report `"insufficient_observations"`.

Not from the literature: this is a project choice. [PW04] recommend no
minimum sample; their results are asymptotic, and their simulations use
samples of 200 and 800 observations, noting extreme values of the circular
bootstrap's estimate at 200. A floor of 20 is far below any sample size they
studied. The automatic estimate relies on sample autocorrelations and a band of
`±2 * sqrt(log10(n) / n)`, which say little in very short samples, and a
percentile interval from a handful of distinct values is mostly
discreteness. The threshold counts periods, so it is 20 days of daily returns
but 20 months of monthly ones; a one-year run never has a monthly interval.
The value is a floor against meaningless output, not a sample size at which
coverage was shown to be adequate.

### Percentile interval

Every interval is the equal-tailed percentile interval of the resampled
statistic [ET93]: the `(1 - confidence) / 2` and `1 - (1 - confidence) / 2`
quantiles of the valid resamples, with NumPy's default linear interpolation.

Replaced, not followed: [LW08] (section 3.2.2) test a Sharpe difference with
a studentized circular block bootstrap. The circular block bootstrap
(Politis and Romano, 1992, as cited in [LW08]) resamples blocks of fixed size
`b` from the observed pairs of returns. The standard error is a HAC estimate
with the prewhitened quadratic spectral kernel (Andrews and Monahan, 1992,
as cited) for the original sample, and is computed from block sums inside
each resample (following Götze and Künsch, 1996, as cited). Studentizing
generally gives better coverage than a percentile interval for a ratio
statistic such as the Sharpe ratio. This package uses the
percentile interval instead because:

- `difference_interval_outcome` accepts any `statistic` callable (a Sharpe
  ratio, a mean, a maximum drawdown, an IC mean). A studentized interval needs
  a standard-error formula for each statistic; the percentile interval needs
  none.
- One rule for single-series intervals, joint tables and paired differences
  keeps the three directly comparable.

Two further departures from [LW08]:

- **Symmetric versus equal-tailed.** [LW08] build a symmetric interval from
  the bootstrap distribution of `|Δ* - Δ̂| / s(Δ*)`, so both ends lie the same
  distance from the estimate. This package takes equal-tailed percentiles,
  which can be asymmetric. A symmetric interval needs the studentized
  statistic; without a standard error there is nothing to symmetrise.
- **Block length.** [LW08] choose the fixed block size by calibration
  (their Algorithm 3.1, after Loh, 1987, as cited): for each candidate size
  they resample from a fitted model, measure the coverage of the resulting
  intervals and keep the size whose coverage is closest to nominal. This
  package uses the automatic estimate of [PW04] instead. Calibration repeats
  the whole bootstrap for every candidate size and needs a model to generate
  data from, which a statistic passed as an arbitrary callable does not
  provide.

Bias-corrected and accelerated (BCa) intervals were not adopted either; their
acceleration constant is normally estimated with an iid jackknife, which does
not fit dependent data without a further blocked construction.

The cost is coverage. The `sharpe_interval_outcome` docstring records a
simulation in which nominal 95% intervals covered the true Sharpe ratio in 70
of 100 samples under regime-switching volatility. Read the intervals as
approximate, never as exact 95% statements.

### Undefined resamples (`valid_reps`)

A resample on which the statistic is undefined (for example a Sharpe ratio of
a resample that happens to be constant) is dropped. The interval is computed
from the remaining draws, and the result reports `reps` and `valid_reps`.
When no draw is valid, the outcome is `"no_valid_resample"`.

Not from the literature. Dropping undefined draws conditions the
distribution on the statistic being defined, which can bias the interval
when many draws are dropped. Compare `valid_reps` with `reps` before reading
an interval.

## Procedures

### Sharpe ratio interval

`sharpe.sharpe_interval_outcome` (and `sharpe_confidence_interval`, which
returns `None` instead of the reason).

- **Literature:** stationary bootstrap [PR94]; automatic block length [PW04]
  with [PPW09]; percentile interval [ET93].
- **As published:** resampling, block-length estimate, percentile quantiles.
- **Combined in this project:** the mean block length is the larger of two
  automatic estimates, one on the returns and one on the squared returns
  (`sharpe._returns_block_size`). The Sharpe ratio depends on the mean and the
  variance. Returns can be nearly uncorrelated while their volatility persists,
  and blocks fitted to the returns alone cut those stretches apart and
  understate the variability. The squared returns carry that persistence. In
  the simulation recorded in the docstring this raised coverage from 49 to 70
  of 100 under regime-switching volatility, kept it on independent and AR(1)
  returns, and on GARCH(1,1) returns gave 89 against 91, within the sampling
  error of 100 samples.
- **Other choices:** minimum sample, percentile interval and dropped
  resamples as above. A caller may pass a fixed `block_size`, which is
  reported back unchanged and does not lift the minimum.

### Single-series intervals for a comparison table

`comparison.interval_table_outcome` (and `interval_table`).

- **Literature:** as for the Sharpe interval.
- **Combined in this project:** all series are aligned to their common
  periods first and resampled jointly with one block length, the largest
  estimate over every series and its squares (the same
  `_returns_block_size` rule). Joint resampling keeps every row on the same
  periods, so the table is a forest plot of one sample.
- These intervals ignore the correlation between series. Non-overlapping
  rows imply a difference; overlapping rows do not imply none. The paired
  difference below is the test.

### Paired difference

`comparison.difference_interval_outcome` (and `difference_interval`).

- **Literature:** paired resampling of the difference of a statistic follows
  the design of [LW08]: both series are resampled with the same indices so
  their correlation is kept. Resampling [PR94]; block length [PW04] with
  [PPW09]; percentile interval [ET93].
- **Replaced:** [LW08] use a circular block bootstrap and a studentized
  statistic; here the stationary bootstrap and a percentile interval are used,
  for the reasons under [Percentile interval](#percentile-interval).
- **Combined:** the block length (next section) and the p-value
  ([p-value](#p-value-of-a-paired-difference)).

#### Block length of a paired difference

The blocks must keep the dependence of the estimator of the difference, not
the dependence of either series. When two strategies share large market
noise, the noise cancels in the difference while a small persistent
difference remains; block lengths fitted to each series follow the
short-memory noise and make an insignificant difference look significant.

This project therefore applies the [PW04] estimate to the jackknife
influence values of the difference (`comparison._jackknife_influence`):
element `t` is `(n - 1)` times the full-sample difference
`statistic(a) - statistic(b)` minus the same difference with period `t` left
out of both series (up to a constant, which the estimate ignores). For the mean this is `a_t - b_t`
up to a constant, so the rule reduces to estimating the block length of the
difference series. The usual centring on the full-sample difference is left
out because the estimate ignores location and scale.

- **Not from the literature as a combination.** [PW04] derive the estimate
  for the sample mean. Applying it to influence values relies on the standard
  linearisation of a smooth statistic as approximately the mean of its
  influence values (see [K89] for the background); no source was found that
  proposes this exact rule.
- **Fallback:** if any leave-one-out statistic is undefined, the raw
  difference `a - b` is used.
- **Limits:** the jackknife does not describe non-smooth statistics (a
  median, a quantile, a maximum drawdown), whose leave-one-out values jump.
  For those, and for statistics too costly to evaluate `2 * n` more times, a
  caller must pass `block_size`. In the docstring's simulation the influence
  rule covered 60 of 100 under regime-switching volatility, below the
  squared-returns rule, because the variance term is diluted by the mean term.

#### p-value of a paired difference

The p-value is the smallest `1 - confidence` at which the percentile interval
excludes zero (`comparison._percentile_p_value`). It inverts the same linearly
interpolated quantiles that give the interval, so the interval at `confidence`
excludes zero exactly when `p_value < 1 - confidence`, for any number of
resamples, up to rounding when the two are equal.

- **Not from the literature.** It is the p-value implied by inverting a
  percentile interval, not a test statistic with a reference distribution,
  and it inherits the percentile interval's coverage error.
- **Few resamples:** it is zero when every resample lies on one side of zero.
  Below `2 / (valid_reps - 1)` it only interpolates between the two most
  extreme resamples, so with few `reps` a small p-value is no more precise
  than that interpolation. The default is `reps = 2000`; draw more when small
  levels matter.

### Probabilistic Sharpe ratio

`sharpe.probabilistic_sharpe_ratio`.

- **Literature:** [BL12]:
  `PSR = Phi((SR - SR*) * sqrt(n - 1) / sqrt(1 - g3 * SR + (g4 - 1) / 4 * SR**2))`,
  with the per-period Sharpe ratio, skewness `g3` and non-excess kurtosis `g4`.
- **As published:** the formula, with a per-period benchmark `SR*`.
- **Choices in this project:** skewness and kurtosis use population moments
  (dividing by `n`). The [BL12] text does not specify the estimator; the
  authors' appendix code (`PSR_Opt.get_Moments`) divides by the number of
  observations, as here. The per-period Sharpe ratio differs: the appendix
  code divides the mean by the population standard deviation, while
  `sharpe._period_sharpe` uses the sample standard deviation (`ddof=1`), the
  same Sharpe ratio as the rest of this package. The two differ by a factor
  of `sqrt(n / (n - 1))`, which is negligible for long samples. Returns whose
  spread is below `2**-60` or whose largest magnitude is above `2**60` are
  rescaled by an exact
  power of two before the moments are taken, which leaves the ratios
  unchanged and avoids overflow of the fourth moment.
- **Undefined cases:** `None` with fewer than two returns, zero dispersion,
  or a variance term that is not positive.

### Deflated Sharpe ratio

`sharpe.deflated_sharpe_ratio`.

- **Literature:** [BL14]. The PSR benchmark is raised to the expected maximum
  Sharpe of `N` skill-less trials,
  `SR*_0 = std(SR_n) * ((1 - gamma) * Z^-1(1 - 1/N) + gamma * Z^-1(1 - 1/(N e)))`.
- **As published:** the expected-maximum formula and `DSR = PSR(SR*_0)`.
- **Choices in this project:**
  - `std(SR_n)` is the sample standard deviation (`ddof=1`) of the trials'
    per-period Sharpe ratios; annualised inputs are converted to per-period
    values so the hurdle is not inflated by `sqrt(periods_per_year)`.
  - `N` is `n_trials`, which the caller must set to the total number of
    trials searched, failed ones included (for plateau input,
    `len(result.trials) + result.dropped_count`). It defaults to the number
    of trial Sharpes given, which undercounts whenever trials were dropped.
    The formula's `N` counts independent trials; estimating an effective number
    of independent trials from correlated ones (for example by clustering
    their returns) is omitted and out of scope. Correlated grid trials have a smaller effective `N`, which
    makes the raw `N` raise the hurdle, but also a smaller measured
    dispersion, which lowers it. The net direction of the error has not been
    measured.
  - With `N = 1`, or equal trial Sharpes, the hurdle is zero and the DSR
    equals the PSR against zero.
  - `None` when the PSR is undefined, or when `N >= 2` but fewer than two
    trial Sharpes are given, so their dispersion cannot be estimated.
  - `Z^-1(1 - p)` is computed as `-Z^-1(p)`, and `1/(N e)` as `(1/N) / e`, to
    keep precision for large `N`.

### Not stacked with Bonferroni

The ML leaderboard's `required_t_stat` (`domain/ml/leaderboard.py`) applies a
two-sided Bonferroni threshold to IC t-statistics. The DSR corrects the
Sharpe ratios of strategy return series. Each already accounts for the whole
selection it covers, so the two are never applied to the same number;
stacking them corrects one selection twice. Bonferroni rests on the union
bound and needs no independence; the DSR's expected maximum assumes
independent trials. Reports that show both label each figure with its
correction.

## What This Page Does Not Claim

- No procedure here has a demonstrated coverage at its nominal level for the
  dependence found in strategy returns. The simulations cited are those
  recorded in the docstrings, with 100 samples each.
- Resampling assumes the series is stationary over the sample. A regime
  change inside the sample is not repaired by any block length.
- These methods quantify sampling uncertainty of one sample. They do not
  correct for in-sample selection outside the trials passed to the DSR, and
  they do not make an in-sample series out-of-sample.
