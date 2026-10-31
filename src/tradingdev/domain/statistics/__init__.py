"""Engine-independent statistics computed from raw run outputs.

Every function takes data the caller supplies (equity curves, timestamps,
trade records, return series or trial rows) and returns frozen result
objects, or, for the period series and buy-and-hold returns, a
``pd.Series`` and, for ``plateau_scores``, a tuple of scores. Nothing here
reads runs, artifacts, caches or an engine's
``metrics``, and nothing writes files. No application service or MCP tool
calls this package yet; ``docs/statistics_leaderboard_integration.md``
lists what the ML leaderboard still needs before it can.

Known limitations
-----------------

Read these first; each one limits what a result may claim.

* **One frequency per analysis.** Every figure is computed on returns of
  one frequency and annualised with one ``periods_per_year``. Daily returns
  are the reference (365 for markets that trade every day, 252 for
  exchange hours). Returns of different frequencies share few or no index
  labels, and a daily and a monthly Sharpe of the same run differ for
  statistical reasons, so they are two views, not two comparable numbers.
* **Short samples get no interval.** Every bootstrap needs
  ``MIN_BOOTSTRAP_OBSERVATIONS = 20`` periods (20 common periods for a
  comparison): 20 days of daily returns, but 20 months of monthly ones, so
  a one-year run never has a monthly interval.
* **Single-series Sharpe intervals may be too narrow.** Their block length
  follows the linear autocorrelation of the returns only, not the
  persistence of volatility. In a review simulation with two volatility
  regimes, the 95% interval covered the true Sharpe in 49 of 100 samples.
  This affects ``sharpe_interval_outcome`` and the individual rows of
  ``interval_table_outcome``; paired differences choose their block length
  from the statistic itself.
* **Plateau scores are local to one search.** They depend on the radius
  and the grid density, so they compare trials within one run only, never
  across runs or radii. A broad plateau fitted to one market regime is
  still overfitted, and plateaus built from training-fold metrics are
  in-sample.
* **The deflated Sharpe assumes independent trials.** Grid trials are
  strongly correlated, so the effective number of trials is smaller than
  ``N``; the ``deflated_sharpe_ratio`` docstring treats using the actual
  ``N`` as an over-correction that leans towards rejecting a real edge.
  That direction is the documented intent, not a measured result:
  correlated trials also have a smaller Sharpe dispersion, which lowers
  the hurdle.
* **Recomputed figures follow their own conventions.** The first period's
  P&L is measured from the first equity value, where the engine uses the
  initial cash; standard deviations use ``ddof=1``, where the engine's
  period P&L uses ``ddof=0``; Sharpe ratios assume a zero risk-free rate.
  A difference from the engine's ``metrics`` is not by itself an error.
* **Statistical is not economic significance**, and only out-of-sample
  returns of strategies run over the same period, instrument and costs
  are comparable. No statistic here repairs a comparison that fails that.

Modules, by the question each answers
-------------------------------------

``periods``
    How does a per-bar equity curve become P&L and returns per UTC day or
    month? The base of every return-based figure.
``performance``
    Do the engine's headline figures (P&L, drawdown, trade counts, daily
    Sharpe) hold up when recomputed from the raw outputs?
``sharpe``
    How sure can we be of this Sharpe ratio, and is the best Sharpe of
    ``N`` trials skill or luck (probabilistic and deflated Sharpe)?
``comparison``
    Is strategy A really better than B on the same data, or is the gap
    within noise? Works for any statistic, not only the Sharpe ratio.
``plateau``
    Is the best parameter set real, on a region where nearby parameters
    also do well, or a lucky spike in the grid?
``breakeven``
    How often must the strategy win to break even with its payoff ratio,
    and how much do turnover and costs take per year?
``benchmark``
    Is the strategy doing more than holding the instrument would (beta,
    alpha, tracking error, information ratio, up/down capture)?

Dependencies and reading order
------------------------------

``a -> b`` means ``a`` imports ``b``. The private ``_validation`` module
holds the shared input checks (real-number arrays, counts, seeds); it is
used by ``periods``, ``sharpe``, ``comparison``, ``plateau`` and
``breakeven``, and ``performance`` and ``benchmark`` reach it through
``periods`` and ``comparison``::

    performance -> periods, sharpe
    benchmark   -> periods, comparison
    comparison  -> sharpe
    plateau, breakeven, periods, sharpe: no other module of the package

Read ``periods`` first for the input conventions (UTC periods, naive
timestamps taken as UTC, returns relative to opening equity), then
``sharpe`` for the bootstrap and :class:`IntervalUnavailable`, then
``comparison``, which builds on both. ``plateau`` stands alone.
``performance``, ``breakeven`` and ``benchmark`` summarise one run and can
be read in any order.

What to call:

* One run's outputs: ``summarize_performance``, ``summarize_breakeven``,
  and ``period_returns(equity, timestamps, "daily")`` for its returns.
* Uncertainty of one Sharpe ratio: ``sharpe_interval_outcome``,
  ``probabilistic_sharpe_ratio``.
* The best of ``N`` trials: ``deflated_sharpe_ratio``.
* A against B: ``difference_interval_outcome``; ``interval_table_outcome``
  for each strategy's own interval in a forest plot.
* Against buy-and-hold: ``buy_and_hold_returns`` then
  ``compare_to_benchmark``.
* A parameter search: ``trials_from_metrics``, then ``plateau_summary``,
  ``plateau_grid`` or ``plateau_scores``.

Failure conventions
-------------------

These are spread over the modules and are the easiest part to misuse.

**Raise or ``None``.** Invalid input raises ``ValueError``: a value of the
wrong kind, non-finite, empty or mismatched in length, unordered
timestamps, an unknown direction or frequency, a return series that is
not a ``pd.Series`` where returns are paired by index, a count that is
not a built-in ``int`` of at least one, or a seed that is not ``None`` or
a built-in ``int`` of at least zero. Nothing is converted on the way:
booleans (also inside lists and object arrays) and numeric strings are not
numbers, NumPy integers are not counts, and numeric timestamps are rejected
because a number has no unit (convert Unix seconds with
``pd.to_datetime(values, unit="s")``). Integers are numbers, though, so a
0/1 signal of integer dtype passed as returns is not caught. A figure
that is undefined for valid input is ``None``, never ``0`` or infinity: a
Sharpe ratio of fewer than two returns or of constant returns, a return
or drawdown percentage when equity is not strictly positive, a win rate
without trades, a ratio whose denominator is zero, a plateau score without
neighbours. Check for ``None`` before formatting any field typed
``float | None``.

**Missing intervals and their reasons.** The bootstrap functions come in
two forms. ``sharpe_interval_outcome``, ``difference_interval_outcome``
and ``interval_table_outcome`` return an :class:`IntervalUnavailable`
instead of an interval, with ``reason`` (``"insufficient_observations"``,
``"undefined_statistic"`` or ``"no_valid_resample"``), the point
``estimate`` when it exists, ``observations`` and ``min_observations``,
and a ready ``message``. ``sharpe_confidence_interval``,
``difference_interval`` and ``interval_table`` collapse every such case to
``None`` (``interval_table`` returns ``None`` for the whole table when the
overlap is short), so use the ``*_outcome`` forms wherever a reader needs
to know why there is no interval.

**Plateau inputs.** ``axes`` is a required keyword of every plateau
function and must list every searched value, failed trials included; pass
``result.axes`` from ``trials_from_metrics`` together with
``result.trials``. Trials must be higher-is-better, so build them with
``trials_from_metrics``, which takes the metric's ``direction`` with no
default and negates ``"minimize"`` metrics; a hand-built :class:`Trial`
of a minimised metric is silently ranked upside down. Failed rows (metric
``None``, NaN or infinite) are dropped from ``trials`` but stay on the
axes; the count is ``result.dropped_count``, also issued as a
``DroppedTrialsWarning``, and must be reported with the result.

**Deflated Sharpe trial count.** ``n_trials`` must be every trial
searched, failed ones included: ``len(result.trials) +
result.dropped_count``. It defaults to the number of trial Sharpes, so
passing only the survivors underestimates ``N`` and under-corrects the
selection bias without any error.

**One ``periods_per_year`` everywhere.** ``sharpe_ratio``, the Sharpe
intervals, ``deflated_sharpe_ratio`` (whose trial Sharpes must be
annualised with the same value), ``summarize_performance`` and
``compare_to_benchmark`` all annualise with it. Use the value that
matches the return frequency, and the same value for every figure that is
compared; the functions cannot detect a mismatch. All of them require it
except ``summarize_performance``, which defaults to 365 for markets that
trade every day; pass it explicitly for anything else. The exception is
``probabilistic_sharpe_ratio``, which works per period: its
``benchmark_sr`` is a per-period Sharpe, so divide an annual benchmark by
``sqrt(periods_per_year)``.

Each module docstring gives its formulas and remaining edge cases.
"""
