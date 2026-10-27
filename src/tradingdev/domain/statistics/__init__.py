"""Engine-independent performance statistics computed from raw run outputs.

Every function takes data the caller supplies (equity curves, timestamps,
trade records, return series or trial rows) and returns frozen result
objects. Nothing here reads runs, artifacts, caches or an engine's
``metrics``, and nothing writes files. No application service or MCP tool
calls this package yet; ``docs/statistics_leaderboard_integration.md``
lists what the ML leaderboard still needs before it can.

Modules, and the question each answers:

``periods``
    How does a per-bar equity curve become P&L and simple returns per UTC
    day or month? The other modules take returns built this way.
``performance``
    What are a run's headline figures (P&L, drawdown, trade counts, daily
    Sharpe) when recomputed from raw outputs, to cross-check the engine?
``sharpe``
    How uncertain is a Sharpe ratio (stationary-bootstrap interval, PSR),
    and is the best Sharpe of ``N`` trials more than luck (deflated
    Sharpe)? Also defines :class:`IntervalUnavailable`, the result every
    bootstrap function returns when no interval exists.
``comparison``
    Is one strategy better than another on the same data? Paired bootstrap
    of the difference of any statistic, and individual intervals for a
    forest plot.
``plateau``
    Is the best parameter set part of a stable region of the search grid,
    or an isolated spike?
``breakeven``
    What win rate does the payoff ratio need to break even, and how much
    turnover and cost does the run carry per year?
``benchmark``
    How do strategy returns relate to buy-and-hold of the instrument (beta,
    alpha, tracking error, information ratio, up/down capture)?

Dependencies between modules (``a -> b``: ``a`` imports ``b``)::

    performance -> periods, sharpe
    benchmark   -> periods, comparison
    comparison  -> sharpe
    plateau, breakeven, periods, sharpe: no package imports

``sharpe`` and ``comparison`` use ``arch`` for the stationary bootstrap and
its block length.

Where to start: read ``periods`` for the input conventions (UTC periods,
naive timestamps taken as UTC, returns relative to opening equity), then
``sharpe`` for the bootstrap and the unavailable-result contract, then
``comparison``. ``plateau`` stands alone. ``performance``, ``breakeven``
and ``benchmark`` are summaries of one run and can be read in any order.

Conventions shared by all modules: invalid input raises ``ValueError``; a
figure that is undefined for valid input, such as a ratio over zero, is
``None``, and the ``*_outcome`` bootstrap functions return an
:class:`~tradingdev.domain.statistics.sharpe.IntervalUnavailable` that
says why. Each module docstring gives its formulas and edge cases.
"""
