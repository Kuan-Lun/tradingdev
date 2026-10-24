"""Generate one-off meeting demo material for domain.statistics.

Outputs (in demo_output/):
  plateau_gridindex.png
                    true plateau vs. false spike heatmaps
  plateau_gridindex_table.md
                    plateau statistics of both cases
  kd_inference.md   inference statistics of the latest cached KD backtest
  kd_equity.png     equity curve and drawdown of that backtest
  comparison_forest.png
                    daily Sharpe intervals of three KD parameter sets and
                    their paired differences

Run: uv run python scripts/demo_statistics.py
"""

from __future__ import annotations

import copy
import itertools
import math
import pickle
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from tradingdev.app.backtest_service import BacktestService
from tradingdev.domain.statistics.breakeven import summarize_breakeven
from tradingdev.domain.statistics.comparison import (
    difference_interval,
    interval_table,
)
from tradingdev.domain.statistics.periods import period_pnl, period_returns
from tradingdev.domain.statistics.plateau import (
    PlateauSummary,
    Trial,
    plateau_grid,
    plateau_summary,
)
from tradingdev.domain.statistics.sharpe import (
    probabilistic_sharpe_ratio,
    sharpe_confidence_interval,
    sharpe_ratio,
)
from tradingdev.shared.utils.config import load_config

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "demo_output"
CACHE_DIR = ROOT / "workspace" / "data" / "processed" / "cache"
KD_CONFIG = ROOT / "src/tradingdev/domain/strategies/bundled/kd_strategy/config.yaml"
KD_VARIANTS = {
    "KD(9, 3, 3)": {"k_period": 9, "d_period": 3, "smooth_k": 3},
    "KD(14, 3, 3)": {"k_period": 14, "d_period": 3, "smooth_k": 3},
    "KD(21, 5, 5)": {"k_period": 21, "d_period": 5, "smooth_k": 5},
}
COMPARISON_REPS = 2000
GRID_SIZE = 5
DAILY_PERIODS_PER_YEAR = 365.0
SEED = 0
BOOTSTRAP_REPS = 1000
FEES = 0.0006
SLIPPAGE = 0.0005
ACCENT = "#2a6fdb"
INK = "#1f2328"
MUTED = "#6e7781"


# ---------------------------------------------------------------- plateau


GRID_AXIS = [float(i) for i in range(GRID_SIZE)]
PLATEAU_AXES = {"x": GRID_AXIS, "y": GRID_AXIS}


def _trials(metric: Callable[[int, int], float]) -> list[Trial]:
    return [
        Trial(params={"x": float(i), "y": float(j)}, metric=metric(i, j))
        for i in range(GRID_SIZE)
        for j in range(GRID_SIZE)
    ]


def plateau_cases() -> dict[str, list[Trial]]:
    rng = np.random.default_rng(SEED)
    plateau_noise = rng.normal(0.0, 0.05, (GRID_SIZE, GRID_SIZE))
    spike_noise = rng.normal(0.0, 0.3, (GRID_SIZE, GRID_SIZE))
    return {
        "True plateau": _trials(
            lambda i, j: 1.5 - 0.3 * abs(i - 2) - 0.3 * abs(j - 2) + plateau_noise[i, j]
        ),
        "False spike": _trials(
            lambda i, j: 2.0 if (i, j) == (1, 3) else float(spike_noise[i, j])
        ),
    }


def _cell(params: Any) -> tuple[float, float]:
    return float(params["x"]), float(params["y"])


def plot_plateau(
    cases: dict[str, list[Trial]], path: Path
) -> dict[str, PlateauSummary]:
    summaries = {
        name: plateau_summary(trials, axes=PLATEAU_AXES)
        for name, trials in cases.items()
    }
    grids = {
        name: plateau_grid(trials, axes=PLATEAU_AXES, x="x", y="y")
        for name, trials in cases.items()
    }
    matrices = {
        name: np.array(
            [[np.nan if v is None else v for v in row] for row in grid.metric]
        )
        for name, grid in grids.items()
    }
    vmin = min(float(np.nanmin(m)) for m in matrices.values())
    vmax = max(float(np.nanmax(m)) for m in matrices.values())

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), constrained_layout=True)
    image = None
    for ax, (name, matrix) in zip(axes, matrices.items(), strict=True):
        summary = summaries[name]
        image = ax.imshow(matrix, origin="lower", cmap="Blues", vmin=vmin, vmax=vmax)
        for (row, col), value in np.ndenumerate(matrix):
            shade = (value - vmin) / (vmax - vmin)
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if shade > 0.55 else INK,
            )
        bx, by = _cell(summary.best_params)
        px, py = _cell(summary.plateau_best_params)
        ax.scatter(
            [bx],
            [by],
            s=900,
            facecolors="none",
            edgecolors="white",
            linewidths=2.5,
            path_effects=[pe.withStroke(linewidth=4.5, foreground=INK)],
        )
        ax.scatter(
            [px],
            [py],
            s=1300,
            marker="s",
            facecolors="none",
            edgecolors="black",
            linewidths=2.5,
            path_effects=[pe.withStroke(linewidth=4.5, foreground="white")],
        )
        ax.set_title(
            f"{name}\nmetric_gap = {summary.metric_gap:.3f}   "
            f"grid_distance = {summary.grid_distance}",
            fontsize=11,
            color=INK,
        )
        ax.set_xticks(range(GRID_SIZE))
        ax.set_yticks(range(GRID_SIZE))
        ax.set_xlabel("Parameter x (grid index)", color=MUTED)
        ax.set_ylabel("Parameter y (grid index)", color=MUTED)
        ax.tick_params(length=0, colors=MUTED)
        for spine in ax.spines.values():
            spine.set_visible(False)

    assert image is not None
    fig.colorbar(image, ax=axes, shrink=0.85, label="Metric (higher is better)")
    legend = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=13,
            markerfacecolor="none",
            markeredgecolor=INK,
            markeredgewidth=2,
            label="Best metric (single best trial)",
        ),
        Line2D(
            [],
            [],
            marker="s",
            linestyle="none",
            markersize=13,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=2,
            label="Plateau best (highest neighbour mean)",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.45, -0.08),
    )
    fig.suptitle(
        "Parameter plateau vs. lucky spike (5x5 grid, radius 1)", fontsize=13, color=INK
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return summaries


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def write_plateau_table(summaries: dict[str, PlateauSummary], path: Path) -> None:
    rows: list[tuple[str, Callable[[PlateauSummary], str]]] = [
        ("best_metric", lambda s: _fmt(s.best_metric)),
        (
            "best_params (x, y)",
            lambda s: "({:.0f}, {:.0f})".format(*_cell(s.best_params)),
        ),
        (
            "plateau_best_params (x, y)",
            lambda s: "({:.0f}, {:.0f})".format(*_cell(s.plateau_best_params)),
        ),
        ("plateau_best_score", lambda s: _fmt(s.plateau_best_score)),
        ("metric_gap", lambda s: _fmt(s.metric_gap)),
        ("grid_distance", lambda s: _fmt(s.grid_distance)),
        ("best_neighborhood_mean", lambda s: _fmt(s.best_neighborhood_mean)),
        ("best_neighborhood_std", lambda s: _fmt(s.best_neighborhood_std)),
        ("best_neighborhood_size", lambda s: _fmt(s.best_neighborhood_size)),
        ("spikiness", lambda s: _fmt(s.spikiness, 2)),
    ]
    names = list(summaries)
    lines = [
        "# Plateau statistics: true plateau vs. false spike",
        "",
        "Synthetic 5x5 grids, radius-1 Chebyshev neighbourhood, seed 0.",
        "",
        "| Statistic | " + " | ".join(names) + " |",
        "| --- | " + " | ".join("---:" for _ in names) + " |",
    ]
    for label, getter in rows:
        lines.append(
            f"| {label} | " + " | ".join(getter(summaries[n]) for n in names) + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- KD run


def latest_cache() -> Path:
    pickles = sorted(CACHE_DIR.glob("*.pkl"), key=lambda p: p.stat().st_mtime)
    if not pickles:
        raise SystemExit(f"no cached pickle in {CACHE_DIR}")
    return pickles[-1]


def write_kd_inference(
    result: Any, snapshot: dict[str, Any], source: Path, path: Path
) -> None:
    equity = np.asarray(result.equity_curve, dtype=float)
    stamps = result.timestamps
    daily_pnl = period_pnl(equity, stamps, "daily")
    monthly_pnl = period_pnl(equity, stamps, "monthly")
    daily_ret = period_returns(equity, stamps, "daily").to_numpy()

    sharpe = sharpe_ratio(daily_ret, periods_per_year=DAILY_PERIODS_PER_YEAR)
    interval = sharpe_confidence_interval(
        daily_ret,
        periods_per_year=DAILY_PERIODS_PER_YEAR,
        reps=BOOTSTRAP_REPS,
        seed=SEED,
    )
    psr = probabilistic_sharpe_ratio(daily_ret)
    t_stat = float(
        daily_pnl.mean() / (daily_pnl.std(ddof=1) / math.sqrt(len(daily_pnl)))
    )

    breakeven = summarize_breakeven(
        result.trades, start=stamps[0], end=stamps[-1], capital=result.init_cash
    )

    cost = breakeven.traded_notional * (FEES + SLIPPAGE)
    # Older engines left `fee = 0`; current ones record fees but fold slippage
    # into fill prices, so either way the full cost is estimated from notional.
    recorded_fees = (
        "Trade records carry `fee = 0` because the engine folds costs into fill "
        "prices, so this is an estimate."
        if breakeven.total_cost == 0.0
        else f"Trade records carry fees of {breakeven.total_cost:,.2f} but no "
        "slippage, which the engine folds into fill prices, so this is an estimate."
    )
    net = float(equity[-1] - equity[0])
    gross = net + cost

    strategy = snapshot.get("strategy", {})
    bt = snapshot.get("backtest", {})
    engine_sharpe = float(result.metrics["sharpe_ratio"])
    block = interval.block_size if interval else "n/a"
    ci = f"[{interval.lower:.2f}, {interval.upper:.2f}]" if interval else "n/a"

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.1%}"

    lines = [
        "# KD strategy: inference statistics",
        "",
        f"Source: `{source.relative_to(ROOT)}` — strategy `{strategy.get('id')}` "
        f"{strategy.get('parameters')}, {bt.get('symbol')} {bt.get('timeframe')}, "
        f"{stamps[0]:%Y-%m-%d} to {stamps[-1]:%Y-%m-%d}, init cash "
        f"{result.init_cash:,.0f}.",
        "",
        "## Sample size",
        "",
        "| Statistic | Value |",
        "| --- | ---: |",
        f"| Real days (UTC days with data) | {len(daily_pnl)} |",
        f"| Months | {len(monthly_pnl)} |",
        f"| Share of months with negative P&L | {pct(float((monthly_pnl < 0).mean()))} "
        f"({int((monthly_pnl < 0).sum())}/{len(monthly_pnl)}) |",
        "",
        "## Sharpe inference (daily returns, annualised x sqrt(365))",
        "",
        "| Statistic | Value |",
        "| --- | ---: |",
        f"| Annualised Sharpe (daily) | {_fmt(sharpe, 2)} |",
        f"| Engine-reported Sharpe (hourly) | {engine_sharpe:.2f} |",
        f"| Stationary bootstrap 95% CI (seed {SEED}, {BOOTSTRAP_REPS} reps) | {ci} |",
        f"| Bootstrap block size (days) | {block} |",
        f"| PSR (P[true SR > 0]) | {_fmt(psr, 4)} |",
        f"| t-statistic of daily P&L | {t_stat:.2f} |",
        "",
        "## Break-even win rate and turnover",
        "",
        "| Statistic | Value |",
        "| --- | ---: |",
        f"| Break-even win rate p* | {pct(breakeven.breakeven_win_rate)} |",
        f"| Realised win rate (decisive trades) | {pct(breakeven.win_rate)} |",
        f"| Gap (realised - p*) | {pct(breakeven.win_rate_edge)} |",
        f"| Payoff ratio W/L | {_fmt(breakeven.payoff_ratio, 2)} |",
        f"| Trades (win / loss) | {breakeven.total_trades} "
        f"({breakeven.winning_trades} / {breakeven.losing_trades}) |",
        f"| Annual turnover (x capital) | {_fmt(breakeven.annual_turnover, 1)} |",
        "",
        "## Cost breakdown",
        "",
        f"Estimated cost = traded notional (entry + exit) x (fees {FEES} + "
        f"slippage {SLIPPAGE}). {recorded_fees} The engine's "
        "`total_volume` metric changed upstream from entry-only to entry plus "
        "exit notional, so the cost is always derived from the trade records, "
        "never from that metric.",
        "",
        "| Statistic | Value |",
        "| --- | ---: |",
        f"| Traded notional (entry + exit) | {breakeven.traded_notional:,.0f} |",
        f"| Estimated cost | {cost:,.2f} |",
        f"| Gross P&L (net + cost) | {gross:,.2f} |",
        f"| Net P&L | {net:,.2f} |",
        f"| Cost / abs(net P&L) | {pct(cost / abs(net)) if net else 'n/a'} |",
        f"| Cost / initial capital | {pct(cost / result.init_cash)} |",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_kd_equity(result: Any, path: Path) -> None:
    equity = np.asarray(result.equity_curve, dtype=float)
    stamps = np.asarray([t.to_pydatetime() for t in result.timestamps])
    drawdown = equity / np.maximum.accumulate(equity) - 1.0

    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(11, 6),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1.3]},
    )
    top.plot(stamps, equity, color=ACCENT, linewidth=2)
    top.axhline(equity[0], color=MUTED, linewidth=1, linestyle="--")
    top.set_ylabel("Equity (USDT)", color=MUTED)
    top.set_title(
        f"KD crossover, BTC/USDT 1h — equity {equity[0]:,.0f} -> {equity[-1]:,.0f}",
        color=INK,
        loc="left",
    )
    bottom.fill_between(
        stamps, drawdown * 100, 0, color=ACCENT, alpha=0.35, linewidth=0
    )
    bottom.plot(stamps, drawdown * 100, color=ACCENT, linewidth=1.2)
    bottom.set_ylabel("Drawdown (%)", color=MUTED)
    bottom.set_xlabel("Date (UTC)", color=MUTED)
    low = int(np.argmin(drawdown))
    bottom.annotate(
        f"Max DD {drawdown[low]:.1%}",
        (stamps[low], drawdown[low] * 100),
        xytext=(-150, 22),
        textcoords="offset points",
        color=INK,
        fontsize=9,
    )
    for ax in (top, bottom):
        ax.grid(axis="y", color="#d0d7de", linewidth=0.6)
        ax.tick_params(colors=MUTED, length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- comparison


def run_kd_variants() -> dict[str, Any]:
    """Backtest each KD parameter set on the bundled config's data."""
    base = load_config(KD_CONFIG)
    service = BacktestService()
    results: dict[str, Any] = {}
    for label, parameters in KD_VARIANTS.items():
        config = copy.deepcopy(base)
        config["strategy"]["parameters"].update(parameters)
        results[label] = service.run_raw_config(config).pipeline.backtest_result
    return results


def plot_comparison_forest(results: dict[str, Any], path: Path) -> None:
    returns = {
        label: period_returns(
            np.asarray(result.equity_curve, dtype=float), result.timestamps, "daily"
        )
        for label, result in results.items()
    }
    statistic = partial(sharpe_ratio, periods_per_year=DAILY_PERIODS_PER_YEAR)
    table = interval_table(returns, statistic, reps=COMPARISON_REPS, seed=SEED)
    if table is None:
        raise SystemExit("KD variants do not overlap enough to compare")
    pairs = {
        f"{a} - {b}": difference_interval(
            returns[a], returns[b], statistic, reps=COMPARISON_REPS, seed=SEED
        )
        for a, b in itertools.combinations(returns, 2)
    }

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 6.4), sharex=True, constrained_layout=True
    )
    singles = [(label, row) for label, row in table.items() if row is not None]
    for y, (_, row) in enumerate(reversed(singles)):
        top.plot([row.lower, row.upper], [y, y], color=ACCENT, linewidth=2)
        top.plot(row.estimate, y, "o", color=ACCENT, markersize=8)
        top.annotate(
            f"{row.estimate:.2f}  [{row.lower:.2f}, {row.upper:.2f}]",
            (row.upper, y),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK,
        )
    top.set_yticks(range(len(singles)), [label for label, _ in reversed(singles)])
    top.set_title(
        "Annualised Sharpe (daily returns), 95% stationary-bootstrap CI",
        color=INK,
        loc="left",
        fontsize=11,
    )

    diffs = [(label, d) for label, d in pairs.items() if d is not None]
    for y, (_, d) in enumerate(reversed(diffs)):
        color = INK if d.lower > 0.0 or d.upper < 0.0 else MUTED
        bottom.plot([d.lower, d.upper], [y, y], color=color, linewidth=2)
        bottom.plot(d.difference, y, "D", color=color, markersize=7)
        bottom.annotate(
            f"{d.difference:+.2f}  [{d.lower:+.2f}, {d.upper:+.2f}]  "
            f"p={d.p_value:.3f}  rho={_fmt(d.correlation, 2)}",
            (d.upper, y),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK,
        )
    bottom.set_yticks(range(len(diffs)), [label for label, _ in reversed(diffs)])
    bottom.set_title(
        "Paired difference in Sharpe (same resampled days), 95% CI",
        color=INK,
        loc="left",
        fontsize=11,
    )
    bottom.set_xlabel("Annualised Sharpe ratio", color=MUTED)

    for ax, count in ((top, len(singles)), (bottom, len(diffs))):
        ax.axvline(0.0, color=MUTED, linewidth=1, linestyle="--")
        ax.set_ylim(-0.6, count - 0.4)
        ax.grid(axis="x", color="#d0d7de", linewidth=0.6)
        ax.tick_params(colors=MUTED, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    left, right = bottom.get_xlim()
    bottom.set_xlim(left, right + 0.45 * (right - left))

    overlap = next(iter(table.values()))
    days = overlap.overlap if overlap is not None else 0
    fig.suptitle(
        f"KD crossover parameter sets, BTC/USDT 1h, {days} common days",
        color=INK,
        fontsize=13,
    )
    fig.text(
        0.01,
        -0.02,
        "Full-sample 2024 returns of untuned parameter sets; real comparisons "
        "must use out-of-sample returns. Black = CI excludes 0, grey = it does not.",
        fontsize=8,
        color=MUTED,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    outputs = {
        "plateau_png": OUTPUT_DIR / "plateau_gridindex.png",
        "plateau_md": OUTPUT_DIR / "plateau_gridindex_table.md",
        "kd_md": OUTPUT_DIR / "kd_inference.md",
        "kd_png": OUTPUT_DIR / "kd_equity.png",
        "comparison_png": OUTPUT_DIR / "comparison_forest.png",
    }
    summaries = plot_plateau(plateau_cases(), outputs["plateau_png"])
    write_plateau_table(summaries, outputs["plateau_md"])

    source = latest_cache()
    with source.open("rb") as handle:
        pipeline = pickle.load(handle)  # noqa: S301 - local trusted cache
    write_kd_inference(
        pipeline.backtest_result, pipeline.config_snapshot, source, outputs["kd_md"]
    )
    plot_kd_equity(pipeline.backtest_result, outputs["kd_png"])
    plot_comparison_forest(run_kd_variants(), outputs["comparison_png"])

    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
