"""Parameter plateau demo on synthetic data: true plateau vs. false spike.

Outputs (under demo_output/):
  plateau_demo.png  side-by-side 5x5 metric heatmaps on a shared colour scale
  plateau_table.md  plateau statistics of both cases

Run with: uv run python scripts/demo_plateau.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from tradingdev.domain.statistics.plateau import (
    PlateauSummary,
    Trial,
    plateau_grid,
    plateau_summary,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "demo_output"
SEED = 0
LOOKBACKS = (10, 20, 30, 40, 50)
THRESHOLDS = (0.01, 0.02, 0.03, 0.04, 0.05)
X, Y = "lookback", "threshold"
AXES = {X: [float(lookback) for lookback in LOOKBACKS], Y: list(THRESHOLDS)}
INK = "#1f2933"
MUTED = "#52606d"


def _trials(metric: Callable[[int, int], float]) -> list[Trial]:
    return [
        Trial(params={X: float(lookback), Y: threshold}, metric=metric(i, j))
        for i, lookback in enumerate(LOOKBACKS)
        for j, threshold in enumerate(THRESHOLDS)
    ]


def build_cases() -> dict[str, list[Trial]]:
    """Case A: smooth plateau peaking at (2, 2); case B: lone spike at (1, 3)."""
    rng = np.random.default_rng(SEED)
    shape = (len(LOOKBACKS), len(THRESHOLDS))
    plateau_noise = rng.normal(0.0, 0.05, shape)
    spike_noise = rng.normal(0.0, 0.3, shape)
    return {
        "A: true plateau": _trials(
            lambda i, j: float(
                1.5 - 0.3 * abs(i - 2) - 0.3 * abs(j - 2) + plateau_noise[i, j]
            )
        ),
        "B: false spike": _trials(
            lambda i, j: 2.0 if (i, j) == (1, 3) else float(spike_noise[i, j])
        ),
    }


def _cell(params: Mapping[str, float] | None) -> tuple[int, int]:
    """Map parameter values to (x_index, y_index) heatmap coordinates."""
    assert params is not None
    return LOOKBACKS.index(int(params[X])), THRESHOLDS.index(params[Y])


def plot(cases: dict[str, list[Trial]], path: Path) -> dict[str, PlateauSummary]:
    summaries = {
        name: plateau_summary(trials, axes=AXES) for name, trials in cases.items()
    }
    matrices = {
        name: np.array(
            [
                [np.nan if value is None else value for value in row]
                for row in plateau_grid(trials, axes=AXES, x=X, y=Y).metric
            ]
        )
        for name, trials in cases.items()
    }
    vmin = min(float(np.nanmin(m)) for m in matrices.values())
    vmax = max(float(np.nanmax(m)) for m in matrices.values())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=True)
    image = None
    for ax, (name, matrix) in zip(axes, matrices.items(), strict=True):
        summary = summaries[name]
        image = ax.imshow(matrix, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        for (row, col), value in np.ndenumerate(matrix):
            shade = (value - vmin) / (vmax - vmin)
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color=INK if shade > 0.6 else "white",
            )
        bx, by = _cell(summary.best_params)
        px, py = _cell(summary.plateau_best_params)
        ax.scatter(
            [px],
            [py],
            s=1700,
            marker="s",
            facecolors="none",
            edgecolors="black",
            linewidths=3,
            path_effects=[pe.withStroke(linewidth=5, foreground="white")],
        )
        ax.scatter(
            [bx],
            [by],
            s=1000,
            facecolors="none",
            edgecolors="white",
            linewidths=3,
            path_effects=[pe.withStroke(linewidth=5, foreground=INK)],
        )
        ax.set_title(
            f"{name}\nmetric_gap = {summary.metric_gap:.3f}    "
            f"grid_distance = {summary.grid_distance}",
            fontsize=12,
            color=INK,
        )
        ax.set_xticks(range(len(LOOKBACKS)), [str(v) for v in LOOKBACKS])
        ax.set_yticks(range(len(THRESHOLDS)), [f"{v:.2f}" for v in THRESHOLDS])
        ax.set_xlabel("lookback", color=MUTED, fontsize=11)
        ax.set_ylabel("threshold", color=MUTED, fontsize=11)
        ax.tick_params(length=0, colors=MUTED, labelsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)

    assert image is not None
    fig.colorbar(image, ax=axes, shrink=0.85, label="Metric (shared scale)")
    legend = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=14,
            markerfacecolor="none",
            markeredgecolor="white",
            markeredgewidth=2.5,
            path_effects=[pe.withStroke(linewidth=4.5, foreground=INK)],
            label="Best trial (highest metric)",
        ),
        Line2D(
            [],
            [],
            marker="s",
            linestyle="none",
            markersize=14,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=2.5,
            label="Plateau best (highest neighbourhood mean)",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.45, -0.09),
    )
    fig.suptitle(
        "Parameter plateau vs. lucky spike (5x5 grid, radius 1, seed 0)",
        fontsize=14,
        color=INK,
        y=1.06,
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return summaries


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}"


ROWS: tuple[tuple[str, Callable[[PlateauSummary], float | int | None]], ...] = (
    ("best_metric", lambda s: s.best_metric),
    ("plateau_best_score", lambda s: s.plateau_best_score),
    ("metric_gap", lambda s: s.metric_gap),
    ("grid_distance", lambda s: s.grid_distance),
    ("best_neighborhood_mean", lambda s: s.best_neighborhood_mean),
    ("best_neighborhood_std", lambda s: s.best_neighborhood_std),
    ("best_neighborhood_size", lambda s: s.best_neighborhood_size),
    ("spikiness", lambda s: s.spikiness),
)


def _params(params: Mapping[str, float] | None) -> str:
    assert params is not None
    return f"({int(params[X])}, {params[Y]:.2f})"


def write_table(summaries: dict[str, PlateauSummary], path: Path) -> str:
    names = list(summaries)
    lines = [
        "# Plateau statistics: true plateau vs. false spike",
        "",
        "Synthetic 5x5 grid (lookback x threshold), radius-1 neighbourhood, seed 0.",
        "",
        "| Statistic | " + " | ".join(names) + " |",
        "| --- | " + " | ".join("---:" for _ in names) + " |",
        "| best_params (lookback, threshold) | "
        + " | ".join(_params(summaries[n].best_params) for n in names)
        + " |",
        "| plateau_best_params (lookback, threshold) | "
        + " | ".join(_params(summaries[n].plateau_best_params) for n in names)
        + " |",
    ]
    lines += [
        f"| {label} | " + " | ".join(_fmt(get(summaries[n])) for n in names) + " |"
        for label, get in ROWS
    ]
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    png = OUTPUT_DIR / "plateau_demo.png"
    table = OUTPUT_DIR / "plateau_table.md"
    summaries = plot(build_cases(), png)
    print(write_table(summaries, table))
    print(f"wrote {png}")
    print(f"wrote {table}")


if __name__ == "__main__":
    main()
