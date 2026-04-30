"""Streamlit dashboard for backtest result visualization (viewer-only).

Launch::

    streamlit run src/tradingdev/dashboard/app.py \
        -- --config configs/xgboost_strategy.yaml

Requires a cached PipelineResult produced by::

    uv run python -m tradingdev.main --config configs/xgboost_strategy.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import plotly.graph_objects as go
import streamlit as st

if TYPE_CHECKING:
    from tradingdev.backtest.pipeline_result import (
        PipelineResult,
    )
    from tradingdev.backtest.result import BacktestResult

from tradingdev.dashboard.analysis import (
    available_months,
    build_equity_series,
    build_trades_df,
    consecutive_loss_counts,
    cumulative_pnl,
    cumulative_pnl_pct,
    filter_by_month,
    monthly_volume,
    rolling_mdd_absolute,
)
from tradingdev.data.data_manager import DataManager
from tradingdev.data.schemas import BacktestConfig, DataConfig
from tradingdev.utils.cache import load_cached_result
from tradingdev.utils.config import load_config

# ---------------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Backtest Dashboard",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
TIMEFRAME_MINUTES: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}

_PIPELINE_KEY = "_pipeline_result"


def _bars_in_30_days(timeframe: str) -> int:
    mins = TIMEFRAME_MINUTES.get(timeframe, 60)
    return 30 * 24 * 60 // mins


def _parse_args() -> Path:
    """Parse ``--config`` from sys.argv."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    known, _ = parser.parse_known_args()
    return Path(known.config)


# ---------------------------------------------------------------------------
# Data loading (cache only)
# ---------------------------------------------------------------------------
def _load_pipeline(config_path: Path) -> PipelineResult:
    """Load PipelineResult from disk cache. Stops app if not found."""
    raw_config = load_config(config_path)
    data_cfg = DataConfig(**raw_config.get("data", {}))
    bt_cfg = BacktestConfig(**raw_config["backtest"])
    manager = DataManager(data_config=data_cfg, backtest_config=bt_cfg)
    processed = manager.effective_processed_path
    cached = load_cached_result(config_path, processed)
    if cached is None:
        st.error(
            "No cached result found. Run the pipeline first:\n\n"
            f"```\nuv run python -m tradingdev.main --config {config_path}\n```"
        )
        st.stop()
        return None  # unreachable; st.stop() raises
    st.toast("Loaded from cache")
    return cached


# ---------------------------------------------------------------------------
# Shared chart rendering
# ---------------------------------------------------------------------------
def _render_charts(  # noqa: PLR0915
    result: BacktestResult,
    unit: str,
    month: str,
    timeframe: str,
    raw_config: dict[str, Any],
) -> None:
    """Render KPI row and 5 chart tabs for a single BacktestResult."""
    metrics = result.metrics
    is_volume_mode = result.mode == "volume"

    # --- KPI row ----------------------------------------------------------
    cols = st.columns(6)
    if is_volume_mode:
        cols[0].metric("Total P&L", f"{metrics['total_pnl']:+,.0f} USDT")
    else:
        cols[0].metric("Total Return", f"{metrics['total_return']:.2%}")
    cols[1].metric("Sharpe", f"{metrics['sharpe_ratio']:.3f}")
    if is_volume_mode:
        cols[2].metric("Max DD", f"{metrics['max_drawdown']:,.0f} USDT")
    else:
        cols[2].metric("Max DD", f"{metrics['max_drawdown']:.2%}")
    cols[3].metric("Win Rate", f"{metrics['win_rate']:.1%}")
    cols[4].metric("Trades", f"{metrics['total_trades']:,}")
    cols[5].metric(
        "Volume",
        f"{metrics['total_volume']:,.0f}",
    )

    # --- Prepare data -----------------------------------------------------
    equity = build_equity_series(result.equity_curve, result.timestamps)
    trades_df = build_trades_df(result.trades, result.timestamps)
    eq_view, tr_view = filter_by_month(equity, trades_df, month)

    # --- Monthly target ---------------------------------------------------
    monthly_target: float | None = None
    if is_volume_mode:
        strat_section = raw_config.get("strategy", {})
        params_section = strat_section.get("parameters", {})
        target_val = params_section.get("monthly_volume_target")
        if isinstance(target_val, (int, float)):
            monthly_target = float(target_val)

    # --- Tabs -------------------------------------------------------------
    tab_pnl, tab_hist, tab_streak, tab_monthly, tab_mdd = st.tabs(
        [
            "Cumulative PnL",
            "Trade PnL Distribution",
            "Consecutive Losses",
            "Monthly Volume",
            "Rolling 30d MDD",
        ]
    )

    # -- Tab 1: Cumulative PnL
    with tab_pnl:
        if unit == "Absolute" or is_volume_mode:
            pnl_series = cumulative_pnl(eq_view, result.init_cash)
            y_label = "Cumulative PnL"
        else:
            pnl_series = cumulative_pnl_pct(eq_view, result.init_cash)
            y_label = "Cumulative PnL (%)"

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=pnl_series.index,
                y=pnl_series.values,
                mode="lines",
                name="PnL",
                line={"color": "#1f77b4"},
            )
        )
        fig.add_hline(y=0, line_dash="dash", line_color="grey")
        fig.update_layout(
            yaxis_title=y_label,
            xaxis_title="Time",
            height=420,
            margin={"t": 30},
        )
        st.plotly_chart(fig, use_container_width=True)

    # -- Tab 2: Trade PnL histogram
    with tab_hist:
        if tr_view.empty:
            st.info("No trades in selected period.")
        else:
            pnl_col = tr_view["net_pnl"]
            if unit == "%" and tr_view["size_quote"].sum() > 0:
                pnl_col = tr_view["net_pnl"] / tr_view["size_quote"] * 100
                x_label = "Per-trade PnL (%)"
            else:
                x_label = "Per-trade PnL"

            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=pnl_col,
                    nbinsx=60,
                    marker_color="#636EFA",
                )
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            fig.update_layout(
                xaxis_title=x_label,
                yaxis_title="Frequency",
                height=420,
                margin={"t": 30},
            )
            st.plotly_chart(fig, use_container_width=True)

    # -- Tab 3: Consecutive loss streaks
    with tab_streak:
        counts = consecutive_loss_counts(tr_view)
        if counts.empty:
            st.info("No losing streaks in selected period.")
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=counts.index.astype(str),
                    y=counts.values,
                    marker_color="#EF553B",
                )
            )
            fig.update_layout(
                xaxis_title="Consecutive Loss Count",
                yaxis_title="Frequency",
                height=420,
                margin={"t": 30},
            )
            st.plotly_chart(fig, use_container_width=True)

    # -- Tab 4: Monthly volume achievement
    with tab_monthly:
        vol_df = monthly_volume(trades_df)
        if vol_df.empty:
            st.info("No trade data with timestamps.")
        else:
            colors = []
            if monthly_target is not None:
                for v in vol_df["volume_quote"]:
                    colors.append("#00CC96" if v >= monthly_target else "#EF553B")
            else:
                colors = ["#636EFA"] * len(vol_df)

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=vol_df["month"],
                    y=vol_df["volume_quote"],
                    marker_color=colors,
                    name="Volume",
                )
            )
            if monthly_target is not None:
                fig.add_hline(
                    y=monthly_target,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text=(f"Target: {monthly_target:,.0f}"),
                )
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Volume",
                height=420,
                margin={"t": 30},
            )
            st.plotly_chart(fig, use_container_width=True)

    # -- Tab 5: Rolling 30-day MDD (absolute)
    with tab_mdd:
        window = _bars_in_30_days(timeframe)
        mdd_series = rolling_mdd_absolute(eq_view, window)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=mdd_series.index,
                y=mdd_series.values,
                mode="lines",
                name="Rolling 30d MDD",
                line={"color": "#EF553B"},
                fill="tozeroy",
                fillcolor="rgba(239,85,59,0.15)",
            )
        )
        fig.update_layout(
            yaxis_title="Max Drawdown",
            xaxis_title="Time",
            height=420,
            margin={"t": 30},
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Mode-specific renderers
# ---------------------------------------------------------------------------
def _render_simple(
    pipeline: PipelineResult,
    config_path: Path,
) -> None:
    """Render dashboard for a simple (non-walk-forward) backtest."""
    raw_config = pipeline.config_snapshot
    result = pipeline.backtest_result
    if result is None:
        st.error("No backtest result in pipeline.")
        st.stop()
        return

    bt_cfg = BacktestConfig(**raw_config["backtest"])
    strategy_name = raw_config["strategy"]["name"]
    is_volume_mode = result.mode == "volume"

    # --- Header
    st.title(f"Backtest Dashboard — {strategy_name}")
    st.caption(f"Mode: **simple** | Config: `{config_path}`")

    # --- Sidebar controls
    with st.sidebar:
        st.header("Controls")
        default_unit = "Absolute" if is_volume_mode else "%"
        unit = st.radio(
            "Display unit",
            ["Absolute", "%"],
            index=0 if default_unit == "Absolute" else 1,
            horizontal=True,
        )
        months = ["全期間"] + available_months(result.timestamps)
        selected_month = st.selectbox("Period", months)

    _render_charts(
        result,
        unit,
        selected_month,
        bt_cfg.timeframe,
        raw_config,
    )


def _render_walk_forward(
    pipeline: PipelineResult,
    config_path: Path,
) -> None:
    """Render dashboard for walk-forward validation results."""
    raw_config = pipeline.config_snapshot
    folds = pipeline.fold_results
    bt_cfg = BacktestConfig(**raw_config["backtest"])
    strategy_name = raw_config["strategy"]["name"]

    # --- Header
    st.title(f"Backtest Dashboard — {strategy_name}")
    st.caption(f"Mode: **walk-forward** | Config: `{config_path}`")

    # --- Sidebar controls
    with st.sidebar:
        st.header("Controls")

        fold_labels = [f"Fold {f.fold_index}" for f in folds]
        selected_fold_label = st.selectbox("Fold", fold_labels)
        fold_idx = fold_labels.index(selected_fold_label)
        fold = folds[fold_idx]

        split = st.radio(
            "Split",
            ["Test (OOS)", "Train (IS)"],
            index=0,
            horizontal=True,
        )
        is_test = split == "Test (OOS)"

        result: BacktestResult | None = (
            fold.test_backtest if is_test else fold.train_backtest
        )
        if result is None:
            st.error("BacktestResult not available for this split.")
            st.stop()
            return

        is_volume_mode = result.mode == "volume"
        default_unit = "Absolute" if is_volume_mode else "%"
        unit = st.radio(
            "Display unit",
            ["Absolute", "%"],
            index=0 if default_unit == "Absolute" else 1,
            horizontal=True,
        )
        months = ["全期間"] + available_months(result.timestamps)
        selected_month = st.selectbox("Period", months)

    # --- Fold info caption
    split_label = "Test (OOS)" if is_test else "Train (IS)"
    st.caption(
        f"**{selected_fold_label}** | {split_label} | "
        f"Train: {fold.train_start} ~ {fold.train_end} | "
        f"Test: {fold.test_start} ~ {fold.test_end}"
    )

    _render_charts(
        result,
        unit,
        selected_month,
        bt_cfg.timeframe,
        raw_config,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    config_path = _parse_args()

    if _PIPELINE_KEY not in st.session_state:
        st.session_state[_PIPELINE_KEY] = _load_pipeline(config_path)

    pipeline: PipelineResult = st.session_state[_PIPELINE_KEY]

    if pipeline.mode == "walk_forward":
        _render_walk_forward(pipeline, config_path)
    else:
        _render_simple(pipeline, config_path)


if __name__ == "__main__":
    main()
