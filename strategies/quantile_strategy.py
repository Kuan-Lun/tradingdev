"""XGBoost path-opportunity volume strategy.

Predicts whether a profitable price move (exceeding fee cost) will
occur within a configurable horizon window, then enters in the
predicted direction with a take-profit exit.

Uses chunked rolling retrain to adapt to changing market conditions.
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed

from quant_backtest.ml.quantile_features import QuantileFeatureEngineer
from quant_backtest.strategies.base import BaseStrategy
from quant_backtest.utils.logger import setup_logger
from quant_backtest.utils.parallel import estimate_n_jobs

if TYPE_CHECKING:
    from quant_backtest.backtest.base_engine import BaseBacktestEngine
    from quant_backtest.data.schemas import (
        ParallelConfig,
        QuantileStrategyConfig,
    )

logger = setup_logger(__name__)

# Grid search tuple: (horizon, min_confidence, min_ratio, edge_for_full_size)
_ParamTuple = tuple[int, float, float, float]


def _train_path_classifiers(
    feat_df: pd.DataFrame,
    feature_names: list[str],
    n_estimators: int = 50,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    subsample: float = 0.7,
) -> tuple[xgb.XGBClassifier, xgb.XGBClassifier]:
    """Train long and short opportunity classifiers.

    Returns:
        Tuple of (long_model, short_model).
    """
    x = feat_df[feature_names].values
    y_long = feat_df["target_long"].values.astype(int)
    y_short = feat_df["target_short"].values.astype(int)

    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "eval_metric": "logloss",
        "verbosity": 0,
        "random_state": 42,
    }

    long_model = xgb.XGBClassifier(**params)
    long_model.fit(x, y_long, verbose=False)

    short_model = xgb.XGBClassifier(**params)
    short_model.fit(x, y_short, verbose=False)

    return long_model, short_model


def _predict_opportunities(
    feat_df: pd.DataFrame,
    feature_names: list[str],
    long_model: xgb.XGBClassifier,
    short_model: xgb.XGBClassifier,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict P(long opportunity) and P(short opportunity).

    Returns:
        Tuple of (p_long, p_short) arrays.
    """
    x = feat_df[feature_names].values
    p_long = long_model.predict_proba(x)[:, 1]
    p_short = short_model.predict_proba(x)[:, 1]
    return p_long, p_short


def _run_path_state_machine(
    close: np.ndarray,
    p_long: np.ndarray,
    p_short: np.ndarray,
    horizon: int,
    min_holding_bars: int,
    min_confidence: float,
    min_ratio: float,
    take_profit: float,
    strategy_sl: float,
    dynamic_sizing: bool,
    min_weight: float,
    edge_for_full_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """State machine for path-opportunity signals.

    Entry uses risk/reward logic:
    - Compute expected edge = P(win) * TP - P(lose) * SL
    - Only enter when edge > 0 AND P(win)/P(lose) > min_ratio
    - Size position proportional to edge

    Exit: strategy stop-loss, or horizon reached.
    Take-profit is handled by the engine.

    Returns:
        Tuple of (signals, size_weights).
    """
    n = len(p_long)
    signals = np.zeros(n, dtype=np.float64)
    size_weights = np.ones(n, dtype=np.float64)

    state = 0
    bars_in_pos = 0
    entry_price = 0.0
    current_weight = 1.0

    for i in range(n):
        pl = p_long[i]
        ps = p_short[i]

        if state == 0:
            # --- FLAT: evaluate entry via risk/reward ---
            # Long: reward if TP hit, risk if SL hit
            long_edge = pl * take_profit - ps * strategy_sl
            short_edge = ps * take_profit - pl * strategy_sl
            # Ratio: how much more likely is our direction vs opposite
            long_ratio = pl / max(ps, 0.01)
            short_ratio = ps / max(pl, 0.01)

            want_long = (
                long_edge > 0
                and long_ratio > min_ratio
                and pl > min_confidence
            )
            want_short = (
                short_edge > 0
                and short_ratio > min_ratio
                and ps > min_confidence
            )

            if want_long and (not want_short or long_edge >= short_edge):
                state = 1
                bars_in_pos = 0
                entry_price = close[i]
                if dynamic_sizing and edge_for_full_size > 0:
                    raw_w = long_edge / edge_for_full_size
                    current_weight = min(max(raw_w, min_weight), 1.0)
                else:
                    current_weight = 1.0
                size_weights[i] = current_weight
            elif want_short:
                state = -1
                bars_in_pos = 0
                entry_price = close[i]
                if dynamic_sizing and edge_for_full_size > 0:
                    raw_w = short_edge / edge_for_full_size
                    current_weight = min(max(raw_w, min_weight), 1.0)
                else:
                    current_weight = 1.0
                size_weights[i] = current_weight
        else:
            # --- IN POSITION ---
            bars_in_pos += 1
            size_weights[i] = current_weight

            if bars_in_pos < min_holding_bars:
                signals[i] = float(state)
                continue

            # Strategy-level stop-loss (tighter than engine SL)
            if entry_price > 0 and strategy_sl > 0:
                pnl_pct = state * (close[i] - entry_price) / entry_price
                if pnl_pct < -strategy_sl:
                    state = 0
                    signals[i] = 0.0
                    continue

            # Horizon exit (TP is handled by the engine)
            if bars_in_pos >= horizon:
                state = 0
                signals[i] = 0.0
                continue

        signals[i] = float(state)

    return signals, size_weights


def _evaluate_path_combo(
    df: pd.DataFrame,
    feature_names: list[str],
    long_model: xgb.XGBClassifier,
    short_model: xgb.XGBClassifier,
    config: QuantileStrategyConfig,
    params: _ParamTuple,
    engine: BaseBacktestEngine,
    target: str,
) -> tuple[_ParamTuple, dict[str, Any]]:
    """Evaluate a parameter combination for grid search."""
    horizon, min_confidence, min_ratio, edge_for_full_size = params
    tp = config.fee_rate * 2  # take profit = round-trip fee

    p_long, p_short = _predict_opportunities(
        df, feature_names, long_model, short_model,
    )

    dyn = config.dynamic_sizing
    min_weight = (
        config.min_position_size / config.position_size if dyn else 0.0
    )

    close_arr = df["close"].astype(float).values

    signals, size_weights = _run_path_state_machine(
        close=close_arr,
        p_long=p_long,
        p_short=p_short,
        horizon=horizon,
        min_holding_bars=config.min_holding_bars,
        min_confidence=min_confidence,
        min_ratio=min_ratio,
        take_profit=tp,
        strategy_sl=config.strategy_sl,
        dynamic_sizing=dyn,
        min_weight=min_weight,
        edge_for_full_size=edge_for_full_size,
    )

    sig_df = df.copy()
    sig_df["signal"] = signals
    sig_df["size_weight"] = size_weights

    result = engine.run(sig_df)
    return params, result.metrics


class QuantileStrategy(BaseStrategy):
    """Path-opportunity strategy for volume trading.

    Predicts whether price will move enough to cover fees within
    a horizon window, enters in the predicted direction, and
    relies on the engine's take-profit to capture the move.

    Supports chunked rolling retrain.
    """

    def __init__(
        self,
        config: QuantileStrategyConfig,
        backtest_engine: BaseBacktestEngine | None = None,
        parallel_config: ParallelConfig | None = None,
    ) -> None:
        self._config = config
        self._backtest_engine = backtest_engine
        self._parallel_config = parallel_config

        self._feature_eng: QuantileFeatureEngineer | None = None
        self._long_model: xgb.XGBClassifier | None = None
        self._short_model: xgb.XGBClassifier | None = None
        self._feature_names: list[str] = []

        # Best params from fit()
        self._best_horizon: int = config.horizon
        self._best_min_confidence: float = config.min_entry_edge
        self._best_min_ratio: float = config.min_ratio
        self._best_edge_for_full_size: float = config.edge_for_full_size

        # Train data for rolling retrain
        self._train_df: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame) -> None:
        """Train classifiers and grid-search strategy parameters."""
        if self._backtest_engine is None:
            logger.warning("No backtest engine; skipping fit()")
            return

        cfg = self._config
        self._train_df = df.copy()

        horizons = cfg.horizon_candidates or [cfg.horizon]
        confidence_candidates = (
            cfg.min_entry_edge_candidates or [cfg.min_entry_edge]
        )
        ratio_candidates = (
            cfg.min_ratio_candidates or [cfg.min_ratio]
        )
        efs_candidates: list[float] = (
            cfg.edge_for_full_size_candidates
            if cfg.dynamic_sizing
            else [cfg.edge_for_full_size]
        )

        best_value = -math.inf
        best_horizon = cfg.horizon
        best_confidence = cfg.min_entry_edge
        best_ratio = cfg.min_ratio
        best_efs = cfg.edge_for_full_size
        target = cfg.target_metric
        min_mp = cfg.min_monthly_pnl

        best_long: xgb.XGBClassifier | None = None
        best_short: xgb.XGBClassifier | None = None
        best_fe: QuantileFeatureEngineer | None = None
        best_fnames: list[str] = []

        for h in horizons:
            logger.info("Training path classifiers for horizon=%d", h)
            fe = QuantileFeatureEngineer(
                horizon=h, profit_target=cfg.fee_rate * 2,
            )
            feat_df = fe.transform(df, include_target=True, target_type="path")
            fnames = [
                f for f in fe.get_feature_names()
                if f not in ("target_long", "target_short", "target")
            ]

            long_m, short_m = _train_path_classifiers(
                feat_df, fnames,
                n_estimators=cfg.n_estimators,
                max_depth=cfg.max_depth,
                learning_rate=cfg.learning_rate,
                subsample=cfg.subsample,
            )

            pos_long = feat_df["target_long"].mean()
            pos_short = feat_df["target_short"].mean()
            logger.info(
                "horizon=%d: long_rate=%.1f%%, short_rate=%.1f%%",
                h, pos_long * 100, pos_short * 100,
            )

            grid: list[_ParamTuple] = list(itertools.product(
                [h], confidence_candidates, ratio_candidates,
                efs_candidates,
            ))

            p_cfg = self._parallel_config
            n_jobs = estimate_n_jobs(
                df,
                safety_factor=p_cfg.safety_factor if p_cfg else 0.6,
                overhead_multiplier=(
                    p_cfg.overhead_multiplier if p_cfg else 3.0
                ),
                reserve_cores=p_cfg.reserve_cores if p_cfg else 2,
            )
            logger.info(
                "Grid search (horizon=%d): %d combos (n_jobs=%d)",
                h, len(grid), n_jobs,
            )

            results: list[tuple[_ParamTuple, dict[str, Any]]] = Parallel(
                n_jobs=n_jobs,
            )(
                delayed(_evaluate_path_combo)(
                    feat_df, fnames, long_m, short_m,
                    cfg, combo, self._backtest_engine, target,
                )
                for combo in grid
            )

            n_filtered = 0
            for params, metrics in results:
                if min_mp is not None:
                    dpm = metrics.get("daily_pnl_mean", -math.inf)
                    monthly_pnl = (
                        dpm * 30
                        if isinstance(dpm, (int, float))
                        else -math.inf
                    )
                    if monthly_pnl < min_mp:
                        n_filtered += 1
                        continue

                value = metrics.get(target, -math.inf)
                if isinstance(value, (int, float)) and value > best_value:
                    best_value = float(value)
                    best_horizon = params[0]
                    best_confidence = params[1]
                    best_ratio = params[2]
                    best_efs = params[3]
                    best_long = long_m
                    best_short = short_m
                    best_fe = fe
                    best_fnames = fnames

            if n_filtered > 0:
                logger.info(
                    "horizon=%d: %d/%d filtered",
                    h, n_filtered, len(grid),
                )

        self._best_horizon = best_horizon
        self._best_min_confidence = best_confidence
        self._best_min_ratio = best_ratio
        self._best_edge_for_full_size = best_efs
        self._long_model = best_long
        self._short_model = best_short
        self._feature_eng = best_fe
        self._feature_names = best_fnames

        logger.info(
            "Fit complete: horizon=%d, confidence=%.4f, "
            "efs=%.4f (%s=%.4f)",
            best_horizon, best_confidence, best_efs,
            target, best_value,
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals with chunked rolling retrain."""
        cfg = self._config

        if self._long_model is None or self._feature_eng is None:
            return self._generate_signals_fixed(df)

        if self._train_df is None:
            return self._generate_signals_with_model(df)

        # --- Chunked rolling retrain ---
        horizon = self._best_horizon
        retrain_interval = cfg.retrain_interval
        train_window = cfg.train_window

        combined = pd.concat(
            [self._train_df, df], ignore_index=True,
        )
        test_offset = len(self._train_df)
        test_len = len(df)

        chunk_starts = list(range(0, test_len, retrain_interval))
        all_signals = np.zeros(test_len, dtype=np.float64)
        all_weights = np.ones(test_len, dtype=np.float64)

        dyn = cfg.dynamic_sizing
        min_weight = (
            cfg.min_position_size / cfg.position_size if dyn else 0.0
        )

        # State carried across chunks
        sm_state = 0
        sm_bars_in_pos = 0
        sm_entry_price = 0.0
        sm_current_weight = 1.0

        for ci, chunk_start in enumerate(chunk_starts):
            chunk_end = min(chunk_start + retrain_interval, test_len)
            abs_start = test_offset + chunk_start
            abs_end = test_offset + chunk_end

            # Training window
            train_begin = max(0, abs_start - train_window)
            train_slice = combined.iloc[train_begin:abs_start].copy()

            if len(train_slice) < 200:
                logger.warning(
                    "Chunk %d: train too small (%d), using fallback",
                    ci, len(train_slice),
                )
                long_m = self._long_model
                short_m = self._short_model
                fnames = self._feature_names
                fe = self._feature_eng
            else:
                logger.info(
                    "Chunk %d/%d: retrain on %d bars",
                    ci + 1, len(chunk_starts), len(train_slice),
                )
                fe = QuantileFeatureEngineer(
                    horizon=horizon,
                    profit_target=cfg.fee_rate * 2,
                )
                train_feat = fe.transform(
                    train_slice, include_target=True, target_type="path",
                )
                fnames = self._feature_names or [
                    f for f in fe.get_feature_names()
                    if f not in ("target_long", "target_short", "target")
                ]
                long_m, short_m = _train_path_classifiers(
                    train_feat, fnames,
                    n_estimators=cfg.n_estimators,
                    max_depth=cfg.max_depth,
                    learning_rate=cfg.learning_rate,
                    subsample=cfg.subsample,
                )

            # Predict on chunk with context for features
            context_start = max(0, abs_start - 60)
            predict_ctx = combined.iloc[context_start:abs_end].copy()
            predict_feat = fe.transform(
                predict_ctx, include_target=False,
            )

            # Align to chunk bars only
            if "timestamp" in predict_feat.columns:
                chunk_ts = combined.iloc[abs_start]["timestamp"]
                mask = predict_feat["timestamp"] >= chunk_ts
                predict_feat = predict_feat[mask].reset_index(drop=True)

            p_long, p_short = _predict_opportunities(
                predict_feat, fnames, long_m, short_m,
            )
            chunk_close = predict_feat["close"].astype(float).values

            # Run state machine for this chunk
            chunk_len = len(p_long)
            for j in range(chunk_len):
                out_idx = chunk_start + j
                if out_idx >= test_len:
                    break

                pl = p_long[j]
                ps = p_short[j]
                c = chunk_close[j]

                if sm_state == 0:
                    tp = cfg.fee_rate * 2
                    sl = cfg.strategy_sl
                    mr = self._best_min_ratio
                    mc = self._best_min_confidence
                    long_edge = pl * tp - ps * sl
                    short_edge = ps * tp - pl * sl
                    long_ratio = pl / max(ps, 0.01)
                    short_ratio = ps / max(pl, 0.01)
                    want_long = (
                        long_edge > 0 and long_ratio > mr and pl > mc
                    )
                    want_short = (
                        short_edge > 0 and short_ratio > mr and ps > mc
                    )

                    if want_long and (
                        not want_short or long_edge >= short_edge
                    ):
                        sm_state = 1
                        sm_bars_in_pos = 0
                        sm_entry_price = c
                        if dyn and self._best_edge_for_full_size > 0:
                            raw_w = (
                                long_edge / self._best_edge_for_full_size
                            )
                            sm_current_weight = min(
                                max(raw_w, min_weight), 1.0,
                            )
                        else:
                            sm_current_weight = 1.0
                        all_weights[out_idx] = sm_current_weight
                    elif want_short:
                        sm_state = -1
                        sm_bars_in_pos = 0
                        sm_entry_price = c
                        if dyn and self._best_edge_for_full_size > 0:
                            raw_w = (
                                short_edge / self._best_edge_for_full_size
                            )
                            sm_current_weight = min(
                                max(raw_w, min_weight), 1.0,
                            )
                        else:
                            sm_current_weight = 1.0
                        all_weights[out_idx] = sm_current_weight
                else:
                    sm_bars_in_pos += 1
                    all_weights[out_idx] = sm_current_weight

                    if sm_bars_in_pos < cfg.min_holding_bars:
                        all_signals[out_idx] = float(sm_state)
                        continue

                    # Strategy-level stop-loss
                    if sm_entry_price > 0 and cfg.strategy_sl > 0:
                        pnl_pct = (
                            sm_state * (c - sm_entry_price) / sm_entry_price
                        )
                        if pnl_pct < -cfg.strategy_sl:
                            sm_state = 0
                            all_signals[out_idx] = 0.0
                            continue

                    if sm_bars_in_pos >= horizon:
                        sm_state = 0
                        all_signals[out_idx] = 0.0
                        continue

                all_signals[out_idx] = float(sm_state)

        result = df.copy()
        if len(all_signals) < len(df):
            pad = len(df) - len(all_signals)
            all_signals = np.concatenate([np.zeros(pad), all_signals])
            all_weights = np.concatenate([np.ones(pad), all_weights])

        result["signal"] = all_signals[: len(df)]
        result["size_weight"] = all_weights[: len(df)]
        return result

    def _generate_signals_fixed(
        self, df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Fallback: train once on df and generate signals."""
        cfg = self._config
        horizon = self._best_horizon

        self._feature_eng = QuantileFeatureEngineer(
            horizon=horizon, profit_target=cfg.fee_rate * 2,
        )
        feat_df = self._feature_eng.transform(
            df, include_target=True, target_type="path",
        )
        self._feature_names = [
            f for f in self._feature_eng.get_feature_names()
            if f not in ("target_long", "target_short", "target")
        ]
        self._long_model, self._short_model = _train_path_classifiers(
            feat_df, self._feature_names,
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
        )
        return self._generate_signals_with_model(df)

    def _generate_signals_with_model(
        self, df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate signals using the current fixed model."""
        cfg = self._config
        assert self._feature_eng is not None  # noqa: S101
        assert self._long_model is not None  # noqa: S101
        assert self._short_model is not None  # noqa: S101

        feat_df = self._feature_eng.transform(df, include_target=False)
        p_long, p_short = _predict_opportunities(
            feat_df, self._feature_names,
            self._long_model, self._short_model,
        )

        dyn = cfg.dynamic_sizing
        min_weight = (
            cfg.min_position_size / cfg.position_size if dyn else 0.0
        )

        close_arr = feat_df["close"].astype(float).values

        tp = cfg.fee_rate * 2

        signals, size_weights = _run_path_state_machine(
            close=close_arr,
            p_long=p_long,
            p_short=p_short,
            horizon=self._best_horizon,
            min_holding_bars=cfg.min_holding_bars,
            min_confidence=self._best_min_confidence,
            min_ratio=self._best_min_ratio,
            take_profit=tp,
            strategy_sl=cfg.strategy_sl,
            dynamic_sizing=dyn,
            min_weight=min_weight,
            edge_for_full_size=self._best_edge_for_full_size,
        )

        result = feat_df.copy()
        result["signal"] = signals
        result["size_weight"] = size_weights
        return result

    def get_parameters(self) -> dict[str, Any]:
        """Return strategy parameters."""
        params = self._config.model_dump()
        params["best_horizon"] = self._best_horizon
        params["best_min_confidence"] = self._best_min_confidence
        params["best_min_ratio"] = self._best_min_ratio
        params["best_edge_for_full_size"] = self._best_edge_for_full_size
        return params
