"""GLFT market-making strategy adapted for signal-based backtesting.

Based on Gueant, Lehalle & Fernandez-Tapia (2013),
"Dealing with the Inventory Risk: A Solution to the Market Making
Problem".  The model computes an optimal spread and reservation price
that adjust dynamically based on inventory risk and volatility.
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from btc_strategy.strategies.base import BaseStrategy
from btc_strategy.utils.logger import setup_logger
from btc_strategy.utils.parallel import estimate_n_jobs

if TYPE_CHECKING:
    import numpy.typing as npt

    from btc_strategy.backtest.base_engine import (
        BaseBacktestEngine,
    )
    from btc_strategy.data.schemas import GLFTStrategyConfig

logger = setup_logger(__name__)

# Type alias for the full parameter tuple searched by fit()
_ParamTuple = tuple[float, float, int, int, int, float, int]


def _evaluate_glft_combo(
    df: pd.DataFrame,
    config: GLFTStrategyConfig,
    params: _ParamTuple,
    engine: BaseBacktestEngine,
    target: str,
) -> tuple[_ParamTuple, dict[str, Any]]:
    """Evaluate a single GLFT parameter combination.

    This is a module-level function so it can be pickled by joblib.

    Returns:
        Tuple of (params, metrics_dict) for the main process to
        apply constraint filtering.
    """
    gamma, kappa, ema_w, max_hold, vol_win, entry_edge, trend_ema = params
    trial = GLFTStrategy(config=config)
    trial._best_gamma = gamma
    trial._best_kappa = kappa
    trial._best_ema_window = ema_w
    trial._best_max_holding_bars = max_hold
    trial._best_vol_window = vol_win
    trial._best_min_entry_edge = entry_edge
    trial._best_trend_ema_window = trend_ema

    signals = trial.generate_signals(df)
    result = engine.run(signals)
    return params, result.metrics


class GLFTStrategy(BaseStrategy):
    """GLFT market-making strategy for signal-based backtesting.

    Uses the Gueant-Lehalle-Fernandez-Tapia optimal market-making
    model to determine entry/exit thresholds based on:

    - **Reservation price**: mid-price adjusted for inventory risk
    - **Optimal spread**: adapts to volatility and time horizon

    The strategy is purely analytical -- no ML model required.
    """

    def __init__(
        self,
        config: GLFTStrategyConfig,
        backtest_engine: BaseBacktestEngine | None = None,
    ) -> None:
        self._config = config
        self._backtest_engine = backtest_engine

        # Populated by fit() -- or use config defaults
        self._best_gamma: float = config.gamma
        self._best_kappa: float = config.kappa
        self._best_ema_window: int = config.ema_window
        self._best_max_holding_bars: int = config.max_holding_bars
        self._best_vol_window: int = config.vol_window
        self._best_min_entry_edge: float = config.min_entry_edge
        self._best_trend_ema_window: int = config.trend_ema_window

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> None:
        """Grid-search over GLFT parameters.

        When ``min_annual_return`` is set, only parameter combinations
        whose ``annual_return`` meets the threshold are considered.
        Among those, the combination with the highest ``target_metric``
        (e.g. ``total_volume_usdt``) is selected.
        """
        if self._backtest_engine is None:
            logger.warning(
                "No backtest engine; skipping fit()",
            )
            return

        best_value = -math.inf
        best_gamma = self._config.gamma
        best_kappa = self._config.kappa
        best_ema = self._config.ema_window
        best_max_hold = self._config.max_holding_bars
        best_vol_win = self._config.vol_window
        best_entry_edge = self._config.min_entry_edge
        best_trend_ema = self._config.trend_ema_window
        target = self._config.target_metric
        min_ar = self._config.min_annual_return

        # When using implied vol, vol_window is irrelevant
        vol_win_candidates = self._config.vol_window_candidates
        if self._config.vol_type == "implied":
            vol_win_candidates = [0]
            logger.info("vol_type='implied': vol_window_candidates ignored")

        grid: list[_ParamTuple] = list(
            itertools.product(
                self._config.gamma_candidates,
                self._config.kappa_candidates,
                self._config.ema_window_candidates,
                self._config.max_holding_bars_candidates,
                vol_win_candidates,
                self._config.min_entry_edge_candidates,
                self._config.trend_ema_candidates,
            ),
        )

        n_jobs = estimate_n_jobs(df)
        logger.info(
            "GLFT grid search: %d combinations (n_jobs=%d)",
            len(grid),
            n_jobs,
        )

        results: list[tuple[_ParamTuple, dict[str, Any]]] = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_glft_combo)(
                df,
                self._config,
                combo,
                self._backtest_engine,
                target,
            )
            for combo in grid
        )

        n_filtered = 0
        for params, metrics in results:
            if min_ar is not None:
                ar = metrics.get("annual_return", -math.inf)
                if not isinstance(ar, (int, float)) or ar < min_ar:
                    n_filtered += 1
                    continue

            value = metrics.get(target, -math.inf)

            if isinstance(value, (int, float)) and value > best_value:
                best_value = float(value)
                best_gamma = params[0]
                best_kappa = params[1]
                best_ema = params[2]
                best_max_hold = params[3]
                best_vol_win = params[4]
                best_entry_edge = params[5]
                best_trend_ema = params[6]

        if n_filtered > 0:
            logger.info(
                "GLFT fit: %d/%d combos filtered (annual_return < %.4f)",
                n_filtered,
                len(grid),
                min_ar if min_ar is not None else 0.0,
            )

        self._best_gamma = best_gamma
        self._best_kappa = best_kappa
        self._best_ema_window = best_ema
        self._best_max_holding_bars = best_max_hold
        self._best_vol_window = best_vol_win
        self._best_min_entry_edge = best_entry_edge
        self._best_trend_ema_window = best_trend_ema

        logger.info(
            "GLFT fit complete: gamma=%.4f, kappa=%.4f, ema=%d, "
            "max_hold=%d, vol_win=%d, entry_edge=%.4f, "
            "trend_ema=%d (%s=%.4f)",
            best_gamma,
            best_kappa,
            best_ema,
            best_max_hold,
            best_vol_win,
            best_entry_edge,
            best_trend_ema,
            target,
            best_value,
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate position-state signals via the GLFT model."""
        close = np.asarray(df["close"].astype(float).values)
        high = np.asarray(df["high"].astype(float).values)
        low = np.asarray(df["low"].astype(float).values)

        # Extract DVOL series when using implied volatility
        dvol: npt.NDArray[np.floating[Any]] | None = None
        if self._config.vol_type == "implied":
            if "dvol" not in df.columns:
                msg = "DataFrame must contain 'dvol' column when vol_type is 'implied'"
                raise ValueError(msg)
            dvol = np.asarray(df["dvol"].astype(float).values)

        ema = np.asarray(
            pd.Series(close)
            .ewm(span=self._best_ema_window, adjust=False)
            .mean()
            .values,
        )
        sigma = self._compute_volatility(close, high, low, dvol=dvol)

        # Trend filter: slow EMA direction (+1=up, -1=down, 0=flat)
        trend_dir: npt.NDArray[np.floating[Any]] | None = None
        if self._best_trend_ema_window > 0:
            slow_ema = np.asarray(
                pd.Series(close)
                .ewm(
                    span=self._best_trend_ema_window,
                    adjust=False,
                )
                .mean()
                .values,
            )
            # Trend = sign of slow EMA slope (diff)
            slope = np.diff(slow_ema, prepend=slow_ema[0])
            trend_dir = np.sign(slope)

        signals = self._run_glft_state_machine(
            close=close,
            ema=ema,
            sigma=sigma,
            gamma=self._best_gamma,
            kappa=self._best_kappa,
            min_hold=self._config.min_holding_bars,
            max_hold=self._best_max_holding_bars,
            min_entry_edge=self._best_min_entry_edge,
            trend_dir=trend_dir,
        )

        result = df.copy()
        result["signal"] = signals
        return result

    def get_parameters(self) -> dict[str, Any]:
        """Return strategy parameters."""
        params = self._config.model_dump()
        params["best_gamma"] = self._best_gamma
        params["best_kappa"] = self._best_kappa
        params["best_ema_window"] = self._best_ema_window
        params["best_max_holding_bars"] = self._best_max_holding_bars
        params["best_vol_window"] = self._best_vol_window
        params["best_min_entry_edge"] = self._best_min_entry_edge
        params["best_trend_ema_window"] = self._best_trend_ema_window
        return params

    # ------------------------------------------------------------------
    # Volatility estimation
    # ------------------------------------------------------------------

    # Minutes in a year (365.25 * 24 * 60) — used for
    # annualized-to-per-bar volatility conversion.
    _MINUTES_PER_YEAR: float = 525_960.0

    def _compute_volatility(
        self,
        close: npt.NDArray[np.floating[Any]],
        high: npt.NDArray[np.floating[Any]],
        low: npt.NDArray[np.floating[Any]],
        dvol: npt.NDArray[np.floating[Any]] | None = None,
    ) -> npt.NDArray[np.floating[Any]]:
        """Compute volatility (sigma).

        When ``vol_type`` is ``"implied"``, *dvol* must be provided
        (annualized % from Deribit DVOL).  Otherwise a rolling
        historical estimator is used.
        """
        if self._config.vol_type == "implied":
            if dvol is None:
                msg = "dvol array is required when vol_type is 'implied'"
                raise ValueError(msg)
            # DVOL is annualized vol in % (e.g. 45 = 45%).
            # Convert to per-bar (1 min) sigma:
            #   sigma = DVOL / 100 / sqrt(minutes_per_year)
            sigma = dvol / 100.0 / np.sqrt(self._MINUTES_PER_YEAR)
        elif self._config.vol_type == "parkinson":
            window = self._best_vol_window
            # Parkinson estimator: uses high-low range
            log_hl = np.log(
                np.maximum(high, 1e-10) / np.maximum(low, 1e-10),
            )
            parkinson_var = log_hl**2 / (4.0 * math.log(2))
            sigma = np.sqrt(
                pd.Series(parkinson_var).rolling(window, min_periods=1).mean().values,
            )
        else:
            window = self._best_vol_window
            # Realized volatility: rolling std of log returns
            log_ret = np.diff(
                np.log(np.maximum(close, 1e-10)),
            )
            log_ret = np.concatenate([[0.0], log_ret])
            sigma = pd.Series(log_ret).rolling(window, min_periods=1).std().values

        # Replace NaN/zero with a tiny positive value
        sigma = np.where(
            np.isnan(sigma) | (sigma <= 0),
            1e-10,
            sigma,
        )
        return np.asarray(sigma, dtype=np.float64)

    # ------------------------------------------------------------------
    # GLFT state machine
    # ------------------------------------------------------------------

    @staticmethod
    def _run_glft_state_machine(
        close: npt.NDArray[np.floating[Any]],
        ema: npt.NDArray[np.floating[Any]],
        sigma: npt.NDArray[np.floating[Any]],
        gamma: float,
        kappa: float,
        min_hold: int,
        max_hold: int,
        min_entry_edge: float = 0.0012,
        trend_dir: npt.NDArray[np.floating[Any]] | None = None,
    ) -> npt.NDArray[np.floating[Any]]:
        """GLFT state machine producing position-state signals.

        ``min_entry_edge`` is the minimum normalized price deviation
        from fair value required to open a position.  It should be
        at least ``fee_rate * 2`` (round-trip cost).

        When ``trend_dir`` is provided, entries are restricted to
        the trend direction (long only in uptrend, short only in
        downtrend).

        Returns an array of ``(1, -1, 0)`` where:

        - ``1``  = long position
        - ``-1`` = short position
        - ``0``  = flat (no position)
        """
        n = len(close)
        signals = np.zeros(n, dtype=np.float64)

        # Pre-compute the constant part of the half-spread
        # (1/gamma) * ln(1 + gamma/kappa)  →  limit γ→0 is 1/kappa
        if gamma < 1e-12:
            spread_const = 1.0 / kappa
        else:
            spread_const = math.log(1.0 + gamma / kappa) / gamma

        state = 0  # 0=flat, 1=long, -1=short
        bars_in_pos = 0

        for i in range(n):
            s = close[i]
            fair = ema[i]
            sig_sq = sigma[i] ** 2

            if fair <= 0 or s <= 0:
                signals[i] = float(state)
                if state != 0:
                    bars_in_pos += 1
                continue

            # Normalized deviation from fair value
            deviation = (s - fair) / fair

            if state == 0:
                # --- FLAT: evaluate entry ---
                tau = float(max_hold)
                # GLFT half-spread in %-space.  σ is in pct
                # (log-return), so σ_abs = σ_pct × fair.
                #   δ_pct = δ_abs / fair
                #         = γ·σ²·fair·τ/2 + spread_const/fair
                glft_hs = gamma * sig_sq * fair * tau / 2.0 + spread_const / fair
                half_spread = max(glft_hs, min_entry_edge)

                want_long = deviation < -half_spread
                want_short = deviation > half_spread

                # Apply trend filter
                if trend_dir is not None:
                    td = trend_dir[i]
                    if td > 0:
                        want_short = False
                    elif td < 0:
                        want_long = False

                if want_long:
                    state = 1
                    bars_in_pos = 0
                elif want_short:
                    state = -1
                    bars_in_pos = 0
            else:
                # --- IN POSITION ---
                bars_in_pos += 1
                tau = float(max(max_hold - bars_in_pos, 1))
                q = float(state)

                if bars_in_pos < min_hold:
                    signals[i] = float(state)
                    continue

                if bars_in_pos >= max_hold:
                    state = 0
                    signals[i] = 0.0
                    continue

                # Inventory-adjusted deviation (%-space)
                inventory_adj = q * gamma * sig_sq * fair * tau
                adjusted_dev = deviation + inventory_adj

                # Exit: GLFT spread in %-space (no entry-edge
                # floor — fees were paid on entry, exit follows
                # inventory risk logic)
                half_spread = gamma * sig_sq * fair * tau / 2.0 + spread_const / fair

                if (state == 1 and adjusted_dev > half_spread) or (
                    state == -1 and adjusted_dev < -half_spread
                ):
                    state = 0

            signals[i] = float(state)

        return signals
