"""Backtest execution engine with signal and volume modes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import vectorbt as vbt

if TYPE_CHECKING:
    import pandas as pd

from btc_strategy.backtest.metrics import (
    calculate_metrics,
    calculate_metrics_from_simulation,
)
from btc_strategy.utils.logger import setup_logger

logger = setup_logger(__name__)


class BacktestEngine:
    """Execute backtests in two modes.

    **Signal mode** (default, for KD etc.):
        Uses vectorbt ``Portfolio.from_signals`` with transition-based
        entry/exit arrays.  Supports all-in or fixed-size positions
        with optional SL/TP.

    **Volume mode** (for market-making):
        Custom bar-by-bar simulation that re-enters immediately after
        SL/TP exits.  Designed to maximise trade count (volume) while
        keeping a fixed position size.

    In both modes the signal is shifted by 1 bar to avoid look-ahead
    bias: a signal at bar *i*'s close is executed at bar *i+1*'s open.
    """

    def __init__(
        self,
        init_cash: float = 10_000.0,
        fees: float = 0.0006,
        slippage: float = 0.0005,
        freq: str = "1h",
        position_size_usdt: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        mode: str = "signal",
    ) -> None:
        self._init_cash = init_cash
        self._fees = fees
        self._slippage = slippage
        self._freq = freq
        self._position_size_usdt = position_size_usdt
        self._stop_loss = stop_loss
        self._take_profit = take_profit
        self._mode = mode

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run backtest on a DataFrame with a ``signal`` column.

        Args:
            df: DataFrame with at least ``close`` and ``signal``
                columns.  Signal values: 1 (long), -1 (short),
                0 (flat).

        Returns:
            Dictionary of performance metrics.
        """
        if self._mode == "volume":
            return self._run_volume(df)
        return self._run_signal(df)

    # ------------------------------------------------------------------
    # signal mode (vectorbt from_signals)
    # ------------------------------------------------------------------

    def _run_signal(self, df: pd.DataFrame) -> dict[str, Any]:
        close = df["close"].astype(float)

        signal = df["signal"].shift(1).fillna(0).astype(int)

        entries = (signal == 1) & (signal.shift(1) != 1)
        exits = (signal != 1) & (signal.shift(1) == 1)
        short_entries = (signal == -1) & (
            signal.shift(1) != -1
        )
        short_exits = (signal != -1) & (
            signal.shift(1) == -1
        )

        logger.info(
            "Running backtest (signal mode): "
            "init_cash=%.0f, fees=%.4f, slippage=%.4f",
            self._init_cash,
            self._fees,
            self._slippage,
        )

        kwargs: dict[str, Any] = {
            "close": close,
            "entries": entries,
            "exits": exits,
            "short_entries": short_entries,
            "short_exits": short_exits,
            "init_cash": self._init_cash,
            "fees": self._fees,
            "slippage": self._slippage,
            "freq": self._freq,
        }

        if self._position_size_usdt is not None:
            kwargs["size"] = self._position_size_usdt
            kwargs["size_type"] = "value"

        if self._stop_loss is not None:
            kwargs["sl_stop"] = self._stop_loss

        if self._take_profit is not None:
            kwargs["tp_stop"] = self._take_profit

        pf = vbt.Portfolio.from_signals(**kwargs)

        metrics = calculate_metrics(pf)
        logger.info(
            "Backtest complete: %d trades",
            metrics["total_trades"],
        )
        return metrics

    # ------------------------------------------------------------------
    # volume mode (custom bar-by-bar simulation)
    # ------------------------------------------------------------------

    def _run_volume(self, df: pd.DataFrame) -> dict[str, Any]:
        """Simulate with forced re-entry after SL/TP exits.

        The position size is always ``position_size_usdt``.  After
        every SL/TP exit the simulator immediately re-enters in the
        direction indicated by the current signal, ensuring maximum
        time-in-market and trade count.
        """
        close_s = df["close"].astype(float)
        high_s = (
            df["high"].astype(float)
            if "high" in df.columns
            else close_s
        )
        low_s = (
            df["low"].astype(float)
            if "low" in df.columns
            else close_s
        )

        signal_s = (
            df["signal"].shift(1).fillna(0).astype(int)
        )

        close = close_s.to_numpy()
        high = high_s.to_numpy()
        low = low_s.to_numpy()
        signal = signal_s.to_numpy()

        n = len(close)
        size_usdt = self._position_size_usdt or 200.0
        sl = self._stop_loss
        tp = self._take_profit
        fee_rate = self._fees + self._slippage

        cash = self._init_cash
        equity = np.empty(n, dtype=np.float64)
        trades: list[dict[str, Any]] = []

        # Position state
        pos_dir = 0  # 0 = flat, 1 = long, -1 = short
        entry_price = 0.0
        pos_qty = 0.0  # BTC quantity held

        logger.info(
            "Running backtest (volume mode): "
            "init_cash=%.0f, size=%.0f, sl=%s, tp=%s",
            self._init_cash,
            size_usdt,
            sl,
            tp,
        )

        for i in range(n):
            sig = int(signal[i])
            c = float(close[i])
            h = float(high[i])
            lo = float(low[i])

            # 1. Check SL / TP if in position
            if pos_dir != 0:
                exit_price = self._check_sl_tp(
                    pos_dir, entry_price, h, lo, sl, tp
                )
                if exit_price is not None:
                    # Close position at SL/TP price
                    trade = self._close_position(
                        pos_dir,
                        pos_qty,
                        entry_price,
                        exit_price,
                        fee_rate,
                    )
                    trades.append(trade)
                    cash += trade["net_pnl"]
                    pos_dir = 0
                    pos_qty = 0.0

                    # Immediate re-entry at close price
                    if sig != 0:
                        entry_price = c
                        pos_qty = size_usdt / c
                        cash -= size_usdt * fee_rate
                        pos_dir = sig

            # 2. Direction change while in position
            if pos_dir != 0 and sig != 0 and sig != pos_dir:
                trade = self._close_position(
                    pos_dir,
                    pos_qty,
                    entry_price,
                    c,
                    fee_rate,
                )
                trades.append(trade)
                cash += trade["net_pnl"]

                # Open reverse position
                entry_price = c
                pos_qty = size_usdt / c
                cash -= size_usdt * fee_rate
                pos_dir = sig

            # 3. Enter new position if flat
            elif pos_dir == 0 and sig != 0:
                entry_price = c
                pos_qty = size_usdt / c
                cash -= size_usdt * fee_rate
                pos_dir = sig

            # 4. Update equity
            if pos_dir != 0:
                unrealised = (
                    pos_dir * pos_qty * (c - entry_price)
                )
                equity[i] = cash + size_usdt + unrealised
            else:
                equity[i] = cash

        # Close any remaining position at last close
        if pos_dir != 0:
            trade = self._close_position(
                pos_dir,
                pos_qty,
                entry_price,
                float(close[-1]),
                fee_rate,
            )
            trades.append(trade)
            cash += trade["net_pnl"]
            equity[-1] = cash

        timestamps = None
        if "timestamp" in df.columns:
            timestamps = df["timestamp"].to_numpy()

        metrics = calculate_metrics_from_simulation(
            equity_curve=equity,
            trades=trades,
            init_cash=self._init_cash,
            timestamps=timestamps,
        )
        logger.info(
            "Backtest complete (volume): %d trades, "
            "volume=%.0f USDT",
            metrics["total_trades"],
            metrics["total_volume_usdt"],
        )
        return metrics

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_sl_tp(
        pos_dir: int,
        entry_price: float,
        high: float,
        low: float,
        sl: float | None,
        tp: float | None,
    ) -> float | None:
        """Return exit price if SL or TP was hit, else None."""
        if pos_dir == 1:
            if sl and low <= entry_price * (1 - sl):
                return entry_price * (1 - sl)
            if tp and high >= entry_price * (1 + tp):
                return entry_price * (1 + tp)
        elif pos_dir == -1:
            if sl and high >= entry_price * (1 + sl):
                return entry_price * (1 + sl)
            if tp and low <= entry_price * (1 - tp):
                return entry_price * (1 - tp)
        return None

    @staticmethod
    def _close_position(
        pos_dir: int,
        pos_qty: float,
        entry_price: float,
        exit_price: float,
        fee_rate: float,
    ) -> dict[str, Any]:
        """Build a trade record for a closed position."""
        gross_pnl = pos_dir * pos_qty * (
            exit_price - entry_price
        )
        exit_value = pos_qty * exit_price
        exit_fee = exit_value * fee_rate
        net_pnl = gross_pnl - exit_fee
        return {
            "direction": pos_dir,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size_usdt": pos_qty * entry_price,
            "gross_pnl": gross_pnl,
            "fee": exit_fee,
            "net_pnl": net_pnl,
        }
