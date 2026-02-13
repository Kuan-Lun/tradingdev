"""Volume-mode backtest engine with bar-by-bar simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from btc_strategy.backtest.base_engine import BaseBacktestEngine
from btc_strategy.backtest.metrics import (
    calculate_metrics_from_simulation,
)
from btc_strategy.backtest.result import BacktestResult
from btc_strategy.utils.logger import setup_logger

if TYPE_CHECKING:
    import pandas as pd

logger = setup_logger(__name__)


class VolumeBacktestEngine(BaseBacktestEngine):
    """Bar-by-bar simulation with forced re-entry after SL/TP.

    Position size is always ``position_size_usdt``.  After every
    SL/TP exit the simulator immediately re-enters in the direction
    indicated by the current signal.
    """

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """Run volume-mode backtest."""
        close_s = df["close"].astype(float)
        high_s = df["high"].astype(float) if "high" in df.columns else close_s
        low_s = df["low"].astype(float) if "low" in df.columns else close_s
        signal_s = df["signal"].shift(1).fillna(0).astype(int)

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

        pos_dir = 0
        entry_price = 0.0
        pos_qty = 0.0

        logger.info(
            "Running backtest (volume mode): init_cash=%.0f, size=%.0f, sl=%s, tp=%s",
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

            if pos_dir != 0:
                exit_price = _check_sl_tp(pos_dir, entry_price, h, lo, sl, tp)
                if exit_price is not None:
                    trade = _close_position(
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

                    if sig != 0:
                        entry_price = c
                        pos_qty = size_usdt / c
                        cash -= size_usdt * fee_rate
                        pos_dir = sig

            if pos_dir != 0 and sig != 0 and sig != pos_dir:
                trade = _close_position(
                    pos_dir,
                    pos_qty,
                    entry_price,
                    c,
                    fee_rate,
                )
                trades.append(trade)
                cash += trade["net_pnl"]
                entry_price = c
                pos_qty = size_usdt / c
                cash -= size_usdt * fee_rate
                pos_dir = sig

            elif pos_dir == 0 and sig != 0:
                entry_price = c
                pos_qty = size_usdt / c
                cash -= size_usdt * fee_rate
                pos_dir = sig

            if pos_dir != 0:
                unrealised = pos_dir * pos_qty * (c - entry_price)
                equity[i] = cash + size_usdt + unrealised
            else:
                equity[i] = cash

        if pos_dir != 0:
            trade = _close_position(
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
            "Backtest complete (volume): %d trades, volume=%.0f USDT",
            metrics["total_trades"],
            metrics["total_volume_usdt"],
        )
        return BacktestResult(
            metrics=metrics,
            equity_curve=equity,
            trades=trades,
            timestamps=timestamps,
            init_cash=self._init_cash,
            mode="volume",
        )


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


def _close_position(
    pos_dir: int,
    pos_qty: float,
    entry_price: float,
    exit_price: float,
    fee_rate: float,
) -> dict[str, Any]:
    """Build a trade record for a closed position."""
    gross_pnl = pos_dir * pos_qty * (exit_price - entry_price)
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
