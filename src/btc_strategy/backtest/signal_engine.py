"""Signal-mode backtest engine using vectorbt."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import vectorbt as vbt

from btc_strategy.backtest.base_engine import BaseBacktestEngine
from btc_strategy.backtest.metrics import calculate_metrics
from btc_strategy.backtest.result import BacktestResult
from btc_strategy.utils.logger import setup_logger

if TYPE_CHECKING:
    import pandas as pd

logger = setup_logger(__name__)


class SignalBacktestEngine(BaseBacktestEngine):
    """Backtest using vectorbt ``Portfolio.from_signals``.

    Supports all-in or fixed-size positions with optional SL/TP.
    The signal is shifted by 1 bar to avoid look-ahead bias.
    """

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """Run signal-mode backtest.

        Execution uses the **open** price of the bar following the
        signal (after ``shift(1)``) to avoid look-ahead bias.
        """
        close = df["close"].astype(float)
        open_ = df["open"].astype(float) if "open" in df.columns else close
        signal = df["signal"].shift(1).fillna(0).astype(int)

        entries = (signal == 1) & (signal.shift(1) != 1)
        exits = (signal != 1) & (signal.shift(1) == 1)
        short_entries = (signal == -1) & (signal.shift(1) != -1)
        short_exits = (signal != -1) & (signal.shift(1) == -1)

        logger.info(
            "Running backtest (signal mode): init_cash=%.0f, fees=%.4f, slippage=%.4f",
            self._init_cash,
            self._fees,
            self._slippage,
        )

        kwargs: dict[str, Any] = {
            "close": close,
            "open": open_,
            "price": open_,
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

        equity_curve = np.asarray(pf.value(), dtype=np.float64)
        trades = _extract_trades(pf)
        timestamps = None
        if "timestamp" in df.columns:
            timestamps = df["timestamp"].to_numpy()

        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_curve,
            trades=trades,
            timestamps=timestamps,
            init_cash=self._init_cash,
            mode="signal",
        )


def _extract_trades(pf: vbt.Portfolio) -> list[dict[str, Any]]:
    """Extract trade records from a vectorbt Portfolio."""
    records = pf.trades.records_readable
    if len(records) == 0:
        return []

    trades: list[dict[str, Any]] = []
    for _, row in records.iterrows():
        entry_price = row.get(
            "Avg Entry Price", row.get("Entry Price", 0.0)
        )
        exit_price = row.get(
            "Avg Exit Price", row.get("Exit Price", 0.0)
        )
        size = row.get("Size", 0.0)
        pnl = row.get("PnL", 0.0)
        direction = 1 if row.get("Direction", "Long") == "Long" else -1

        trades.append(
            {
                "direction": direction,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "size_usdt": float(size * entry_price),
                "gross_pnl": float(pnl),
                "fee": 0.0,
                "net_pnl": float(pnl),
            }
        )
    return trades
