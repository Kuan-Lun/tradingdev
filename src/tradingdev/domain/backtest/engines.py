"""Facade for backtest engine implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tradingdev.domain.backtest.base_engine import BaseBacktestEngine
from tradingdev.domain.backtest.signal_engine import SignalBacktestEngine
from tradingdev.domain.backtest.volume_engine import VolumeBacktestEngine

if TYPE_CHECKING:
    from tradingdev.domain.backtest.schemas import BacktestConfig


def create_backtest_engine(config: BacktestConfig) -> BaseBacktestEngine:
    """Create the concrete backtest engine requested by config."""
    if config.mode == "volume":
        return VolumeBacktestEngine(
            fees=config.fees,
            slippage=config.slippage,
            freq=config.timeframe,
            position_size=config.position_size,
            stop_loss=config.stop_loss,
            take_profit=config.take_profit,
            signal_as_position=config.signal_as_position,
            re_entry_after_sl=config.re_entry_after_sl,
            monthly_max_loss=config.monthly_max_loss,
        )
    return SignalBacktestEngine(
        init_cash=config.init_cash,
        fees=config.fees,
        slippage=config.slippage,
        freq=config.timeframe,
        position_size=config.position_size,
        stop_loss=config.stop_loss,
        take_profit=config.take_profit,
        signal_as_position=config.signal_as_position,
        re_entry_after_sl=config.re_entry_after_sl,
    )


__all__ = [
    "BaseBacktestEngine",
    "SignalBacktestEngine",
    "VolumeBacktestEngine",
    "create_backtest_engine",
]
