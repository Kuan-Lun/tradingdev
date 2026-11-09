"""Final cash, native fills and costs agree after mandatory liquidation."""

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.backtest.signal_engine import SignalBacktestEngine


def _market(signals: list[int], final_close: float = 110.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, tz="UTC"),
            "open": [100.0, 100.0, 100.0],
            "close": [100.0, 100.0, final_close],
            "signal": signals,
        }
    )


@pytest.mark.parametrize(
    ("direction", "entry_price", "exit_price", "pnl"),
    [(1, 100.5, 109.45, 6.8505), (-1, 99.5, 110.55, -13.1505)],
)
@pytest.mark.parametrize("final_bar_entry", [False, True])
def test_terminal_fill_closes_long_and_short_with_both_costs(
    direction: int,
    entry_price: float,
    exit_price: float,
    pnl: float,
    final_bar_entry: bool,
) -> None:
    signals = [0 if final_bar_entry else direction, direction, direction]
    result = SignalBacktestEngine(
        init_cash=1000.0, position_size=100.0, fees=0.01, slippage=0.005
    ).run(_market(signals))
    records = result.execution_records
    accounts = result.account_history
    assert records is not None and accounts is not None
    assert len(records) == 2
    entry, exit_ = records
    assert entry.bar_index == (2 if final_bar_entry else 1)
    assert entry.filled_size == pytest.approx(1.0)
    assert entry.filled_price == pytest.approx(entry_price)
    assert exit_.bar_index == 2
    assert exit_.requested_price == exit_.market.close == 110.0
    assert exit_.filled_price == pytest.approx(exit_price)
    assert exit_.filled_size == pytest.approx(1.0)
    assert exit_.requested_slippage == 0.005
    assert exit_.after.position == 0.0
    assert len(accounts) == len(result.equity_curve) == 3
    assert accounts[-1].position == accounts[-1].asset_value == 0.0
    assert accounts[-1].cash == pytest.approx(1000.0 + pnl)
    assert accounts[-1].equity == result.equity_curve[-1]
    assert result.metrics["total_pnl"] == pytest.approx(pnl)
    assert result.metrics["total_fees"] == pytest.approx(
        (entry_price + exit_price) * 0.01
    )
    assert result.metrics["total_volume"] == pytest.approx(entry_price + exit_price)
    assert result.metrics["total_trades"] == 1
    assert result.metrics["open_trades"] == 0
    assert result.trades[0]["status"] == "closed"
    assert result.trades[0]["net_pnl"] == pytest.approx(pnl)
    assert result.timestamps is not None
    assert len(np.unique(result.timestamps)) == 3


def test_flat_price_charges_both_sides_and_preserves_initial_capital_basis() -> None:
    result = SignalBacktestEngine(
        init_cash=1000.0, position_size=100.0, fees=0.01, slippage=0.0
    ).run(_market([1, 1, 1], final_close=100.0))
    assert result.equity_curve.tolist() == pytest.approx([1000.0, 999.0, 998.0])
    assert result.metrics["total_pnl"] == -2.0
    assert result.trades[0]["net_pnl"] == -2.0


def test_final_open_reversal_is_preserved_before_closing_the_new_short() -> None:
    result = SignalBacktestEngine(
        init_cash=1000.0, position_size=100.0, fees=0.0, slippage=0.0
    ).run(_market([1, -1, -1]))
    assert result.execution_records is not None
    assert [r.bar_index for r in result.execution_records] == [1, 2, 2]
    assert [r.side for r in result.execution_records] == ["buy", "sell", "buy"]
    assert result.execution_records[1].after.position == -1.0
    assert result.execution_records[-1].after.position == 0.0
    assert result.metrics["total_trades"] == 2
    assert [t["net_pnl"] for t in result.trades] == pytest.approx([0.0, -10.0])


def test_final_open_exit_does_not_receive_a_duplicate_closing_fill() -> None:
    result = SignalBacktestEngine(
        init_cash=1000.0, position_size=100.0, fees=0.01, slippage=0.0
    ).run(_market([1, 0, 0]))
    assert result.execution_records is not None
    assert len(result.execution_records) == 2
    assert result.execution_records[-1].requested_price == 100.0
    assert result.metrics["total_fees"] == 2.0
    assert result.metrics["total_pnl"] == -2.0


def test_unfunded_short_close_fails_instead_of_publishing_an_open_position() -> None:
    engine = SignalBacktestEngine(
        init_cash=100.0, position_size=1000.0, fees=0.0, slippage=0.0
    )
    with pytest.raises(ValueError, match="Terminal liquidation could not close"):
        engine.run(_market([-1, -1, -1], final_close=1000.0))
