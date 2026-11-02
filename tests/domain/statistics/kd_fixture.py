"""Load the versioned KD(14,3,3) BTC/USDT 1h 2024 fixture (see its README)."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "kd_btcusdt_1h_2024"


@dataclass(frozen=True)
class KDFixture:
    """Raw outputs of the pinned KD backtest, as the statistics modules read them."""

    timestamps: pd.DatetimeIndex
    equity: npt.NDArray[np.float64]
    trades: list[dict[str, Any]]
    run: dict[str, Any]


def load_kd_fixture() -> KDFixture:
    bars = pd.read_parquet(FIXTURE_DIR / "equity.parquet")
    trades = pd.read_parquet(FIXTURE_DIR / "trades.parquet")
    run = json.loads((FIXTURE_DIR / "run.json").read_text(encoding="utf-8"))
    return KDFixture(
        timestamps=pd.DatetimeIndex(bars["timestamp"]),
        equity=bars["equity"].to_numpy(dtype=np.float64),
        trades=[
            {str(key): value for key, value in record.items()}
            for record in trades.to_dict("records")
        ],
        run=run,
    )
