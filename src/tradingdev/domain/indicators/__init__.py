"""Technical indicators.

``tradingdev.domain.indicators`` is the only module under ``src/`` that calls
pandas-ta; feature engineering and bundled strategies use these functions
instead of calling the library directly. Generated strategies in a user
workspace may import ``pandas_ta`` themselves; the strategy contract asks
them to follow the same rules (column-name selection, ``talib=False``), but
validation does not enforce those rules.
"""

from tradingdev.domain.indicators.technical import (
    ADXResult,
    BollingerBands,
    MACDResult,
    StochasticResult,
    adx,
    atr,
    bollinger_bands,
    ema,
    indicator_column,
    macd,
    rsi,
    sma,
    stochastic,
)

__all__ = [
    "ADXResult",
    "BollingerBands",
    "MACDResult",
    "StochasticResult",
    "adx",
    "atr",
    "bollinger_bands",
    "ema",
    "indicator_column",
    "macd",
    "rsi",
    "sma",
    "stochastic",
]
