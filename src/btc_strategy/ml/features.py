"""Feature engineering for ML trading strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas_ta as ta

from btc_strategy.utils.logger import setup_logger

if TYPE_CHECKING:
    import pandas as pd

logger = setup_logger(__name__)

# Rolling statistic windows
_ROLL_WINDOWS = [6, 12, 24]
# SMA ratio windows
_SMA_WINDOWS = [7, 14, 21]


class FeatureEngineer:
    """Generate feature matrices from OHLCV data for ML models.

    Features generated:
        - Lagged log-returns (``t-1`` to ``t-lookback``).
        - Rolling mean / std of returns over multiple windows.
        - ``close / SMA(N)`` ratios.
        - Volume change rate and volume / SMA_volume(N).
        - RSI(14), MACD histogram, Bollinger Band %B.
        - Target column: ``sign(next bar return)``.
    """

    def __init__(self, lookback: int = 24) -> None:
        self._lookback = lookback
        self._feature_names: list[str] = []

    @property
    def lookback(self) -> int:
        """Return the lookback window size."""
        return self._lookback

    def transform(
        self,
        df: pd.DataFrame,
        *,
        include_target: bool = True,
    ) -> pd.DataFrame:
        """Transform OHLCV DataFrame into a feature DataFrame.

        Args:
            df: OHLCV DataFrame with ``open``, ``high``, ``low``,
                ``close``, ``volume`` columns.
            include_target: If ``True``, append a ``target`` column
                with ``sign(next bar return)``. Set to ``False``
                during prediction to prevent data leakage.

        Returns:
            DataFrame with feature columns (and optionally ``target``).
            Rows with NaN from lookback are dropped.
        """
        result = df.copy()
        close = result["close"].astype(float)
        volume = result["volume"].astype(float)

        # Log returns (apply -> diff keeps Series type for mypy)
        log_ret = close.apply(np.log).diff()
        result["log_return"] = log_ret

        # Lagged returns
        for lag in range(1, self._lookback + 1):
            result[f"ret_lag_{lag}"] = log_ret.shift(lag)

        # Rolling statistics on returns
        for w in _ROLL_WINDOWS:
            result[f"ret_mean_{w}"] = log_ret.rolling(w).mean()
            result[f"ret_std_{w}"] = log_ret.rolling(w).std()

        # Price relative to SMA
        for w in _SMA_WINDOWS:
            sma = close.rolling(w).mean()
            result[f"close_sma_ratio_{w}"] = close / sma

        # Volume features
        result["volume_change"] = volume.pct_change()
        for w in _SMA_WINDOWS:
            vol_sma = volume.rolling(w).mean()
            result[f"vol_sma_ratio_{w}"] = volume / vol_sma

        # Technical indicators via pandas-ta
        rsi = ta.rsi(close, length=14)
        if rsi is not None:
            result["rsi_14"] = rsi

        macd_df = ta.macd(close)
        if macd_df is not None:
            # MACD histogram is the 3rd column
            result["macd_hist"] = macd_df.iloc[:, 2]

        bbands = ta.bbands(close, length=20)
        if bbands is not None:
            # %B = (close - lower) / (upper - lower)
            upper = bbands.iloc[:, 0]
            lower = bbands.iloc[:, 2]
            band_width = upper - lower
            result["bb_pctb"] = (close - lower) / band_width.replace(
                0, float("nan")
            )

        # Target: direction of next bar
        if include_target:
            next_ret = close.shift(-1) / close - 1
            result["target"] = np.sign(next_ret).fillna(0).astype(int)

        # Drop rows with NaN from lookback / indicator warm-up
        result = result.dropna().reset_index(drop=True)

        # Cache feature names (exclude non-feature columns)
        exclude = {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "target",
        }
        self._feature_names = [
            c for c in result.columns if c not in exclude
        ]

        return result

    def get_feature_names(self) -> list[str]:
        """Return the list of generated feature column names."""
        return list(self._feature_names)
