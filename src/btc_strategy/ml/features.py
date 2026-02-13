"""Feature engineering for ML trading strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta

from btc_strategy.utils.logger import setup_logger

logger = setup_logger(__name__)

# Rolling statistic windows
_ROLL_WINDOWS = [6, 12, 24]
# SMA ratio windows
_SMA_WINDOWS = [7, 14, 21]


_DENSE_THRESHOLD = 32


def _sampled_lags(lookback: int) -> list[int]:
    """Return lag positions: dense up to 32, then exponentially spaced.

    - Lags 1..min(32, lookback): every step (preserves short-term
      microstructure).
    - Beyond 32: powers-of-2 spacing (64, 128, 256, …) up to lookback.
    - Always includes the exact lookback value as the final lag.
    """
    dense_end = min(_DENSE_THRESHOLD, lookback)
    lags = list(range(1, dense_end + 1))

    # Exponentially spaced beyond the dense region
    lag = _DENSE_THRESHOLD * 2
    while lag <= lookback:
        lags.append(lag)
        lag *= 2

    if lags[-1] != lookback:
        lags.append(lookback)
    return lags


class FeatureEngineer:
    """Generate feature matrices from OHLCV data for ML models.

    Features generated:
        - Sampled lagged log-returns (exponentially spaced up to lookback).
        - Rolling mean / std of returns over multiple windows.
        - ``close / SMA(N)`` ratios.
        - Volume change rate and volume / SMA_volume(N).
        - RSI(14), MACD histogram, Bollinger Band %B.
        - Target column: ``sign(next bar return)``.
    """

    def __init__(
        self, lookback: int = 24, target_horizon: int = 1
    ) -> None:
        self._lookback = lookback
        self._target_horizon = target_horizon
        self._feature_names: list[str] = []

    @property
    def lookback(self) -> int:
        """Return the lookback window size."""
        return self._lookback

    @property
    def target_horizon(self) -> int:
        """Return the target prediction horizon."""
        return self._target_horizon

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
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)

        # Log returns
        log_ret = close.apply(np.log).diff()

        # Build all feature columns in a dict, then concat once
        features: dict[str, pd.Series] = {"log_return": log_ret}

        # Sampled lagged returns (exponentially spaced)
        for lag in _sampled_lags(self._lookback):
            features[f"ret_lag_{lag}"] = log_ret.shift(lag)

        # Rolling statistics on returns
        for w in _ROLL_WINDOWS:
            features[f"ret_mean_{w}"] = log_ret.rolling(w).mean()
            features[f"ret_std_{w}"] = log_ret.rolling(w).std()

        # Price relative to SMA
        for w in _SMA_WINDOWS:
            sma = close.rolling(w).mean()
            features[f"close_sma_ratio_{w}"] = close / sma

        # Volume features
        features["volume_change"] = volume.pct_change()
        for w in _SMA_WINDOWS:
            vol_sma = volume.rolling(w).mean()
            features[f"vol_sma_ratio_{w}"] = volume / vol_sma

        # Technical indicators via pandas-ta
        rsi = ta.rsi(close, length=14)
        if rsi is not None:
            features["rsi_14"] = rsi

        macd_df = ta.macd(close)
        if macd_df is not None:
            features["macd_hist"] = macd_df.iloc[:, 2]

        bbands = ta.bbands(close, length=20)
        if bbands is not None:
            upper = bbands.iloc[:, 0]
            lower = bbands.iloc[:, 2]
            band_width = upper - lower
            features["bb_pctb"] = (close - lower) / band_width.replace(
                0, float("nan")
            )

        # Target: direction of future return over target_horizon bars
        if include_target:
            next_ret = (
                close.shift(-self._target_horizon) / close - 1
            )
            features["target"] = np.sign(next_ret).fillna(0).astype(int)

        # Concat all features at once (avoids fragmentation)
        result = pd.concat(
            [df, pd.DataFrame(features, index=df.index)], axis=1
        )

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
