"""Tests for the GLFT market-making strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from pydantic import ValidationError

from btc_strategy.backtest.volume_engine import VolumeBacktestEngine
from btc_strategy.data.schemas import GLFTStrategyConfig
from btc_strategy.strategies.glft_strategy import GLFTStrategy

if TYPE_CHECKING:
    import pandas as pd

# ── Helpers ──────────────────────────────────────────────────


def _make_config(**overrides: object) -> GLFTStrategyConfig:
    """Create a lightweight config for fast tests."""
    defaults: dict[str, object] = {
        "gamma": 0.01,
        "kappa": 1.5,
        "ema_window": 10,
        "vol_window": 10,
        "min_holding_bars": 3,
        "max_holding_bars": 15,
        "gamma_candidates": [0.005, 0.01, 0.05],
        "kappa_candidates": [1.0, 1.5],
        "ema_window_candidates": [10, 21],
        "max_holding_bars_candidates": [10, 15],
        "vol_window_candidates": [10, 20],
    }
    defaults.update(overrides)
    return GLFTStrategyConfig(**defaults)  # type: ignore[arg-type]


# ── Config validation tests ──────────────────────────────────


class TestGLFTStrategyConfig:
    def test_default_values(self) -> None:
        """Default config should have valid parameter values."""
        config = GLFTStrategyConfig()
        assert config.gamma == 0.01
        assert config.kappa == 1.5
        assert config.ema_window == 21
        assert config.min_holding_bars == 5
        assert config.max_holding_bars == 30

    def test_gamma_must_be_non_negative(self) -> None:
        """gamma < 0 should raise validation error; 0 is allowed."""
        GLFTStrategyConfig(gamma=0.0)  # should NOT raise
        with pytest.raises(ValidationError):
            GLFTStrategyConfig(gamma=-0.01)

    def test_kappa_must_be_positive(self) -> None:
        """kappa <= 0 should raise validation error."""
        with pytest.raises(ValidationError):
            GLFTStrategyConfig(kappa=0.0)
        with pytest.raises(ValidationError):
            GLFTStrategyConfig(kappa=-1.0)

    def test_max_holding_gt_min_holding(self) -> None:
        """max_holding_bars must be > min_holding_bars."""
        with pytest.raises(ValidationError):
            GLFTStrategyConfig(
                min_holding_bars=10,
                max_holding_bars=5,
            )
        with pytest.raises(ValidationError):
            GLFTStrategyConfig(
                min_holding_bars=10,
                max_holding_bars=10,
            )

    def test_implied_vol_type_accepted(self) -> None:
        """vol_type='implied' is valid when dvol_processed_path set."""
        config = GLFTStrategyConfig(
            vol_type="implied",
            dvol_processed_path="data/processed/dvol.parquet",
        )
        assert config.vol_type == "implied"

    def test_implied_requires_dvol_path(self) -> None:
        """vol_type='implied' without dvol_processed_path raises."""
        with pytest.raises(ValidationError, match="dvol_processed_path"):
            GLFTStrategyConfig(vol_type="implied")


# ── State machine tests ──────────────────────────────────────


class TestGLFTStateMachine:
    def test_min_holding_enforced(self) -> None:
        """Cannot exit before min_holding_bars."""
        n = 20
        # Price starts well below EMA, then jumps above
        close = np.concatenate(
            [
                np.full(5, 99.0),  # below EMA → should enter long
                np.full(15, 102.0),  # above EMA → would want to exit
            ]
        )
        ema = np.full(n, 100.0)
        sigma = np.full(n, 0.005)

        signals = GLFTStrategy._run_glft_state_machine(
            close=close,
            ema=ema,
            sigma=sigma,
            gamma=0.01,
            kappa=1.5,
            min_hold=5,
            max_hold=30,
        )

        # Find first entry
        entry_idx = -1
        for i in range(n):
            if signals[i] != 0:
                entry_idx = i
                break

        assert entry_idx >= 0, "Should have entered a position"
        # Must hold for at least min_hold bars after entry
        for j in range(entry_idx, min(entry_idx + 5, n)):
            assert signals[j] != 0, f"Exited too early at bar {j}"

    def test_max_holding_forced_exit(self) -> None:
        """Position must exit at max_holding_bars."""
        n = 30
        # Price stays slightly below EMA the entire time
        close = np.full(n, 99.5)
        ema = np.full(n, 100.0)
        sigma = np.full(n, 0.001)

        signals = GLFTStrategy._run_glft_state_machine(
            close=close,
            ema=ema,
            sigma=sigma,
            gamma=0.001,
            kappa=0.5,
            min_hold=2,
            max_hold=10,
        )

        # Find entry and check forced exit
        in_pos = False
        hold_count = 0
        for i in range(n):
            if not in_pos and signals[i] != 0:
                in_pos = True
                hold_count = 0
            elif in_pos:
                if signals[i] != 0:
                    hold_count += 1
                else:
                    # Exited — should be at or before max_hold
                    assert hold_count <= 10
                    break

    def test_flat_when_within_spread(self) -> None:
        """No entry when price is near EMA (within spread)."""
        n = 50
        close = np.full(n, 100.0)  # price == EMA
        ema = np.full(n, 100.0)
        sigma = np.full(n, 0.01)

        signals = GLFTStrategy._run_glft_state_machine(
            close=close,
            ema=ema,
            sigma=sigma,
            gamma=0.1,
            kappa=1.5,
            min_hold=5,
            max_hold=30,
        )

        # Should stay flat
        assert np.all(signals == 0)

    def test_long_entry_when_cheap(self) -> None:
        """Price below EMA by more than spread triggers long."""
        n = 20
        close = np.full(n, 95.0)  # 5% below EMA
        ema = np.full(n, 100.0)
        sigma = np.full(n, 0.001)  # low vol → tight spread

        signals = GLFTStrategy._run_glft_state_machine(
            close=close,
            ema=ema,
            sigma=sigma,
            gamma=0.001,
            kappa=1.0,
            min_hold=2,
            max_hold=15,
        )

        # Should enter long
        assert signals[0] == 1

    def test_short_entry_when_expensive(self) -> None:
        """Price above EMA by more than spread triggers short."""
        n = 20
        close = np.full(n, 105.0)  # 5% above EMA
        ema = np.full(n, 100.0)
        sigma = np.full(n, 0.001)

        signals = GLFTStrategy._run_glft_state_machine(
            close=close,
            ema=ema,
            sigma=sigma,
            gamma=0.001,
            kappa=1.0,
            min_hold=2,
            max_hold=15,
        )

        assert signals[0] == -1


# ── Signal generation tests ──────────────────────────────────


class TestGLFTSignalGeneration:
    def test_signal_values_in_valid_range(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        """All signals must be -1, 0, or 1."""
        config = _make_config()
        strategy = GLFTStrategy(config=config)
        result = strategy.generate_signals(sample_ohlcv_df)
        assert set(result["signal"].unique()).issubset(
            {-1.0, 0.0, 1.0},
        )

    def test_no_nan_in_signals(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        """Signals should not contain NaN."""
        config = _make_config()
        strategy = GLFTStrategy(config=config)
        result = strategy.generate_signals(sample_ohlcv_df)
        assert not result["signal"].isna().any()

    def test_signal_column_added(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        """generate_signals() should add a signal column."""
        config = _make_config()
        strategy = GLFTStrategy(config=config)
        result = strategy.generate_signals(sample_ohlcv_df)
        assert "signal" in result.columns
        assert len(result) == len(sample_ohlcv_df)

    def test_original_df_not_modified(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        """generate_signals() should not modify the input df."""
        config = _make_config()
        strategy = GLFTStrategy(config=config)
        cols_before = set(sample_ohlcv_df.columns)
        strategy.generate_signals(sample_ohlcv_df)
        assert set(sample_ohlcv_df.columns) == cols_before


# ── fit() tests ──────────────────────────────────────────────


class TestGLFTFit:
    def test_fit_selects_from_candidates(
        self,
        large_ohlcv_df: pd.DataFrame,
    ) -> None:
        """After fit(), best params should be from candidates."""
        engine = VolumeBacktestEngine(
            init_cash=10_000,
            fees=0.0,
            slippage=0.0,
            position_size_usdt=100.0,
            signal_as_position=True,
        )
        config = _make_config()
        strategy = GLFTStrategy(
            config=config,
            backtest_engine=engine,
        )
        strategy.fit(large_ohlcv_df)
        params = strategy.get_parameters()
        assert params["best_gamma"] in config.gamma_candidates
        assert params["best_kappa"] in config.kappa_candidates
        assert params["best_ema_window"] in config.ema_window_candidates
        assert params["best_max_holding_bars"] in config.max_holding_bars_candidates
        assert params["best_vol_window"] in config.vol_window_candidates

    def test_fit_without_engine_uses_defaults(
        self,
        large_ohlcv_df: pd.DataFrame,
    ) -> None:
        """fit() without engine should use default params."""
        config = _make_config()
        strategy = GLFTStrategy(config=config)
        strategy.fit(large_ohlcv_df)  # should not raise
        params = strategy.get_parameters()
        assert params["best_gamma"] == config.gamma
        assert params["best_kappa"] == config.kappa


# ── get_parameters() tests ───────────────────────────────────


class TestGLFTGetParameters:
    def test_includes_config_fields(self) -> None:
        """get_parameters() should include config fields."""
        config = _make_config()
        strategy = GLFTStrategy(config=config)
        params = strategy.get_parameters()
        assert "gamma" in params
        assert "kappa" in params
        assert "ema_window" in params
        assert "min_holding_bars" in params

    def test_includes_best_params(self) -> None:
        """get_parameters() should include best_* fields."""
        config = _make_config()
        strategy = GLFTStrategy(config=config)
        params = strategy.get_parameters()
        assert "best_gamma" in params
        assert "best_kappa" in params
        assert "best_ema_window" in params
        assert "best_max_holding_bars" in params
        assert "best_vol_window" in params


# ── Integration tests ────────────────────────────────────────


class TestGLFTIntegration:
    def test_end_to_end_with_volume_engine(
        self,
        large_ohlcv_df: pd.DataFrame,
    ) -> None:
        """Full pipeline: fit -> generate_signals -> engine.run."""
        engine = VolumeBacktestEngine(
            init_cash=10_000,
            fees=0.0006,
            slippage=0.0005,
            position_size_usdt=100.0,
            stop_loss=0.03,
            signal_as_position=True,
            re_entry_after_sl=False,
        )
        config = _make_config()
        strategy = GLFTStrategy(
            config=config,
            backtest_engine=engine,
        )
        strategy.fit(large_ohlcv_df.iloc[:800])
        signals = strategy.generate_signals(
            large_ohlcv_df.iloc[800:],
        )
        result = engine.run(signals)

        assert result.metrics["total_trades"] >= 0
        assert "total_return" in result.metrics


# ── Implied volatility (DVOL) tests ─────────────────────────


def _make_implied_config(
    **overrides: object,
) -> GLFTStrategyConfig:
    """Config with vol_type='implied' for fast tests."""
    defaults: dict[str, object] = {
        "gamma": 0.01,
        "kappa": 1.5,
        "ema_window": 10,
        "vol_window": 0,
        "vol_type": "implied",
        "dvol_processed_path": "dummy.parquet",
        "min_holding_bars": 3,
        "max_holding_bars": 15,
        "gamma_candidates": [0.005, 0.01],
        "kappa_candidates": [1.0, 1.5],
        "ema_window_candidates": [10],
        "max_holding_bars_candidates": [15],
        "vol_window_candidates": [10, 20],
    }
    defaults.update(overrides)
    return GLFTStrategyConfig(**defaults)  # type: ignore[arg-type]


class TestGLFTImpliedVolatility:
    def test_compute_volatility_implied_conversion(
        self,
    ) -> None:
        """sigma = dvol / 100 / sqrt(525960)."""
        config = _make_implied_config()
        strategy = GLFTStrategy(config=config)

        dvol = np.array([45.0, 60.0, 30.0])
        close = np.array([40000.0, 40000.0, 40000.0])
        high = close.copy()
        low = close.copy()

        sigma = strategy._compute_volatility(close, high, low, dvol=dvol)

        expected = dvol / 100.0 / np.sqrt(525_960.0)
        np.testing.assert_allclose(sigma, expected, rtol=1e-10)

    def test_compute_volatility_implied_no_dvol_raises(
        self,
    ) -> None:
        """Calling implied vol without dvol array raises."""
        config = _make_implied_config()
        strategy = GLFTStrategy(config=config)

        close = np.array([40000.0])
        high = close.copy()
        low = close.copy()

        with pytest.raises(ValueError, match="dvol array"):
            strategy._compute_volatility(close, high, low)

    def test_generate_signals_with_dvol(
        self,
        sample_ohlcv_with_dvol_df: pd.DataFrame,
    ) -> None:
        """End-to-end signal generation with implied vol."""
        config = _make_implied_config()
        strategy = GLFTStrategy(config=config)
        result = strategy.generate_signals(sample_ohlcv_with_dvol_df)

        assert "signal" in result.columns
        assert not result["signal"].isna().any()
        assert set(result["signal"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_generate_signals_implied_missing_dvol_raises(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        """Missing dvol column raises ValueError."""
        config = _make_implied_config()
        strategy = GLFTStrategy(config=config)

        with pytest.raises(ValueError, match="dvol"):
            strategy.generate_signals(sample_ohlcv_df)

    def test_fit_implied_ignores_vol_window(
        self,
        large_ohlcv_df: pd.DataFrame,
    ) -> None:
        """fit() with implied vol should use [0] for vol_window."""
        # Add a dvol column so generate_signals works
        df = large_ohlcv_df.copy()
        rng = np.random.default_rng(seed=99)
        df["dvol"] = rng.uniform(30.0, 80.0, size=len(df))

        engine = VolumeBacktestEngine(
            init_cash=10_000,
            fees=0.0,
            slippage=0.0,
            position_size_usdt=100.0,
            signal_as_position=True,
        )
        config = _make_implied_config()
        strategy = GLFTStrategy(config=config, backtest_engine=engine)
        strategy.fit(df)
        params = strategy.get_parameters()
        # vol_window should be 0 (the sentinel for implied)
        assert params["best_vol_window"] == 0
