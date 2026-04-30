"""Tests for the persistent disk cache utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np

from tradingdev.domain.backtest.pipeline_result import PipelineResult
from tradingdev.domain.backtest.result import BacktestResult

if TYPE_CHECKING:
    from pathlib import Path

    import pytest
from tradingdev.shared.utils.cache import (
    _code_fingerprint,
    cache_dir,
    clear_cache,
    compute_cache_key,
    load_cached_result,
    save_cached_result,
)


def _make_pipeline() -> PipelineResult:
    result = BacktestResult(
        metrics={"total_return": 0.05},
        equity_curve=np.array([10000.0, 10050.0]),
        trades=[{"net_pnl": 50.0}],
        init_cash=10_000.0,
        mode="signal",
    )
    return PipelineResult(
        mode="simple",
        backtest_result=result,
        config_snapshot={"strategy": {"name": "test"}},
    )


class TestCodeFingerprint:
    """Tests for _code_fingerprint()."""

    def test_returns_16_hex_chars(self) -> None:
        fp = _code_fingerprint()
        assert len(fp) == 16
        assert all(c in "0123456789abcdef" for c in fp)

    def test_deterministic_when_code_unchanged(self) -> None:
        fp1 = _code_fingerprint()
        fp2 = _code_fingerprint()
        assert fp1 == fp2

    def test_fallback_on_git_failure(self) -> None:
        """When git is unavailable, returns random (non-deterministic) value."""
        with patch(
            "tradingdev.shared.utils.cache._run_git",
            return_value=None,
        ):
            fp1 = _code_fingerprint()
            fp2 = _code_fingerprint()
        assert len(fp1) == 16
        assert len(fp2) == 16
        # Two random UUIDs should differ.
        assert fp1 != fp2


class TestComputeCacheKey:
    """Tests for compute_cache_key()."""

    def test_same_inputs_same_key(self, tmp_path: Path) -> None:
        config = tmp_path / "cfg.yaml"
        config.write_text("param: 1")
        data = tmp_path / "data.parquet"
        data.write_bytes(b"fake")

        key1 = compute_cache_key(config, data)
        key2 = compute_cache_key(config, data)
        assert key1 == key2

    def test_config_change_invalidates(self, tmp_path: Path) -> None:
        config = tmp_path / "cfg.yaml"
        data = tmp_path / "data.parquet"
        data.write_bytes(b"fake")

        config.write_text("param: 1")
        key1 = compute_cache_key(config, data)
        config.write_text("param: 2")
        key2 = compute_cache_key(config, data)
        assert key1 != key2

    def test_missing_data_file(self, tmp_path: Path) -> None:
        config = tmp_path / "cfg.yaml"
        config.write_text("param: 1")
        missing = tmp_path / "no_such_file.parquet"
        # Should not raise — just skips data stat.
        key = compute_cache_key(config, missing)
        assert len(key) == 16


class TestSaveLoadRoundtrip:
    """Tests for save/load/clear cache functions."""

    def test_default_cache_dir_uses_workspace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("TRADINGDEV_DATA_ROOT", raising=False)
        monkeypatch.setattr("tradingdev.shared.utils.cache.CACHE_DIR", None)

        assert cache_dir() == tmp_path / "workspace" / "data" / "processed" / "cache"

    def test_data_root_overrides_cache_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data_root = tmp_path / "runtime_data"
        monkeypatch.setenv("TRADINGDEV_DATA_ROOT", str(data_root))
        monkeypatch.setattr("tradingdev.shared.utils.cache.CACHE_DIR", None)

        assert cache_dir() == data_root / "processed" / "cache"

    def test_roundtrip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "tradingdev.shared.utils.cache.CACHE_DIR",
            tmp_path / "cache",
        )
        config = tmp_path / "cfg.yaml"
        config.write_text("param: 1")
        data = tmp_path / "data.parquet"
        data.write_bytes(b"fake")

        pipeline = _make_pipeline()
        path = save_cached_result(pipeline, config, data)
        assert path.exists()

        loaded = load_cached_result(config, data)
        assert loaded is not None
        assert loaded.mode == pipeline.mode
        assert loaded.config_snapshot == pipeline.config_snapshot

    def test_load_returns_none_on_miss(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "tradingdev.shared.utils.cache.CACHE_DIR",
            tmp_path / "cache",
        )
        config = tmp_path / "cfg.yaml"
        config.write_text("param: 1")
        data = tmp_path / "data.parquet"
        data.write_bytes(b"fake")

        assert load_cached_result(config, data) is None

    def test_clear_cache(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cache_dir = tmp_path / "cache"
        monkeypatch.setattr(
            "tradingdev.shared.utils.cache.CACHE_DIR",
            cache_dir,
        )
        config = tmp_path / "cfg.yaml"
        config.write_text("param: 1")
        data = tmp_path / "data.parquet"
        data.write_bytes(b"fake")

        save_cached_result(_make_pipeline(), config, data)
        assert any(cache_dir.glob("*.pkl"))

        count = clear_cache()
        assert count == 1
        assert not any(cache_dir.glob("*.pkl"))
