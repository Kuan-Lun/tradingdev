"""Tests for the persistent disk cache utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from pathlib import Path

    import pytest
from tradingdev.shared.utils.cache import (
    _code_fingerprint,
    cache_dir,
    clear_cache,
    compute_cache_key,
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
        data = tmp_path / "data.parquet"
        data.write_bytes(b"fake")

        key1 = compute_cache_key(manifest_hash="1" * 64, processed_path=data)
        key2 = compute_cache_key(manifest_hash="1" * 64, processed_path=data)
        assert key1 == key2

    def test_manifest_change_invalidates(self, tmp_path: Path) -> None:
        data = tmp_path / "data.parquet"
        data.write_bytes(b"fake")

        key1 = compute_cache_key(manifest_hash="1" * 64, processed_path=data)
        key2 = compute_cache_key(manifest_hash="2" * 64, processed_path=data)
        assert key1 != key2

    def test_data_change_invalidates(self, tmp_path: Path) -> None:
        data = tmp_path / "data.parquet"
        data.write_bytes(b"fake")
        key1 = compute_cache_key(manifest_hash="1" * 64, processed_path=data)
        data.write_bytes(b"longer data fixture")
        key2 = compute_cache_key(manifest_hash="1" * 64, processed_path=data)
        assert key1 != key2

    def test_missing_data_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "no_such_file.parquet"
        # Should not raise — just skips data stat.
        key = compute_cache_key(manifest_hash="1" * 64, processed_path=missing)
        assert len(key) == 16


class TestCacheDirectory:
    """Tests for locating and clearing the cache directory."""

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

    def test_clear_cache(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cache_dir = tmp_path / "cache"
        monkeypatch.setattr(
            "tradingdev.shared.utils.cache.CACHE_DIR",
            cache_dir,
        )
        cache_dir.mkdir()
        (cache_dir / "pipeline.pkl").write_bytes(b"cached pipeline fixture")
        metadata = cache_dir / "unrelated.json"
        metadata.write_text("{}", encoding="utf-8")

        count = clear_cache()
        assert count == 1
        assert not any(cache_dir.glob("*.pkl"))
        assert metadata.exists()
