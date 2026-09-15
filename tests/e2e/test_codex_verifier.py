"""Offline checks for the live-test verifier and timeout cleanup."""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import pytest
import yaml

from tests.e2e.codex_harness import runtime_environment, verify_generated_strategy
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.strategy_service import StrategyService

if TYPE_CHECKING:
    from pytest import MonkeyPatch

_CODE = '''\
"""Deterministic fixture for the isolated verifier."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tradingdev.domain.strategies.base import BaseStrategy

if TYPE_CHECKING:
    import pandas as pd


class FixtureStrategy(BaseStrategy):
    def __init__(self, *, fast_period: int, slow_period: int) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        fast = result["close"].rolling(self.fast_period).mean()
        slow = result["close"].rolling(self.slow_period).mean()
        result["signal"] = 0
        result.loc[fast > slow, "signal"] = 1
        result.loc[fast < slow, "signal"] = -1
        return result

    def get_parameters(self) -> dict[str, Any]:
        return {"fast_period": self.fast_period, "slow_period": self.slow_period}
'''


def _fixture_strategy(root: Path) -> Path:
    workspace = WorkspacePaths(root / "workspace")
    saved = StrategyService(workspace).save_draft(
        "codex_sma_integration",
        _CODE,
        yaml.safe_dump(
            {
                "strategy": {
                    "id": "codex_sma_integration",
                    "class_name": "FixtureStrategy",
                    "parameters": {"fast_period": 5, "slow_period": 20},
                },
            }
        ),
    )
    assert saved.success
    metadata_path = workspace.generated_strategies / "codex_sma_integration.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["status"] = "runnable"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return Path(saved.source_path)


def test_verifier_checks_generated_strategy_in_a_separate_process(
    monkeypatch: MonkeyPatch,
) -> None:
    with TemporaryDirectory(prefix="tradingdev-verifier-") as temporary:
        root = Path(temporary)
        for key, value in runtime_environment(root).items():
            monkeypatch.setenv(key, value)
        _fixture_strategy(root)
        verify_generated_strategy(root)
    assert not root.exists()


def test_verifier_terminates_hanging_generated_code_and_removes_artifacts(
    monkeypatch: MonkeyPatch,
) -> None:
    with TemporaryDirectory(prefix="tradingdev-verifier-") as temporary:
        root = Path(temporary)
        for key, value in runtime_environment(root).items():
            monkeypatch.setenv(key, value)
        source = _fixture_strategy(root)
        marker = root / "verifier.pid"
        source.write_text(
            _CODE
            + "\nimport os\nfrom pathlib import Path\n"
            + f"Path({str(marker)!r}).write_text(str(os.getpid()))\n"
            + "while True:\n    pass\n",
            encoding="utf-8",
        )
        group_signals: list[int] = []

        def deny_group_signal(pid: int, sig: int) -> None:
            group_signals.append(sig)
            raise PermissionError("Simulate restricted macOS process group signals")

        monkeypatch.setattr(os, "killpg", deny_group_signal)
        with pytest.raises(AssertionError, match="verification exceeded"):
            verify_generated_strategy(
                root, timeout_seconds=3, startup_timeout_seconds=30
            )
        assert group_signals == [signal.SIGTERM]
        assert marker.exists(), "Verifier never reached the hanging strategy"
        pid = int(marker.read_text(encoding="utf-8"))
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    assert not root.exists()
