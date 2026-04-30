"""Strategy lifecycle service tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.strategy_service import StrategyService

if TYPE_CHECKING:
    from pathlib import Path

_STRATEGY_CODE = """\
from __future__ import annotations

from typing import Any

import pandas as pd

from tradingdev.domain.strategies.base import BaseStrategy


class FixtureStrategy(BaseStrategy):
    def __init__(
        self,
        backtest_engine: object | None = None,
        threshold: float = 0.0,
    ) -> None:
        self._engine = backtest_engine
        self._threshold = threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["signal"] = 0
        moves = result["close"].pct_change().fillna(0)
        result.loc[moves > self._threshold, "signal"] = 1
        return result

    def get_parameters(self) -> dict[str, Any]:
        return {"threshold": self._threshold}
"""

_YAML = """\
strategy:
  id: "fixture_strategy"
  version: "0.1.0"
  class_name: "FixtureStrategy"
  source_path: "workspace/generated_strategies/fixture_strategy.py"
  parameters:
    threshold: 0.0
backtest:
  symbol: "BTC/USDT"
  timeframe: "1h"
  start_date: "2024-01-01"
  end_date: "2024-01-31"
  init_cash: 10000.0
data:
  source: "binance_api"
  requirements:
    market:
      source: "binance_api"
      symbol: "BTC/USDT"
      timeframe: "1h"
    features: []
"""


def test_strategy_service_lifecycle(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    service = StrategyService(workspace)
    service._quality_gate_diagnostics = lambda _path: []  # type: ignore[assignment,method-assign]

    saved = service.save_draft("fixture_strategy", _STRATEGY_CODE, _YAML)

    assert saved.success is True
    assert saved.status == "draft"
    validated = service.validate("fixture_strategy")
    assert validated["success"] is True
    assert validated["status"] == "validated"
    assert validated["signal_analysis"]["rows"] == 80
    dry_run = service.dry_run("fixture_strategy")
    assert dry_run["success"] is True
    assert dry_run["status"] == "runnable"
    assert dry_run["signal_analysis"]["transition_count"] >= 1
    promoted = service.promote("fixture_strategy")
    assert promoted == {
        "success": True,
        "strategy_id": "fixture_strategy",
        "status": "promoted",
    }


def test_list_strategies_includes_requirements_and_recent_runs(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    service = StrategyService(workspace, store=store)
    assert service.save_draft("fixture_strategy", _STRATEGY_CODE, _YAML).success
    store.create_run(
        run_id="run_a",
        job_id="job_a",
        strategy_id="fixture_strategy",
        artifact_dir=workspace.runs / "run_a",
        metrics={"total_return": 0.1},
        dataset_id="dataset-a",
    )

    item = next(
        item
        for item in service.list_strategies()
        if item["strategy_id"] == "fixture_strategy"
    )

    assert item["data_requirements"]["market"]["symbol"] == "BTC/USDT"
    assert item["recent_runs"][0]["run_id"] == "run_a"
    assert item["recent_runs"][0]["metrics"]["total_return"] == 0.1


def test_strategy_service_rejects_banned_import(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    service = StrategyService(workspace)
    service._quality_gate_diagnostics = lambda _path: []  # type: ignore[assignment,method-assign]

    code = "import os\n" + _STRATEGY_CODE
    assert service.save_draft(
        "bad_strategy", code, _YAML.replace("fixture_strategy", "bad_strategy")
    ).success

    validated = service.validate("bad_strategy")

    assert validated["success"] is False
    messages = [item["message"] for item in validated["diagnostics"]]
    codes = [item["code"] for item in validated["diagnostics"]]
    assert "banned import: os" in messages
    assert "banned_import" in codes


def test_strategy_service_validate_reports_syntax_errors(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    service = StrategyService(workspace)
    saved = service.save_draft("syntax_strategy", _STRATEGY_CODE, _YAML)
    assert saved.success is True
    source_path = workspace.generated_strategies / "syntax_strategy.py"
    source_path.write_text("def broken(:\n", encoding="utf-8")

    validated = service.validate("syntax_strategy")

    assert validated["success"] is False
    assert [item["code"] for item in validated["diagnostics"]] == ["syntax_error"]


def test_strategy_service_rejects_non_allowlisted_import(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    service = StrategyService(workspace)
    service._quality_gate_diagnostics = lambda _path: []  # type: ignore[assignment,method-assign]

    code = _STRATEGY_CODE.replace(
        "import pandas as pd",
        "import talib\n\nimport pandas as pd",
    )
    assert service.save_draft(
        "bad_import_strategy",
        code,
        _YAML.replace("fixture_strategy", "bad_import_strategy"),
    ).success

    validated = service.validate("bad_import_strategy")

    assert validated["success"] is False
    codes = [item["code"] for item in validated["diagnostics"]]
    assert "import_not_allowed" in codes


def test_strategy_service_rejects_invalid_signal_values(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    service = StrategyService(workspace)
    service._quality_gate_diagnostics = lambda _path: []  # type: ignore[assignment,method-assign]

    code = _STRATEGY_CODE.replace(
        'result.loc[moves > self._threshold, "signal"] = 1',
        'result.loc[moves > self._threshold, "signal"] = 2',
    )
    assert service.save_draft(
        "bad_signal_strategy",
        code,
        _YAML.replace("fixture_strategy", "bad_signal_strategy"),
    ).success

    validated = service.validate("bad_signal_strategy")

    assert validated["success"] is False
    codes = [item["code"] for item in validated["diagnostics"]]
    assert "invalid_signal_values" in codes
    assert validated["signal_analysis"]["rows"] == 80


def test_strategy_service_rejects_input_mutation(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    service = StrategyService(workspace)
    service._quality_gate_diagnostics = lambda _path: []  # type: ignore[assignment,method-assign]

    code = _STRATEGY_CODE.replace("result = df.copy()", "result = df")
    assert service.save_draft(
        "mutating_strategy",
        code,
        _YAML.replace("fixture_strategy", "mutating_strategy"),
    ).success

    validated = service.validate("mutating_strategy")

    assert validated["success"] is False
    codes = [item["code"] for item in validated["diagnostics"]]
    assert "input_mutated" in codes
