"""Feature-dependent rolling errors are rejected before market data is loaded."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.backtest_service import BacktestService
from tradingdev.app.data_service import DataService
from tradingdev.app.strategy_service import StrategyService
from tradingdev.domain.execution import ExecutionManifest
from tradingdev.domain.strategies.bundled.safety_volume_strategy.config import (
    SafetyVolumeStrategyConfig,
)
from tradingdev.domain.strategies.bundled.xgboost_strategy.config import (
    XGBoostStrategyConfig,
)
from tradingdev.domain.strategies.loader import StrategyLoader
from tradingdev.shared.utils.config import load_config


@pytest.mark.parametrize(
    ("strategy", "lookback", "prior_bars"),
    [("xgboost_strategy", 12, 20), ("safety_volume_strategy", 24, 48)],
)
@pytest.mark.parametrize(
    "entrypoint", ["prepare", "cli_service", "saved_run", "saved_dataset"]
)
def test_impossible_feature_context_fails_before_data_loading(
    monkeypatch: pytest.MonkeyPatch,
    strategy: str,
    lookback: int,
    prior_bars: int,
    entrypoint: str,
) -> None:
    def forbidden_load(*args: object, **kwargs: object) -> None:
        pytest.fail("Offline-invalid rolling geometry must not load market data")

    monkeypatch.setattr(DataService, "load", forbidden_load)
    root = Path(__file__).resolve().parents[2]
    config = load_config(
        root / "src/tradingdev/domain/strategies/bundled" / strategy / "config.yaml"
    )
    config["strategy"]["parameters"].update(
        lookback_candidates=[lookback],
        rolling_gen={"enabled": True, "predict_lookback": prior_bars},
    )
    config["validation"] = {
        "mode": "automatic",
        "n_splits": 2,
        "train_ratio": 0.6,
    }
    with TemporaryDirectory(prefix="tradingdev-rolling-preflight-") as temporary:
        workspace = WorkspacePaths(Path(temporary))
        service = BacktestService(
            data_service=DataService(workspace),
            strategy_loader=StrategyLoader(workspace_root=workspace.root),
            strategy_gate=StrategyService(workspace),
        )
        saved = None
        if entrypoint.startswith("saved_"):
            config_type = (
                XGBoostStrategyConfig
                if strategy == "xgboost_strategy"
                else SafetyVolumeStrategyConfig
            )
            # Simulate a prior version accepting this configuration. Preserve a
            # valid hash and all captured values; rejection must not rely on a
            # tampered manifest or silently repair it with new defaults.
            with monkeypatch.context() as prior_version:
                prior_version.setattr(
                    config_type, "validate_rolling_geometry", lambda _config: None
                )
                captured = service.prepare_execution(config, kind="walk_forward")
            saved = ExecutionManifest.model_validate_json(captured.model_dump_json())
            saved.verify()
        with pytest.raises(ValueError, match="predict_lookback must be at least"):
            if entrypoint == "cli_service":
                service.run_raw_config(config, walk_forward=True)
            elif entrypoint == "prepare":
                service.prepare_execution(config, kind="walk_forward")
            else:
                assert saved is not None
                if entrypoint == "saved_run":
                    service.run_manifest(saved)
                else:
                    service.load_execution_dataset(saved)
