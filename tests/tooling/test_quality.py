"""Exercise the configured Mypy boundary with real accepted and rejected types."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from tooling.process_guard import process_workspace, run_captured

ROOT = Path(__file__).resolve().parents[2]
CONCRETE_MODULES = (
    "src/tradingdev/domain/indicators/technical.py",
    "src/tradingdev/domain/optimization/grid_search.py",
    "src/tradingdev/domain/ml/models/registry.py",
    "src/tradingdev/domain/ml/validation/schemes.py",
)


@pytest.mark.parametrize(
    "source, rejected",
    [
        (
            "import pandas as pd\n"
            "def first(values: pd.Series[float]) -> float:\n"
            "    return values.iloc[0]\n"
            "def opaque(value: object) -> object:\n"
            "    return value\n",
            False,
        ),
        (
            "from typing import Any\n"
            "def consume(value: Any) -> None:\n"
            "    value.nonexistent_method()\n",
            True,
        ),
        (
            "import pandas as pd\n"
            "def consume(values: pd.Series) -> None:\n"
            "    values.iloc[0].nonexistent_method()\n",
            True,
        ),
    ],
    ids=["concrete-and-opaque", "explicit-any", "series-default-any"],
)
def test_concrete_module_policy(source: str, rejected: bool) -> None:
    # Shadow the real paths so Mypy resolves the configured module names without
    # modifying the checkout or replacing the policy with test-only flags.
    with process_workspace(
        prefix="tradingdev-type-policy-", label="Mypy probe"
    ) as root:
        probe = root / "probe.py"
        probe.write_text(source, encoding="utf-8")
        shadows = [
            argument
            for module in CONCRETE_MODULES
            for argument in ("--shadow-file", module, str(probe))
        ]
        result = run_captured(
            [
                sys.executable,
                "-m",
                "mypy",
                "--config-file=pyproject.toml",
                "--no-incremental",
                f"--cache-dir={os.devnull}",
                *shadows,
                *CONCRETE_MODULES,
            ],
            cwd=ROOT,
            env=os.environ.copy(),
            timeout=120,
        )
        assert result.returncode == int(rejected), result.stdout + result.stderr
        if rejected:
            errors = [line for line in result.stdout.splitlines() if ": error:" in line]
            assert len(errors) == len(CONCRETE_MODULES), result.stdout
            assert all("[explicit-any]" in line for line in errors), result.stdout
            assert {line.split(":", 1)[0] for line in errors} == set(CONCRETE_MODULES)
    assert not root.exists()
