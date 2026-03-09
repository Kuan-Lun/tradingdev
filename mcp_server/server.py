"""FastMCP server exposing 5 backtesting tools.

Conversational workflow
-----------------------
1. LLM generates strategy Python code + YAML config in the conversation.
2. LLM calls ``save_strategy`` → files written to strategies/ and configs/.
3. LLM confirms logic with the user and asks whether to run.
4. User confirms → LLM calls ``start_backtest`` → worker subprocess spawned,
   job_id returned immediately (non-blocking).
5. User asks for progress → LLM calls ``get_job_status`` → returns status,
   elapsed time, or final metrics when done.

Claude Desktop configuration
-----------------------------
Add the following to
~/Library/Application Support/Claude/claude_desktop_config.json:

{
  "mcpServers": {
    "quant-backtest": {
      "command": "uv",
      "args": ["run", "python", "mcp_server/server.py"],
      "cwd": "/Users/kuanlun_wang/Desktop/git-repo/tradingdev.clone"
    }
  }
}

Strategy template for LLM
--------------------------
LLM-generated strategy files must follow this pattern:

    from __future__ import annotations
    from typing import TYPE_CHECKING, Any
    import pandas as pd
    from quant_backtest.strategies.base import BaseStrategy
    if TYPE_CHECKING:
        from quant_backtest.backtest.base_engine import BaseBacktestEngine

    class MyStrategy(BaseStrategy):
        def __init__(self, backtest_engine: "BaseBacktestEngine") -> None:
            self._engine = backtest_engine

        def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
            result = df.copy()
            result["signal"] = 0  # populate with 1 / -1 / 0
            return result

        def get_parameters(self) -> dict[str, Any]:
            return {}

Corresponding YAML must include:

    strategy:
      name: "my_strategy"
      class: "MyStrategy"                       # Python class name
      file: "strategies/my_strategy.py"        # relative to project root
      description: "..."
    backtest:
      symbol: "BTC/USDT"
      timeframe: "1h"
      start_date: "2024-01-01"
      end_date:   "2024-12-31"
      init_cash:  10000.0       # required for signal mode
      fees:       0.0006
      slippage:   0.0005
      mode:       "signal"
    data:
      source: "binance_api"
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

# Ensure project root and src/ are on sys.path so 'mcp_server' and
# 'quant_backtest' are importable regardless of invocation method.
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
for _p in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import yaml  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.server.fastmcp.server import (  # type: ignore[attr-defined]  # noqa: E402
    TransportSecuritySettings,
)

from mcp_server import job_store  # noqa: E402
from quant_backtest.utils.logger import setup_logger  # noqa: E402

logger = setup_logger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()

mcp = FastMCP(
    name="quant-backtest",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
    instructions="""
You are a quantitative trading strategy assistant backed by a local
backtesting framework (vectorbt).

Available tools
---------------
• get_strategy_template – **call this FIRST** to get BaseStrategy source,
  example code, and YAML template; follow these patterns exactly
• list_available_data   – see what OHLCV data is cached locally
• save_strategy         – persist LLM-generated Python + YAML to disk
• start_backtest        – launch a background backtest (non-blocking)
• get_job_status        – poll progress or retrieve completed results
• list_jobs             – show all past / running jobs

Workflow
--------
Step 1: Call get_strategy_template() to get the reference code.
Step 2: Call list_available_data() to check what data is cached.
Step 3: Write strategy code + YAML following the template exactly.
Step 4: Call save_strategy() to persist files.
Step 5: Show the user the strategy logic and confirm before running.
Step 6: Call start_backtest() and tell the user to wait.
Step 7: When the user asks, call get_job_status() to retrieve results.

IMPORTANT: You MUST call get_strategy_template() before writing any
strategy code. Do NOT guess the class structure or YAML format.
""",
)


# ---------------------------------------------------------------------------
# Helper: lightweight data availability check
# ---------------------------------------------------------------------------
def _check_data_available(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
) -> bool:
    """Return True if all yearly parquet files exist in data/processed/."""
    processed_dir = _PROJECT_ROOT / "data" / "processed"
    if not processed_dir.exists():
        return False

    sym = symbol.replace("/", "").lower()
    try:
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
    except (ValueError, IndexError):
        return False

    for year in range(start_year, end_year + 1):
        complete = processed_dir / f"{sym}_{timeframe}_{year}.parquet"
        partial = processed_dir / f"{sym}_{timeframe}_{year}_partial.parquet"
        if not complete.exists() and not partial.exists():
            return False
    return True


# ---------------------------------------------------------------------------
# Helper: PID liveness check
# ---------------------------------------------------------------------------
def _is_pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


# ---------------------------------------------------------------------------
# Tool 0: get_strategy_template
# ---------------------------------------------------------------------------
@mcp.tool()
def get_strategy_template() -> dict[str, str]:
    """Return reference code and YAML template for writing a new strategy.

    **IMPORTANT**: Always call this tool FIRST before writing any strategy
    code.  It returns the BaseStrategy ABC source, a complete example
    strategy, and a matching YAML config so you can follow the exact
    patterns required by this project.
    """
    base_path = _PROJECT_ROOT / "src" / "quant_backtest" / "strategies" / "base.py"
    base_source = base_path.read_text(encoding="utf-8") if base_path.exists() else ""

    example_code = '''\
"""Example: Simple moving-average crossover strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from quant_backtest.strategies.base import BaseStrategy

if TYPE_CHECKING:
    from quant_backtest.backtest.base_engine import BaseBacktestEngine


class SmaCrossoverStrategy(BaseStrategy):
    """Buy when fast SMA crosses above slow SMA, sell on reverse."""

    def __init__(
        self,
        backtest_engine: BaseBacktestEngine | None = None,
        fast_period: int = 10,
        slow_period: int = 30,
    ) -> None:
        self._engine = backtest_engine
        self._fast_period = fast_period
        self._slow_period = slow_period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        fast = result["close"].rolling(self._fast_period).mean()
        slow = result["close"].rolling(self._slow_period).mean()
        fast_prev = fast.shift(1)
        slow_prev = slow.shift(1)

        result["signal"] = 0
        result.loc[(fast > slow) & (fast_prev <= slow_prev), "signal"] = 1
        result.loc[(fast < slow) & (fast_prev >= slow_prev), "signal"] = -1
        return result

    def get_parameters(self) -> dict[str, Any]:
        return {
            "fast_period": self._fast_period,
            "slow_period": self._slow_period,
        }
'''

    example_yaml = """\
strategy:
  name: "sma_crossover"
  class: "SmaCrossoverStrategy"          # must match Python class name
  file: "strategies/sma_crossover.py"    # relative to project root
  description: "Simple moving-average crossover"
  parameters:
    fast_period: 10
    slow_period: 30

backtest:
  symbol: "BTC/USDT"
  timeframe: "1h"
  start_date: "2024-01-01"
  end_date: "2024-12-31"
  init_cash: 10000.0
  fees: 0.0006
  slippage: 0.0005
  mode: "signal"

data:
  source: "binance_api"
"""

    api_reference = """\
## DataFrame columns (OHLCV)
Input df always has: timestamp, open, high, low, close, volume

## Available imports for strategy code

### Required
from quant_backtest.strategies.base import BaseStrategy

### Optional — built-in indicator
from quant_backtest.indicators.kd import KDIndicator
  KDIndicator(k_period=14, d_period=3, smooth_k=3)
  .calculate(df) → adds 'stoch_k', 'stoch_d' columns

### Optional — logging (use instead of print)
from quant_backtest.utils.logger import setup_logger
  logger = setup_logger(__name__)

### TYPE_CHECKING only (for type hints)
from quant_backtest.backtest.base_engine import BaseBacktestEngine

## pandas methods for computing indicators (no external library needed)
- SMA:  df["close"].rolling(window).mean()
- EMA:  df["close"].ewm(span=period, adjust=False).mean()
- STD:  df["close"].rolling(window).std()
- Shift: series.shift(1)  # previous bar value
- Min/Max: df["low"].rolling(window).min() / df["high"].rolling(window).max()

## YAML config fields
### Required
strategy.name        — snake_case strategy name
strategy.class       — exact Python class name
strategy.file        — "strategies/<name>.py" (relative to project root)
backtest.symbol      — e.g. "BTC/USDT"
backtest.timeframe   — e.g. "1h", "1d", "15m"
backtest.start_date  — "YYYY-MM-DD"
backtest.end_date    — "YYYY-MM-DD"
backtest.init_cash   — required when mode is "signal"
backtest.mode        — "signal" (default)

### Optional
strategy.description — human-readable description
strategy.parameters  — dict of strategy params (for documentation)
backtest.fees        — default 0.0006 (0.06%)
backtest.slippage    — default 0.0005 (0.05%)
backtest.stop_loss   — optional stop-loss ratio
backtest.take_profit — optional take-profit ratio
data.source          — "binance_api" (default)
"""

    return {
        "base_strategy_source": base_source,
        "example_strategy_code": example_code,
        "example_yaml_config": example_yaml,
        "api_reference": api_reference,
        "notes": (
            "1. strategy.class must match the Python class name exactly. "
            "2. strategy.file must be 'strategies/<name>.py'. "
            "3. Constructor should accept backtest_engine as optional kwarg. "
            "4. generate_signals() must add a 'signal' column (1/-1/0) to df. "
            "5. Use pandas built-in methods (rolling, ewm, shift) for indicators — "
            "do NOT import external indicator libraries like ta-lib or pandas-ta. "
            "6. backtest.init_cash is required when mode is 'signal'."
        ),
    }


# ---------------------------------------------------------------------------
# Tool 1: list_available_data
# ---------------------------------------------------------------------------
@mcp.tool()
def list_available_data() -> list[dict[str, Any]]:
    """List OHLCV data cached in data/processed/.

    Returns a list of available symbols with their timeframes and years.
    Call this before start_backtest to verify the required data exists.
    """
    processed_dir = _PROJECT_ROOT / "data" / "processed"
    if not processed_dir.exists():
        return []

    # Filename pattern: {symbol}_{timeframe}_{year}[_partial].parquet
    pattern = re.compile(r"^([a-z0-9]+)_([a-z0-9]+)_(\d{4})(_partial)?\.parquet$")

    grouped: dict[tuple[str, str], set[int]] = {}
    for f in sorted(processed_dir.glob("*.parquet")):
        m = pattern.match(f.name)
        if not m:
            continue
        symbol, timeframe = m.group(1), m.group(2)
        year = int(m.group(3))
        key = (symbol, timeframe)
        grouped.setdefault(key, set()).add(year)

    return [
        {
            "symbol": sym.upper(),
            "timeframe": tf,
            "years_available": sorted(years),
        }
        for (sym, tf), years in sorted(grouped.items())
    ]


# ---------------------------------------------------------------------------
# Tool 2: save_strategy
# ---------------------------------------------------------------------------
@mcp.tool()
def save_strategy(
    name: str,
    code: str,
    yaml_config: str,
) -> dict[str, Any]:
    """Save LLM-generated strategy code and YAML config to disk.

    Args:
        name:        Strategy name in snake_case (e.g. "rsi_reversal").
        code:        Full Python source for the strategy class.
        yaml_config: YAML configuration string.

    Returns:
        {"success": bool, "py_path": str, "yaml_path": str, "error": str | None}
    """

    def _fail(msg: str) -> dict[str, Any]:
        return {"success": False, "py_path": "", "yaml_path": "", "error": msg}

    # Validate name format
    if not re.match(r"^[a-z][a-z0-9_]*$", name):
        return _fail(
            f"Invalid name '{name}': must be lowercase snake_case "
            f"(letters, digits, underscores; must start with a letter)."
        )

    # Validate Python syntax
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return _fail(f"Python syntax error: {exc}")

    # Validate YAML syntax
    try:
        parsed: dict[str, Any] = yaml.safe_load(yaml_config)
    except yaml.YAMLError as exc:
        return _fail(f"YAML parse error: {exc}")

    if not isinstance(parsed, dict):
        return _fail("YAML must be a mapping at the top level.")

    # Validate required YAML fields
    strat_section = parsed.get("strategy", {})
    if not isinstance(strat_section, dict):
        return _fail("YAML 'strategy' must be a mapping.")
    if "class" not in strat_section:
        return _fail("YAML missing required field: strategy.class")
    if "file" not in strat_section:
        return _fail("YAML missing required field: strategy.file")

    bt_section = parsed.get("backtest", {})
    if isinstance(bt_section, dict):
        mode = bt_section.get("mode", "signal")
        if mode == "signal" and bt_section.get("init_cash") is None:
            return _fail("YAML backtest.init_cash is required when mode is 'signal'.")

    # Write files
    strategies_dir = _PROJECT_ROOT / "strategies"
    configs_dir = _PROJECT_ROOT / "configs"
    strategies_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    py_path = strategies_dir / f"{name}.py"
    yaml_path = configs_dir / f"{name}.yaml"

    try:
        py_path.write_text(code, encoding="utf-8")
        yaml_path.write_text(yaml_config, encoding="utf-8")
    except OSError as exc:
        return _fail(f"File write error: {exc}")

    logger.info("Saved strategy '%s' → %s + %s", name, py_path, yaml_path)
    return {
        "success": True,
        "py_path": str(py_path.relative_to(_PROJECT_ROOT)),
        "yaml_path": str(yaml_path.relative_to(_PROJECT_ROOT)),
        "error": None,
    }


# ---------------------------------------------------------------------------
# Tool 3: start_backtest
# ---------------------------------------------------------------------------
@mcp.tool()
def start_backtest(
    strategy_name: str,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    """Launch a background backtest job (non-blocking).

    The worker subprocess handles data download if needed.  Poll progress
    with get_job_status(job_id).

    Args:
        strategy_name: Matches a file in configs/ (e.g. "rsi_reversal").
        symbol:        Trading pair as used by Binance (e.g. "BTC/USDT").
        timeframe:     Candle interval (e.g. "1h", "15m", "1d").
        start_date:    ISO date string "YYYY-MM-DD".
        end_date:      ISO date string "YYYY-MM-DD".

    Returns:
        {"job_id": str, "message": str, "data_available": bool}
    """
    config_path = _PROJECT_ROOT / "configs" / f"{strategy_name}.yaml"
    strategy_file = _PROJECT_ROOT / "strategies" / f"{strategy_name}.py"

    if not config_path.exists():
        return {
            "job_id": "",
            "message": (
                f"Config not found: configs/{strategy_name}.yaml. "
                "Call save_strategy first."
            ),
            "data_available": False,
        }
    if not strategy_file.exists():
        return {
            "job_id": "",
            "message": (
                f"Strategy file not found: strategies/{strategy_name}.py. "
                "Call save_strategy first."
            ),
            "data_available": False,
        }

    data_available = _check_data_available(symbol, timeframe, start_date, end_date)

    job_id = uuid4().hex[:12]
    job_store.create_job(
        job_id=job_id,
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        config_path=str(config_path.relative_to(_PROJECT_ROOT)),
    )

    worker_script = Path(__file__).parent / "backtest_worker.py"
    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, str(worker_script), job_id, str(config_path)],
        cwd=str(_PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # detach from MCP server process group
    )
    job_store.update_job(job_id, pid=proc.pid)
    logger.info("Spawned worker PID=%d for job %s", proc.pid, job_id)

    data_msg = (
        "Data already cached locally."
        if data_available
        else "Data not fully cached — worker will download it automatically."
    )
    return {
        "job_id": job_id,
        "message": f"Backtest started. Job ID: {job_id}. {data_msg}",
        "data_available": data_available,
    }


# ---------------------------------------------------------------------------
# Tool 4: get_job_status
# ---------------------------------------------------------------------------
@mcp.tool()
def get_job_status(job_id: str) -> dict[str, Any]:
    """Return the current status of a backtest job.

    When done, includes the full metrics dict.
    When still running, includes elapsed_seconds and data_downloaded flag.
    When failed, includes the error message.

    Args:
        job_id: The ID returned by start_backtest.
    """
    job = job_store.get_job(job_id)
    if job is None:
        return {"status": "not_found", "error": f"No job with ID: {job_id}"}

    status: str = job["status"]

    # Auto-recover orphaned worker processes
    if status in ("downloading_data", "running_backtest") and not _is_pid_alive(
        job.get("pid")
    ):
        status = "failed"
        job_store.update_job(
            job_id,
            status="failed",
            error="Worker process terminated unexpectedly.",
            end_time=datetime.now(UTC).isoformat(),
        )

    start_time = datetime.fromisoformat(job["start_time"])
    elapsed = round((datetime.now(UTC) - start_time).total_seconds(), 1)

    response: dict[str, Any] = {
        "status": status,
        "strategy_name": job["strategy_name"],
        "symbol": job["symbol"],
        "timeframe": job["timeframe"],
        "start_date": job["start_date"],
        "end_date": job["end_date"],
        "elapsed_seconds": elapsed,
    }

    if status == "done":
        response["metrics"] = job_store.load_result(job["result_path"])
        response["end_time"] = job.get("end_time")
    elif status == "failed":
        response["error"] = job.get("error", "Unknown error")
    else:
        response["data_downloaded"] = job.get("data_downloaded", False)

    return response


# ---------------------------------------------------------------------------
# Tool 5: list_jobs
# ---------------------------------------------------------------------------
@mcp.tool()
def list_jobs() -> list[dict[str, Any]]:
    """List all backtest jobs sorted by start time (newest first).

    Useful for tracking multiple runs or retrieving a forgotten job_id.
    """
    jobs = job_store.list_all_jobs()
    now = datetime.now(UTC)
    summaries = []
    for job in jobs:
        start_time = datetime.fromisoformat(job["start_time"])
        elapsed = round((now - start_time).total_seconds(), 1)
        summaries.append(
            {
                "job_id": job["job_id"],
                "status": job["status"],
                "strategy_name": job["strategy_name"],
                "symbol": job["symbol"],
                "timeframe": job["timeframe"],
                "start_date": job["start_date"],
                "end_date": job["end_date"],
                "elapsed_seconds": elapsed,
                "data_downloaded": job.get("data_downloaded", False),
            }
        )
    return summaries


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quant Backtest MCP Server")
    parser.add_argument(
        "--web",
        action="store_true",
        help="Run as HTTP server (for claude.ai web). Default: stdio.",
    )
    parser.add_argument(
        "--transport",
        choices=["sse", "streamable-http"],
        default="streamable-http",
        help="HTTP transport when --web is set (default: streamable-http)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP port when --web is set (default: 8000)",
    )
    args = parser.parse_args()

    if args.web:
        mcp.settings.port = args.port
        mcp.run(transport=args.transport)
    else:
        mcp.run()
