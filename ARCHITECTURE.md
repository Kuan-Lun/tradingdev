# Architecture Overview

TradingDev is being refactored into an MCP-first quantitative strategy
development server. MCP tools are the main product API. CLI and dashboard
entry points are adapters over the same package, not separate products.

Checkpoint 1 establishes the package boundary:

```text
src/tradingdev/
  mcp/
    server.py
    job_store.py
    workers/
      backtest.py
      optimization.py
  data/
    data_manager.py
    loader.py
    processor.py
    schemas.py
  domain/
    data/crawlers/
    ml/
      base.py
      models/
      features/
      thesis_validator.py
  backtest/
  dashboard/
  indicators/
  strategies/
  validation/
  utils/
```

Root-level `strategies/`, `configs/`, `data/`, and `.backtest_jobs/` are
temporary runtime locations kept for Checkpoint 1 compatibility. Later
checkpoints will move generated artifacts and runtime state under
`workspace/`, split bundled strategy configs from generated configs, and
replace `.backtest_jobs/jobs.json` with SQLite job/run/artifact metadata.

## Package Flow

```mermaid
flowchart TB
    user[LLM or user] --> mcp[tradingdev.mcp.server]
    mcp --> jobs[tradingdev.mcp.job_store]
    mcp --> workers[tradingdev.mcp.workers]
    workers --> data[tradingdev.data.DataManager]
    workers --> strategy[root strategies]
    workers --> engine[tradingdev.backtest engines]
    data --> crawlers[tradingdev.domain.data.crawlers]
    strategy --> ml[tradingdev.domain.ml]
    engine --> result[BacktestResult and metrics]
    workers --> jobs

    cli[uv run tradingdev] --> data
    cli --> strategy
    cli --> engine
    dashboard[Streamlit dashboard] --> cache[data/cache PipelineResult]
```

## Runtime Entrypoints

- MCP stdio: `uv run tradingdev-mcp`
- MCP HTTP: `uv run tradingdev-mcp --web --transport streamable-http --port 8000`
- CLI backtest: `uv run python -m tradingdev.main --config configs/<strategy>.yaml`
- Module MCP launch: `uv run python -m tradingdev.mcp.server`

The MCP server launches background workers with module execution:

```text
python -m tradingdev.mcp.workers.backtest <job_id> <config_path>
python -m tradingdev.mcp.workers.optimization <job_id>
```

The server no longer mutates `sys.path` to import package code. Project-root
runtime paths are resolved from `TRADINGDEV_PROJECT_ROOT` when set, otherwise
from the current working directory.

## Current Domain Components

```mermaid
classDiagram
    direction TB

    class FastMCPServer {
        +list_strategies()
        +get_strategy()
        +get_strategy_template()
        +save_strategy()
        +start_backtest()
        +start_optimization()
        +confirm_optimization()
        +get_job_status()
        +list_jobs()
    }

    class JobRecord {
        <<pydantic>>
        +job_id: str
        +job_type: str
        +status: str
        +strategy_name: str
        +symbol: str
        +timeframe: str
        +config_path: str
        +pid: int | None
        +result_path: str
    }

    class DataManager {
        +load() tuple
        +effective_processed_path: Path
    }

    class BaseCrawler {
        <<abstract>>
        +fetch(symbol, timeframe, start, end) DataFrame
        +save_raw(df, output_path) None
    }

    class BinanceAPICrawler
    class BinanceVisionCrawler
    class BinanceDerivativesCrawler
    class DeribitDVOLCrawler

    class BaseBacktestEngine {
        <<abstract>>
        +run(df) BacktestResult
    }

    class SignalBacktestEngine
    class VolumeBacktestEngine
    class BacktestResult

    class BaseStrategy {
        <<abstract>>
        +fit(df) None
        +generate_signals(df) DataFrame
        +get_parameters() dict
    }

    class BaseModel {
        <<abstract>>
        +train(df, eval_df) None
        +predict(df) Series
        +predict_proba(df) DataFrame
    }

    FastMCPServer --> JobRecord
    FastMCPServer --> DataManager
    FastMCPServer --> BaseBacktestEngine
    DataManager --> BaseCrawler
    BaseCrawler <|-- BinanceAPICrawler
    BaseCrawler <|-- BinanceVisionCrawler
    BaseCrawler <|-- BinanceDerivativesCrawler
    BaseCrawler <|-- DeribitDVOLCrawler
    BaseBacktestEngine <|-- SignalBacktestEngine
    BaseBacktestEngine <|-- VolumeBacktestEngine
    SignalBacktestEngine --> BacktestResult
    VolumeBacktestEngine --> BacktestResult
    BaseStrategy --> BaseBacktestEngine
    BaseModel <|-- XGBoostDirectionModel
    BaseModel <|-- AutoGluonDirectionModel
```

## Checkpoint Notes

- `tradingdev.domain.data.crawlers` owns external data crawlers.
- `tradingdev.domain.ml.models` owns model wrappers.
- `tradingdev.domain.ml.features` owns feature engineering.
- `tradingdev.mcp.job_store.JobRecord` gives the temporary JSON job store a
  typed schema while keeping filesystem storage until the SQLite checkpoint.
- `src/tradingdev/data/schemas.py` is still an aggregate schema module. Schema
  split happens in the next checkpoint.
- Root-level `strategies/registry.py` is still used by the CLI path. Unified
  strategy loading and workspace-generated strategies happen in the next
  checkpoint.
