# Architecture Overview

TradingDev 的主要產品邊界是 MCP tools。MCP、CLI 與 dashboard 不各自組回測流程，
而是呼叫 `tradingdev.app` services；domain 層保存交易、資料、策略與模型邏輯；
adapters 層負責 FastMCP、CLI、dashboard、SQLite、filesystem 與 subprocess。

## Module Layout

```text
src/tradingdev/
  mcp/
    server.py
    schemas.py
    tools/
    workers/
  app/
    strategy_service.py
    data_service.py
    backtest_service.py
    optimization_service.py
    job_service.py
    run_service.py
    artifact_service.py
    feature_request_service.py
    capability_service.py
    job_store.py
  domain/
    strategies/
    backtest/
    data/
    indicators/
    ml/
    validation/
  adapters/
    cli/
    dashboard/
    execution/
    storage/
  shared/
    utils/
```

## Runtime Flow

```mermaid
flowchart TB
    user[LLM or user] --> mcp[FastMCP server]
    mcp --> tools[mcp.tools thin wrappers]
    tools --> app[application services]
    cli[CLI adapter] --> app
    dashboard[Dashboard adapter] --> app

    app --> loader[StrategyLoader]
    app --> data[DataService]
    app --> backtest[BacktestService]
    app --> jobs[JobService]
    app --> runs[RunService]
    app --> artifacts[ArtifactService]

    loader --> bundled[bundled strategies]
    loader --> generated[workspace generated strategies]
    data --> manager[domain.data.DataManager]
    data --> requirements[DataRequirement]
    backtest --> engines[domain.backtest engines]
    jobs --> workers[mcp.workers subprocesses]
    jobs --> sqlite[(workspace/tradingdev.sqlite)]
    runs --> sqlite
    artifacts --> sqlite
    artifacts --> files[workspace/runs and workspace artifacts]
```

## Strategy Lifecycle

```mermaid
stateDiagram-v2
    [*] --> draft: save_strategy
    draft --> validated: validate_strategy
    draft --> draft: validation failed
    validated --> runnable: dry_run_strategy
    runnable --> promoted: promote_strategy
    promoted --> promoted: bundled strategies start here
    runnable --> running: start_backtest/start_walk_forward
    promoted --> running: start_backtest/start_walk_forward
    running --> done
    running --> failed
```

Generated strategies must live in `workspace/generated_strategies/`. Bundled
strategies live next to their git-versioned configs under
`src/tradingdev/domain/strategies/bundled/`.

## Storage

```mermaid
classDiagram
    class SQLiteStore {
        +upsert_job(record)
        +get_job(job_id)
        +list_jobs()
        +create_run(...)
        +list_runs()
        +get_run(run_id)
        +create_artifact(...)
        +list_artifacts(run_id)
    }

    class WorkspacePaths {
        +generated_strategies
        +configs
        +runs
        +feature_requests
        +raw_data
        +processed_data
    }

    class JobRecord {
        <<pydantic>>
        +job_id
        +status
        +job_type
        +strategy_name
        +config_path
        +result_path
    }

    SQLiteStore --> JobRecord
    SQLiteStore --> WorkspacePaths
```

SQLite stores metadata. Filesystem stores generated code/config, result JSON,
feature requests and data caches. `workspace/runs/<run_id>/` is linked from the
`runs.artifact_dir` column.

## Domain Contracts

- Strategy signal convention: `1` long, `-1` short, `0` flat.
- Strategy parameters live in YAML `strategy.parameters`.
- Data requirements live in YAML `data.requirements`.
- `start_backtest` rejects configs with `validation:`; use `start_walk_forward`.
- Generated strategies must pass static policy checks before execution.
- Runtime cache defaults to `workspace/data/`; `TRADINGDEV_DATA_ROOT` can override
  raw/processed data root.
