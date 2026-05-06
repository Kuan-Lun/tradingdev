# CLAUDE.md

This file guides Claude Code when working in this repository.

## Project Overview

TradingDev is an MCP-first quantitative strategy development server for crypto
futures research. MCP tools are the primary product entry point. CLI and
dashboard are adapters over the same `tradingdev.app` services. Scope is
historical backtesting, strategy research, optimization, and artifact tracking;
there is no live trading, credential handling, or order placement.

## Commands

```bash
# Install dependencies
uv sync

# Tests
uv run pytest tests/
uv run pytest tests/ -v

# Code quality
./scripts/hooks/finalize-python.sh
./scripts/hooks/finalize-markdown.sh
uv run black src tests scripts
uv run ruff check --fix src tests scripts
uv run mypy src tests scripts

# MCP entry points
uv run tradingdev-mcp
uv run tradingdev-mcp --web --transport streamable-http --port 8000
uv run python -m tradingdev.mcp.server --help

# CLI adapter
uv run python -m tradingdev --config \
  src/tradingdev/domain/strategies/bundled/kd_strategy/config.yaml
uv run python -m tradingdev --config \
  src/tradingdev/domain/strategies/bundled/xgboost_strategy/config.yaml \
  --walk-forward
```

Always use `uv run python`, never bare `python`.

## Project Structure

```text
src/tradingdev/
  mcp/                 FastMCP server, prompts, schemas, thin tool wrappers, workers
  app/                 application services and typed job store facade
  domain/              backtest, data, indicators, strategies, ML, validation
  adapters/            CLI, dashboard, storage, process execution
  shared/utils/        shared logger/config/cache/parallel helpers
workspace/             runtime generated strategies, configs, data, runs, SQLite
tests/                 app/domain/mcp/adapters/integration/shared layered tests
```

## Key Design Decisions

- **Signal convention**: `1` = long, `-1` = short, `0` = flat.
- **No look-ahead bias**: signals may only use data at index `t` and earlier.
- **Strategy parameters in YAML only**: tunable values live in
  `strategy.parameters`.
- **Strategy lifecycle**: `save_strategy` creates draft; `validate_strategy`
  creates validated; `dry_run_strategy` creates runnable; execution accepts only
  runnable or promoted strategies.
- **Generated strategy execution safety**: `validate_strategy` and
  `dry_run_strategy` currently execute generated Python code during contract
  checks. Sandboxed execution isolation is future work and must be addressed
  before treating untrusted code as isolated.
- **Bundled vs generated**: bundled strategy code/config is git-versioned under
  `src/tradingdev/domain/strategies/bundled/`; generated strategy code/config is
  runtime state under `workspace/generated_strategies/` and `workspace/configs/`.
- **Data requirements**: runtime feature inputs are declared in
  `data.requirements`, not inferred from strategy parameter names.
- **Dataset inspection**: `inspect_dataset(config_path)` reports market cache
  availability and declared feature source health from the same data root used
  by backtests.
- **Runtime data cache**: defaults to `workspace/data/raw` and
  `workspace/data/processed`; `TRADINGDEV_DATA_ROOT` may override raw/processed
  data root.
- **Hook wrappers**: `.Codex/hooks/` contains Codex-facing delegation wrappers
  only; canonical hook implementations live in `scripts/hooks/`.
- **Job/run/artifact storage**: metadata lives in `workspace/tradingdev.sqlite`;
  run result, config snapshot, strategy source snapshot, and dataset fingerprint
  files live under `workspace/runs/<run_id>/`. Backtest runs also store a
  `pipeline_result` artifact for the dashboard.
- **Dashboard data source**: dashboard reads `RunService` / `ArtifactService`,
  never old cache files directly.
- **Logging**: use logging helpers, not `print()`.

## Document Maintenance

- Update [ARCHITECTURE.md](ARCHITECTURE.md) when module boundaries, services,
  Pydantic models, Mermaid diagrams, or storage contracts change.
- Update [docs/strategy_contract.md](docs/strategy_contract.md) when generated
  strategy requirements change.
- Update [docs/run_artifacts.md](docs/run_artifacts.md) when job/run/artifact
  schema or workspace layout changes.
- Update [docs/strategies/](docs/strategies/) when bundled strategy interface,
  parameters, or signal logic changes.

## Git Flow

- **main**: merge-only, no direct commits.
- **Feature branches**: one branch per vibe coding session (`feature/<name>`).
- **Merging**: `git merge --no-ff` into main. Delete the feature branch after
  merging with `git branch -d feature/<name>`.
- **Tags**: milestones as `v<major>.<minor>.<patch>`.
- **Commit prefixes**: `feat:` `fix:` `refactor:` `test:` `docs:` `chore:`.

**Always use the git flow script** — never run `git commit` directly:

```bash
scripts/git-flow-commit.sh <branch-suffix> "$(cat <<'EOF'
<prefix>: <title>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)" [files...]
```

The script creates the feature branch, commits, merges `--no-ff` into main,
and deletes the branch atomically. A `pre-commit` hook blocks any attempt to
commit directly to main.
