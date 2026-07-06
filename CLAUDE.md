# CLAUDE.md

This file guides Claude Code when working in this repository.

## Project Overview

TradingDev is an MCP-first quantitative strategy development server. MCP tools
are the primary product entry point. CLI and dashboard are adapters over the
same `tradingdev.app` services. Scope is historical backtesting, strategy
research, optimization, and artifact tracking; there is no live trading,
credential handling, or order placement. Market data is multi-asset through a
crawler registry (crypto via Binance sources; stocks/futures/indices/FX via
Yahoo Finance); adding a source only requires a new `BaseCrawler` adapter
registered in `domain/data/crawlers/registry.py`.

## Communication

- Claude 必須以繁體中文回答所有對話內容，不論使用者以何種語言提問；程式碼、指令、檔名、專有名詞等仍維持原文。

## Commands

```bash
# Install dependencies (--all-extras: src/tests/scripts includes dashboard
# code, so mypy/black/ruff need streamlit/plotly installed to type-check it
# correctly — a bare `uv sync`/`uv run` drops optional extras)
uv sync --all-extras

# Tests
uv run --all-extras pytest tests/
uv run --all-extras pytest tests/ -v

# Code quality
./scripts/hooks/finalize-python.sh
./scripts/hooks/finalize-markdown.sh
uv run --all-extras black src tests scripts
uv run --all-extras ruff check --fix src tests scripts
uv run --all-extras mypy src tests scripts

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

## Architecture

This project is pre-1.0 and the sections below describe today's design, not a
contract to preserve. If a change intentionally replaces one of these patterns,
update or delete the stale part of this doc in the same change rather than
working around it.

## Design Principles

- Follow SOLID principles: single responsibility, open/closed, Liskov
  substitution, interface segregation, dependency inversion.

## Key Design Decisions

- **Signal convention**: `1` = long, `-1` = short, `0` = flat.
- **No look-ahead bias**: signals may only use data at index `t` and earlier.
- **Strategy parameters in YAML only**: tunable values live in
  `strategy.parameters`.
- **Strategy lifecycle**: `save_strategy` creates draft; `validate_strategy`
  creates validated; `dry_run_strategy` creates runnable; execution accepts only
  runnable or promoted strategies. The gate lives in
  `StrategyService.resolve_executable` and is enforced at execution time by
  `BacktestService`, so MCP jobs, the CLI, and workers cannot bypass it.
  `validate` and `dry_run` share one `SignalContractChecker` run at two fixture
  depths.
- **Generated strategy execution safety**: `validate_strategy` and
  `dry_run_strategy` currently execute generated Python code during contract
  checks. Sandboxed execution isolation is future work and must be addressed
  before treating untrusted code as isolated.
- **Bundled vs generated**: bundled strategy code/config is git-versioned under
  `src/tradingdev/domain/strategies/bundled/`; generated strategy code/config is
  runtime state under `workspace/generated_strategies/` and `workspace/configs/`.
- **Data requirements**: runtime feature inputs are declared in
  `data.requirements`, not inferred from strategy parameter names.
- **Data source selection**: `data.requirements.market.source` picks the
  crawler from `domain/data/crawlers/registry.py` (`binance_vision` default,
  `binance_api`, `yahoo_finance`); when omitted it inherits legacy
  `data.source`. `DataManager` takes a `MarketDataRequest` plus an injected
  crawler and does not depend on `domain.backtest`.
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

## Code Style

- **Sync obligation for tooling configuration:** the IDE save pipeline and the
  Stop hook pipeline are kept in lockstep across the locations below. Any
  change to one of them requires matching updates to the others in the same
  change.
  - Python formatting/lint/type-check:
    [.vscode/settings.json](.vscode/settings.json) (`[python]` block),
    [mypy.ini](mypy.ini) (strict mode), the `[lint]` section of
    [ruff.toml](ruff.toml), all auto-discovered by both the IDE and
    `uv run`, and the shared implementation at
    [scripts/hooks/finalize-python.sh](scripts/hooks/finalize-python.sh),
    registered as a Claude Stop hook in
    [.claude/settings.local.json](.claude/settings.local.json).
  - Markdown formatting: [.vscode/settings.json](.vscode/settings.json)
    (`[markdown]` block), the shared implementation at
    [scripts/hooks/finalize-markdown.sh](scripts/hooks/finalize-markdown.sh),
    and the same Claude Stop-hook registration in
    [.claude/settings.local.json](.claude/settings.local.json).
  - Tool versions: the `dev` group of `[dependency-groups]` in
    [pyproject.toml](pyproject.toml) pins `black`, `ruff`, `mypy`, and
    `pymarkdownlnt`. Both the IDE pipeline (when invoked via `uv run`) and the
    Stop-hook scripts resolve to these venv-installed versions, so bumping any
    of them must be done here — not via Homebrew or any other system-wide
    install.
- Ruff's `E2xx` whitespace rules (e.g. `E271`/`E272`
  multiple-spaces-before/after-keyword) are preview-only in this Ruff version
  and stay off even with `select = ["E", ...]` unless `preview = true` is set.
  Don't be surprised if the CLI/hook misses a whitespace nit that an IDE
  extension flags separately.
- Python version range: refer to `requires-python` in
  [pyproject.toml](pyproject.toml)
- **Comments:** default to none. Only add one when the *why* isn't obvious
  from the code itself (a hidden constraint, a non-obvious invariant, a
  workaround for a specific bug). Never frame a comment around the current
  change, refactor, or task ("moved here for X", "changed from Y to Z",
  "added for the Z flow") — write it as a timeless statement of the
  constraint, since that context rots as the codebase evolves but the
  underlying constraint doesn't. Prefer one line over a multi-line block.

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
