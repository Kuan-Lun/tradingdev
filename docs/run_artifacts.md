# Run Artifacts

TradingDev stores runtime metadata in `workspace/tradingdev.sqlite` and files
under `workspace/`.

## Workspace Layout

```text
workspace/
  tradingdev.sqlite
  generated_strategies/
  configs/
  data/
    raw/
    processed/
  runs/
    <run_id>/
      result.json
      config.yaml
      strategy.py
      dataset_fingerprint.json
      pipeline_result.pkl
  feature_requests/
```

## SQLite Tables

- `jobs`: generic background job status, job type, pid,
  `created_at`/`started_at`/`ended_at`, error, and payload. Backtest-specific
  fields such as strategy, symbol, timeframe, date range, and config path are
  nullable columns mirrored in the JSON payload when present.
- `runs`: completed run metadata, metrics JSON, config hash, source hash,
  random seed, dataset id, and artifact directory.
- `artifacts`: run and non-run artifact metadata, path, sha256, and metadata JSON.
- `events`: job-scoped structured events.

`job_id` and `run_id` are currently the same for completed backtest and
optimization jobs. `get_job_status(job_id)` returns the `run_id` once a run is
done.

## MCP Lookup

- `list_jobs` / `get_job_status` / `cancel_job`: operational progress and
  cancellation.
- `list_runs` / `get_run`: completed research results.
- `compare_runs`: side-by-side numeric metric comparison.
- `list_artifacts` / `get_artifact`: result JSON and other artifact lookup.
- `promote_strategy`: generated strategy artifact promotion.
- `record_feature_request`: structured unsupported feature requests.

Each completed run writes files under `workspace/runs/<run_id>/` and records
matching SQLite metadata:

- `result.json`: serialized metrics, stored as `result_json`.
- `config.yaml`: effective config snapshot used for the run, stored as
  `config_snapshot` with `config_hash`. For MCP-launched backtest and
  walk-forward jobs this snapshot includes the symbol, timeframe, and date range
  supplied to `start_backtest` / `start_walk_forward`.
- `strategy.py`: generated or bundled strategy source snapshot when
  `strategy.source_path` is available, stored as `strategy_source` with
  source hash metadata. The same hash is indexed in `runs.source_hash` for
  direct SQL lookup.
- `dataset_fingerprint.json`: dataset id, symbol, timeframe, date range, and a
  fingerprint hash, stored as `dataset_fingerprint`.
- `pipeline_result.pkl`: full `PipelineResult` for dashboard rendering, stored
  as `pipeline_result` for completed backtest and walk-forward jobs.

The SQLite `runs.artifact_dir` value must match the corresponding
`workspace/runs/<run_id>/` directory. `runs.random_seed` records an explicit
top-level/backtest `random_seed`, or the unique `random_seed`/`random_state`/
`seed` value found under `strategy.parameters` when one exists. Run comparison,
dashboard rendering, and artifact lookup always read through SQLite first, then
resolve files from the recorded artifact paths.
