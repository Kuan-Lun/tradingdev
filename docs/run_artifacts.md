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
  feature_requests/
```

## SQLite Tables

- `jobs`: background job status, strategy, config path, pid,
  `created_at`/`started_at`/`ended_at`, and payload.
- `runs`: completed run metadata, metrics JSON, config hash, dataset id, and
  artifact directory.
- `artifacts`: run and non-run artifact metadata, path, sha256, and metadata JSON.
- `events`: job-scoped structured events.

`job_id` and `run_id` are currently the same for completed backtest and
optimization jobs. `get_job_status(job_id)` returns the `run_id` once a run is
done.

## MCP Lookup

- `list_jobs` / `get_job_status`: operational progress.
- `list_runs` / `get_run`: completed research results.
- `compare_runs`: side-by-side numeric metric comparison.
- `list_artifacts` / `get_artifact`: result JSON and other artifact lookup.
- `record_feature_request`: structured unsupported feature requests.

Each completed run writes files under `workspace/runs/<run_id>/` and records
matching SQLite metadata:

- `result.json`: serialized metrics, stored as `result_json`.
- `config.yaml`: config snapshot used for the run, stored as `config_snapshot`
  with `config_hash`.
- `strategy.py`: generated or bundled strategy source snapshot when
  `strategy.source_path` is available, stored as `strategy_source` with
  source hash metadata.
- `dataset_fingerprint.json`: dataset id, symbol, timeframe, date range, and a
  fingerprint hash, stored as `dataset_fingerprint`.

The SQLite `runs.artifact_dir` value must match the corresponding
`workspace/runs/<run_id>/` directory. Run comparison and artifact lookup always
read through SQLite first, then resolve files from the recorded artifact paths.
