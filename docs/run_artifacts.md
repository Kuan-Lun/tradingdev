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
  feature_requests/
```

## SQLite Tables

- `jobs`: background job status, strategy, config path, pid, timing, and payload.
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

Each completed run writes `workspace/runs/<run_id>/result.json` and records a
matching `result_json` artifact with sha256.
