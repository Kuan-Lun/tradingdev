# Run Artifacts

TradingDev stores runtime metadata in `workspace/tradingdev.sqlite` and files
under `workspace/`.

## Workspace Layout

```text
workspace/
  tradingdev.sqlite
  .workers/
    <launch_token>/
      start.json
      ready.json
      stop
      finished.json
  generated_strategies/
    <strategy_id>/
      current.json
      revisions/
        <revision_id>/
          strategy.py
          config.yaml
          metadata.json
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
  nullable columns mirrored in the JSON payload when present. The payload also
  records `process_create_time`, the supervisor's OS creation time in Unix seconds,
  and `worker_control_id`, its unique launch token. The PID belongs to the
  supervisor, not the strategy-executing child.
- `runs`: completed run metadata, metrics JSON, config hash, source hash,
  random seed, dataset id, artifact directory, and selected `revision_id`.
- `artifacts`: run and non-run artifact metadata, path, sha256, and metadata JSON.
- `events`: job-scoped structured events.

`job_id` and `run_id` are currently the same for completed backtest and
optimization jobs. `get_job_status(job_id)` returns the `run_id` once a run is
done. Generated strategy jobs and runs carry their selected `revision_id`;
bundled strategies and historical records without revision identity expose
`null`. A later save changes only the current pointer and does not change the
revision used by an existing job or completed run.

`.workers/<launch_token>` stores the startup identity, worker startup response,
optional cancellation request (`stop`), and final cleanup acknowledgement. These
small records remain in a regular workspace so later sessions can verify cleanup
without trusting a potentially reused PID. Test workspaces remove them with the
rest of the temporary workspace after verified cleanup.

Cancellation of an active job without a complete control identity fails without
signalling or claiming that the process stopped. A queued job that has not started
a process can be cancelled directly. Status checks mark a started job failed when
its supervisor identity is missing or no longer matches; the first response
includes the same error persisted in the database. Spawn failures also persist
`failed`, an explanatory error and `ended_at` before propagating the exception.

## MCP Lookup

- `list_jobs` / `get_job_status` / `cancel_job`: operational progress and
  cancellation.
- `list_runs` / `get_run`: completed research results.
- `compare_runs`: side-by-side numeric metric comparison.
- `list_artifacts` / `get_artifact`: result JSON and other artifact lookup.
- `promote_strategy`: generated strategy artifact promotion.
- `record_feature_request`: structured unsupported feature requests.

The generated revision's `config.yaml` holds its base settings;
`runs/<run_id>/config.yaml` holds effective execution settings and includes
`strategy.revision_id`. Runtime market/date or optimization parameter overrides
do not rewrite the base revision. Generated backtest and walk-forward parameters
must match the base revision; only optimization search parameters can vary.
Revision identity does not freeze imported
Python dependencies or market data and is not a complete execution manifest.

Each completed background job writes files under `workspace/runs/<run_id>/` and records
matching SQLite metadata:

- `result.json`: serialized metrics, stored as `result_json`.
- `config.yaml`: effective config snapshot used for the run, stored as
  `config_snapshot` with `config_hash`. Backtest/walk-forward snapshots come
  from the executed `PipelineResult.config_snapshot`; optimization supplies
  the config held by its worker. Completion does not reload a potentially
  changed config file for these execution paths. For MCP-launched backtest and
  walk-forward and optimization jobs this snapshot includes the symbol, timeframe,
  and date range supplied to the start tool. Optimization includes the entire
  final calendar day; ordinary backtest bounds remain timestamps.
- `strategy.py`: generated or bundled strategy source snapshot when
  `strategy.source_path` is available, stored as `strategy_source` with
  source hash metadata. Generated source snapshots come from the selected
  immutable revision, not whichever revision is current at completion.
  The same hash is indexed in `runs.source_hash` for
  direct SQL lookup.
- `dataset_fingerprint.json`: dataset id, symbol, timeframe, date range, and a
  fingerprint hash, stored as `dataset_fingerprint`.
- `pipeline_result.pkl`: full `PipelineResult` for dashboard rendering, stored
  as `pipeline_result` for completed backtest and walk-forward jobs.

For background jobs, SQLite `runs.artifact_dir` must match the corresponding
`workspace/runs/<run_id>/` directory. `runs.config_hash` is the SHA-256 of the
serialized effective config snapshot, not the hash of the immutable revision's
base config. `runs.random_seed` records an explicit
top-level/backtest `random_seed`, or the unique `random_seed`/`random_state`/
`seed` value found under `strategy.parameters` when one exists. Run comparison,
dashboard rendering, and artifact lookup always read through SQLite first, then
resolve files from the recorded artifact paths.

CLI runs instead store their `pipeline_result` cache under
`workspace/data/processed/cache` (or `$TRADINGDEV_DATA_ROOT/processed/cache`),
with that directory as `runs.artifact_dir`. Their config snapshot is embedded in
the pipeline result. The run's config hash and revision identity come from this
executed snapshot. Before caching, the CLI checks the requested config against
the executed config (excluding the service-managed `source_hash`); a change to
strategy identity, dates, costs, or other settings rejects caching. It does not
associate an old result with a changed config, even for the same revision.

Optimization `result.json` includes the best parameters, training metrics and
out-of-sample metrics. For newly saved results, its parsed JSON matches the
`metrics` object in the run record returned by `get_run`, not the entire MCP
response. Parameters outside the search grid retain their YAML values; selection
uses training results, and only the selected parameters are evaluated out of
sample.

Before writing `result.json` or SQLite `runs.metrics`, metric values are
recursively normalized: `NaN`, `Infinity`, and `-Infinity` become JSON `null`,
including values inside nested objects and arrays such as optimization metrics.

Legacy results are also normalized in memory when `JobStore.load_result` reads a
result file or `SQLiteStore` reads run metrics. These reads do not rewrite the
stored file or database row. Consequently, `get_run`, `list_runs`, and completed
`get_job_status` responses expose legacy non-finite metric values as `null`.

In contrast, `get_artifact(include_content=true)` returns the original UTF-8 file
text without parsing or normalizing its contents. A legacy `result.json` may
therefore still contain `NaN` or `Infinity` even when result lookup returns `null`;
the content-equivalence statement above does not apply to these legacy artifacts.
The CLI displays missing or non-finite metrics as `N/A`, while keeping finite zero
values visible as zero.
