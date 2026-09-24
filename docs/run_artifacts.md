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
      manifest.json
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
  supervisor, not the strategy-executing child. New submissions also record
  `manifest_hash`, the expected digest of their fixed execution specification.
- `runs`: completed run metadata, metrics JSON, config hash, source hash,
  random seed, dataset id, artifact directory, selected `revision_id`, and
  execution `manifest_hash`.
- `artifacts`: run and non-run artifact metadata, path, sha256, and metadata JSON.
- `events`: job-scoped structured events.

`job_id` and `run_id` are currently the same for completed backtest and
optimization jobs. `get_job_status(job_id)` returns the `run_id` once a run is
done. Generated strategy jobs and runs carry their selected `revision_id`;
bundled strategies and historical records without revision identity expose
`null`. A later save changes only the current pointer and does not change the
revision used by an existing job or completed run.

Successful `start_backtest`, `start_walk_forward`, and `start_optimization`
responses include the required `manifest_hash`. Job status, job lists, and run
records carry the same digest; historical entries without an execution manifest
return `null`. Historical results remain readable, but old jobs cannot resume
through the new workers or proceed through optimization confirmation. Submit a
new job to execute them under a fixed specification.

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

## Execution Manifest

Before spawning a background worker, the service publishes
`runs/<job_id>/manifest.json` and records its hash in the job. Publication never
replaces a different existing manifest. Atomic publication applies to the manifest
file; file publication and the database job write do not form one transaction.
Schema version 2 contains:

- `kind`: `backtest`, `walk_forward`, or `optimization`.
- `config`: the effective strategy identity and source hash, backtest settings,
  optional walk-forward settings, parallel policy, seed settings, and data
  requirements with resolved absolute directories and feature paths. Typed
  execution defaults are materialized and dates are JSON strings.
- `strategy_execution`: the strategy kind (`bundled` or `generated`) and
  `constructor_kwargs`. Bundled `config` and optional `fit_config` objects include
  nested model defaults; generated keyword arguments include declared constructor
  defaults. Injected engine and parallel policy objects remain outside this JSON
  object and are supplied from the manifest's execution configuration.
- `optimization`: `null` for other kinds; otherwise the parameter ranges,
  optimization metric, `train_start`/`train_end`/`test_start`/`test_end`,
  `direction: maximize`, `trial_timeout_seconds: 300`,
  `confirmation_timeout_seconds: 1800`, and `confirmation_poll_interval: 2.0`.
  These timeout and polling values are fixed at submission. Parameter names are
  sorted; the caller's candidate order within each range is preserved.
- `manifest_hash`: SHA-256 of canonical JSON over all the preceding fields and
  `schema_version: 2`, excluding the hash field itself. Object keys are sorted;
  job IDs and creation timestamps are not part of the specification.

Typed backtest, walk-forward, parallel, and data defaults have the same digest
whether implicit or explicit. The strategy mapping is retained under its revision
contract. Its effective values are expanded separately into `strategy_execution`,
so filling defaults does not modify the validated revision's original declaration.
Because the declaration is also retained, omitted and explicit strategy parameters
can still produce different digests even when their effective values are equal.
Nested optimization candidates override only their specified leaves, preserving
the remaining fixed values; candidate structure and bundled model contracts are
validated before creating a job, without running strategy constructors. Constructor
or execution errors can still occur during the trial. Unsupported defaults or
candidates requiring new implicit fields are rejected. Integer candidates for
float fields are accepted only when conversion preserves their numeric value;
string/boolean coercion and precision loss are rejected.
Non-finite numbers in execution configuration or search values are rejected instead of
being converted to `null`. Invalid resolved execution settings return
`invalid_execution_request` for backtest/walk-forward submissions or
`invalid_optimization_request` for optimization, without creating a job.
These codes also cover strategy identity mismatches, revision-file read failures
during binding, and failures of the repeated executable-state check during
preparation. Failure to select an executable strategy at the start of the request
instead returns `strategy_not_executable`.
The mode must match the presence of walk-forward
validation settings. Optimization has its own training/test split and rejects
configs containing `validation` with `invalid_optimization_request`.

Manifest decoding preserves stored JSON without reapplying today's config defaults.
Before execution, current runtime models must accept the saved config and effective
strategy settings without adding fields or changing values. New required/defaulted
fields or incompatible normalization cause execution to fail, not reinterpret the
request. Schema version 2 requires the version and effective strategy fields;
version 1 or unknown versions require a new submission. No manifest is migrated
in place, and historical result metadata remains readable.

Workers receive only a job ID and load this fixed manifest path. They verify the
supported schema, content digest, expected job hash, and strategy/mode identity
before execution. Optimization confirmation and result persistence verify the
same binding. Missing, modified, or unsupported manifests fail execution;
confirmation returns `execution_manifest_invalid`. Changing the original YAML,
the current strategy pointer, or the neighboring `config.yaml` does not redefine
the submitted request. A changed request requires a new job.

`config.yaml` is a readable projection of `manifest.config`; workers do not use
it as their source of execution settings. `config_hash` still hashes the YAML
projection (including the original strategy declaration); resolved constructor
settings are available in the manifest's `strategy_execution` object.
`manifest_hash` covers the complete execution request. The
artifact's `sha256` hashes the actual `manifest.json` file bytes, including its
own `manifest_hash` field, and therefore is a separate digest.

This specification fixes the request, not the contents of referenced data files,
installed dependencies, engine environment, or RNG state. Absolute paths and
fingerprints do not make those inputs immutable or guarantee identical results
on a later run. Integrity verification assumes the job's expected digest remains
trusted; it is not a security boundary against someone controlling both the
database and workspace files.

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
The execution manifest additionally fixes runtime overrides and search settings
without rewriting the base revision.

Each completed background job writes files under `workspace/runs/<run_id>/` and records
matching SQLite metadata:

- `result.json`: serialized metrics, stored as `result_json`.
- `manifest.json`: the specification published at submission, registered after
  successful result persistence as `execution_manifest`. Its artifact metadata
  includes `manifest_hash` and `schema_version`. Use `list_artifacts` and
  `get_artifact(include_content=true)` to inspect the JSON after completion.
- `config.yaml`: effective config snapshot used for the run, stored as
  `config_snapshot` with `config_hash`. It projects the executed manifest config;
  persistence checks the worker's snapshot against that manifest and the job's
  digest rather than reloading a possibly changed YAML file. It includes the
  start tool's symbol, timeframe, and date range together with resolved defaults.
  Optimization includes the entire final calendar day; ordinary backtest bounds
  remain timestamps.
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
while `runs.artifact_dir` points to `workspace/runs/cli_<cache_key>/`, where the
execution manifest is published and registered as an artifact. The pipeline
result embeds both its config snapshot and execution manifest. The run's config
hash, manifest hash, and revision identity come from the executed specification.
Saving uses this manifest without re-reading the original config file; editing
that file after execution does not prevent saving the original result. The
original `config_path` remains descriptive artifact metadata.

CLI cache identity combines the manifest hash, processed-data file size/mtime,
and a Git source-code fingerprint. New settings produce a different manifest and
therefore a different cache key. These file-stat and code fingerprints help
invalidate caches; they are not immutable data or environment versions and do
not guarantee full reproducibility.

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
