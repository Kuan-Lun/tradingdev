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
    .locks/
      <strategy_id>.lock
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
      performance.json
      observations.json
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

Draft cleanup scans raw job identities and payloads, run references and all stored
manifests, including manifests left without a database job. Failed and cancelled
jobs still retain their source references. A malformed or ambiguous history blocks
cleanup. Only explicitly selected non-current drafts without references can be
deleted; runnable sources and execution artifacts remain. Cleanup does not dedupe
or migrate old revisions. The per-strategy `.locks` files coordinate saves,
validation, status updates and cleanup; they stay outside deletable revision
directories and are never unlinked while another process might use their inode.

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
Schema version 4 contains:

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
  catalog-defined `direction: maximize|minimize`, `trial_timeout_seconds: 300`,
  `confirmation_timeout_seconds: 1800`, and `confirmation_poll_interval: 2.0`.
  These timeout and polling values are fixed at submission. Parameter names are
  sorted; the caller's candidate order within each range is preserved.
- `manifest_hash`: SHA-256 of canonical JSON over all the preceding fields and
  `schema_version: 4`, excluding the hash field itself. Object keys are sorted;
  job IDs and creation timestamps are not part of the specification.

Typed backtest, walk-forward, parallel, and data defaults have the same digest
whether implicit or explicit. Strategy identity and non-parameter declarations
remain bound to the revision. Without an MCP `parameters` override, or with
`parameters: null`, `config.strategy.parameters` retains the input configuration's
parameter mapping while constructor defaults are expanded separately into
`strategy_execution`. CLI execution likewise retains the parameter mapping from
its submitted YAML, which may already contain experiment settings.

An explicit MCP `parameters` mapping, including `{}`, recursively overlays the
captured values and replaces `config.strategy.parameters` with the complete
effective experiment parameters, including defaults. For bundled strategies this
is the captured constructor's `config` mapping; for generated strategies it is
the captured constructor keyword mapping. The run's `config.yaml` projection
reflects that replacement. The revision's original YAML and evidence stay unchanged.
Overrides can target only keys already captured from base parameters or constructor
defaults; unknown keys at any recursively merged mapping level are rejected.

Without overrides, omitted versus explicit constructor defaults in the input
parameter mapping can produce different digests despite equal effective values.
An explicit empty override can also differ from an omitted override if it expands
defaults absent from that mapping. Conversely, explicit override mappings that
produce identical full parameters have the same digest when the rest of the
manifest is identical. The hash records the stored specification, not whether
the caller explicitly supplied a particular override key.

Nested optimization candidates override only their specified leaves, preserving
the remaining fixed values; candidate structure and bundled model contracts are
validated before creating a job without constructing each candidate. Submission
also constructs generated strategies with the base execution parameters and runs
80- and 240-row signal-contract fixtures in the calling process before job creation,
even without parameter overrides. Worker supervision and trial timeouts do not
cover these submission-time checks. Each generated candidate is checked again
with its own parameters in the worker; constructor or execution errors can still
occur during the trial. Unsupported defaults or
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
request. Schema version 4 requires the version, effective strategy fields,
performance settings, the fixed optimization direction, and the canonical
top-level `random_seed`; versions 1/2/3 or unknown versions require a new submission. No manifest is migrated
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
projection (including the execution's strategy identity, non-parameter declarations,
and parameter mapping described above); resolved constructor
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
- `list_runs` / `get_run`: completed result summaries and metric/scope discovery.
- `get_metric_catalog`: current definitions, units, applicability, summary choices,
  annualization requirements and optimization directions.
- `get_run_metrics`: complete or selected metrics in a saved scope, with definition
  snapshots, provider versions, settings and unavailable-value reasons.
- `compare_runs`: scoped metric comparison with explicit comparability reasons.
- `list_artifacts` / `get_artifact`: result JSON and other artifact lookup.
- `promote_strategy`: generated strategy artifact promotion.
- `record_feature_request`: structured unsupported feature requests.

The generated revision's `config.yaml` holds its base settings;
`runs/<run_id>/config.yaml` holds effective execution settings and includes
`strategy.revision_id`. Runtime market/date and parameter experiments
do not rewrite the base revision. Backtest and walk-forward requests may supply
`parameters`, recursively merged into the base parameters. CLI runtime configs
may change `strategy.parameters` while preserving other strategy declarations.
The same revision can therefore have many runs with different parameters and
manifest hashes, without duplicate generated source revisions. Both short and
long signal-contract fixtures check effective generated settings before execution.
The execution manifest additionally fixes runtime overrides and search settings
without rewriting the base revision.

Each completed background job writes files under `workspace/runs/<run_id>/` and records
matching SQLite metadata:

- `result.json`: serialized metrics, stored as `result_json`.
- `performance.json`: all metric values by scope, definitions and calculation
  metadata, stored as `performance_json`.
- `observations.json`: aligned timestamps, equity, per-bar returns and trade
  records by scope, stored as `observations_json`.
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
base config. `runs.random_seed` records only the canonical
top-level `random_seed`; strategy-specific model parameters are not inferred as a
run seed. Run comparison,
dashboard rendering, and artifact lookup always read through SQLite first, then
resolve files from the recorded artifact paths.

CLI runs instead store their `pipeline_result` cache under
`workspace/data/processed/cache` (or `$TRADINGDEV_DATA_ROOT/processed/cache`),
while `runs.artifact_dir` points to
`workspace/runs/cli_<cache_key>_<execution_id>/`, where the
execution manifest is published and registered as an artifact. The pipeline
shares the same performance/observations JSON storage in that run directory;
the pickle is only needed for dashboard rendering. The pipeline
result embeds both its config snapshot and execution manifest. The run's config
hash, manifest hash, and revision identity come from the executed specification.
Saving uses this manifest without re-reading the original config file; editing
that file after execution does not prevent saving the original result. The
original `config_path` remains descriptive artifact metadata. Each save has a
fresh execution ID and a separate `<run_id>.pkl`; same-request reruns can preserve
different results. Clearing the pickle cache leaves historical performance JSON
readable; the cleared historical dashboard artifact is unavailable, while a new
execution writes its own complete cache.

The CLI cache fingerprint requires the executed manifest hash and combines it with
processed-data file size/mtime and a Git source-code fingerprint. There is no
YAML-based key fallback. New settings produce a different manifest and therefore
a different cache key. Completed results are loaded by run ID through the
registered artifact path, without recomputing a key from current YAML, data, or
code. The unused YAML-based `load_cached_result` and `save_cached_result` helpers
have been removed; this storage path does not automatically reuse prior backtests.
These file-stat and code fingerprints describe the execution inputs; they are not
immutable data or environment versions and do not guarantee full reproducibility.

Optimization `result.json` includes the best parameters, training metrics,
out-of-sample metrics, objective and fixed direction. Its complete numerical
projection remains in SQLite; `get_run` and completed `get_job_status` select a
summary at read time, so their `metrics` are not the entire `result.json` payload.
Parameters outside the search grid retain their YAML values; selection
uses training results, and only the selected parameters are evaluated out of
sample. Every training trial retains its full result in the performance and
observation artifacts; selection references the saved winning trial without
discarding other trials.

## Performance Scopes and Provenance

`performance.json` schema version 1 records `run_id`, `manifest_hash`,
`default_scope`, `scopes`, and `definitions`. Each scope contains `values` and
`metadata`, its engine `mode`, `kind`, `split`, optional fold/trial index, and
parameter values. Definition snapshots describe unit, provider, meaning,
applicability, summary selection and optimization direction. They are saved with
the result; queries do not substitute the current catalog.

| Run type | Scopes | Default |
| --- | --- | --- |
| Backtest | `full` | `full` |
| Walk-forward | `fold/<index>/train`, `fold/<index>/test`, `test_summary` | `test_summary` |
| Optimization | `trial/<index>/train`, `test` | `test` |

New backtest and walk-forward scopes record the manifest's fixed, effective
constructor parameters, including defaults, in `parameters`. This includes
`test_summary`: its parameters describe the initial execution settings, not an
aggregation of learned fold settings. Generated strategies use their constructor
arguments; bundled strategies use the constructor's `config` object. Actual
values reported by `strategy.get_parameters()` are saved separately in
`metadata.strategy_parameters` for the full backtest or each train/test fold.
Walk-forward captures each fold's values independently after fitting; later
folds cannot overwrite earlier snapshots. The summary retains those values in
`metadata.fold_metadata`.

Optimization keeps its existing meaning: `parameters` contains the trial's
searched values (or the selected values for `test`), while
`metadata.strategy_parameters` contains the strategy's complete reported values.
Previously published artifacts are read unchanged. In older artifacts, full and
summary parameters may be empty and fold parameters may contain fitted values;
queries do not infer missing values from current strategy files.

Indices are zero-based. Optimization's `selected_train_scope` identifies the
winning training trial. A `backtest` scope contains scalar values; the
`fold_summary` scope contains `mean`, `std`, `min`, `max` and `valid_count` per
metric. Fold summaries are descriptive distributions, not portfolio returns over
the concatenated test periods, and have no fabricated observation series.

Metadata records provider versions, annualization and rate settings, daily return
sampling, bar drawdown sampling, UTC calendar aggregation, cost model, execution
context, and unavailable-value reasons. Examples include `not_applicable`,
`missing_annualization`, `missing_timestamps`, `no_trades`, `zero_denominator`,
`unbounded`, `unsupported_daily_sampling` (bars coarser than daily), and
`unknown_bar_frequency`. Numeric JSON values remain finite; unavailable values
use `null`.

Annualized metrics, `daily_max_drawdown`, and daily/monthly PnL statistics require
a declared bar frequency of one day or finer. Coarser or unrecognized frequencies
retain bar observations, total return/PnL, bar drawdown, observed date/month counts
and trade statistics. Their `return_sampling` and `calmar_drawdown_sampling`
settings are `unavailable`, and converted per-period rates are `null`. Frequency
validation does not infer bar duration from missing dates or synthesize daily
equity. Existing saved results remain snapshots; this correction does not
recalculate previously published artifacts.

`observations.json` has the same run and manifest identity. Backtest scopes retain
initial capital, equity, per-bar returns, UTC ISO timestamps and normalized trade
records. Volume returns and missing timestamps remain `null`. Available equity,
return and timestamp arrays align by bar; trade indices refer to that scope's
arrays. Open marks are separate from executed exits. Raw observations are readable
through artifact lookup without Python pickle.

`get_run_metrics` defaults to the run's saved default scope; omit `metric_ids` to
read that scope in full. Unknown scopes and metric IDs produce structured errors
with discovery information. Summary omission never affects calculation or storage.
Comparisons retain the requested values and report comparability and reasons;
differences in mode, definitions, provider versions, sampling, annualization,
capital or execution context cannot silently become equivalent rankings.

Detailed reads verify supported schema, run/manifest identity and recorded
artifact digest. Missing or corrupted artifacts are errors rather than a request
to rerun a strategy. Publication validates and encodes data before writing;
existing different content is not overwritten. Retrying a background job's same
result also checks all registered artifact files and hashes before reporting
success; missing or corrupted files are explicit errors. File publication and SQLite writes
are not a cross-system transaction. A per-run `.result-publication` marker prevents
concurrent publishers. Handled exceptions restore the prior files and remove new
run/artifact records. An externally terminated process can leave the marker;
queries report the publication as busy instead of treating incomplete output as
legacy data. The reader does not delete the marker or repair stored data.

Historical runs without performance artifacts remain readable through ordinary
run lookup, marked as lacking detailed provenance. `get_run_metrics` reports
`performance_artifact_unavailable`; reads never migrate, unpickle or recompute
those results. Historical values are not asserted to follow the new definitions.

## JSON Values

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
