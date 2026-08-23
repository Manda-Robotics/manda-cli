# Artifact storage operations

The Manda CLI treats a private object-storage bucket as the canonical artifact store. GPU hosts
use local disks as scratch space; producers publish immutable completed runs, and consumers select
metadata, analysis, or full download profiles.

## Configure a client

Install `rclone`, copy `examples/storage.env.example` to a private location, and give every
benchmark producer a stable, unique `ROBOLAB_STORAGE_PRODUCER_ID`. Store the Backblaze application
key in a separate file with mode `600`; do not put the secret in the environment file.

```bash
chmod 600 /path/to/storage.env /path/to/backblaze-app-key
set -a
source /path/to/storage.env
set +a
manda artifacts doctor
```

Use a separate prefix-restricted key and producer ID on every marketplace VM. Rotate or revoke the
key after returning a machine, and do not distribute an account master key.

To confirm write access, upload and read back a tiny expiring probe:

```bash
manda artifacts doctor --write-test
```

## Checkpoint and resume

Checkpoint files that have been quiescent for at least two minutes:

```bash
manda artifacts checkpoint --run-dir "$RUN_DIR"
```

The generated run ID is stored in `$RUN_DIR/.artifact-run-id`, so repeated checkpoints use the same
working prefix. A different machine can resume it:

```bash
manda artifacts resume \
  --producer-id vast-metz \
  --run-id "$RUN_ID" \
  --destination "$RUN_DIR"
```

When `manda execute` runs a planned job, these opt-in variables enable periodic checkpoints,
failure checkpoints, and automatic final publication:

```bash
export ROBOLAB_STORAGE_AUTO_FINALIZE=1
export ROBOLAB_STORAGE_CHECKPOINT_ON_FAILURE=1
export ROBOLAB_STORAGE_CHECKPOINT_INTERVAL_SECONDS=600
manda execute \
  --config /path/to/experiment.toml \
  --plan /path/to/plan.jsonl \
  --yes
```

The periodic checkpoint only copies files older than the safety window, avoiding actively written
HDF5 files.

## Publish a completed run

After all writers have stopped, publish with either a standalone study ID:

```bash
manda artifacts push \
  --run-dir "$RUN_DIR" \
  --study-id my-study \
  --stage pilot \
  --policy pi05
```

Or pass an experiment config to validate its expected episode and task counts:

```bash
manda artifacts push \
  --run-dir "$RUN_DIR" \
  --config /path/to/experiment.toml \
  --stage pilot \
  --policy pi05
```

Finalization creates `run.json` and a per-file SHA-256 manifest, uploads only manifest-listed files,
checks the remote payload, and writes `_COMPLETE` last. A completed prefix is immutable. Repeating
the operation verifies and returns the existing commit instead of rewriting it.

## Discover and consume runs

```bash
manda artifacts list --study-id my-study
manda artifacts list --all-producers

manda artifacts pull --run-id "$RUN_ID" --destination ./run --profile metadata
manda artifacts pull --run-id "$RUN_ID" --destination ./run --profile analysis
manda artifacts pull --run-id "$RUN_ID" --destination ./run --profile full
```

- `metadata` includes run, result, configuration, provenance, and event-log data.
- `analysis` adds HDF5 and other non-media data but excludes videos and images.
- `full` materializes the exact manifest payload, including media.

Every pull validates selected files against the committed SHA-256 manifest. Analysis commands also
accept a remote run ID instead of a local directory and cache the analysis profile under
`ROBOLAB_ARTIFACT_CACHE` or the platform's default cache directory:

```bash
manda validate-results \
  --config /path/to/experiment.toml \
  --stage pilot \
  --run-dir "$RUN_ID"

manda extract-trajectories \
  --run-dir "$RUN_ID" \
  --policy pi05 \
  --output derived/pi05-trajectories.jsonl
```

## Publish the analysis catalog

Install the analysis extra, then publish an immutable catalog version and update
`catalog/latest.json`:

```bash
python -m pip install -e '.[analysis]'
manda artifacts publish-catalog \
  --study-id my-study \
  --output-dir derived/catalogs
```

Catalog publication needs a trusted read/write key scoped to the shared `robolab/v1/` root so it
can discover all producer prefixes and update shared `derived/` and `catalog/` objects. A catalog
contains run, task, episode, and asset JSON/JSONL, optional Parquet, a manifest, and `_COMPLETE`;
each row records the source run and manifest digest. Asset rows contain the private B2 object key,
content type, size, role, and SHA-256 digest—not a public URL.

## Bucket administration

`configure-b2` prints a dry-run plan by default:

```bash
manda artifacts configure-b2 \
  --producer-id metz-local \
  --producer-id vast-metz \
  --producer-id runpod-colleague
```

The plan enables Backblaze-managed AES-256 encryption for future uploads, expires current working
checkpoints after 14 days, deletes hidden working versions one day later, deletes other prior
versions after 30 days, and cancels unfinished multipart uploads after seven days. Current
completed objects do not expire.

Applying lifecycle rules permanently deletes eligible versions. Review the complete plan first,
then explicitly apply it:

```bash
manda artifacts configure-b2 \
  --producer-id metz-local \
  --producer-id vast-metz \
  --producer-id runpod-colleague \
  --yes
```

The provisioner validates the configured bucket name and optional bucket ID, preserves unrelated
lifecycle rules, and uses Backblaze bucket revisions to avoid overwriting concurrent changes.
