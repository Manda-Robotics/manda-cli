# Authenticated artifact operations

Manda uses Backblaze B2 as the canonical store for immutable experiment outputs and Supabase as
the user identity and dashboard read model. GPU disks are scratch space. A signed-in user can
checkpoint or publish data only for studies where they are a contributor or owner, and can
discover or download completed runs only for studies where they are a member.

## Configure and sign in

Install `rclone`, then copy `examples/auth.env.example` to a local file. The Supabase URL and
publishable key are public client values. A benchmark host must not receive a Supabase secret key,
Backblaze administration key, or permanent producer key.

```bash
set -a
source /path/to/manda-auth.env
set +a

manda auth login --email you@example.org --client-label runpod-gpu-01
manda auth status
```

Login sends a one-time code to an account that an administrator has already invited. The CLI saves
the access and rotating refresh tokens under `~/.config/manda/session.json` by default and enforces
mode `600` on that file. `MANDA_SESSION_FILE` can give isolated workers separate sessions.

For a non-interactive worker, place a short-lived Supabase user access token in a mode-`600` file:

```bash
manda auth login --access-token-file /run/secrets/manda-access-token \
  --client-label vast-worker-03
```

This session cannot refresh itself; replace it after expiry. Do not pass a token directly on the
command line because process listings and shell history may retain it.

On first login, the access service registers a random producer ID owned by that user and client.
The same local session reuses it. A new VM receives a new producer ID, which keeps concurrent
writes isolated without asking users to invent globally unique names.

## Credential model

Each storage command asks the authenticated access service for an expiring B2 application key.
The broker checks the user's study membership and client ownership, then scopes the key to one of:

- a study-specific doctor probe prefix;
- one working checkpoint for one study and run;
- one immutable completed-run prefix; or
- one completed run selected through the membership-protected database.

The temporary secret remains in memory and is injected only into the `rclone` child process. The
CLI never writes it to its session file, logs, JSON output, or command arguments.

## Check storage access

```bash
manda artifacts doctor --study-id my-study
manda artifacts doctor --study-id my-study --write-test
```

The write test uploads and reads back a tiny probe under its restricted prefix. The probe is
eligible for the private infrastructure lifecycle policy.

## Checkpoint and resume

Checkpoint files that have been quiescent for at least two minutes:

```bash
manda artifacts checkpoint \
  --run-dir "$RUN_DIR" \
  --study-id my-study
```

The generated run ID is stored in `$RUN_DIR/.artifact-run-id`, so later checkpoints use the same
working prefix. Another machine signed in as the same user can resume it when given the original
producer ID shown in the checkpoint receipt:

```bash
manda artifacts resume \
  --study-id my-study \
  --producer-id u-original-producer-id \
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

Periodic checkpoints copy only files older than the safety window, avoiding actively written HDF5
files. Each operation obtains a freshly scoped lease, so a long-running benchmark does not depend
on a credential that may have expired hours earlier.

## Publish a completed run

After all writers have stopped, publish with a standalone study ID:

```bash
manda artifacts push \
  --run-dir "$RUN_DIR" \
  --study-id my-study \
  --stage pilot \
  --policy pi05
```

Or pass an experiment config to validate expected episode and task counts:

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

The private indexer projects committed B2 runs into Supabase. Until that synchronization has run,
the object is durable but will not yet appear in `artifacts list` or the dashboard. This deliberate
boundary prevents an unverified client request from becoming dashboard data. The infrastructure
repository documents the synchronization command and is the right place to automate it on a
schedule or queue.

## Discover and consume runs

```bash
manda artifacts list --study-id my-study

manda artifacts pull \
  --run-id "$RUN_ID" \
  --study-id my-study \
  --destination ./run \
  --profile metadata

manda artifacts pull --run-id "$RUN_ID" --destination ./run --profile analysis
manda artifacts pull --run-id "$RUN_ID" --destination ./run --profile full
```

- `metadata` includes run, result, configuration, provenance, and event-log data.
- `analysis` adds HDF5 and other non-media data but excludes videos and images.
- `full` materializes the exact manifest payload, including media.

Every pull validates selected files against the committed SHA-256 manifest. If a run ID happens to
be reused in more than one accessible study, pass `--study-id` to disambiguate it.

Analysis commands also accept an indexed run ID instead of a local directory and cache the
analysis profile under `ROBOLAB_ARTIFACT_CACHE` or the platform's default cache directory:

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

End a user session and revoke its current refresh token with:

```bash
manda auth logout
```

Bucket lifecycle configuration, all-producer catalog publication, Supabase ingestion, membership
administration, and the B2 key broker are privileged operations in the private `manda-infra`
repository.
