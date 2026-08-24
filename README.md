# Manda CLI

`manda` is the standalone command-line client for Manda's RoboLab benchmark workflow. It owns
provider-neutral artifact storage, experiment planning and execution, runtime checks, provenance,
result validation, and trajectory analysis. The original `robot-audit` executable remains as a
compatibility alias.

The study repository is still responsible for experiment TOML files, benchmark-specific scripts,
and scientific documentation. This repository contains the reusable client.

## Install

Python 3.11 or newer is required. `rclone` is required for artifact transfers.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[all]'
manda --help
```

With `uv`:

```bash
uv sync --extra all --extra dev
uv run manda --help
```

## Command families

```text
manda validate-config       Validate an experiment TOML file
manda plan                  Build a deterministic execution plan
manda doctor                Check RoboLab and policy-server readiness
manda execute               Dry-run or execute a planned benchmark job
manda auth ...              Sign in, inspect, and end a user session
manda teams ...             Create teams and invite members
manda studies ...           Create, discover, and share studies
manda quota status          Inspect personal artifact storage use
manda artifacts ...         Checkpoint, publish, list, pull, and verify artifacts
manda validate-results      Validate a local or remotely materialized run
manda extract-trajectories  Derive HDF5 trajectory metrics
manda compare-trajectories  Detect action and simulator-state drift
manda compare               Rank task-level disagreements
manda capture-provenance    Record repository, host, and GPU provenance
```

The command layout leaves room for a future `manda vast ...` provider wrapper without coupling the
artifact and analysis layers to Vast.ai.

## User authentication and artifact storage

Clients authenticate as the same Supabase user used by the web dashboard. The official hosted
service URL and publishable key are built into the client, so users do not need to set environment
variables or handle infrastructure credentials.

```bash
manda auth login --email you@example.org --client-label vast-gpu-01
manda auth status
manda studies list
manda quota status
manda artifacts doctor --write-test
manda artifacts checkpoint --run-dir "$RUN_DIR"
manda artifacts push --run-dir "$RUN_DIR" --stage pilot --policy pi05
manda artifacts list --study-id my-study
manda artifacts pull --run-id "$RUN_ID" --destination ./run --profile analysis --study-id my-study
```

Successful interactive login prints a compact workspace welcome. Use `manda auth login --json`
when a script needs the complete machine-readable session report.

If the account has exactly one writable study, upload commands select it automatically. Pass
`--study-id` when several are available. Team owners can set up and share work without SQL:

```bash
manda teams create --name "Robotics team"  # ID inferred as robotics-team
manda teams invite robotics-team colleague@example.org --role member
manda studies create --title "My benchmark" --owner-team robotics-team
manda studies grant my-study --email reviewer@example.org --role viewer
```

`manda auth configure` supports a custom/self-hosted service; the example environment file remains
available for development overrides. Public client configuration is never a substitute for RLS.

Email login is invite-only and prompts for the one-time code. The refreshable session is stored in
a mode-`600` local file. For a non-interactive worker, `--access-token-file` accepts a short-lived
user access token from a mode-`600` file; the token must be replaced when it expires.

Every transfer requests a short-lived, operation- and path-scoped B2 application key from the
authenticated access service. The key exists only in the CLI process and is passed to `rclone`
through its child environment. Benchmark hosts never receive the long-lived B2 administration
key or a Supabase secret key. Upload leases reserve the estimated transfer size against the
user's server-side quota; the hosted default is 5 GiB, and committed runs are reconciled from
their verified manifests.

Passing `--config /path/to/experiment.toml` to `artifacts push` adds expected episode/task checks.
Passing only `--study-id` keeps storage publication independent of a study checkout. See the
[artifact operations guide](docs/artifact-storage.md) for the full workflow.

Privileged database ingestion, bucket configuration, schema migrations, access-broker deployment,
media signing, and dashboard deployment belong to the private `manda-infra` repository. This
public client never accepts a Supabase secret key or long-lived B2 credential.

## Planning and analysis

The repository includes representative configuration examples, but installed commands always use
explicit paths:

```bash
manda validate-config --config /path/to/experiment.toml
manda plan --config /path/to/experiment.toml --stage smoke --output plans/smoke.jsonl
manda execute --config /path/to/experiment.toml --plan plans/smoke.jsonl
manda validate-results --config /path/to/experiment.toml --stage smoke --run-dir "$RUN_DIR"
manda extract-trajectories --run-dir "$RUN_DIR" --policy pi05 --output derived/pi05.jsonl
```

Run `manda execute ... --yes` only after reviewing its dry-run output.

## Development

```bash
uv sync --extra all --extra dev
uv run pytest
uv run ruff check src tests
```

Real credentials, private environment files, artifact caches, derived data, and plans are ignored.
