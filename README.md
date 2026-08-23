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
manda artifacts ...         Checkpoint, publish, list, pull, and verify artifacts
manda validate-results      Validate a local or remotely materialized run
manda extract-trajectories  Derive HDF5 trajectory metrics
manda compare-trajectories  Detect action and simulator-state drift
manda compare               Rank task-level disagreements
manda capture-provenance    Record repository, host, and GPU provenance
```

The command layout leaves room for a future `manda vast ...` provider wrapper without coupling the
artifact and analysis layers to Vast.ai.

## Artifact storage

Copy [examples/storage.env.example](examples/storage.env.example) to a private file outside Git,
give every producer a unique ID, and point the application-key variable at a `chmod 600` secret
file:

```bash
set -a
source /path/to/storage.env
set +a

manda artifacts doctor --write-test
manda artifacts list --all-producers
manda artifacts checkpoint --run-dir "$RUN_DIR"
manda artifacts push --run-dir "$RUN_DIR" --study-id my-study --stage pilot --policy pi05
manda artifacts pull --run-id "$RUN_ID" --destination ./run --profile analysis
manda artifacts verify --run-id "$RUN_ID"
```

Passing `--config /path/to/experiment.toml` to `artifacts push` adds expected episode/task checks.
Passing only `--study-id` keeps storage publication independent of a study checkout. See the
[artifact operations guide](docs/artifact-storage.md) for the full workflow.

Privileged database ingestion, schema migrations, media signing, and dashboard deployment belong
to the private `manda-infra` repository. This public client never accepts a Supabase service key.

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
