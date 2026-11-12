# Manda CLI
**Share, verify and analyze robot policy evaluations from the command line.**

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://docs.astral.sh/ruff/)

`manda` is the client for Manda's robot evaluation service. It signs you in with the same account
as the Manda dashboard, publishes benchmark runs to shared studies, and pulls, verifies and
analyzes them again later. It also plans and runs RoboLab benchmark jobs on your own GPU.

Accounts are invite-only for now. Without one, the planning and analysis commands still work on
local runs.

[Artifact operations guide](docs/artifact-storage.md) · [Example experiment](examples/experiment.toml)

## Installation

Requires **Python 3.11+**, and [rclone](https://rclone.org/install/) for artifact transfers.
Install from source using [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
git clone https://github.com/Manda-Robotics/manda-cli.git
cd manda-cli
uv sync --extra all
uv run manda --help
```

Or use pip in a virtual environment: `python -m pip install -e '.[all]'`.

The core has no dependencies. Extras add what a workflow needs:

| Extra | Purpose |
|---|---|
| `dashboard` | HDF5 timeseries for the dashboard's episode charts when you push a run |
| `analysis` | Trajectory extraction and comparison |
| `all` | Both of the above |
| `dev` | Tests and Ruff |

## Quickstart

Sign in with your invited email address; the command prompts for a one-time code. Then check
your studies and storage, and publish a finished run:

```sh
manda auth login --email you@example.org --client-label my-workstation
manda studies list
manda quota status
manda artifacts push --run-dir "$RUN_DIR" --study-id my-study
```

A successful push is verified and immediately visible to every member of the study, in the
dashboard and through `manda artifacts list`.

## Commands

| Command | Purpose |
|---|---|
| `manda auth ...` | Sign in, inspect and end a session |
| `manda teams ...` | Create teams and invite members |
| `manda studies ...` | Create, discover, share and publish studies |
| `manda evals ...` | Run, follow and stop hosted evaluations |
| `manda quota status` | Inspect your artifact storage use |
| `manda artifacts ...` | Checkpoint, push, list, pull and verify runs |
| `manda validate-config` | Validate an experiment TOML file |
| `manda plan` | Build a deterministic execution plan |
| `manda doctor` | Check RoboLab and policy-server readiness |
| `manda execute` | Dry-run or execute a planned benchmark job |
| `manda validate-results` | Validate a local or downloaded run |
| `manda extract-trajectories` | Derive HDF5 trajectory metrics |
| `manda compare-trajectories` | Detect action and simulator-state drift |
| `manda compare` | Rank task-level disagreements |
| `manda capture-provenance` | Record repository, host and GPU provenance |

## Hosted evaluations

Run a RoboLab evaluation on a GPU node that Manda rents for you. The budget is held from your
credit while the job runs, you are charged for the compute it used, and verified episodes appear
in the study as they finish:

```sh
manda evals credit
manda evals submit --study my-study --easiest 5 --episodes 10 --budget 5          # plan only
manda evals submit --study my-study --easiest 5 --episodes 10 --budget 5 --yes    # start it
manda evals show JOB_ID --watch
manda evals list --active
manda evals cancel JOB_ID
```

Choose tasks with `--task` (repeatable), `--easiest N` or `--all-tasks`, from the task catalog of
the RoboLab revision the nodes run. `--policy` selects the pinned pi0.5 reference (`pi05`, the
default), Rung on Manda's endpoint (`rung`, with `--model gemini-4-argon` (default),
`claude-fable-5-2` or `gpt-6.1-sol`, each over its vendor's paid API), or your own OpenPI WebSocket server (`openpi` with
`--name` and `--endpoint`, optionally labelled with `--version`, `--checkpoint` and
`--training-config`). Parallel environments follow from the node and episode count. Every command
prints JSON, and `submit` without `--yes` only shows the plan.

## Studies and sharing

A study is a shared folder of runs. Team owners can set up and share work without help from Manda:

```sh
manda teams create --name "Robotics team"      # ID inferred as robotics-team
manda teams invite robotics-team colleague@example.org --role member
manda studies create --title "My benchmark" --owner-team robotics-team
manda studies grant my-study --email reviewer@example.org --role viewer
manda studies publish my-study                 # make ready runs public
manda studies unpublish my-study
```

If your account has exactly one writable study, upload commands select it automatically. Pass
`--study-id` when there are several. `manda auth login --json` prints the full session report
for scripts.

## Artifacts and security

```sh
manda artifacts doctor --write-test
manda artifacts checkpoint --run-dir "$RUN_DIR"
manda artifacts push --run-dir "$RUN_DIR"
manda artifacts list --study-id my-study
manda artifacts list --mine
manda artifacts pull --run-id "$RUN_ID" --destination ./run --profile analysis --study-id my-study
```

The service address and its publishable key are built in, so there is nothing to configure and no
infrastructure credential to handle. Access is enforced by the service's row-level security, not
by this client.

- **Sessions:** the refreshable session is stored in a local file readable only by you. For a
  non-interactive worker, `--access-token-file` takes a short-lived access token from such a
  file; replace it before it expires.
- **Transfers:** every transfer uses a short-lived storage key scoped to one operation and path.
  It exists only in the CLI process and is handed to `rclone` through its environment. This
  client never accepts a long-lived storage key or a service secret.
- **Quota:** uploads reserve their estimated size against your storage quota first (5 GiB by
  default). A push is verified against the stored objects before it becomes visible.
- **Self-hosting:** `manda auth configure` points the client at another deployment; see
  [`examples/auth.env.example`](examples/auth.env.example) for development overrides.

Before a run is sealed, `artifacts push` derives one compact JSON file per episode with its event
timeline and, with the `dashboard` extra, downsampled timeseries for the dashboard. A missing
optional dependency never discards the source trajectory; it is reported as a validation warning.
Stage and policy are inferred from run metadata and the output directory name; `--stage`,
`--policy` and `--config` override or check them. See the
[artifact operations guide](docs/artifact-storage.md) for the full workflow.

## Running benchmarks

Planning and execution take explicit paths to an experiment file, such as
[`examples/experiment.toml`](examples/experiment.toml). They expect a RoboLab installation and a
policy server on the machine that executes:

```sh
manda validate-config --config experiment.toml
manda plan --config experiment.toml --stage smoke --output plans/smoke.jsonl
manda execute --config experiment.toml --plan plans/smoke.jsonl
manda validate-results --config experiment.toml --stage smoke --run-dir "$RUN_DIR"
manda extract-trajectories --run-dir "$RUN_DIR" --policy pi05 --output derived/pi05.jsonl
```

`manda execute` is a dry run until you add `--yes`. Review its output first.

## Development

```sh
uv sync --extra all --extra dev
uv run pytest
uv run ruff check src tests
```

Credentials, private environment files, artifact caches, derived data and plans are ignored by Git.

## Credits

Built by [Manda Robotics](https://mandarobotics.com). Benchmark execution integrates with NVIDIA's
[RoboLab](https://github.com/NVlabs/RoboLab), Isaac Sim and Isaac Lab. For running frontier
models as robot policies, see [Rung](https://github.com/Manda-Robotics/rung).

## License

[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution.
