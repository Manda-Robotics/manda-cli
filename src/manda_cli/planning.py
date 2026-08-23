from __future__ import annotations

import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import ConfigError, ExperimentConfig, PolicyConfig, StageConfig


@dataclass(frozen=True)
class Job:
    job_id: str
    study_id: str
    stage: str
    policy: str
    policy_display_name: str
    shard_index: int
    shard_count: int
    tasks: tuple[str, ...]
    expected_task_count: int
    expected_episodes: int
    output_folder: str
    working_directory_env: str
    simulator_gpu: int
    server_host: str
    server_port: int
    command: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["command_preview"] = shlex.join(self.command)
        return value


def _partition(items: tuple[str, ...], count: int) -> list[tuple[str, ...]]:
    if count < 1:
        raise ConfigError("shards must be >= 1")
    if count > len(items):
        raise ConfigError(f"Cannot split {len(items)} tasks into {count} non-empty shards")
    return tuple(tuple(items[index::count]) for index in range(count))  # type: ignore[return-value]


def _command(config: ExperimentConfig, policy: PolicyConfig, stage: StageConfig) -> list[str]:
    command = ["uv", "run", "python", policy.runner, *policy.runner_args]
    command.extend(
        [
            "--headless",
            "--num-envs",
            str(stage.num_envs),
            "--num-runs",
            str(stage.num_runs),
            "--instruction-type",
            config.study.instruction_type,
            "--video-mode",
            stage.video_mode,
            "--renderer",
            config.study.renderer,
            "--rendering-type",
            config.study.rendering_type,
        ]
    )
    if config.study.isaac_stack == "isaac60":
        command.append("--kit_args=--/UJITSO/enabled=false")
    if policy.requires_server:
        command.extend(
            [
                "--remote-host",
                policy.server_host(),
                "--remote-port",
                str(policy.server_port()),
            ]
        )
    if stage.record_image_data:
        command.append("--record-image-data")
    return command


def build_jobs(
    config: ExperimentConfig,
    stage_name: str,
    *,
    policy_names: list[str] | None = None,
    shards: int = 1,
    include_disabled: bool = False,
) -> list[Job]:
    try:
        stage = config.stages[stage_name]
    except KeyError as exc:
        raise ConfigError(f"Unknown stage {stage_name!r}") from exc
    if stage.blocked_by:
        raise ConfigError(f"Stage {stage_name!r} is blocked by {', '.join(stage.blocked_by)}")
    if not stage.tasks and stage.task_selector != "all":
        raise ConfigError(f"Stage {stage_name!r} has no selected tasks")
    if stage.task_selector == "all" and shards != 1:
        raise ConfigError(
            "The all-task selector cannot be sharded until RoboLab task metadata is imported "
            "(FEATURE-003)"
        )

    if policy_names:
        missing = sorted(set(policy_names) - set(config.policies))
        if missing:
            raise ConfigError(f"Unknown policies: {', '.join(missing)}")
        policies = [config.policies[name] for name in policy_names]
    else:
        policies = list(config.policies.values())

    if not include_disabled:
        policies = [policy for policy in policies if policy.enabled]
    if not policies:
        raise ConfigError("No runnable policies selected")
    for policy in policies:
        if policy.blocked_by:
            raise ConfigError(
                f"Policy {policy.name!r} is blocked by {', '.join(policy.blocked_by)}"
            )

    task_shards = _partition(stage.tasks, shards) if stage.tasks else [tuple()]
    jobs: list[Job] = []
    for policy in policies:
        for shard_index, tasks in enumerate(task_shards):
            suffix = f"_s{shard_index + 1:02d}of{shards:02d}" if shards > 1 else ""
            output_folder = f"{config.study.id}_{stage.name}_{policy.name}{suffix}"
            command = _command(config, policy, stage)
            command.extend(["--output-folder-name", output_folder])
            if tasks:
                command.extend(["--task", *tasks])
            task_count = len(tasks) if tasks else stage.expected_task_count
            jobs.append(
                Job(
                    job_id=f"{stage.name}-{policy.name}{suffix.replace('_', '-')}",
                    study_id=config.study.id,
                    stage=stage.name,
                    policy=policy.name,
                    policy_display_name=policy.display_name,
                    shard_index=shard_index,
                    shard_count=shards,
                    tasks=tasks,
                    expected_task_count=task_count,
                    expected_episodes=task_count * stage.num_envs * stage.num_runs,
                    output_folder=output_folder,
                    working_directory_env=config.study.robolab_root_env,
                    simulator_gpu=policy.simulation_gpu,
                    server_host=policy.server_host(),
                    server_port=policy.server_port(),
                    command=tuple(command),
                )
            )
    return jobs


def write_plan(jobs: list[Job], path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(json.dumps(job.as_dict(), sort_keys=True) + "\n")
    return output


def read_plan(path: str | Path) -> list[dict[str, object]]:
    plan_path = Path(path).expanduser().resolve()
    jobs: list[dict[str, object]] = []
    with plan_path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"Invalid JSON on {plan_path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict) or not isinstance(value.get("job_id"), str):
                raise ConfigError(f"Invalid job on {plan_path}:{line_number}")
            jobs.append(value)
    return jobs
