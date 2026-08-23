from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


@dataclass(frozen=True)
class StudyConfig:
    id: str
    title: str
    robolab_root_env: str
    output_root_env: str
    robolab_expected_revision: str
    isaac_stack: str
    renderer: str
    rendering_type: str
    instruction_type: str
    expected_task_count: int
    simulator_gpu: int
    seed_schedule: tuple[int, ...]

    def robolab_root(self) -> Path | None:
        value = os.environ.get(self.robolab_root_env)
        return Path(value).expanduser() if value else None

    def output_root(self) -> Path | None:
        value = os.environ.get(self.output_root_env)
        return Path(value).expanduser() if value else None


@dataclass(frozen=True)
class PolicyConfig:
    name: str
    enabled: bool
    display_name: str
    runner: str
    runner_args: tuple[str, ...]
    requires_server: bool
    server_host_env: str | None
    server_port_env: str | None
    server_cwd_env: str | None
    server_command: tuple[str, ...]
    simulation_gpu: int
    blocked_by: tuple[str, ...]
    status_note: str

    def server_host(self) -> str:
        if not self.server_host_env:
            return "127.0.0.1"
        return os.environ.get(self.server_host_env, "127.0.0.1")

    def server_port(self) -> int:
        if not self.server_port_env:
            return 8000
        raw = os.environ.get(self.server_port_env, "8000")
        try:
            return int(raw)
        except ValueError as exc:
            raise ConfigError(f"{self.server_port_env} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class StageConfig:
    name: str
    description: str
    tasks: tuple[str, ...]
    task_selector: str | None
    num_envs: int
    num_runs: int
    video_mode: str
    record_image_data: bool
    expected_task_count: int
    blocked_by: tuple[str, ...]

    @property
    def expected_episodes(self) -> int:
        return self.expected_task_count * self.num_envs * self.num_runs


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    schema_version: int
    study: StudyConfig
    policies: dict[str, PolicyConfig]
    stages: dict[str, StageConfig]

    def enabled_policies(self) -> list[PolicyConfig]:
        return [policy for policy in self.policies.values() if policy.enabled]


def _require_table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid [{key}] table")
    return value


def _require_str(data: dict[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{context}.{key} must be a non-empty string")
    return value


def _int(data: dict[str, Any], key: str, context: str, *, minimum: int = 0) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ConfigError(f"{context}.{key} must be an integer >= {minimum}")
    return value


def _strings(data: dict[str, Any], key: str, context: str) -> tuple[str, ...]:
    value = data.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{context}.{key} must be a list of strings")
    return tuple(value)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"Configuration does not exist: {config_path}")

    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    schema_version = raw.get("schema_version")
    if schema_version != 1:
        raise ConfigError(f"Unsupported schema_version {schema_version!r}; expected 1")

    study_raw = _require_table(raw, "study")
    seeds = study_raw.get("seed_schedule", [])
    if not isinstance(seeds, list) or not seeds or not all(isinstance(seed, int) for seed in seeds):
        raise ConfigError("study.seed_schedule must be a non-empty list of integers")
    if len(seeds) != len(set(seeds)):
        raise ConfigError("study.seed_schedule contains duplicates")

    study = StudyConfig(
        id=_require_str(study_raw, "id", "study"),
        title=_require_str(study_raw, "title", "study"),
        robolab_root_env=_require_str(study_raw, "robolab_root_env", "study"),
        output_root_env=_require_str(study_raw, "output_root_env", "study"),
        robolab_expected_revision=_require_str(study_raw, "robolab_expected_revision", "study"),
        isaac_stack=_require_str(study_raw, "isaac_stack", "study"),
        renderer=_require_str(study_raw, "renderer", "study"),
        rendering_type=_require_str(study_raw, "rendering_type", "study"),
        instruction_type=_require_str(study_raw, "instruction_type", "study"),
        expected_task_count=_int(study_raw, "expected_task_count", "study", minimum=1),
        simulator_gpu=_int(study_raw, "simulator_gpu", "study"),
        seed_schedule=tuple(seeds),
    )
    if study.isaac_stack not in {"isaac50", "isaac51", "isaac60"}:
        raise ConfigError("study.isaac_stack must be isaac50, isaac51, or isaac60")
    if study.renderer not in {"realtime", "pathtracing"}:
        raise ConfigError("study.renderer must be realtime or pathtracing")

    policies_raw = _require_table(raw, "policies")
    policies: dict[str, PolicyConfig] = {}
    for name, policy_raw in policies_raw.items():
        if not isinstance(policy_raw, dict):
            raise ConfigError(f"policies.{name} must be a table")
        context = f"policies.{name}"
        policy = PolicyConfig(
            name=name,
            enabled=bool(policy_raw.get("enabled", False)),
            display_name=_require_str(policy_raw, "display_name", context),
            runner=_require_str(policy_raw, "runner", context),
            runner_args=_strings(policy_raw, "runner_args", context),
            requires_server=bool(policy_raw.get("requires_server", True)),
            server_host_env=policy_raw.get("server_host_env"),
            server_port_env=policy_raw.get("server_port_env"),
            server_cwd_env=policy_raw.get("server_cwd_env"),
            server_command=_strings(policy_raw, "server_command", context),
            simulation_gpu=_int(policy_raw, "simulation_gpu", context),
            blocked_by=_strings(policy_raw, "blocked_by", context),
            status_note=str(policy_raw.get("status_note", "")),
        )
        if policy.enabled and policy.blocked_by:
            raise ConfigError(
                f"{context} is enabled but still blocked by {', '.join(policy.blocked_by)}"
            )
        if policy.enabled and policy.requires_server and not policy.server_command:
            raise ConfigError(f"{context} is enabled but has no server_command")
        policies[name] = policy
    if not policies:
        raise ConfigError("At least one policy must be configured")

    stages_raw = _require_table(raw, "stages")
    stages: dict[str, StageConfig] = {}
    for name, stage_raw in stages_raw.items():
        if not isinstance(stage_raw, dict):
            raise ConfigError(f"stages.{name} must be a table")
        context = f"stages.{name}"
        tasks = _strings(stage_raw, "tasks", context)
        selector = stage_raw.get("task_selector")
        if selector is not None and selector != "all":
            raise ConfigError(f"{context}.task_selector currently supports only 'all'")
        if tasks and selector:
            raise ConfigError(f"{context} cannot define both tasks and task_selector")
        stage = StageConfig(
            name=name,
            description=_require_str(stage_raw, "description", context),
            tasks=tasks,
            task_selector=selector,
            num_envs=_int(stage_raw, "num_envs", context, minimum=1),
            num_runs=_int(stage_raw, "num_runs", context, minimum=1),
            video_mode=_require_str(stage_raw, "video_mode", context),
            record_image_data=bool(stage_raw.get("record_image_data", False)),
            expected_task_count=_int(stage_raw, "expected_task_count", context),
            blocked_by=_strings(stage_raw, "blocked_by", context),
        )
        if stage.video_mode not in {"all", "sensor", "viewport", "none"}:
            raise ConfigError(f"{context}.video_mode has an unsupported value")
        if tasks and len(tasks) != len(set(tasks)):
            raise ConfigError(f"{context}.tasks contains duplicates")
        if tasks and stage.expected_task_count != len(tasks):
            raise ConfigError(
                f"{context}.expected_task_count is {stage.expected_task_count}, "
                f"but {len(tasks)} tasks are listed"
            )
        stages[name] = stage
    if not stages:
        raise ConfigError("At least one stage must be configured")

    return ExperimentConfig(
        path=config_path,
        schema_version=schema_version,
        study=study,
        policies=policies,
        stages=stages,
    )
