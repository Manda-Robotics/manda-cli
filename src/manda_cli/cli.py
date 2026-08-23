from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .artifacts import ArtifactError, ArtifactManager, verify_local_run
from .b2_admin import apply_bucket_configuration, configuration_plan
from .catalog import publish_catalog
from .config import ConfigError, load_config
from .doctor import run_doctor
from .execution import execute_job
from .interesting import compare_runs, write_comparison
from .planning import build_jobs, write_plan
from .provenance import capture_provenance, write_provenance
from .results import ResultsError, validate_results
from .storage import StorageError, StorageSettings
from .trajectories import (
    AnalysisDependencyError,
    compare_trajectory_runs,
    extract_trajectory_metrics,
    write_jsonl,
    write_trajectory_comparison,
)


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="manda",
        description=(
            "Manage Manda benchmark artifacts and plan, execute, validate, and analyze "
            "RoboLab experiments."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="Validate experiment TOML")
    validate.add_argument("--config", required=True)

    plan = subparsers.add_parser("plan", help="Create a deterministic JSONL execution plan")
    plan.add_argument("--config", required=True)
    plan.add_argument("--stage", required=True)
    plan.add_argument("--policy", action="append", dest="policies")
    plan.add_argument("--shards", type=int, default=1)
    plan.add_argument("--include-disabled", action="store_true")
    plan.add_argument("--output", required=True)

    doctor = subparsers.add_parser("doctor", help="Check local or remote runtime readiness")
    doctor.add_argument("--config", required=True)
    doctor.add_argument("--policy")
    doctor.add_argument("--skip-server", action="store_true")

    execute = subparsers.add_parser("execute", help="Execute one planned job")
    execute.add_argument("--config", required=True)
    execute.add_argument("--plan", required=True)
    execute.add_argument("--job-id")
    execute.add_argument("--study-root", default=".")
    execute.add_argument(
        "--yes", action="store_true", help="Actually execute; otherwise print a dry run"
    )

    results = subparsers.add_parser("validate-results", help="Validate a RoboLab run folder")
    results.add_argument("--config", required=True)
    results.add_argument("--run-dir", required=True)
    results.add_argument("--stage", required=True)

    trajectories = subparsers.add_parser(
        "extract-trajectories", help="Extract behavioral metrics from HDF5 trajectories"
    )
    trajectories.add_argument("--run-dir", required=True)
    trajectories.add_argument("--policy", required=True)
    trajectories.add_argument("--output", required=True)

    trajectory_compare = subparsers.add_parser(
        "compare-trajectories",
        help="Compare matched HDF5 actions and simulator trajectories across RoboLab runs",
    )
    trajectory_compare.add_argument("--reference", required=True)
    trajectory_compare.add_argument("--candidate", required=True)
    trajectory_compare.add_argument("--output", required=True)
    trajectory_compare.add_argument("--position-atol-m", type=float, default=1.0e-3)
    trajectory_compare.add_argument("--joint-atol-rad", type=float, default=1.0e-3)
    trajectory_compare.add_argument("--rotation-atol-deg", type=float, default=0.25)
    trajectory_compare.add_argument("--velocity-atol", type=float, default=1.0e-2)
    trajectory_compare.add_argument("--action-atol", type=float, default=1.0e-5)
    trajectory_compare.add_argument("--value-atol", type=float, default=1.0e-4)
    trajectory_compare.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Exit with status 1 when any episode exceeds the configured tolerances",
    )

    compare = subparsers.add_parser(
        "compare", help="Rank task-level policy disagreements for follow-up"
    )
    compare.add_argument("--left", required=True, help="Left episode_results.jsonl")
    compare.add_argument("--right", required=True, help="Right episode_results.jsonl")
    compare.add_argument("--left-policy", required=True)
    compare.add_argument("--right-policy", required=True)
    compare.add_argument("--output", required=True)

    provenance = subparsers.add_parser(
        "capture-provenance", help="Capture repositories, config, host, and GPU provenance"
    )
    provenance.add_argument("--config", required=True)
    provenance.add_argument("--study-root", default=".")
    provenance.add_argument("--policy-root")
    provenance.add_argument("--output", required=True)

    artifacts = subparsers.add_parser(
        "artifacts", help="Checkpoint, publish, discover, and retrieve cloud artifacts"
    )
    artifact_subparsers = artifacts.add_subparsers(dest="artifact_command", required=True)

    artifact_doctor = artifact_subparsers.add_parser(
        "doctor", help="Check rclone, credentials, bucket access, and optional write/read-back"
    )
    artifact_doctor.add_argument("--write-test", action="store_true")

    configure_b2 = artifact_subparsers.add_parser(
        "configure-b2",
        help="Dry-run or apply SSE-B2 and bounded lifecycle rules to the configured bucket",
    )
    configure_b2.add_argument("--producer-id", action="append", dest="producer_ids")
    configure_b2.add_argument(
        "--yes", action="store_true", help="Apply the displayed bucket-wide configuration"
    )

    checkpoint = artifact_subparsers.add_parser(
        "checkpoint", help="Copy quiescent run files into resumable working storage"
    )
    checkpoint.add_argument("--run-dir", required=True)
    checkpoint.add_argument("--run-id")
    checkpoint.add_argument("--min-age-seconds", type=int, default=120)

    resume = artifact_subparsers.add_parser(
        "resume", help="Restore an incomplete working checkpoint onto another machine"
    )
    resume.add_argument("--run-id", required=True)
    resume.add_argument("--destination", required=True)
    resume.add_argument("--producer-id")

    push = artifact_subparsers.add_parser(
        "push", help="Validate, manifest, upload, verify, and commit an immutable run"
    )
    push.add_argument("--run-dir", required=True)
    push.add_argument("--run-id")
    push.add_argument("--config", help="Optional experiment TOML used for validation metadata")
    push.add_argument("--study-id")
    push.add_argument("--stage")
    push.add_argument("--policy")
    push.add_argument("--audit-dir")

    artifact_list = artifact_subparsers.add_parser(
        "list", help="List completed runs visible to the configured key"
    )
    artifact_list.add_argument("--study-id")
    artifact_list.add_argument("--all-producers", action="store_true")

    pull = artifact_subparsers.add_parser(
        "pull", help="Materialize a completed run using a named download profile"
    )
    pull.add_argument("--run-id", required=True)
    pull.add_argument("--destination", required=True)
    pull.add_argument("--profile", choices=["metadata", "analysis", "full"], default="analysis")
    pull.add_argument("--producer-id")

    verify = artifact_subparsers.add_parser(
        "verify", help="Verify a local materialization or a remote completion marker"
    )
    verify_target = verify.add_mutually_exclusive_group(required=True)
    verify_target.add_argument("--run-dir")
    verify_target.add_argument("--run-id")
    verify.add_argument("--profile", choices=["metadata", "analysis", "full"], default="full")
    verify.add_argument("--producer-id")
    verify.add_argument("--study-id")
    verify.add_argument("--expected-manifest-sha256")

    catalog = artifact_subparsers.add_parser(
        "publish-catalog", help="Publish versioned run, episode, task, and Parquet catalogs"
    )
    catalog.add_argument("--output-dir", required=True)
    catalog.add_argument("--study-id")
    catalog.add_argument("--version")
    catalog.add_argument("--no-parquet", action="store_true")
    catalog.add_argument("--local-only", action="store_true")

    return parser


def _validate_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    value = {
        "valid": True,
        "config": str(config.path),
        "study_id": config.study.id,
        "enabled_policies": [policy.name for policy in config.enabled_policies()],
        "disabled_policies": [
            policy.name for policy in config.policies.values() if not policy.enabled
        ],
        "stages": {
            name: {
                "expected_tasks": stage.expected_task_count,
                "expected_episodes": stage.expected_episodes,
                "blocked_by": list(stage.blocked_by),
            }
            for name, stage in config.stages.items()
        },
    }
    _json(value)
    return 0


def _plan_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    jobs = build_jobs(
        config,
        args.stage,
        policy_names=args.policies,
        shards=args.shards,
        include_disabled=args.include_disabled,
    )
    output = write_plan(jobs, args.output)
    _json(
        {
            "plan": str(output),
            "jobs": [job.as_dict() for job in jobs],
            "total_expected_episodes": sum(job.expected_episodes for job in jobs),
        }
    )
    return 0


def _doctor_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.policy and args.policy not in config.policies:
        raise ConfigError(f"Unknown policy {args.policy!r}")
    report = run_doctor(config, policy_name=args.policy, check_server=not args.skip_server)
    _json(report)
    return 0 if report["ready"] else 1


def _execute_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    return execute_job(
        config,
        args.plan,
        job_id=args.job_id,
        yes=args.yes,
        study_root=Path(args.study_root).expanduser().resolve(),
    )


def _results_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.stage not in config.stages:
        raise ConfigError(f"Unknown stage {args.stage!r}")
    stage = config.stages[args.stage]
    run_dir = _materialize_run_reference(args.run_dir)
    report = validate_results(
        run_dir,
        expected_episodes=stage.expected_episodes,
        expected_tasks=stage.tasks or None,
    )
    _json(report)
    return 0 if report["valid"] else 1


def _trajectories_command(args: argparse.Namespace) -> int:
    run_dir = _materialize_run_reference(args.run_dir)
    rows = extract_trajectory_metrics(run_dir, args.policy)
    output = write_jsonl(rows, args.output)
    _json({"output": str(output), "episodes": len(rows)})
    return 0


def _trajectory_compare_command(args: argparse.Namespace) -> int:
    report = compare_trajectory_runs(
        args.reference,
        args.candidate,
        position_atol_m=args.position_atol_m,
        joint_atol_rad=args.joint_atol_rad,
        rotation_atol_deg=args.rotation_atol_deg,
        velocity_atol=args.velocity_atol,
        action_atol=args.action_atol,
        value_atol=args.value_atol,
    )
    output = write_trajectory_comparison(report, args.output)
    summary = dict(report["summary"])
    summary["output"] = str(output)
    _json(summary)
    return 1 if args.fail_on_drift and not report["summary"]["passed"] else 0


def _compare_command(args: argparse.Namespace) -> int:
    rows = compare_runs(
        args.left,
        args.right,
        left_policy=args.left_policy,
        right_policy=args.right_policy,
    )
    output = write_comparison(rows, args.output)
    _json({"output": str(output), "shared_tasks": len(rows), "top": rows[:10]})
    return 0


def _provenance_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    robolab = config.study.robolab_root()
    policy_root = Path(args.policy_root).expanduser().resolve() if args.policy_root else None
    value = capture_provenance(
        config_path=config.path,
        study_root=Path(args.study_root).expanduser().resolve(),
        robolab_root=robolab,
        policy_root=policy_root,
    )
    output = write_provenance(value, args.output)
    _json({"output": str(output)})
    return 0


def _materialize_run_reference(value: str) -> Path:
    local = Path(value).expanduser()
    if local.exists():
        return local.resolve()
    return ArtifactManager.from_environment(require_producer=False).materialize(
        value,
        profile="analysis",
    )


def _artifact_study_id(args: argparse.Namespace) -> str:
    if getattr(args, "study_id", None):
        return str(args.study_id)
    config_path = getattr(args, "config", None)
    if config_path:
        return load_config(config_path).study.id
    raise ArtifactError("Pass --study-id or --config")


def _artifacts_command(args: argparse.Namespace) -> int:
    action = args.artifact_command
    if action == "doctor":
        manager = ArtifactManager.from_environment(require_producer=args.write_test)
        report = manager.doctor(write_test=args.write_test)
    elif action == "configure-b2":
        settings = StorageSettings.from_environment()
        producers = args.producer_ids or [settings.producer_id]
        report = (
            apply_bucket_configuration(settings, producers)
            if args.yes
            else {"applied": False, "plan": configuration_plan(settings, producers)}
        )
    elif action == "checkpoint":
        manager = ArtifactManager.from_environment()
        report = manager.checkpoint(
            args.run_dir,
            run_id=args.run_id,
            min_age_seconds=args.min_age_seconds,
        )
    elif action == "resume":
        manager = ArtifactManager.from_environment(require_producer=False)
        report = manager.resume(
            args.run_id,
            destination=args.destination,
            producer_id=args.producer_id,
        )
    elif action == "push":
        config = load_config(args.config) if args.config else None
        study_id = _artifact_study_id(args)
        manager = ArtifactManager.from_environment()
        expected_episodes = None
        expected_tasks = None
        if args.stage and config:
            if args.stage not in config.stages:
                raise ConfigError(f"Unknown stage {args.stage!r}")
            stage_config = config.stages[args.stage]
            expected_episodes = stage_config.expected_episodes
            expected_tasks = stage_config.tasks or None
        report = manager.finalize(
            args.run_dir,
            study_id=study_id,
            run_id=args.run_id,
            stage=args.stage,
            policy=args.policy,
            expected_episodes=expected_episodes,
            expected_tasks=expected_tasks,
            audit_dir=args.audit_dir,
        )
    elif action == "list":
        manager = ArtifactManager.from_environment(require_producer=False)
        runs = manager.list_runs(
            study_id=args.study_id,
            all_producers=args.all_producers,
        )
        report = {"runs": runs, "count": len(runs)}
    elif action == "pull":
        manager = ArtifactManager.from_environment(require_producer=False)
        report = manager.pull(
            args.run_id,
            destination=args.destination,
            profile=args.profile,
            producer_id=args.producer_id,
        )
    elif action == "verify":
        if args.run_dir:
            report = verify_local_run(args.run_dir, profile=args.profile)
        else:
            manager = ArtifactManager.from_environment(require_producer=False)
            report = manager.verify_remote(
                args.run_id,
                expected_manifest_sha256=args.expected_manifest_sha256,
                study_id=args.study_id,
                producer_id=args.producer_id,
            )
    elif action == "publish-catalog":
        manager = ArtifactManager.from_environment(require_producer=False)
        report = publish_catalog(
            manager,
            output_dir=args.output_dir,
            study_id=args.study_id,
            version=args.version,
            parquet=not args.no_parquet,
            upload=not args.local_only,
        )
    else:  # pragma: no cover - argparse enforces the choices
        raise ArtifactError(f"Unknown artifacts command: {action}")
    _json(report)
    return 0 if report.get("valid", True) else 1


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    commands = {
        "validate-config": _validate_command,
        "plan": _plan_command,
        "doctor": _doctor_command,
        "execute": _execute_command,
        "validate-results": _results_command,
        "extract-trajectories": _trajectories_command,
        "compare-trajectories": _trajectory_compare_command,
        "compare": _compare_command,
        "capture-provenance": _provenance_command,
        "artifacts": _artifacts_command,
    }
    try:
        return commands[args.command](args)
    except (
        ConfigError,
        ResultsError,
        AnalysisDependencyError,
        ArtifactError,
        StorageError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
