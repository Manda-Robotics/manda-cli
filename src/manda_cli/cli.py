from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .access import AccessError, MandaAccessClient, auth_settings_report
from .artifacts import ArtifactError, ensure_run_id, estimate_upload_bytes, verify_local_run
from .auth import AuthError, AuthManager, AuthSettings, SessionStore, save_client_config
from .config import ConfigError, load_config
from .doctor import run_doctor
from .execution import execute_job
from .interesting import compare_runs, write_comparison
from .planning import build_jobs, write_plan
from .provenance import capture_provenance, write_provenance
from .results import ResultsError, validate_results
from .storage import StorageError
from .trajectories import (
    AnalysisDependencyError,
    compare_trajectory_runs,
    extract_trajectory_metrics,
    write_jsonl,
    write_trajectory_comparison,
)


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _styled(text: str, *codes: str, color: bool) -> str:
    if not color:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"


def _format_bytes(value: object) -> str:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return "unknown"
    amount = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    unit = units[0]
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            break
        amount /= 1024
    if unit == "B":
        return f"{int(amount)} {unit}"
    precision = 0 if amount.is_integer() else 1
    return f"{amount:.{precision}f} {unit}"


def _login_welcome(report: dict[str, Any], *, color: bool | None = None) -> None:
    if color is None:
        color = sys.stdout.isatty() and "NO_COLOR" not in os.environ

    user = report.get("user") if isinstance(report.get("user"), dict) else {}
    email = user.get("email") or "Manda user"
    producer_id = report.get("producer_id") or "not registered"
    teams = report.get("teams") if isinstance(report.get("teams"), list) else []
    studies = report.get("studies") if isinstance(report.get("studies"), list) else []
    quota = report.get("quota") if isinstance(report.get("quota"), dict) else {}
    available = _format_bytes(quota.get("available_bytes"))
    total = _format_bytes(quota.get("quota_bytes"))

    brand = _styled("MANDA", "1", "36", color=color)
    connected = _styled("CONNECTED", "1", "32", color=color)
    rule = _styled("━" * 48, "2", "36", color=color)

    def label(value: str) -> str:
        return _styled(f"{value:<10}", "2", color=color)

    print()
    print(f"  {brand}  {_styled('•', '2', color=color)}  {connected}")
    print(f"  {rule}")
    print()
    print(f"  {_styled('Welcome,', '1', color=color)} {email}")
    print()
    print(f"  {label('Client')}{producer_id}")
    print(f"  {label('Storage')}{available} available of {total}")
    print(f"  {label('Access')}{len(studies)} studies · {len(teams)} teams")

    if not studies:
        print()
        print(f"  {_styled('Ready for your first study?', '1', color=color)}")
        if not teams:
            print("    manda teams create --id my-team --name \"My team\"")
            print(
                "    manda studies create --id my-study --title \"My study\" "
                "--owner-team my-team"
            )
        else:
            print("    manda studies create --id my-study --title \"My study\"")
    print()


def _secure_token_file(path: str) -> str:
    token_path = Path(path).expanduser().resolve()
    if not token_path.is_file():
        raise AuthError(f"Access-token file does not exist: {token_path}")
    if token_path.stat().st_mode & 0o077:
        raise AuthError(f"Access-token file must have mode 600: {token_path}")
    token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise AuthError("Access-token file is empty")
    return token


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

    auth = subparsers.add_parser("auth", help="Sign in and manage this CLI user session")
    auth_subparsers = auth.add_subparsers(dest="auth_command", required=True)
    auth_login = auth_subparsers.add_parser(
        "login", help="Sign in with an invite-only email OTP or an access-token file"
    )
    auth_login.add_argument("--email")
    auth_login.add_argument("--access-token-file")
    auth_login.add_argument("--client-label")
    auth_login.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    auth_subparsers.add_parser("status", help="Show the current user, memberships, and client")
    auth_subparsers.add_parser("logout", help="Revoke and remove this CLI session")
    auth_configure = auth_subparsers.add_parser(
        "configure", help="Use a custom Manda service or restore the hosted defaults"
    )
    auth_configure.add_argument("--supabase-url")
    auth_configure.add_argument("--publishable-key-file")
    auth_configure.add_argument("--api-url")
    auth_configure.add_argument("--hosted", action="store_true")

    teams = subparsers.add_parser("teams", help="Create teams and manage their members")
    teams_subparsers = teams.add_subparsers(dest="teams_command", required=True)
    teams_subparsers.add_parser("list", help="List teams visible to the signed-in user")
    team_create = teams_subparsers.add_parser("create", help="Create a team owned by this user")
    team_create.add_argument("--id", required=True, dest="team_id")
    team_create.add_argument("--name", required=True)
    team_members = teams_subparsers.add_parser("members", help="List team members")
    team_members.add_argument("team_id")
    team_invite = teams_subparsers.add_parser("invite", help="Invite or grant a team member")
    team_invite.add_argument("team_id")
    team_invite.add_argument("email")
    team_invite.add_argument("--role", choices=["member", "admin", "owner"], default="member")

    studies = subparsers.add_parser("studies", help="Create, discover, and share studies")
    studies_subparsers = studies.add_subparsers(dest="studies_command", required=True)
    studies_subparsers.add_parser("list", help="List studies visible to the signed-in user")
    study_show = studies_subparsers.add_parser("show", help="Show one accessible study")
    study_show.add_argument("study_id")
    study_create = studies_subparsers.add_parser("create", help="Create a study")
    study_create.add_argument("--id", required=True, dest="study_id")
    study_create.add_argument("--title", required=True)
    study_create.add_argument("--owner-team")
    study_grant = studies_subparsers.add_parser("grant", help="Share a study with a user or team")
    study_grant.add_argument("study_id")
    grant_target = study_grant.add_mutually_exclusive_group(required=True)
    grant_target.add_argument("--email")
    grant_target.add_argument("--team", dest="team_id")
    study_grant.add_argument(
        "--role", choices=["viewer", "contributor", "owner"], default="viewer"
    )

    quota = subparsers.add_parser("quota", help="Inspect storage use and reservations")
    quota.add_subparsers(dest="quota_command", required=True).add_parser(
        "status", help="Show this user's storage quota"
    )

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
        "doctor", help="Check authenticated storage access and optional write/read-back"
    )
    artifact_doctor.add_argument("--study-id")
    artifact_doctor.add_argument("--write-test", action="store_true")

    checkpoint = artifact_subparsers.add_parser(
        "checkpoint", help="Copy quiescent run files into resumable working storage"
    )
    checkpoint.add_argument("--run-dir", required=True)
    checkpoint.add_argument("--study-id")
    checkpoint.add_argument("--run-id")
    checkpoint.add_argument("--min-age-seconds", type=int, default=120)

    resume = artifact_subparsers.add_parser(
        "resume", help="Restore an incomplete working checkpoint onto another machine"
    )
    resume.add_argument("--run-id", required=True)
    resume.add_argument("--study-id")
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
        "list", help="List completed runs visible to the signed-in user"
    )
    artifact_list.add_argument("--study-id")

    pull = artifact_subparsers.add_parser(
        "pull", help="Materialize a completed run using a named download profile"
    )
    pull.add_argument("--run-id", required=True)
    pull.add_argument("--destination", required=True)
    pull.add_argument("--profile", choices=["metadata", "analysis", "full"], default="analysis")
    pull.add_argument("--study-id")

    verify = artifact_subparsers.add_parser(
        "verify", help="Verify a local materialization or a remote completion marker"
    )
    verify_target = verify.add_mutually_exclusive_group(required=True)
    verify_target.add_argument("--run-dir")
    verify_target.add_argument("--run-id")
    verify.add_argument("--profile", choices=["metadata", "analysis", "full"], default="full")
    verify.add_argument("--study-id")
    verify.add_argument("--expected-manifest-sha256")

    return parser


def _auth_command(args: argparse.Namespace) -> int:
    if args.auth_command == "configure":
        current = AuthSettings.from_environment()
        if args.hosted:
            if args.supabase_url or args.publishable_key_file or args.api_url:
                raise AuthError("Pass --hosted by itself")
            current.config_file.unlink(missing_ok=True)
            SessionStore(current.session_file).clear()
            report = AuthSettings.from_environment().report()
            report["session_removed"] = True
            _json(report)
            return 0
        if not args.supabase_url or not args.publishable_key_file:
            raise AuthError(
                "Custom configuration requires --supabase-url and --publishable-key-file"
            )
        key_file = Path(args.publishable_key_file).expanduser().resolve()
        if not key_file.is_file():
            raise AuthError(f"Publishable-key file does not exist: {key_file}")
        key = key_file.read_text(encoding="utf-8").strip()
        path = save_client_config(
            supabase_url=args.supabase_url,
            publishable_key=key,
            api_url=args.api_url,
        )
        SessionStore(current.session_file).clear()
        report = AuthSettings.from_environment().report()
        report.update({"config_file": str(path), "session_removed": True})
        _json(report)
        return 0

    manager = AuthManager.from_environment()
    if args.auth_command == "login":
        if args.email and args.access_token_file:
            raise AuthError("Pass either --email or --access-token-file, not both")
        if args.access_token_file:
            session = manager.login_access_token(_secure_token_file(args.access_token_file))
        else:
            email = (args.email or input("Email: ")).strip()
            manager.client.start_email_otp(email)
            print("A one-time sign-in code was sent to the invited account.")
            otp = getpass.getpass("Email OTP: ").strip()
            session = manager.login_email(email, otp)
        access = MandaAccessClient(manager)
        producer_id = access.ensure_producer(args.client_label)
        remote = access.session_report()
        report = {
            "authenticated": True,
            "user": {"user_id": session.user_id, "email": session.email},
            "producer_id": producer_id,
            "session_file": str(manager.settings.session_file),
            "teams": remote.get("teams", []),
            "studies": remote.get("studies", []),
            "quota": remote.get("quota"),
        }
    elif args.auth_command == "status":
        session = manager.ensure_session()
        report = {
            "authenticated": True,
            "configuration": auth_settings_report(),
            "local_session": session.report(),
            "remote_session": MandaAccessClient(manager).session_report(),
        }
    elif args.auth_command == "logout":
        manager.logout()
        report = {"authenticated": False, "session_removed": True}
    else:  # pragma: no cover - argparse enforces the choices
        raise AuthError(f"Unknown auth command: {args.auth_command}")
    if args.auth_command == "login" and not args.json:
        _login_welcome(report)
    else:
        _json(report)
    return 0


def _teams_command(args: argparse.Namespace) -> int:
    access = MandaAccessClient.from_environment()
    if args.teams_command == "list":
        teams = access.list_teams()
        report = {"teams": teams, "count": len(teams)}
    elif args.teams_command == "create":
        report = access.create_team(args.team_id, args.name)
    elif args.teams_command == "members":
        members = access.list_team_members(args.team_id)
        report = {"team_id": args.team_id, "members": members, "count": len(members)}
    elif args.teams_command == "invite":
        report = access.invite_team_member(args.team_id, args.email, args.role)
    else:  # pragma: no cover - argparse enforces the choices
        raise AccessError(f"Unknown teams command: {args.teams_command}")
    _json(report)
    return 0


def _studies_command(args: argparse.Namespace) -> int:
    access = MandaAccessClient.from_environment()
    if args.studies_command == "list":
        studies = access.list_studies()
        report = {"studies": studies, "count": len(studies)}
    elif args.studies_command == "show":
        matches = [
            study for study in access.list_studies() if study.get("study_id") == args.study_id
        ]
        if not matches:
            raise AccessError(f"Study {args.study_id!r} is not accessible")
        report = matches[0]
    elif args.studies_command == "create":
        report = access.create_study(
            args.study_id,
            args.title,
            owner_team_id=args.owner_team,
        )
    elif args.studies_command == "grant":
        report = access.grant_study(
            args.study_id,
            args.role,
            email=args.email,
            team_id=args.team_id,
        )
    else:  # pragma: no cover - argparse enforces the choices
        raise AccessError(f"Unknown studies command: {args.studies_command}")
    _json(report)
    return 0


def _quota_command(args: argparse.Namespace) -> int:
    if args.quota_command != "status":  # pragma: no cover - argparse enforces the choices
        raise AccessError(f"Unknown quota command: {args.quota_command}")
    _json(MandaAccessClient.from_environment().quota_status())
    return 0


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
    access = MandaAccessClient.from_environment()
    run = access.find_run(value)
    cache_root = Path(
        os.environ.get(
            "ROBOLAB_ARTIFACT_CACHE",
            Path.home() / ".cache" / "manda" / "runs",
        )
    ).expanduser().resolve()
    destination = cache_root / run.run_id
    access.download_manager(run).pull(
        run.run_id,
        destination=destination,
        profile="analysis",
        producer_id=run.producer_id,
        study_id=run.study_id,
    )
    return destination


def _artifact_study_id(args: argparse.Namespace, access: MandaAccessClient) -> str:
    if getattr(args, "study_id", None):
        return str(args.study_id)
    config_path = getattr(args, "config", None)
    if config_path:
        return load_config(config_path).study.id
    return access.resolve_study(None, writable=True)


def _artifacts_command(args: argparse.Namespace) -> int:
    action = args.artifact_command
    if action == "doctor":
        access = MandaAccessClient.from_environment()
        study_id = access.resolve_study(args.study_id, writable=True)
        manager = access.upload_manager(study_id, purpose="doctor", requested_bytes=1024)
        report = manager.doctor(write_test=args.write_test)
    elif action == "checkpoint":
        run_dir = Path(args.run_dir).expanduser().resolve()
        if not run_dir.is_dir():
            raise ArtifactError(f"Run directory does not exist: {run_dir}")
        access = MandaAccessClient.from_environment()
        study_id = access.resolve_study(args.study_id, writable=True)
        producer_id = access.ensure_producer()
        run_id = ensure_run_id(run_dir, producer_id, args.run_id)
        manager = access.upload_manager(
            study_id,
            purpose="checkpoint",
            run_id=run_id,
            producer_id=producer_id,
            requested_bytes=estimate_upload_bytes(
                run_dir, min_age_seconds=args.min_age_seconds
            ),
        )
        report = manager.checkpoint(
            run_dir,
            run_id=run_id,
            min_age_seconds=args.min_age_seconds,
        )
    elif action == "resume":
        access = MandaAccessClient.from_environment()
        study_id = access.resolve_study(args.study_id, writable=True)
        producer_id = args.producer_id or access.ensure_producer()
        manager = access.resume_manager(
            study_id,
            args.run_id,
            producer_id,
        )
        report = manager.resume(
            args.run_id,
            destination=args.destination,
            producer_id=producer_id,
        )
    elif action == "push":
        config = load_config(args.config) if args.config else None
        run_dir = Path(args.run_dir).expanduser().resolve()
        if not run_dir.is_dir():
            raise ArtifactError(f"Run directory does not exist: {run_dir}")
        access = MandaAccessClient.from_environment()
        study_id = _artifact_study_id(args, access)
        producer_id = access.ensure_producer()
        run_id = ensure_run_id(run_dir, producer_id, args.run_id)
        requested_bytes = estimate_upload_bytes(run_dir)
        if args.audit_dir:
            requested_bytes += estimate_upload_bytes(args.audit_dir, metadata_allowance=0)
        manager = access.upload_manager(
            study_id,
            purpose="finalize",
            run_id=run_id,
            producer_id=producer_id,
            requested_bytes=requested_bytes,
        )
        expected_episodes = None
        expected_tasks = None
        if args.stage and config:
            if args.stage not in config.stages:
                raise ConfigError(f"Unknown stage {args.stage!r}")
            stage_config = config.stages[args.stage]
            expected_episodes = stage_config.expected_episodes
            expected_tasks = stage_config.tasks or None
        report = manager.finalize(
            run_dir,
            study_id=study_id,
            run_id=run_id,
            stage=args.stage,
            policy=args.policy,
            expected_episodes=expected_episodes,
            expected_tasks=expected_tasks,
            audit_dir=args.audit_dir,
        )
    elif action == "list":
        runs = MandaAccessClient.from_environment().list_runs(study_id=args.study_id)
        report = {"runs": runs, "count": len(runs)}
    elif action == "pull":
        access = MandaAccessClient.from_environment()
        run = access.find_run(args.run_id, study_id=args.study_id)
        report = access.download_manager(run).pull(
            run.run_id,
            destination=args.destination,
            profile=args.profile,
            producer_id=run.producer_id,
            study_id=run.study_id,
        )
    elif action == "verify":
        if args.run_dir:
            report = verify_local_run(args.run_dir, profile=args.profile)
        else:
            access = MandaAccessClient.from_environment()
            run = access.find_run(args.run_id, study_id=args.study_id)
            report = access.download_manager(run).verify_remote(
                run.run_id,
                expected_manifest_sha256=args.expected_manifest_sha256,
                study_id=run.study_id,
                producer_id=run.producer_id,
            )
    else:  # pragma: no cover - argparse enforces the choices
        raise ArtifactError(f"Unknown artifacts command: {action}")
    _json(report)
    return 0 if report.get("valid", True) else 1


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    commands = {
        "auth": _auth_command,
        "teams": _teams_command,
        "studies": _studies_command,
        "quota": _quota_command,
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
        AccessError,
        ArtifactError,
        AuthError,
        StorageError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
