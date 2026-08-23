from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import threading
from pathlib import Path
from typing import Any

from .access import AccessError, MandaAccessClient
from .artifacts import ArtifactError, ArtifactManager, ensure_run_id
from .auth import AuthError
from .config import ConfigError, ExperimentConfig
from .planning import read_plan
from .provenance import capture_provenance, write_provenance
from .storage import StorageError


def select_job(plan_path: str | Path, job_id: str | None) -> dict[str, Any]:
    jobs = read_plan(plan_path)
    if job_id:
        matches = [job for job in jobs if job["job_id"] == job_id]
        if len(matches) != 1:
            raise ConfigError(f"Expected one job named {job_id!r}, found {len(matches)}")
        return matches[0]  # type: ignore[return-value]
    if len(jobs) != 1:
        raise ConfigError("Plan contains multiple jobs; pass --job-id")
    return jobs[0]  # type: ignore[return-value]


def _check_server(host: str, port: int) -> None:
    try:
        with socket.create_connection((host, port), timeout=3):
            pass
    except OSError as exc:
        raise ConfigError(f"Policy server is not reachable at {host}:{port}: {exc}") from exc


def execute_job(
    config: ExperimentConfig,
    plan_path: str | Path,
    *,
    job_id: str | None,
    yes: bool,
    study_root: Path,
) -> int:
    job = select_job(plan_path, job_id)
    command = job.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise ConfigError("Selected job has an invalid command")
    if not yes:
        print(shlex.join(command))
        print("Dry run only. Pass --yes to execute.")
        return 0

    robolab_root = config.study.robolab_root()
    if robolab_root is None or not robolab_root.is_dir():
        raise ConfigError(f"{config.study.robolab_root_env} is not a valid directory")
    output_root = config.study.output_root()
    if output_root is None:
        raise ConfigError(f"{config.study.output_root_env} is not set")
    output_root.mkdir(parents=True, exist_ok=True)

    host = str(job.get("server_host", "127.0.0.1"))
    port = int(job.get("server_port", 8000))
    _check_server(host, port)

    policy_name = str(job["policy"])
    policy = config.policies[policy_name]
    policy_root = None
    if policy.server_cwd_env and os.environ.get(policy.server_cwd_env):
        policy_root = Path(os.environ[policy.server_cwd_env]).expanduser().resolve()

    output_folder = job.get("output_folder")
    if not isinstance(output_folder, str) or not output_folder:
        raise ConfigError("Selected job has no output folder")
    run_dir = output_root / output_folder
    audit_dir = run_dir / "_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    provenance = capture_provenance(
        config_path=config.path,
        study_root=study_root,
        robolab_root=robolab_root,
        policy_root=policy_root,
        job=job,
    )
    write_provenance(provenance, audit_dir / "provenance.json")

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(job.get("simulator_gpu", 0))
    environment.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
    log_path = audit_dir / "run.log"
    checkpoint_interval_raw = os.environ.get("ROBOLAB_STORAGE_CHECKPOINT_INTERVAL_SECONDS", "0")
    try:
        checkpoint_interval = int(checkpoint_interval_raw)
    except ValueError as exc:
        raise ConfigError("ROBOLAB_STORAGE_CHECKPOINT_INTERVAL_SECONDS must be an integer") from exc
    if checkpoint_interval < 0:
        raise ConfigError("ROBOLAB_STORAGE_CHECKPOINT_INTERVAL_SECONDS must be >= 0")
    access_client: MandaAccessClient | None = None
    producer_id: str | None = None

    def artifact_manager(purpose: str) -> tuple[ArtifactManager, str]:
        nonlocal access_client, producer_id
        if access_client is None:
            access_client = MandaAccessClient.from_environment()
        if producer_id is None:
            producer_id = access_client.ensure_producer()
        run_id = ensure_run_id(run_dir, producer_id)
        return (
            access_client.upload_manager(
                str(job["study_id"]),
                purpose=purpose,
                run_id=run_id,
                producer_id=producer_id,
            ),
            run_id,
        )

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=robolab_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        checkpoint_stop = threading.Event()
        checkpoint_thread = None

        def checkpoint_loop() -> None:
            while not checkpoint_stop.wait(checkpoint_interval):
                try:
                    manager, run_id = artifact_manager("checkpoint")
                    report = manager.checkpoint(run_dir, run_id=run_id)
                    print(f"Periodic artifact checkpoint: {report['run_id']}")
                except (AccessError, ArtifactError, AuthError, StorageError) as exc:
                    print(f"Warning: periodic artifact checkpoint failed: {exc}")

        if checkpoint_interval:
            checkpoint_thread = threading.Thread(
                target=checkpoint_loop,
                name="artifact-checkpoint",
                daemon=True,
            )
            checkpoint_thread.start()
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
                log.flush()
            return_code = process.wait()
        finally:
            checkpoint_stop.set()
            if checkpoint_thread is not None:
                checkpoint_thread.join(timeout=30)

    completion = {"job_id": job["job_id"], "exit_code": return_code}
    with (audit_dir / "completion.json").open("w", encoding="utf-8") as handle:
        json.dump(completion, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if return_code != 0:
        if os.environ.get("ROBOLAB_STORAGE_CHECKPOINT_ON_FAILURE") == "1":
            try:
                manager, run_id = artifact_manager("checkpoint")
                report = manager.checkpoint(run_dir, run_id=run_id)
                print(
                    "Failure checkpoint: "
                    + json.dumps(
                        {"run_id": report["run_id"], "remote": report["remote"]},
                        sort_keys=True,
                    )
                )
            except (AccessError, ArtifactError, AuthError, StorageError) as exc:
                print(f"Warning: failure checkpoint did not complete: {exc}")
        return return_code

    if os.environ.get("ROBOLAB_STORAGE_AUTO_FINALIZE") == "1":
        manager, run_id = artifact_manager("finalize")
        task_values = job.get("tasks")
        expected_tasks = (
            tuple(str(value) for value in task_values)
            if isinstance(task_values, list) and task_values
            else None
        )
        expected_raw = job.get("expected_episodes")
        expected_episodes = int(expected_raw) if isinstance(expected_raw, int) else None
        try:
            report = manager.finalize(
                run_dir,
                study_id=str(job["study_id"]),
                run_id=run_id,
                stage=str(job["stage"]),
                policy=str(job["policy"]),
                expected_episodes=expected_episodes,
                expected_tasks=expected_tasks,
            )
        except (AccessError, ArtifactError, AuthError, StorageError) as exc:
            raise ConfigError(
                f"Experiment completed, but durable artifact finalization failed: {exc}"
            ) from exc
        receipts = output_root / ".artifact-receipts"
        receipts.mkdir(parents=True, exist_ok=True)
        receipt_path = receipts / f"{report['run_id']}.json"
        with receipt_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(
            "Durable artifact: "
            + json.dumps(
                {
                    "run_id": report["run_id"],
                    "manifest_sha256": report["manifest_sha256"],
                    "receipt": str(receipt_path),
                },
                sort_keys=True,
            )
        )
    return return_code
