from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _command_output(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_state(path: Path | None) -> dict[str, Any] | None:
    if path is None or not (path / ".git").exists():
        return None
    revision = _command_output(["git", "rev-parse", "HEAD"], cwd=path)
    status = _command_output(["git", "status", "--short"], cwd=path)
    remote = _command_output(["git", "remote", "get-url", "origin"], cwd=path)
    return {
        "path": str(path),
        "revision": revision,
        "dirty": bool(status),
        "status": status.splitlines() if status else [],
        "origin": remote,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_provenance(
    *,
    config_path: Path,
    study_root: Path,
    robolab_root: Path | None,
    policy_root: Path | None = None,
    job: dict[str, object] | None = None,
) -> dict[str, Any]:
    allowlisted_environment = {}
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "OMNI_KIT_ACCEPT_EULA",
        "ROBOLAB_ROOT",
        "ROBOLAB_OUTPUT_ROOT",
        "ROBOLAB_REPOSITORY_URL",
        "ROBOLAB_REF",
        "ISAAC_STACK",
        "MINIMUM_NVIDIA_DRIVER_MAJOR",
        "OPENPI_ROOT",
        "DREAMZERO_ROOT",
    ):
        if key in os.environ:
            allowlisted_environment[key] = os.environ[key]

    gpu = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "job": job,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "hostname": platform.node(),
            "gpu": gpu.splitlines() if gpu else [],
        },
        "environment": allowlisted_environment,
        "repositories": {
            "study": git_state(study_root),
            "robolab": git_state(robolab_root),
            "policy": git_state(policy_root),
        },
    }


def write_provenance(value: dict[str, Any], path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output
