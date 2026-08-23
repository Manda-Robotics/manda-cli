from __future__ import annotations

import os
import platform
import shutil
import socket
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import ExperimentConfig
from .provenance import git_state


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def _path_check(name: str, path: Path | None, required: str | None = None) -> Check:
    if path is None:
        return Check(name, "error", "Environment variable is not set")
    if not path.is_dir():
        return Check(name, "error", f"Directory does not exist: {path}")
    if required and not (path / required).is_file():
        return Check(name, "error", f"Missing {required} below {path}")
    return Check(name, "ok", str(path))


def _server_check(host: str, port: int) -> Check:
    try:
        with socket.create_connection((host, port), timeout=1.5):
            pass
    except OSError as exc:
        return Check("policy_server", "error", f"Cannot reach {host}:{port}: {exc}")
    return Check("policy_server", "ok", f"Reachable at {host}:{port}")


def run_doctor(
    config: ExperimentConfig,
    *,
    policy_name: str | None = None,
    check_server: bool = True,
) -> dict[str, object]:
    checks: list[Check] = []
    if platform.system() == "Linux" and platform.machine() in {"x86_64", "AMD64"}:
        checks.append(Check("platform", "ok", platform.platform()))
    else:
        checks.append(
            Check(
                "platform",
                "warning",
                f"{platform.platform()} is suitable for planning, not RoboLab execution",
            )
        )

    for executable in ("git", "uv", "ffmpeg"):
        found = shutil.which(executable)
        status = "ok" if found else "error"
        checks.append(Check(executable, status, found or "Not found on PATH"))
    nvidia = shutil.which("nvidia-smi")
    checks.append(
        Check(
            "nvidia-smi",
            "ok" if nvidia else "warning",
            nvidia or "Not present (expected on the remote GPU host only)",
        )
    )

    robolab_root = config.study.robolab_root()
    checks.append(_path_check("robolab_root", robolab_root, "pyproject.toml"))
    output_root = config.study.output_root()
    if output_root is None:
        checks.append(Check("output_root", "error", "Output root environment variable is not set"))
    elif output_root.exists() and output_root.is_dir():
        checks.append(Check("output_root", "ok", str(output_root)))
    else:
        checks.append(
            Check("output_root", "warning", f"Will be created when executing: {output_root}")
        )

    if config.study.robolab_expected_revision == "UNPINNED":
        checks.append(
            Check(
                "robolab_revision",
                "warning",
                "Revision is not frozen; resolve DEC-003 after the smoke test",
            )
        )
    else:
        state = git_state(robolab_root)
        actual = state.get("revision") if state else None
        checks.append(
            Check(
                "robolab_revision",
                "ok" if actual == config.study.robolab_expected_revision else "error",
                f"expected={config.study.robolab_expected_revision} actual={actual}",
            )
        )

    policies = [config.policies[policy_name]] if policy_name else config.enabled_policies()
    for policy in policies:
        if robolab_root is not None:
            checks.append(_path_check(f"runner:{policy.name}", robolab_root, policy.runner))
        if policy.server_cwd_env:
            raw = os.environ.get(policy.server_cwd_env)
            path = Path(raw).expanduser() if raw else None
            checks.append(_path_check(f"server_repo:{policy.name}", path))
        if check_server and policy.requires_server:
            server = _server_check(policy.server_host(), policy.server_port())
            checks.append(Check(f"policy_server:{policy.name}", server.status, server.detail))

    counts = {
        status: sum(check.status == status for check in checks)
        for status in ("ok", "warning", "error")
    }
    return {
        "ready": counts["error"] == 0,
        "counts": counts,
        "checks": [asdict(check) for check in checks],
    }
