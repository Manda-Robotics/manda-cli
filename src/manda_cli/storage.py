from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class StorageError(ValueError):
    """Raised when artifact storage is not configured or a transfer fails."""


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def safe_identifier(value: str, label: str) -> str:
    if not _SAFE_ID.fullmatch(value):
        raise StorageError(
            f"{label} must start with an alphanumeric character and contain only "
            "letters, digits, '.', '_', or '-': {value!r}"
        )
    return value


def remote_join(root: str, *parts: str) -> str:
    """Join slash-delimited object keys without damaging the rclone `name:` prefix."""
    clean = [str(part).strip("/") for part in parts if str(part).strip("/")]
    suffix = "/".join(clean)
    return f"{root.rstrip('/')}/{suffix}" if suffix else root.rstrip("/")


@dataclass(frozen=True)
class StorageSettings:
    remote: str
    prefix: str
    producer_id: str
    bucket_id: str | None = None
    key_id: str | None = None
    application_key_file: Path | None = None
    application_key: str | None = field(default=None, repr=False)
    endpoint: str | None = None
    region: str | None = None
    rclone_binary: str = "rclone"
    lease_expires_at: str | None = None
    lease_name_prefix: str | None = None
    lease_study_id: str | None = None
    lease_reservation_id: str | None = None
    lease_requested_bytes: int | None = None

    @classmethod
    def from_environment(cls, *, require_producer: bool = True) -> StorageSettings:
        remote = os.environ.get("ROBOLAB_STORAGE_REMOTE", "").strip()
        bucket = os.environ.get("ROBOLAB_STORAGE_BUCKET", "").strip()
        if not remote:
            if not bucket:
                raise StorageError(
                    "Set ROBOLAB_STORAGE_REMOTE (for a configured rclone remote) or "
                    "ROBOLAB_STORAGE_BUCKET"
                )
            remote = f"robolab:{bucket}"
        if ":" not in remote:
            raise StorageError("ROBOLAB_STORAGE_REMOTE must include an rclone remote name and ':'")

        producer = os.environ.get("ROBOLAB_STORAGE_PRODUCER_ID", "").strip()
        if require_producer and not producer:
            raise StorageError("Set ROBOLAB_STORAGE_PRODUCER_ID")
        if producer:
            safe_identifier(producer, "producer ID")

        prefix = os.environ.get("ROBOLAB_STORAGE_PREFIX", "robolab/v1").strip("/")
        if not prefix or any(part in {".", ".."} for part in prefix.split("/")):
            raise StorageError("ROBOLAB_STORAGE_PREFIX must be a safe, non-empty object prefix")

        key_file_raw = os.environ.get("ROBOLAB_STORAGE_APPLICATION_KEY_FILE", "").strip()
        key_file = Path(key_file_raw).expanduser().resolve() if key_file_raw else None
        bucket_id = os.environ.get("ROBOLAB_STORAGE_BUCKET_ID", "").strip() or None
        key_id = os.environ.get("ROBOLAB_STORAGE_KEY_ID", "").strip() or None
        endpoint = os.environ.get("ROBOLAB_STORAGE_ENDPOINT", "").strip() or None
        region = os.environ.get("ROBOLAB_STORAGE_REGION", "").strip() or None
        if region is None and endpoint:
            match = re.search(r"s3\.([^.]+)\.backblazeb2\.com", endpoint)
            region = match.group(1) if match else None

        supplied = (key_file is not None, key_id is not None, endpoint is not None)
        if any(supplied) and not all(supplied):
            raise StorageError(
                "Direct B2 configuration requires ROBOLAB_STORAGE_KEY_ID, "
                "ROBOLAB_STORAGE_APPLICATION_KEY_FILE, and ROBOLAB_STORAGE_ENDPOINT"
            )
        if key_file is not None:
            if not key_file.is_file():
                raise StorageError(f"Application-key file does not exist: {key_file}")
            mode = key_file.stat().st_mode & 0o777
            if mode & 0o077:
                raise StorageError(
                    f"Application-key file must not be group/world accessible: {key_file}"
                )

        binary = os.environ.get("ROBOLAB_STORAGE_RCLONE_BINARY", "rclone").strip()
        return cls(
            remote=remote,
            prefix=prefix,
            producer_id=producer,
            bucket_id=bucket_id,
            key_id=key_id,
            application_key_file=key_file,
            endpoint=endpoint,
            region=region,
            rclone_binary=binary,
        )

    @classmethod
    def from_lease(cls, lease: dict[str, Any]) -> StorageSettings:
        def required(name: str) -> str:
            value = lease.get(name)
            if not isinstance(value, str) or not value.strip():
                raise StorageError(f"Storage lease is missing {name}")
            return value.strip()

        bucket = required("bucket_name")
        prefix = required("prefix").strip("/")
        producer = required("producer_id")
        safe_identifier(producer, "producer ID")
        if not prefix or any(part in {".", ".."} for part in prefix.split("/")):
            raise StorageError("Storage lease contains an unsafe object prefix")
        endpoint = required("endpoint").rstrip("/")
        if not endpoint.startswith("https://"):
            raise StorageError("Storage lease endpoint must use HTTPS")
        region = lease.get("region")
        if region is not None and not isinstance(region, str):
            raise StorageError("Storage lease region must be a string or null")
        reservation_id = lease.get("reservation_id")
        if reservation_id is not None and not isinstance(reservation_id, str):
            raise StorageError("Storage lease reservation_id must be a string or null")
        requested_bytes = lease.get("requested_bytes")
        if requested_bytes is not None and (
            not isinstance(requested_bytes, int) or requested_bytes < 0
        ):
            raise StorageError("Storage lease requested_bytes must be a non-negative integer")
        return cls(
            remote=f"manda-lease:{bucket}",
            prefix=prefix,
            producer_id=producer,
            bucket_id=required("bucket_id"),
            key_id=required("key_id"),
            application_key=required("application_key"),
            endpoint=endpoint,
            region=region,
            rclone_binary=os.environ.get("ROBOLAB_STORAGE_RCLONE_BINARY", "rclone").strip(),
            lease_expires_at=required("expires_at"),
            lease_name_prefix=required("name_prefix").strip("/"),
            lease_study_id=required("study_id"),
            lease_reservation_id=reservation_id,
            lease_requested_bytes=requested_bytes,
        )

    @property
    def root(self) -> str:
        return remote_join(self.remote, self.prefix)

    @property
    def lease_root(self) -> str | None:
        if not self.lease_name_prefix:
            return None
        return remote_join(self.remote, self.lease_name_prefix)

    @property
    def producer_root(self) -> str:
        if not self.producer_id:
            raise StorageError("A producer ID is required for this operation")
        return remote_join(self.root, "producers", self.producer_id)

    def run_root(self, study_id: str, run_id: str) -> str:
        safe_identifier(study_id, "study ID")
        safe_identifier(run_id, "run ID")
        return remote_join(self.producer_root, "runs", study_id, run_id)

    def working_root(self, run_id: str) -> str:
        safe_identifier(run_id, "run ID")
        if self.lease_study_id:
            safe_identifier(self.lease_study_id, "study ID")
            return remote_join(self.producer_root, "working", self.lease_study_id, run_id)
        return remote_join(self.producer_root, "working", run_id)


class RcloneClient:
    """Small subprocess boundary for provider-neutral object transfers.

    Credentials are injected only into the child environment. They are never placed on the
    command line or included in exceptions.
    """

    def __init__(self, settings: StorageSettings):
        self.settings = settings

    def _environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        if self.settings.application_key_file is None and self.settings.application_key is None:
            return environment

        assert self.settings.key_id is not None
        assert self.settings.endpoint is not None
        secret = self.settings.application_key
        if secret is None:
            assert self.settings.application_key_file is not None
            secret = self.settings.application_key_file.read_text(encoding="utf-8").strip()
        if not secret:
            raise StorageError("Application-key file is empty")
        remote_name = self.settings.remote.split(":", 1)[0]
        config_prefix = "RCLONE_CONFIG_" + re.sub(r"[^A-Za-z0-9]", "_", remote_name).upper()
        variables = {
            f"{config_prefix}_TYPE": "s3",
            f"{config_prefix}_PROVIDER": "Other",
            f"{config_prefix}_ENV_AUTH": "false",
            f"{config_prefix}_ACCESS_KEY_ID": self.settings.key_id,
            f"{config_prefix}_SECRET_ACCESS_KEY": secret,
            f"{config_prefix}_ENDPOINT": self.settings.endpoint,
            f"{config_prefix}_ACL": "private",
            f"{config_prefix}_NO_CHECK_BUCKET": "true",
        }
        if self.settings.region:
            variables[f"{config_prefix}_REGION"] = self.settings.region
        environment.update(variables)
        return environment

    def available(self) -> bool:
        return shutil.which(self.settings.rclone_binary) is not None

    def run(
        self,
        arguments: list[str],
        *,
        check: bool = True,
        input_text: str | None = None,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                [self.settings.rclone_binary, *arguments],
                check=False,
                capture_output=True,
                text=True,
                input=input_text,
                timeout=timeout,
                env=self._environment(),
            )
        except FileNotFoundError as exc:
            raise StorageError(
                f"Missing {self.settings.rclone_binary!r}; install rclone or set "
                "ROBOLAB_STORAGE_RCLONE_BINARY"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise StorageError(f"rclone timed out while running {arguments[0]!r}") from exc
        if check and result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
            raise StorageError(f"rclone {arguments[0]} failed: {detail}")
        return result

    def version(self) -> str:
        result = self.run(["version"], timeout=15)
        return result.stdout.splitlines()[0] if result.stdout else "rclone"

    def list_files(self, remote: str, *, include: str | None = None) -> list[tuple[str, int]]:
        command = [
            "lsf",
            remote,
            "--recursive",
            "--files-only",
            "--format",
            "ps",
            "--separator",
            "\t",
        ]
        if include:
            command.extend(["--include", include])
        result = self.run(command)
        rows: list[tuple[str, int]] = []
        for raw in result.stdout.splitlines():
            if not raw.strip():
                continue
            try:
                path, size = raw.rsplit("\t", 1)
                rows.append((path, int(size)))
            except ValueError as exc:
                raise StorageError(f"Unexpected rclone lsf row: {raw!r}") from exc
        return rows

    def cat(self, remote: str, *, missing_ok: bool = False) -> str | None:
        if missing_ok:
            stat = self.run(["lsjson", remote, "--stat"], check=False)
            if stat.returncode != 0 or not stat.stdout.strip() or stat.stdout.strip() == "null":
                return None
            try:
                metadata = json.loads(stat.stdout)
            except json.JSONDecodeError as exc:
                raise StorageError(f"Unexpected rclone lsjson response for {remote}") from exc
            if not isinstance(metadata, dict) or metadata.get("IsDir") is True:
                return None
        result = self.run(["cat", remote])
        return result.stdout

    def copy(self, source: str | Path, destination: str, *extra: str) -> None:
        self.run(["copy", str(source), destination, *extra])

    def copyto(self, source: str | Path, destination: str) -> None:
        self.run(["copyto", str(source), destination])

    def check(self, source: str | Path, destination: str, *extra: str) -> None:
        self.run(["check", str(source), destination, *extra])

    def configuration_report(self) -> dict[str, Any]:
        return {
            "remote": self.settings.remote,
            "prefix": self.settings.prefix,
            "producer_id": self.settings.producer_id or None,
            "direct_credentials": self.settings.application_key_file is not None,
            "temporary_credentials": self.settings.application_key is not None,
            "endpoint": self.settings.endpoint,
            "region": self.settings.region,
            "lease_expires_at": self.settings.lease_expires_at,
            "lease_reservation_id": self.settings.lease_reservation_id,
            "lease_requested_bytes": self.settings.lease_requested_bytes,
            "application_key_file": (
                str(self.settings.application_key_file)
                if self.settings.application_key_file is not None
                else None
            ),
        }
