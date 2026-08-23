from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .results import task_name, validate_results
from .storage import RcloneClient, StorageSettings, remote_join, safe_identifier

MANIFEST_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 1
IDENTITY_FILE = ".artifact-run-id"
MANIFEST_FILE = "manifest.json"
RUN_FILE = "run.json"
COMPLETE_FILE = "_COMPLETE"


class ArtifactError(ValueError):
    """Raised when an artifact run cannot be prepared, transferred, or verified."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Cannot read {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"Expected an object in {label}: {path}")
    return value


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return slug[:80] or "run"


def _role(path: str) -> str:
    name = Path(path).name
    suffix = Path(path).suffix.lower()
    if path.startswith("_audit/"):
        return "provenance"
    if name in {RUN_FILE, IDENTITY_FILE}:
        return "metadata"
    if name == "episode_results.jsonl":
        return "result"
    if name == "env_cfg.json" or name.endswith(".toml"):
        return "config"
    if name.startswith("log_") and suffix == ".json":
        return "event_log"
    if suffix in {".hdf5", ".h5"}:
        return "trajectory"
    if suffix == ".mp4":
        return "video"
    if suffix in {".png", ".jpg", ".jpeg", ".gif"}:
        return "image"
    if suffix in {".json", ".jsonl", ".csv", ".parquet"}:
        return "data"
    return "other"


def _content_type(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".json": "application/json",
        ".jsonl": "application/x-ndjson",
        ".hdf5": "application/x-hdf5",
        ".h5": "application/x-hdf5",
        ".mp4": "video/mp4",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".csv": "text/csv",
        ".parquet": "application/vnd.apache.parquet",
        ".toml": "application/toml",
        ".txt": "text/plain",
        ".log": "text/plain",
    }.get(suffix, "application/octet-stream")


def _manifest_payload_files(root: Path) -> list[Path]:
    excluded = {MANIFEST_FILE, COMPLETE_FILE}
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ArtifactError(f"Symlinks are not allowed in archived runs: {path}")
        relative = path.relative_to(root)
        if ".DS_Store" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.is_file() and relative.as_posix() not in excluded:
            files.append(path)
    return sorted(files, key=lambda value: value.relative_to(root).as_posix())


def create_manifest(root: str | Path, run_record: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(root).expanduser().resolve()
    entries: list[dict[str, Any]] = []
    roles: Counter[str] = Counter()
    total_bytes = 0
    for path in _manifest_payload_files(run_dir):
        relative = path.relative_to(run_dir).as_posix()
        size = path.stat().st_size
        role = _role(relative)
        roles[role] += 1
        total_bytes += size
        entries.append(
            {
                "path": relative,
                "bytes": size,
                "sha256": file_sha256(path),
                "content_type": _content_type(relative),
                "role": role,
            }
        )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": utc_now(),
        "run_id": run_record["run_id"],
        "study_id": run_record["study_id"],
        "producer_id": run_record["producer_id"],
        "object_count": len(entries),
        "total_bytes": total_bytes,
        "roles": dict(sorted(roles.items())),
        "files": entries,
    }


def profile_includes(role: str, profile: str) -> bool:
    if profile == "full":
        return True
    if profile == "analysis":
        return role not in {"video", "image"}
    if profile == "metadata":
        return role in {"metadata", "result", "config", "provenance", "event_log"}
    raise ArtifactError(f"Unknown artifact profile: {profile!r}")


def verify_local_run(root: str | Path, *, profile: str = "full") -> dict[str, Any]:
    run_dir = Path(root).expanduser().resolve()
    manifest_path = run_dir / MANIFEST_FILE
    manifest = _load_json(manifest_path, "manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ArtifactError(f"Unsupported manifest schema in {manifest_path}")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ArtifactError(f"Manifest has no files list: {manifest_path}")

    errors: list[str] = []
    checked = 0
    checked_bytes = 0
    expected_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("manifest contains a non-object file entry")
            continue
        relative = entry.get("path")
        role = entry.get("role")
        if not isinstance(relative, str) or not isinstance(role, str):
            errors.append("manifest entry has an invalid path or role")
            continue
        if not profile_includes(role, profile):
            continue
        expected_paths.add(relative)
        path = run_dir / relative
        try:
            resolved = path.resolve()
            resolved.relative_to(run_dir)
        except (OSError, ValueError):
            errors.append(f"unsafe path: {relative}")
            continue
        if not path.is_file():
            errors.append(f"missing: {relative}")
            continue
        expected_size = entry.get("bytes")
        if not isinstance(expected_size, int) or path.stat().st_size != expected_size:
            errors.append(f"size mismatch: {relative}")
            continue
        expected_hash = entry.get("sha256")
        if not isinstance(expected_hash, str) or file_sha256(path) != expected_hash:
            errors.append(f"SHA-256 mismatch: {relative}")
            continue
        checked += 1
        checked_bytes += expected_size

    return {
        "valid": not errors,
        "run_dir": str(run_dir),
        "run_id": manifest.get("run_id"),
        "manifest_sha256": file_sha256(manifest_path),
        "profile": profile,
        "checked_files": checked,
        "checked_bytes": checked_bytes,
        "expected_files": len(expected_paths),
        "errors": errors,
    }


def _copy_audit_bundle(source: Path, run_dir: Path) -> None:
    if not source.is_dir():
        raise ArtifactError(f"Audit bundle does not exist: {source}")
    destination = run_dir / "_audit"
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_symlink():
            raise ArtifactError(f"Symlinks are not allowed in audit bundles: {path}")
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _audit_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "_audit" / "provenance.json"
    if not path.is_file():
        return {}
    return _load_json(path, "provenance")


def ensure_run_id(run_dir: Path, producer_id: str, requested: str | None = None) -> str:
    identity = run_dir / IDENTITY_FILE
    existing = identity.read_text(encoding="utf-8").strip() if identity.is_file() else None
    if requested:
        safe_identifier(requested, "run ID")
        if existing and existing != requested:
            raise ArtifactError(
                f"Run already has ID {existing!r}; refusing requested ID {requested!r}"
            )
        run_id = requested
    elif existing:
        safe_identifier(existing, "stored run ID")
        run_id = existing
    else:
        date = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{date}-{_slug(run_dir.name)}-{producer_id}-{uuid.uuid4().hex[:8]}"
        safe_identifier(run_id, "generated run ID")
    if not existing:
        identity.write_text(run_id + "\n", encoding="utf-8")
    return run_id


class ArtifactManager:
    def __init__(self, settings: StorageSettings, client: RcloneClient | None = None):
        self.settings = settings
        self.client = client or RcloneClient(settings)

    @classmethod
    def from_environment(cls, *, require_producer: bool = True) -> ArtifactManager:
        return cls(StorageSettings.from_environment(require_producer=require_producer))

    def doctor(self, *, write_test: bool = False) -> dict[str, Any]:
        report = self.client.configuration_report()
        report["rclone"] = self.client.version()
        report["root"] = self.settings.root
        read_target = (
            self.settings.producer_root
            if self.settings.producer_id
            else self.settings.root
        )
        self.client.run(["lsf", read_target, "--max-depth", "1"])
        report["readable"] = True
        report["write_test"] = None
        if write_test:
            if not self.settings.producer_id:
                raise ArtifactError("A producer ID is required for the write test")
            payload = json.dumps(
                {
                    "schema_version": 1,
                    "producer_id": self.settings.producer_id,
                    "checked_at": utc_now(),
                },
                sort_keys=True,
            ) + "\n"
            with tempfile.TemporaryDirectory() as raw:
                local = Path(raw) / "probe.json"
                local.write_text(payload, encoding="utf-8")
                remote = remote_join(
                    self.settings.producer_root,
                    "working",
                    "_doctor",
                    f"probe-{uuid.uuid4().hex}.json",
                )
                self.client.copyto(local, remote)
                downloaded = self.client.cat(remote)
            if downloaded != payload:
                raise ArtifactError("Storage write test read-back did not match uploaded payload")
            report["write_test"] = {"valid": True, "remote": remote}
        report["ready"] = True
        return report

    def checkpoint(
        self,
        run_dir: str | Path,
        *,
        run_id: str | None = None,
        min_age_seconds: int = 120,
    ) -> dict[str, Any]:
        root = Path(run_dir).expanduser().resolve()
        if not root.is_dir():
            raise ArtifactError(f"Run directory does not exist: {root}")
        if min_age_seconds < 0:
            raise ArtifactError("min_age_seconds must be >= 0")
        identity = ensure_run_id(root, self.settings.producer_id, run_id)
        remote = self.settings.working_root(identity)
        extra = ["--exclude", COMPLETE_FILE]
        if min_age_seconds:
            extra.extend(["--min-age", f"{min_age_seconds}s"])
        self.client.copy(root, remote, *extra)
        return {
            "run_id": identity,
            "run_dir": str(root),
            "remote": remote,
            "min_age_seconds": min_age_seconds,
            "committed": False,
        }

    def _run_record(
        self,
        root: Path,
        *,
        run_id: str,
        study_id: str,
        stage: str | None,
        policy: str | None,
        validation: dict[str, Any],
    ) -> dict[str, Any]:
        provenance = _audit_metadata(root)
        job = provenance.get("job") if isinstance(provenance.get("job"), dict) else {}
        repositories = (
            provenance.get("repositories")
            if isinstance(provenance.get("repositories"), dict)
            else {}
        )
        portable_repositories = {
            name: (
                {key: value for key, value in repository.items() if key != "path"}
                if isinstance(repository, dict)
                else repository
            )
            for name, repository in repositories.items()
        }
        portable_validation = {
            key: value for key, value in validation.items() if key != "run_dir"
        }
        return {
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": run_id,
            "study_id": study_id,
            "producer_id": self.settings.producer_id,
            "source_folder": root.name,
            "stage": stage or job.get("stage"),
            "policy": policy or job.get("policy"),
            "job_id": job.get("job_id"),
            "created_at": utc_now(),
            "validation": portable_validation,
            "repositories": portable_repositories,
        }

    def finalize(
        self,
        run_dir: str | Path,
        *,
        study_id: str,
        run_id: str | None = None,
        stage: str | None = None,
        policy: str | None = None,
        expected_episodes: int | None = None,
        expected_tasks: list[str] | tuple[str, ...] | None = None,
        audit_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        safe_identifier(study_id, "study ID")
        root = Path(run_dir).expanduser().resolve()
        if not root.is_dir():
            raise ArtifactError(f"Run directory does not exist: {root}")
        local_completion_path = root / COMPLETE_FILE
        if local_completion_path.is_file():
            if audit_dir is not None:
                raise ArtifactError("Cannot add an audit bundle to an already completed run")
            marker = _load_json(local_completion_path, "completion marker")
            identity = ensure_run_id(root, self.settings.producer_id, run_id)
            if marker.get("run_id") != identity or marker.get("study_id") != study_id:
                raise ArtifactError("Local completion marker does not match the requested run")
            local_report = verify_local_run(root)
            if not local_report["valid"]:
                raise ArtifactError(
                    "Completed local run failed verification: "
                    + "; ".join(local_report["errors"])
                )
            report = self.verify_remote(
                identity,
                expected_manifest_sha256=str(marker.get("manifest_sha256") or ""),
                study_id=study_id,
                producer_id=self.settings.producer_id,
            )
            report.update({"run_dir": str(root), "idempotent": True})
            return report
        if audit_dir is not None:
            _copy_audit_bundle(Path(audit_dir).expanduser().resolve(), root)

        identity = ensure_run_id(root, self.settings.producer_id, run_id)
        validation = validate_results(
            root,
            expected_episodes=expected_episodes,
            expected_tasks=expected_tasks,
        )
        if not validation["valid"]:
            errors = "; ".join(validation["errors"])
            raise ArtifactError(f"Refusing to finalize invalid run: {errors}")

        record = self._run_record(
            root,
            run_id=identity,
            study_id=study_id,
            stage=stage,
            policy=policy,
            validation=validation,
        )
        _write_json(root / RUN_FILE, record)
        manifest = create_manifest(root, record)
        _write_json(root / MANIFEST_FILE, manifest)
        manifest_sha = file_sha256(root / MANIFEST_FILE)
        remote = self.settings.run_root(study_id, identity)

        existing_complete = self.client.cat(remote_join(remote, COMPLETE_FILE), missing_ok=True)
        if existing_complete is not None:
            try:
                existing = json.loads(existing_complete)
            except json.JSONDecodeError as exc:
                raise ArtifactError(f"Remote completion marker is invalid for {identity}") from exc
            if existing.get("manifest_sha256") != manifest_sha:
                raise ArtifactError(
                    f"Completed remote run {identity!r} has a different manifest; "
                    "completed runs are immutable"
                )
            remote_report = self.verify_remote(
                identity,
                expected_manifest_sha256=manifest_sha,
                study_id=study_id,
                producer_id=self.settings.producer_id,
            )
            remote_report["idempotent"] = True
            return remote_report

        transfer_paths = [entry["path"] for entry in manifest["files"]]
        transfer_paths.append(MANIFEST_FILE)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            files_from = Path(handle.name)
            for path in transfer_paths:
                handle.write(path + "\n")
        try:
            filters = ("--files-from-raw", str(files_from))
            self.client.copy(root, remote, *filters)
            self.client.check(root, remote, "--one-way", "--download", *filters)
        finally:
            files_from.unlink(missing_ok=True)
        remote_manifest = self.client.cat(remote_join(remote, MANIFEST_FILE))
        if remote_manifest is None or text_sha256(remote_manifest) != manifest_sha:
            raise ArtifactError("Remote manifest digest differs after upload")

        completion = {
            "schema_version": 1,
            "run_id": identity,
            "study_id": study_id,
            "producer_id": self.settings.producer_id,
            "manifest_sha256": manifest_sha,
            "object_count": manifest["object_count"],
            "total_bytes": manifest["total_bytes"],
            "committed_at": utc_now(),
        }
        _write_json(root / COMPLETE_FILE, completion)
        self.client.copyto(root / COMPLETE_FILE, remote_join(remote, COMPLETE_FILE))
        remote_completion = self.client.cat(remote_join(remote, COMPLETE_FILE))
        if remote_completion is None or text_sha256(remote_completion) != file_sha256(
            root / COMPLETE_FILE
        ):
            raise ArtifactError("Remote completion marker differs after upload")

        report = self.verify_remote(
            identity,
            expected_manifest_sha256=manifest_sha,
            study_id=study_id,
            producer_id=self.settings.producer_id,
        )
        report.update({"run_dir": str(root), "idempotent": False})
        return report

    def _complete_paths(
        self,
        *,
        producer_id: str | None = None,
        all_producers: bool = False,
    ) -> list[str]:
        producer = None if all_producers else (producer_id or self.settings.producer_id or None)
        if producer:
            safe_identifier(producer, "producer ID")
            root = remote_join(self.settings.root, "producers", producer)
            return [
                f"{producer}/{path}"
                for path, _ in self.client.list_files(root, include=f"**/{COMPLETE_FILE}")
                if path.endswith(f"/{COMPLETE_FILE}")
            ]
        root = remote_join(self.settings.root, "producers")
        return [
            path
            for path, _ in self.client.list_files(root, include=f"**/{COMPLETE_FILE}")
            if path.endswith(f"/{COMPLETE_FILE}")
        ]

    def locate_run(self, run_id: str, *, producer_id: str | None = None) -> str:
        safe_identifier(run_id, "run ID")
        matches = []
        for complete in self._complete_paths(producer_id=producer_id):
            parts = complete.split("/")
            if len(parts) < 5 or parts[-1] != COMPLETE_FILE or parts[-2] != run_id:
                continue
            if producer_id is not None and parts[0] != producer_id:
                continue
            matches.append(complete[: -len(f"/{COMPLETE_FILE}")])
        if not matches:
            raise ArtifactError(f"No completed remote run found for {run_id!r}")
        if len(matches) > 1:
            raise ArtifactError(
                f"Run ID {run_id!r} is ambiguous across producers; specify --producer-id"
            )
        return remote_join(self.settings.root, "producers", matches[0])

    def list_runs(
        self, *, study_id: str | None = None, all_producers: bool = False
    ) -> list[dict[str, Any]]:
        if study_id:
            safe_identifier(study_id, "study ID")
        runs: list[dict[str, Any]] = []
        for complete in self._complete_paths(all_producers=all_producers):
            parts = complete.split("/")
            if len(parts) < 5 or parts[1] != "runs":
                continue
            producer, _, study, run_id = parts[-5:-1]
            if study_id and study != study_id:
                continue
            remote = remote_join(
                self.settings.root, "producers", producer, "runs", study, run_id
            )
            raw = self.client.cat(remote_join(remote, RUN_FILE))
            try:
                record = json.loads(raw or "")
            except json.JSONDecodeError as exc:
                raise ArtifactError(f"Invalid remote run record at {remote}") from exc
            marker_raw = self.client.cat(remote_join(remote, COMPLETE_FILE))
            try:
                marker = json.loads(marker_raw or "")
            except json.JSONDecodeError as exc:
                raise ArtifactError(f"Invalid remote completion marker at {remote}") from exc
            record["remote"] = remote
            record["manifest_sha256"] = marker.get("manifest_sha256")
            record["committed_at"] = marker.get("committed_at")
            runs.append(record)
        return sorted(runs, key=lambda item: (str(item.get("created_at")), str(item["run_id"])))

    def verify_remote(
        self,
        run_id: str,
        *,
        expected_manifest_sha256: str | None = None,
        study_id: str | None = None,
        producer_id: str | None = None,
    ) -> dict[str, Any]:
        if study_id is not None and producer_id is not None:
            safe_identifier(study_id, "study ID")
            safe_identifier(producer_id, "producer ID")
            remote = remote_join(
                self.settings.root,
                "producers",
                producer_id,
                "runs",
                study_id,
                run_id,
            )
        else:
            remote = self.locate_run(run_id, producer_id=producer_id)
        manifest_raw = self.client.cat(remote_join(remote, MANIFEST_FILE))
        marker_raw = self.client.cat(remote_join(remote, COMPLETE_FILE))
        if manifest_raw is None or marker_raw is None:
            raise ArtifactError(f"Remote run is not complete: {run_id}")
        try:
            manifest = json.loads(manifest_raw)
            marker = json.loads(marker_raw)
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"Remote run metadata is invalid: {run_id}") from exc
        manifest_sha = text_sha256(manifest_raw)
        errors = []
        if marker.get("manifest_sha256") != manifest_sha:
            errors.append("completion marker does not match manifest")
        if expected_manifest_sha256 and expected_manifest_sha256 != manifest_sha:
            errors.append("manifest does not match expected digest")

        remote_sizes = dict(self.client.list_files(remote))
        entries = manifest.get("files") if isinstance(manifest.get("files"), list) else []
        expected_sizes = {
            entry["path"]: entry["bytes"]
            for entry in entries
            if isinstance(entry, dict)
            and isinstance(entry.get("path"), str)
            and isinstance(entry.get("bytes"), int)
        }
        expected_sizes[MANIFEST_FILE] = len(manifest_raw.encode("utf-8"))
        expected_sizes[COMPLETE_FILE] = len(marker_raw.encode("utf-8"))
        for path, size in expected_sizes.items():
            if remote_sizes.get(path) != size:
                errors.append(f"remote size mismatch: {path}")
        unexpected = sorted(set(remote_sizes) - set(expected_sizes))
        if unexpected:
            errors.append(f"remote contains {len(unexpected)} unexpected objects")
        return {
            "valid": not errors,
            "committed": not errors,
            "run_id": run_id,
            "remote": remote,
            "manifest_sha256": manifest_sha,
            "object_count": manifest.get("object_count"),
            "total_bytes": manifest.get("total_bytes"),
            "errors": errors,
        }

    def pull(
        self,
        run_id: str,
        *,
        destination: str | Path,
        profile: str = "analysis",
        producer_id: str | None = None,
    ) -> dict[str, Any]:
        if profile not in {"metadata", "analysis", "full"}:
            raise ArtifactError(f"Unknown artifact profile: {profile!r}")
        remote = self.locate_run(run_id, producer_id=producer_id)
        target = Path(destination).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        manifest_raw = self.client.cat(remote_join(remote, MANIFEST_FILE))
        marker_raw = self.client.cat(remote_join(remote, COMPLETE_FILE))
        if manifest_raw is None or marker_raw is None:
            raise ArtifactError(f"Remote run is incomplete: {run_id}")
        try:
            manifest = json.loads(manifest_raw)
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"Remote manifest is invalid: {run_id}") from exc
        entries = manifest.get("files")
        if not isinstance(entries, list):
            raise ArtifactError(f"Remote manifest has no files list: {run_id}")
        selected = [
            entry["path"]
            for entry in entries
            if isinstance(entry, dict)
            and isinstance(entry.get("path"), str)
            and isinstance(entry.get("role"), str)
            and profile_includes(entry["role"], profile)
        ]
        selected.extend([MANIFEST_FILE, COMPLETE_FILE])
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            files_from = Path(handle.name)
            for path in sorted(set(selected)):
                handle.write(path + "\n")
        try:
            self.client.copy(remote, str(target), "--files-from-raw", str(files_from))
        finally:
            files_from.unlink(missing_ok=True)
        report = verify_local_run(target, profile=profile)
        if not report["valid"]:
            raise ArtifactError(
                "Downloaded run failed verification: " + "; ".join(report["errors"])
            )
        report.update({"remote": remote, "destination": str(target)})
        return report

    def resume(
        self,
        run_id: str,
        *,
        destination: str | Path,
        producer_id: str | None = None,
    ) -> dict[str, Any]:
        safe_identifier(run_id, "run ID")
        producer = producer_id or self.settings.producer_id
        if not producer:
            raise ArtifactError("A producer ID is required to locate a working checkpoint")
        safe_identifier(producer, "producer ID")
        remote = remote_join(
            self.settings.root,
            "producers",
            producer,
            "working",
            run_id,
        )
        if not self.client.list_files(remote):
            raise ArtifactError(f"No working checkpoint found for {producer}/{run_id}")
        target = Path(destination).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        self.client.copy(remote, str(target))
        identity = target / IDENTITY_FILE
        if not identity.is_file() or identity.read_text(encoding="utf-8").strip() != run_id:
            raise ArtifactError("Restored checkpoint is missing its matching run identity")
        return {
            "run_id": run_id,
            "producer_id": producer,
            "remote": remote,
            "destination": str(target),
            "committed": False,
        }

    def materialize(
        self,
        reference: str | Path,
        *,
        profile: str = "analysis",
        cache_dir: str | Path | None = None,
    ) -> Path:
        local = Path(reference).expanduser()
        if local.exists():
            return local.resolve()
        run_id = str(reference)
        safe_identifier(run_id, "run ID")
        cache_root = Path(
            cache_dir
            or os.environ.get(
                "ROBOLAB_ARTIFACT_CACHE",
                Path.home() / ".cache" / "robot-policy-audit" / "runs",
            )
        ).expanduser().resolve()
        destination = cache_root / run_id
        self.pull(run_id, destination=destination, profile=profile)
        return destination


def summarize_run_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = Counter(name for row in rows if (name := task_name(row)))
    successes = sum(row.get("success") is True for row in rows)
    return {
        "episode_count": len(rows),
        "task_count": len(tasks),
        "successes": successes,
        "success_rate": successes / len(rows) if rows else None,
        "task_episode_counts": dict(sorted(tasks.items())),
    }


def read_remote_jsonl(client: RcloneClient, remote: str) -> list[dict[str, Any]]:
    raw = client.cat(remote)
    if raw is None:
        raise ArtifactError(f"Missing remote JSONL: {remote}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"Invalid JSON at {remote}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ArtifactError(f"Expected an object at {remote}:{line_number}")
        rows.append(value)
    return rows
