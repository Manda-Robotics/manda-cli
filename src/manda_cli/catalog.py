from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifacts import ArtifactError, ArtifactManager, file_sha256, read_remote_jsonl, utc_now
from .storage import remote_join, safe_identifier


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _parquet_row(row: dict[str, Any]) -> dict[str, Any]:
    """Keep Parquet columns stable even when metric/event dictionaries evolve."""
    return {
        key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        for key, value in row.items()
    }


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ArtifactError(
            "Parquet publication requires the analysis extra: uv sync --extra analysis"
        ) from exc
    table = pa.Table.from_pylist([_parquet_row(row) for row in rows])
    pq.write_table(table, path, compression="zstd")


def _task_rows(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        task = row.get("task_name") or row.get("env_name") or "unknown"
        grouped[(str(row["source_run_id"]), str(task))].append(row)

    output = []
    for (run_id, task), rows in sorted(grouped.items()):
        successes = sum(row.get("success") is True for row in rows)
        scores = [
            float(row["score"])
            for row in rows
            if isinstance(row.get("score"), (int, float))
            and not isinstance(row.get("score"), bool)
        ]
        output.append(
            {
                "source_run_id": run_id,
                "source_producer_id": rows[0].get("source_producer_id"),
                "source_manifest_sha256": rows[0].get("source_manifest_sha256"),
                "study_id": rows[0].get("source_study_id"),
                "producer_id": rows[0].get("source_producer_id"),
                "stage": rows[0].get("source_stage"),
                "policy": rows[0].get("source_policy"),
                "task_name": task,
                "episode_count": len(rows),
                "successes": successes,
                "success_rate": successes / len(rows),
                "mean_score": sum(scores) / len(scores) if scores else None,
            }
        )
    return output


def _catalog_manifest(output: Path) -> dict[str, Any]:
    files = []
    total = 0
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name in {"manifest.json", "_COMPLETE"}:
            continue
        size = path.stat().st_size
        total += size
        files.append({"path": path.name, "bytes": size, "sha256": file_sha256(path)})
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "object_count": len(files),
        "total_bytes": total,
        "files": files,
    }


def _remote_json(manager: ArtifactManager, remote: str, name: str) -> dict[str, Any]:
    raw = manager.client.cat(remote_join(remote, name))
    try:
        value = json.loads(raw or "")
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"Invalid {name} at {remote}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"Expected an object in {name} at {remote}")
    return value


def _object_location(remote: str, path: str) -> tuple[str, str]:
    joined = remote_join(remote, path)
    if ":" not in joined:
        raise ArtifactError(f"Remote has no provider prefix: {remote}")
    bucket_and_key = joined.split(":", 1)[1].lstrip("/")
    if "/" not in bucket_and_key:
        raise ArtifactError(f"Remote has no object key: {joined}")
    bucket_name, object_key = bucket_and_key.split("/", 1)
    return bucket_name, object_key


def collect_catalog_rows(
    manager: ArtifactManager,
    *,
    study_id: str | None = None,
    version: str | None = None,
    all_producers: bool = True,
) -> dict[str, Any]:
    """Collect a portable relational catalog from committed B2 runs."""
    if study_id:
        safe_identifier(study_id, "study ID")
    catalog_version = version or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_identifier(catalog_version, "catalog version")
    generated_at = utc_now()

    runs = manager.list_runs(study_id=study_id, all_producers=all_producers)
    episodes: list[dict[str, Any]] = []
    assets: list[dict[str, Any]] = []
    run_catalog: list[dict[str, Any]] = []
    for run in runs:
        remote = str(run["remote"])
        rows = read_remote_jsonl(manager.client, remote_join(remote, "episode_results.jsonl"))
        for row in rows:
            enriched = dict(row)
            enriched.update(
                {
                    "source_run_id": run["run_id"],
                    "source_manifest_sha256": run.get("manifest_sha256"),
                    "source_study_id": run.get("study_id"),
                    "source_producer_id": run.get("producer_id"),
                    "source_stage": run.get("stage"),
                    "source_policy": run.get("policy"),
                }
            )
            episodes.append(enriched)

        manifest = _remote_json(manager, remote, "manifest.json")
        entries = manifest.get("files")
        if not isinstance(entries, list):
            raise ArtifactError(f"Manifest has no files list at {remote}")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ArtifactError(f"Manifest contains an invalid file entry at {remote}")
            path = entry.get("path")
            if not isinstance(path, str):
                raise ArtifactError(f"Manifest contains an invalid path at {remote}")
            bucket_name, object_key = _object_location(remote, path)
            first_part = path.split("/", 1)[0]
            task = first_part if "/" in path and not first_part.startswith("_") else None
            assets.append(
                {
                    "source_run_id": run["run_id"],
                    "source_manifest_sha256": run.get("manifest_sha256"),
                    "source_study_id": run.get("study_id"),
                    "source_producer_id": run.get("producer_id"),
                    "path": path,
                    "bucket_name": bucket_name,
                    "object_key": object_key,
                    "role": entry.get("role"),
                    "content_type": entry.get("content_type"),
                    "bytes": entry.get("bytes"),
                    "sha256": entry.get("sha256"),
                    "task_name": task,
                }
            )
        run_catalog.append(
            {
                **{key: value for key, value in run.items() if key != "validation"},
                "validation": run.get("validation"),
                "episode_count": len(rows),
                "object_count": manifest.get("object_count"),
                "total_bytes": manifest.get("total_bytes"),
            }
        )

    return {
        "schema_version": 1,
        "catalog_version": catalog_version,
        "generated_at": generated_at,
        "study_id": study_id,
        "runs": run_catalog,
        "episodes": episodes,
        "tasks": _task_rows(episodes),
        "assets": assets,
    }


def publish_catalog(
    manager: ArtifactManager,
    *,
    output_dir: str | Path,
    study_id: str | None = None,
    version: str | None = None,
    parquet: bool = True,
    upload: bool = True,
) -> dict[str, Any]:
    catalog = collect_catalog_rows(manager, study_id=study_id, version=version)
    catalog_version = str(catalog["catalog_version"])
    output = Path(output_dir).expanduser().resolve() / catalog_version
    if output.exists() and any(output.iterdir()):
        raise ArtifactError(f"Catalog output version already exists and is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    run_catalog = catalog["runs"]
    episodes = catalog["episodes"]
    tasks = catalog["tasks"]
    assets = catalog["assets"]
    _write_json(
        output / "runs.json",
        {
            "schema_version": 1,
            "catalog_version": catalog_version,
            "generated_at": catalog["generated_at"],
            "study_id": study_id,
            "runs": run_catalog,
        },
    )
    _write_jsonl(output / "episodes.jsonl", episodes)
    _write_jsonl(output / "tasks.jsonl", tasks)
    _write_jsonl(output / "assets.jsonl", assets)
    if parquet:
        _write_parquet(output / "episodes.parquet", episodes)
        _write_parquet(output / "tasks.parquet", tasks)
        _write_parquet(output / "assets.parquet", assets)

    manifest = _catalog_manifest(output)
    _write_json(output / "manifest.json", manifest)
    manifest_sha = file_sha256(output / "manifest.json")
    completion = {
        "schema_version": 1,
        "catalog_version": catalog_version,
        "manifest_sha256": manifest_sha,
        "committed_at": utc_now(),
    }
    _write_json(output / "_COMPLETE", completion)

    derived_remote = remote_join(manager.settings.root, "derived", catalog_version)
    if upload:
        existing = manager.client.cat(remote_join(derived_remote, "_COMPLETE"), missing_ok=True)
        if existing is not None:
            raise ArtifactError(f"Derived catalog version already exists: {catalog_version}")
        manager.client.copy(output, derived_remote, "--exclude", "_COMPLETE")
        manager.client.check(
            output,
            derived_remote,
            "--one-way",
            "--download",
            "--exclude",
            "_COMPLETE",
        )
        manager.client.copyto(output / "_COMPLETE", remote_join(derived_remote, "_COMPLETE"))
        remote_completion = manager.client.cat(remote_join(derived_remote, "_COMPLETE"))
        if remote_completion != (output / "_COMPLETE").read_text(encoding="utf-8"):
            raise ArtifactError("Remote derived-catalog completion marker differs after upload")

        catalog_root = remote_join(manager.settings.root, "catalog")
        manager.client.copyto(output / "runs.json", remote_join(catalog_root, "runs.json"))
        latest = output.parent / f".{catalog_version}-latest.json"
        _write_json(
            latest,
            {
                "schema_version": 1,
                "catalog_version": catalog_version,
                "derived_prefix": f"derived/{catalog_version}",
                "manifest_sha256": manifest_sha,
                "updated_at": utc_now(),
            },
        )
        try:
            manager.client.copyto(latest, remote_join(catalog_root, "latest.json"))
        finally:
            latest.unlink(missing_ok=True)

    return {
        "catalog_version": catalog_version,
        "output_dir": str(output),
        "remote": derived_remote if upload else None,
        "manifest_sha256": manifest_sha,
        "run_count": len(run_catalog),
        "episode_count": len(episodes),
        "task_row_count": len(tasks),
        "asset_count": len(assets),
        "parquet": parquet,
        "uploaded": upload,
    }
