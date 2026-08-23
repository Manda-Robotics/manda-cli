from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


class SupabaseError(ValueError):
    """Raised when dashboard database configuration or synchronization fails."""


class SupabaseHttpError(SupabaseError):
    def __init__(self, status: int, code: str, message: str):
        self.status = status
        self.code = code
        super().__init__(f"Supabase request failed ({status}, {code}): {message}")


@dataclass(frozen=True)
class SupabaseSettings:
    url: str
    secret_key: str
    publishable_key: str | None = None
    jwks_url: str | None = None

    @classmethod
    def from_environment(cls) -> SupabaseSettings:
        url = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
        secret = os.environ.get("SUPABASE_SECRET_KEY", "").strip()
        publishable = os.environ.get("SUPABASE_PUBLISHABLE_KEY", "").strip() or None
        jwks = os.environ.get("SUPABASE_JWKS_URL", "").strip() or None
        if not url or not secret:
            raise SupabaseError("Set SUPABASE_URL and SUPABASE_SECRET_KEY")
        parsed = urlparse(url)
        if parsed.scheme not in {"https", "http"} or not parsed.netloc:
            raise SupabaseError("SUPABASE_URL must be an absolute HTTP(S) URL")
        if parsed.scheme != "https" and parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise SupabaseError("SUPABASE_URL must use HTTPS except for a local development server")
        return cls(
            url=url,
            secret_key=secret,
            publishable_key=publishable,
            jwks_url=jwks,
        )

    def report(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "secret_key_configured": True,
            "publishable_key_configured": self.publishable_key is not None,
            "jwks_url_configured": self.jwks_url is not None,
        }


class SupabaseRestClient:
    """Minimal PostgREST client that keeps the Supabase secret out of logs and URLs."""

    def __init__(self, settings: SupabaseSettings):
        self.settings = settings

    def request(
        self,
        method: str,
        table: str,
        *,
        query: dict[str, str | int] | None = None,
        payload: Any = None,
        prefer: str | None = None,
    ) -> Any:
        suffix = f"?{urlencode(query)}" if query else ""
        url = f"{self.settings.url}/rest/v1/{table}{suffix}"
        headers = {
            "apikey": self.settings.secret_key,
            "Accept": "application/json",
            "User-Agent": "manda-cli/0.1",
        }
        data = None
        if payload is not None:
            data = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if prefer:
            headers["Prefer"] = prefer
        request = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=30) as response:  # noqa: S310 - validated URL
                raw = response.read()
        except HTTPError as exc:
            raw = exc.read()
            try:
                error = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                error = {}
            code = str(error.get("code") or f"HTTP_{exc.code}")
            message = str(error.get("message") or exc.reason)
            raise SupabaseHttpError(exc.code, code, message) from exc
        except (URLError, TimeoutError) as exc:
            reason = getattr(exc, "reason", exc)
            raise SupabaseError(f"Cannot reach Supabase: {reason}") from exc
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise SupabaseError("Supabase returned a non-JSON response") from exc

    def select(
        self,
        table: str,
        *,
        columns: str = "*",
        filters: dict[str, str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        query: dict[str, str | int] = {"select": columns}
        query.update(filters or {})
        if limit is not None:
            query["limit"] = limit
        result = self.request("GET", table, query=query)
        if not isinstance(result, list):
            raise SupabaseError(f"Expected a row list from {table}")
        return result

    def upsert(
        self,
        table: str,
        rows: list[dict[str, Any]],
        *,
        on_conflict: str,
        batch_size: int = 500,
    ) -> int:
        if batch_size < 1:
            raise SupabaseError("batch_size must be at least 1")
        written = 0
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset : offset + batch_size]
            self.request(
                "POST",
                table,
                query={"on_conflict": on_conflict},
                payload=batch,
                prefer="resolution=merge-duplicates,return=minimal",
            )
            written += len(batch)
        return written

    def delete(self, table: str, *, filters: dict[str, str]) -> None:
        self.request("DELETE", table, query=filters, prefer="return=minimal")

    def update(
        self,
        table: str,
        payload: dict[str, Any],
        *,
        filters: dict[str, str],
    ) -> None:
        self.request("PATCH", table, query=filters, payload=payload, prefer="return=minimal")

    def doctor(self) -> dict[str, Any]:
        report = self.settings.report()
        missing = []
        rows: list[dict[str, Any]] = []
        for table, columns in (
            ("studies", "study_id"),
            ("runs", "run_key"),
            ("task_results", "run_key"),
            ("episodes", "run_key"),
            ("assets", "run_key"),
            ("catalog_imports", "catalog_version"),
        ):
            try:
                sample = self.select(table, columns=columns, limit=1)
                if table == "runs":
                    rows = sample
            except SupabaseHttpError as exc:
                if exc.status == 404 or exc.code in {"42P01", "PGRST205"}:
                    missing.append(table)
                    continue
                raise
        if missing:
            report.update(
                {
                    "reachable": True,
                    "schema_ready": False,
                    "ready": False,
                    "missing_tables": missing,
                    "migration": "supabase/migrations/20260823000100_dashboard_read_model.sql",
                }
            )
            return report
        report.update(
            {
                "reachable": True,
                "schema_ready": True,
                "ready": True,
                "sample_run_count": len(rows),
            }
        )
        return report


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _required_text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise SupabaseError(f"Catalog row is missing {key}")
    return value


def _finite_number(value: Any, *, default: float | None = None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    return number if math.isfinite(number) else default


def _run_key(row: dict[str, Any]) -> str:
    producer = _required_text(row, "producer_id")
    run_id = _required_text(row, "run_id")
    return f"{producer}/{run_id}"


def _source_run_key(row: dict[str, Any]) -> str:
    producer = _required_text(row, "source_producer_id")
    run_id = _required_text(row, "source_run_id")
    return f"{producer}/{run_id}"


def _remote_prefix(remote: str) -> str:
    if ":" not in remote:
        raise SupabaseError(f"Run remote has no provider prefix: {remote}")
    bucket_and_key = remote.split(":", 1)[1].lstrip("/")
    if "/" not in bucket_and_key:
        raise SupabaseError(f"Run remote has no object prefix: {remote}")
    return bucket_and_key.split("/", 1)[1]


def _episode_key(row: dict[str, Any]) -> str:
    task = str(row.get("task_name") or row.get("env_name") or "unknown")
    coordinates = [row.get("run_index"), row.get("env_id"), row.get("episode")]
    if any(value is not None for value in coordinates):
        values = ["" if value is None else str(value) for value in coordinates]
        return "|".join([task, *values])
    canonical = json.dumps(_json_value(row), sort_keys=True, separators=(",", ":"))
    return f"{task}|sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


def catalog_sync_plan(catalog: dict[str, Any]) -> dict[str, Any]:
    return {
        "catalog_version": catalog.get("catalog_version"),
        "generated_at": catalog.get("generated_at"),
        "study_id": catalog.get("study_id"),
        "run_count": len(catalog.get("runs") or []),
        "task_count": len(catalog.get("tasks") or []),
        "episode_count": len(catalog.get("episodes") or []),
        "asset_count": len(catalog.get("assets") or []),
        "note": "No database rows are deleted for runs absent from this catalog.",
    }


class DashboardIndexer:
    def __init__(self, client: SupabaseRestClient):
        self.client = client

    @staticmethod
    def _run_payload(run: dict[str, Any], catalog_version: str) -> dict[str, Any]:
        run_key = _run_key(run)
        remote = _required_text(run, "remote")
        return {
            "run_key": run_key,
            "run_id": _required_text(run, "run_id"),
            "study_id": _required_text(run, "study_id"),
            "producer_id": _required_text(run, "producer_id"),
            "stage": run.get("stage"),
            "policy": run.get("policy"),
            "manifest_sha256": _required_text(run, "manifest_sha256"),
            "catalog_version": catalog_version,
            "remote_prefix": _remote_prefix(remote),
            "episode_count": int(run.get("episode_count") or 0),
            "object_count": run.get("object_count"),
            "total_bytes": run.get("total_bytes"),
            "validation": _json_value(run.get("validation")),
            "metadata": _json_value(run),
            "index_schema_version": 1,
            "index_status": "indexing",
            "source_created_at": run.get("created_at"),
            "committed_at": run.get("committed_at"),
        }

    @staticmethod
    def _task_payload(row: dict[str, Any]) -> dict[str, Any]:
        run_key = _source_run_key(row)
        known = {"episode_count", "successes", "success_rate", "mean_score"}
        return {
            "run_key": run_key,
            "task_name": str(row.get("task_name") or "unknown"),
            "episode_count": int(row.get("episode_count") or 0),
            "successes": int(row.get("successes") or 0),
            "success_rate": _finite_number(row.get("success_rate"), default=0.0),
            "mean_score": _finite_number(row.get("mean_score")),
            "metrics": _json_value({key: value for key, value in row.items() if key not in known}),
        }

    @staticmethod
    def _episode_payload(row: dict[str, Any]) -> dict[str, Any]:
        run_key = _source_run_key(row)
        task = str(row.get("task_name") or row.get("env_name") or "unknown")
        score = _finite_number(row.get("score"))
        duration = _finite_number(row.get("duration"))
        if duration is None:
            duration = _finite_number(row.get("elapsed_time"))
        attributes = row.get("attributes") if isinstance(row.get("attributes"), list) else []
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        return {
            "run_key": run_key,
            "episode_key": _episode_key(row),
            "task_name": task,
            "episode_index": row.get("episode") if isinstance(row.get("episode"), int) else None,
            "env_id": row.get("env_id") if isinstance(row.get("env_id"), int) else None,
            "run_index": row.get("run_index") if isinstance(row.get("run_index"), int) else None,
            "success": row.get("success") if isinstance(row.get("success"), bool) else None,
            "score": score,
            "duration_seconds": duration,
            "instruction": row.get("instruction"),
            "attributes": _json_value(attributes),
            "metrics": _json_value(metrics),
            "raw": _json_value(row),
        }

    @staticmethod
    def _asset_payload(row: dict[str, Any]) -> dict[str, Any]:
        run_key = _source_run_key(row)
        return {
            "run_key": run_key,
            "path": _required_text(row, "path"),
            "bucket_name": _required_text(row, "bucket_name"),
            "object_key": _required_text(row, "object_key"),
            "role": _required_text(row, "role"),
            "content_type": _required_text(row, "content_type"),
            "bytes": int(row.get("bytes") or 0),
            "sha256": _required_text(row, "sha256"),
            "task_name": row.get("task_name"),
            "metadata": _json_value(row),
        }

    def sync(self, catalog: dict[str, Any], *, batch_size: int = 500) -> dict[str, Any]:
        plan = catalog_sync_plan(catalog)
        catalog_version = _required_text(catalog, "catalog_version")
        runs = catalog.get("runs")
        if not isinstance(runs, list):
            raise SupabaseError("Catalog runs must be a list")
        grouped_tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
        grouped_episodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
        grouped_assets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in catalog.get("tasks") or []:
            payload = self._task_payload(row)
            grouped_tasks[payload["run_key"]].append(payload)
        for row in catalog.get("episodes") or []:
            payload = self._episode_payload(row)
            grouped_episodes[payload["run_key"]].append(payload)
        for row in catalog.get("assets") or []:
            payload = self._asset_payload(row)
            grouped_assets[payload["run_key"]].append(payload)

        study_ids = sorted({_required_text(run, "study_id") for run in runs})
        self.client.upsert(
            "studies",
            [{"study_id": study_id} for study_id in study_ids],
            on_conflict="study_id",
            batch_size=batch_size,
        )
        synced = 0
        skipped = 0
        for run in runs:
            payload = self._run_payload(run, catalog_version)
            run_key = str(payload["run_key"])
            existing = self.client.select(
                "runs",
                columns="manifest_sha256,index_schema_version,index_status",
                filters={"run_key": f"eq.{run_key}"},
                limit=1,
            )
            if (
                existing
                and existing[0].get("manifest_sha256") == payload["manifest_sha256"]
                and existing[0].get("index_schema_version") == payload["index_schema_version"]
                and existing[0].get("index_status") == "ready"
            ):
                skipped += 1
                continue
            self.client.upsert("runs", [payload], on_conflict="run_key", batch_size=batch_size)
            try:
                for table in ("task_results", "episodes", "assets"):
                    self.client.delete(table, filters={"run_key": f"eq.{run_key}"})
                self.client.upsert(
                    "task_results",
                    grouped_tasks[run_key],
                    on_conflict="run_key,task_name",
                    batch_size=batch_size,
                )
                self.client.upsert(
                    "episodes",
                    grouped_episodes[run_key],
                    on_conflict="run_key,episode_key",
                    batch_size=batch_size,
                )
                self.client.upsert(
                    "assets",
                    grouped_assets[run_key],
                    on_conflict="run_key,path",
                    batch_size=batch_size,
                )
                self.client.update(
                    "runs",
                    {"index_status": "ready", "catalog_version": catalog_version},
                    filters={"run_key": f"eq.{run_key}"},
                )
                synced += 1
            except SupabaseError:
                with suppress(SupabaseError):
                    self.client.update(
                        "runs",
                        {"index_status": "failed"},
                        filters={"run_key": f"eq.{run_key}"},
                    )
                raise

        self.client.upsert(
            "catalog_imports",
            [
                {
                    "catalog_version": catalog_version,
                    "generated_at": catalog.get("generated_at"),
                    "run_count": plan["run_count"],
                    "task_count": plan["task_count"],
                    "episode_count": plan["episode_count"],
                    "asset_count": plan["asset_count"],
                    "metadata": {"study_id": catalog.get("study_id")},
                }
            ],
            on_conflict="catalog_version",
            batch_size=batch_size,
        )
        return {**plan, "applied": True, "synced_runs": synced, "skipped_runs": skipped}
