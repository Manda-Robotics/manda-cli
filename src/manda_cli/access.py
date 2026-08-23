from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .artifacts import ArtifactError, ArtifactManager
from .auth import AuthManager, AuthSettings
from .storage import StorageSettings


class AccessError(ValueError):
    """Raised when the authenticated Manda API denies or cannot broker an operation."""


class AccessHttpError(AccessError):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(f"Manda access request failed ({status}): {message}")


@dataclass(frozen=True)
class RunAccess:
    run_key: str
    run_id: str
    study_id: str
    producer_id: str
    row: dict[str, Any]


class MandaAccessClient:
    def __init__(self, auth: AuthManager):
        self.auth = auth
        self.settings = auth.settings

    @classmethod
    def from_environment(cls) -> MandaAccessClient:
        return cls(AuthManager.from_environment())

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        query: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        session = self.auth.ensure_session()
        suffix = f"?{urlencode(query)}" if query else ""
        url = f"{self.settings.api_url}/{path.lstrip('/')}{suffix}"
        headers = {
            "apikey": self.settings.publishable_key,
            "Authorization": f"Bearer {session.access_token}",
            "Accept": "application/json",
            "User-Agent": "manda-cli/0.1",
        }
        data = None
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=45) as response:  # noqa: S310 - validated URL
                raw = response.read()
        except HTTPError as exc:
            raw = exc.read()
            try:
                error = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                error = {}
            message = str(error.get("error") or error.get("message") or exc.reason)
            raise AccessHttpError(exc.code, message) from exc
        except (URLError, TimeoutError) as exc:
            reason = getattr(exc, "reason", exc)
            raise AccessError(f"Cannot reach the Manda access service: {reason}") from exc
        if not raw:
            return {}
        try:
            value = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise AccessError("Manda access service returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise AccessError("Manda access service returned an invalid response")
        return value

    def session_report(self) -> dict[str, Any]:
        return self._request("GET", "session")

    def ensure_producer(self, label: str | None = None) -> str:
        session = self.auth.ensure_session()
        if session.producer_id:
            return session.producer_id
        client_label = (label or socket.gethostname() or "manda-client").strip()[:100]
        value = self._request("POST", "producers", payload={"label": client_label})
        producer_id = value.get("producer_id")
        if not isinstance(producer_id, str) or not producer_id:
            raise AccessError("Producer registration returned no producer ID")
        self.auth.update_producer(producer_id, client_label)
        return producer_id

    def list_runs(self, *, study_id: str | None = None) -> list[dict[str, Any]]:
        query = {"study_id": study_id} if study_id else None
        value = self._request("GET", "runs", query=query)
        rows = value.get("runs")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise AccessError("Manda access service returned an invalid run list")
        return rows

    def find_run(self, run_id: str, *, study_id: str | None = None) -> RunAccess:
        matches = [row for row in self.list_runs(study_id=study_id) if row.get("run_id") == run_id]
        if not matches:
            raise ArtifactError(f"No accessible completed run found for {run_id!r}")
        if len(matches) > 1:
            raise ArtifactError(f"Run ID {run_id!r} is ambiguous; pass --study-id")
        row = matches[0]
        values = {name: row.get(name) for name in ("run_key", "run_id", "study_id", "producer_id")}
        if not all(isinstance(value, str) and value for value in values.values()):
            raise AccessError("Accessible run metadata is missing its identity")
        return RunAccess(
            run_key=str(values["run_key"]),
            run_id=str(values["run_id"]),
            study_id=str(values["study_id"]),
            producer_id=str(values["producer_id"]),
            row=row,
        )

    def upload_manager(
        self,
        study_id: str,
        *,
        purpose: str,
        run_id: str | None = None,
        producer_id: str | None = None,
        producer_label: str | None = None,
    ) -> ArtifactManager:
        producer_id = producer_id or self.ensure_producer(producer_label)
        payload = {
            "operation": "upload",
            "study_id": study_id,
            "producer_id": producer_id,
            "purpose": purpose,
        }
        if run_id:
            payload["run_id"] = run_id
        value = self._request(
            "POST",
            "storage/lease",
            payload=payload,
        )
        return ArtifactManager(StorageSettings.from_lease(_lease(value)))

    def download_manager(self, run: RunAccess) -> ArtifactManager:
        value = self._request(
            "POST",
            "storage/lease",
            payload={"operation": "download", "run_key": run.run_key},
        )
        return ArtifactManager(StorageSettings.from_lease(_lease(value)))


def _lease(value: dict[str, Any]) -> dict[str, Any]:
    lease = value.get("lease")
    if not isinstance(lease, dict):
        raise AccessError("Manda access service returned no storage lease")
    return lease


def auth_settings_report() -> dict[str, Any]:
    return AuthSettings.from_environment().report()
