from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from manda_cli.access import MandaAccessClient
from manda_cli.auth import (
    AuthError,
    AuthManager,
    AuthSession,
    AuthSettings,
    SessionStore,
)
from manda_cli.storage import RcloneClient, StorageSettings


class FakeAuthClient:
    def __init__(self, refreshed: AuthSession):
        self.refreshed = refreshed
        self.refresh_calls: list[str] = []

    def refresh(self, refresh_token: str) -> AuthSession:
        self.refresh_calls.append(refresh_token)
        return self.refreshed


class RecordingAccessClient(MandaAccessClient):
    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict[str, object] | None]] = []

    def _request(  # type: ignore[override]
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, object] | None = None,
        query: dict[str, str] | None = None,
    ) -> dict[str, object]:
        self.requests.append((method, path, payload))
        return {
            "lease": {
                "bucket_name": "bucket",
                "bucket_id": "bucket-id",
                "endpoint": "https://s3.us-east-005.backblazeb2.com",
                "region": "us-east-005",
                "prefix": "robolab/v1",
                "producer_id": "producer-a",
                "study_id": "study-a",
                "name_prefix": "robolab/v1/producers/producer-a/runs/study-a/run-a/",
                "key_id": "key-id",
                "application_key": "temporary-secret",
                "expires_at": "2026-08-24T00:00:00Z",
            }
        }


def _settings(session_file: Path) -> AuthSettings:
    return AuthSettings(
        supabase_url="https://project.supabase.co",
        publishable_key="sb_publishable_test",
        api_url="https://project.supabase.co/functions/v1/manda-access",
        session_file=session_file,
    )


class AuthTests(unittest.TestCase):
    def test_public_auth_settings_require_https_and_do_not_report_key(self) -> None:
        environment = {
            "MANDA_SUPABASE_URL": "https://project.supabase.co",
            "MANDA_SUPABASE_PUBLISHABLE_KEY": "sb_publishable_test",
            "MANDA_SESSION_FILE": "/tmp/manda-test-session.json",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = AuthSettings.from_environment()
        self.assertNotIn(settings.publishable_key, str(settings.report()))
        self.assertEqual(
            settings.api_url,
            "https://project.supabase.co/functions/v1/manda-access",
        )
        with (
            patch.dict(
                os.environ,
                {
                    "MANDA_SUPABASE_URL": "http://example.com",
                    "MANDA_SUPABASE_PUBLISHABLE_KEY": "key",
                },
                clear=True,
            ),
            self.assertRaisesRegex(AuthError, "HTTPS"),
        ):
            AuthSettings.from_environment()

    def test_session_store_is_mode_600_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "config" / "session.json"
            store = SessionStore(path)
            session = AuthSession(
                access_token="access",
                refresh_token="refresh",
                expires_at=int(time.time()) + 3600,
                user_id="user-id",
                email="user@example.com",
                producer_id="producer-a",
            )
            store.save(session)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(store.load(), session)
            path.chmod(0o644)
            with self.assertRaisesRegex(AuthError, "mode 600"):
                store.load()

    def test_session_store_does_not_chmod_an_existing_parent(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            parent = Path(raw) / "existing"
            parent.mkdir(mode=0o755)
            parent.chmod(0o755)
            store = SessionStore(parent / "session.json")
            store.save(
                AuthSession(
                    access_token="access",
                    refresh_token=None,
                    expires_at=int(time.time()) + 3600,
                    user_id="user-id",
                    email=None,
                )
            )
            self.assertEqual(parent.stat().st_mode & 0o777, 0o755)

    def test_refresh_rotates_token_and_preserves_client_identity(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "session.json"
            store = SessionStore(path)
            store.save(
                AuthSession(
                    access_token="expired",
                    refresh_token="refresh-1",
                    expires_at=1,
                    user_id="user-id",
                    email="user@example.com",
                    producer_id="producer-a",
                    producer_label="vast-node",
                )
            )
            refreshed = AuthSession(
                access_token="access-2",
                refresh_token="refresh-2",
                expires_at=int(time.time()) + 3600,
                user_id="user-id",
                email="user@example.com",
            )
            client = FakeAuthClient(refreshed)
            manager = AuthManager(_settings(path), store=store, client=client)  # type: ignore[arg-type]
            session = manager.ensure_session()
            self.assertEqual(client.refresh_calls, ["refresh-1"])
            self.assertEqual(session.producer_id, "producer-a")
            self.assertEqual(session.producer_label, "vast-node")
            self.assertEqual(store.load().refresh_token, "refresh-2")

    def test_temporary_b2_lease_is_in_memory_and_redacted(self) -> None:
        settings = StorageSettings.from_lease(
            {
                "bucket_name": "bucket",
                "bucket_id": "bucket-id",
                "endpoint": "https://s3.us-east-005.backblazeb2.com",
                "region": "us-east-005",
                "prefix": "robolab/v1",
                "producer_id": "producer-a",
                "study_id": "study-a",
                "name_prefix": "robolab/v1/producers/producer-a/runs/study-a/run-a/",
                "key_id": "key-id",
                "application_key": "temporary-secret",
                "expires_at": "2026-08-24T00:00:00Z",
            }
        )
        self.assertNotIn("temporary-secret", repr(settings))
        report = RcloneClient(settings).configuration_report()
        self.assertTrue(report["temporary_credentials"])
        self.assertFalse(report["direct_credentials"])
        self.assertNotIn("temporary-secret", str(report))

    def test_upload_lease_requests_one_exact_purpose_and_run(self) -> None:
        access = RecordingAccessClient()
        manager = access.upload_manager(
            "study-a",
            purpose="finalize",
            run_id="run-a",
            producer_id="producer-a",
        )
        self.assertEqual(
            access.requests,
            [
                (
                    "POST",
                    "storage/lease",
                    {
                        "operation": "upload",
                        "study_id": "study-a",
                        "producer_id": "producer-a",
                        "purpose": "finalize",
                        "run_id": "run-a",
                    },
                )
            ],
        )
        self.assertEqual(
            manager.settings.lease_root,
            "manda-lease:bucket/robolab/v1/producers/producer-a/runs/study-a/run-a",
        )

    def test_checkpoint_lease_uses_study_scoped_working_path(self) -> None:
        settings = StorageSettings.from_lease(
            {
                "bucket_name": "bucket",
                "bucket_id": "bucket-id",
                "endpoint": "https://s3.us-east-005.backblazeb2.com",
                "region": "us-east-005",
                "prefix": "robolab/v1",
                "producer_id": "producer-a",
                "study_id": "study-a",
                "name_prefix": "robolab/v1/producers/producer-a/working/study-a/run-a/",
                "key_id": "key-id",
                "application_key": "temporary-secret",
                "expires_at": "2026-08-24T00:00:00Z",
            }
        )
        self.assertEqual(
            settings.working_root("run-a"),
            "manda-lease:bucket/robolab/v1/producers/producer-a/working/study-a/run-a",
        )


if __name__ == "__main__":
    unittest.main()
