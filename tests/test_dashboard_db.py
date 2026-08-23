from __future__ import annotations

import math
import os
import unittest
from unittest.mock import patch

from manda_cli.dashboard_db import (
    DashboardIndexer,
    SupabaseError,
    SupabaseSettings,
    catalog_sync_plan,
)


def _catalog() -> dict:
    return {
        "catalog_version": "catalog-1",
        "generated_at": "2026-08-23T00:00:00+00:00",
        "study_id": "study",
        "runs": [
            {
                "run_id": "run-1",
                "study_id": "study",
                "producer_id": "node-a",
                "stage": "pilot",
                "policy": "pi05",
                "manifest_sha256": "a" * 64,
                "remote": "test:bucket/robolab/v1/producers/node-a/runs/study/run-1",
                "episode_count": 1,
                "object_count": 2,
                "total_bytes": 30,
                "validation": {"valid": True},
            }
        ],
        "tasks": [
            {
                "source_run_id": "run-1",
                "source_producer_id": "node-a",
                "task_name": "TaskA",
                "episode_count": 1,
                "successes": 1,
                "success_rate": 1.0,
                "mean_score": 0.75,
            }
        ],
        "episodes": [
            {
                "source_run_id": "run-1",
                "source_producer_id": "node-a",
                "task_name": "TaskA",
                "episode": 0,
                "success": True,
                "score": 0.75,
            }
        ],
        "assets": [
            {
                "source_run_id": "run-1",
                "source_producer_id": "node-a",
                "path": "TaskA/episode_0.mp4",
                "bucket_name": "bucket",
                "object_key": (
                    "robolab/v1/producers/node-a/runs/study/run-1/TaskA/episode_0.mp4"
                ),
                "role": "video",
                "content_type": "video/mp4",
                "bytes": 10,
                "sha256": "b" * 64,
                "task_name": "TaskA",
            }
        ],
    }


class FakeRestClient:
    def __init__(self, existing: list[dict] | None = None):
        self.existing = existing or []
        self.upserts: list[tuple[str, list[dict], str]] = []
        self.deletes: list[tuple[str, dict]] = []
        self.updates: list[tuple[str, dict, dict]] = []

    def select(self, table, *, columns="*", filters=None, limit=None):
        del columns, filters, limit
        return self.existing if table == "runs" else []

    def upsert(self, table, rows, *, on_conflict, batch_size=500):
        del batch_size
        self.upserts.append((table, rows, on_conflict))
        return len(rows)

    def delete(self, table, *, filters):
        self.deletes.append((table, filters))

    def update(self, table, payload, *, filters):
        self.updates.append((table, payload, filters))


class DashboardDatabaseTests(unittest.TestCase):
    def test_settings_require_server_secret_and_secure_url(self) -> None:
        environment = {
            "SUPABASE_URL": "https://example.supabase.co",
            "SUPABASE_SECRET_KEY": "sb_secret_test",
            "SUPABASE_PUBLISHABLE_KEY": "sb_publishable_test",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = SupabaseSettings.from_environment()
        self.assertEqual(settings.url, environment["SUPABASE_URL"])
        self.assertNotIn(settings.secret_key, str(settings.report()))
        with (
            patch.dict(
                os.environ,
                {"SUPABASE_URL": "http://example.com", "SUPABASE_SECRET_KEY": "secret"},
                clear=True,
            ),
            self.assertRaisesRegex(SupabaseError, "HTTPS"),
        ):
            SupabaseSettings.from_environment()

    def test_sync_replaces_children_and_marks_run_ready(self) -> None:
        client = FakeRestClient()
        report = DashboardIndexer(client).sync(_catalog())  # type: ignore[arg-type]
        self.assertEqual(report["synced_runs"], 1)
        self.assertEqual(report["asset_count"], 1)
        self.assertEqual(
            [table for table, _ in client.deletes],
            ["task_results", "episodes", "assets"],
        )
        tables = [table for table, _, _ in client.upserts]
        self.assertEqual(
            tables,
            ["studies", "runs", "task_results", "episodes", "assets", "catalog_imports"],
        )
        run_payload = client.upserts[1][1][0]
        self.assertEqual(run_payload["run_key"], "node-a/run-1")
        self.assertEqual(
            run_payload["remote_prefix"],
            "robolab/v1/producers/node-a/runs/study/run-1",
        )
        self.assertEqual(client.updates[-1][1]["index_status"], "ready")

    def test_same_manifest_and_index_schema_is_idempotently_skipped(self) -> None:
        client = FakeRestClient(
            [{"manifest_sha256": "a" * 64, "index_schema_version": 1, "index_status": "ready"}]
        )
        report = DashboardIndexer(client).sync(_catalog())  # type: ignore[arg-type]
        self.assertEqual(report["synced_runs"], 0)
        self.assertEqual(report["skipped_runs"], 1)
        self.assertFalse(client.deletes)
        self.assertEqual([table for table, _, _ in client.upserts], ["studies", "catalog_imports"])

    def test_dry_run_plan_reports_all_relations(self) -> None:
        plan = catalog_sync_plan(_catalog())
        self.assertEqual(plan["run_count"], 1)
        self.assertEqual(plan["task_count"], 1)
        self.assertEqual(plan["episode_count"], 1)
        self.assertEqual(plan["asset_count"], 1)

    def test_non_finite_metrics_are_normalized_for_json(self) -> None:
        catalog = _catalog()
        catalog["tasks"][0]["mean_score"] = math.nan
        catalog["episodes"][0]["score"] = math.inf
        catalog["episodes"][0]["duration"] = math.nan
        client = FakeRestClient()
        DashboardIndexer(client).sync(catalog)  # type: ignore[arg-type]
        task = next(rows[0] for table, rows, _ in client.upserts if table == "task_results")
        episode = next(rows[0] for table, rows, _ in client.upserts if table == "episodes")
        self.assertIsNone(task["mean_score"])
        self.assertIsNone(episode["score"])
        self.assertIsNone(episode["duration_seconds"])


if __name__ == "__main__":
    unittest.main()
