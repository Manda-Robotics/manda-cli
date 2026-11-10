from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from manda_cli.artifacts import create_manifest, profile_includes
from manda_cli.dashboard_artifacts import prepare_dashboard_artifacts


class DashboardArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.task = self.root / "TaskA"
        self.task.mkdir()
        (self.root / "episode_results.jsonl").write_text(
            json.dumps(
                {
                    "task_name": "TaskA",
                    "episode": 2,
                    "run": 1,
                    "success": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (self.task / "env_cfg.json").write_text(
            json.dumps({"sim": {"dt": 0.01}, "decimation": 2}),
            encoding="utf-8",
        )
        (self.task / "log_1_env2.json").write_text(
            json.dumps(
                {
                    "events": [
                        {
                            "step": 5,
                            "code": 7,
                            "name": "TARGET_OBJECT_DROPPED",
                            "info": "lost grasp",
                            "score": 0.5,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_derives_timed_events_and_manifest_role(self) -> None:
        report = prepare_dashboard_artifacts(self.root)
        self.assertEqual(report["episode_count"], 1)
        self.assertEqual(report["event_count"], 1)
        output = self.task / "_dashboard" / "v1" / "run_1_env2.json"
        payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 1)
        self.assertAlmostEqual(payload["dt"], 0.02)
        self.assertAlmostEqual(payload["events"][0]["time_s"], 0.1)
        self.assertEqual(payload["events"][0]["severity"], "failure")

        manifest = create_manifest(
            self.root,
            {"run_id": "run-1", "study_id": "study", "producer_id": "producer"},
        )
        role_by_path = {row["path"]: row["role"] for row in manifest["files"]}
        self.assertEqual(role_by_path["TaskA/_dashboard/v1/run_1_env2.json"], "dashboard")
        self.assertTrue(profile_includes("dashboard", "metadata"))

    def test_uses_existing_env_directory_but_keeps_logical_task_name(self) -> None:
        (self.root / "episode_results.jsonl").write_text(
            json.dumps(
                {
                    "task_name": "LogicalTask",
                    "env_name": "TaskA",
                    "episode": 2,
                    "run": 1,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        report = prepare_dashboard_artifacts(self.root)
        self.assertEqual(report["event_count"], 1)
        output = self.root / "LogicalTask" / "_dashboard" / "v1" / "run_1_env2.json"
        payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["task_name"], "LogicalTask")
        self.assertEqual(payload["events"][0]["name"], "TARGET_OBJECT_DROPPED")

    @unittest.skipUnless(
        importlib.util.find_spec("h5py") and importlib.util.find_spec("numpy"),
        "HDF5 analysis dependencies are not installed",
    )
    def test_derives_downsampled_timeseries(self) -> None:
        import h5py
        import numpy as np

        with h5py.File(self.task / "run_1.hdf5", "w") as handle:
            episode = handle.create_group("data/demo_2")
            episode.create_dataset("actions", data=np.arange(1000).reshape(100, 10))
        report = prepare_dashboard_artifacts(self.root, max_points=12, max_dimensions=4)
        self.assertEqual(report["timeseries_count"], 1)
        payload = json.loads(
            (self.task / "_dashboard" / "v1" / "run_1_env2.json").read_text(
                encoding="utf-8"
            )
        )
        actions = payload["timeseries"]["series"]["actions"]
        self.assertLessEqual(len(actions["data"]), 12)
        self.assertEqual(actions["total_dimensions"], 10)
        self.assertEqual(actions["labels"], ["d0", "d1", "d2", "d3"])

    def test_skip_existing_leaves_finished_episodes_untouched(self) -> None:
        prepare_dashboard_artifacts(self.root)
        output = self.task / "_dashboard" / "v1" / "run_1_env2.json"
        before = output.stat()
        report = prepare_dashboard_artifacts(self.root, skip_existing=True)
        after = output.stat()
        self.assertEqual((before.st_ino, before.st_mtime_ns), (after.st_ino, after.st_mtime_ns))
        self.assertEqual(report["generated"], ["TaskA/_dashboard/v1/run_1_env2.json"])
        self.assertEqual(report["timeseries_count"], 0)


if __name__ == "__main__":
    unittest.main()
