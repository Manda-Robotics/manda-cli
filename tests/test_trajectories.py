from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from manda_cli.trajectories import compare_trajectory_runs, extract_trajectory_metrics

HAS_ANALYSIS = bool(importlib.util.find_spec("h5py") and importlib.util.find_spec("numpy"))


@unittest.skipUnless(HAS_ANALYSIS, "analysis extra is not installed")
class TrajectoryTests(unittest.TestCase):
    @staticmethod
    def _write_regression_run(
        root: Path,
        *,
        position_offset: float = 0.0,
        quaternion_sign: float = 1.0,
        success: bool = True,
    ) -> None:
        import h5py
        import numpy as np

        task_dir = root / "TaskA"
        task_dir.mkdir()
        with h5py.File(task_dir / "run_0.hdf5", "w") as handle:
            data = handle.create_group("data")
            data.attrs["isaacsim_version"] = "6.0.1.0"
            demo = data.create_group("demo_0")
            demo.attrs["success"] = success
            demo.attrs["seed"] = 0
            demo.attrs["num_samples"] = 3
            demo.create_dataset(
                "actions",
                data=np.array([[0, 0], [1, 1], [2, 0]], dtype=np.float32),
            )
            position = np.array(
                [[0, 0, 0], [1 + position_offset, 0, 0], [2, 0, 0]],
                dtype=np.float32,
            )
            demo.create_dataset("ee_pose/position", data=position)
            orientation = np.tile(
                np.array([quaternion_sign, 0, 0, 0], dtype=np.float32), (3, 1)
            )
            demo.create_dataset("ee_pose/orientation", data=orientation)
            root_pose = np.concatenate([position, orientation], axis=1)
            demo.create_dataset("states/rigid_object/banana/root_pose", data=root_pose)
            demo.create_dataset(
                "states/rigid_object/banana/root_velocity",
                data=np.zeros((3, 6), dtype=np.float32),
            )
            demo.create_dataset(
                "initial_state/rigid_object/banana/root_pose",
                data=root_pose[:1],
            )

    def test_extracts_metrics_and_initial_state_hash(self) -> None:
        import h5py
        import numpy as np

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task_dir = root / "TaskA"
            task_dir.mkdir()
            path = task_dir / "run_0.hdf5"
            with h5py.File(path, "w") as handle:
                demo = handle.create_group("data/demo_0")
                demo.create_dataset(
                    "ee_pose/position",
                    data=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32),
                )
                demo.create_dataset(
                    "ee_pose/linear_velocity",
                    data=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
                )
                demo.create_dataset(
                    "actions",
                    data=np.array([[0, 0], [1, 1], [0, 0]], dtype=np.float32),
                )
                demo.create_dataset(
                    "subtask/score", data=np.array([0, 0.5, 0.25], dtype=np.float32)
                )
                demo.create_dataset(
                    "initial_state/rigid_object/banana/root_pose",
                    data=np.zeros((1, 7), dtype=np.float32),
                )

            rows = extract_trajectory_metrics(root, "policy-a")
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["ee_path_length_m"], 2.0)
            self.assertEqual(rows[0]["subtask_score_regressions"], 1)
            self.assertEqual(len(rows[0]["initial_state_sha256"]), 64)

    def test_compares_actions_states_and_sign_invariant_quaternions(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            reference = root / "reference"
            candidate = root / "candidate"
            reference.mkdir()
            candidate.mkdir()
            self._write_regression_run(reference)
            self._write_regression_run(candidate, quaternion_sign=-1.0)

            report = compare_trajectory_runs(reference, candidate)
            episode = report["episodes"][0]
            self.assertTrue(report["summary"]["passed"])
            self.assertTrue(episode["actions_exact"])
            self.assertFalse(episode["initial_state_sha256"]["exact"])
            self.assertTrue(episode["initial_state_within_tolerance"])
            self.assertTrue(episode["physics_state_within_tolerance"])
            orientation = next(
                channel
                for channel in episode["channels"]
                if channel["path"] == "ee_pose/orientation"
            )
            self.assertEqual(orientation["max_angle_deg"], 0.0)

    def test_reports_first_step_that_exceeds_position_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            reference = root / "reference"
            candidate = root / "candidate"
            reference.mkdir()
            candidate.mkdir()
            self._write_regression_run(reference)
            self._write_regression_run(candidate, position_offset=0.005)

            report = compare_trajectory_runs(reference, candidate, position_atol_m=0.001)
            episode = report["episodes"][0]
            self.assertFalse(report["summary"]["passed"])
            self.assertEqual(episode["first_exceed_step"], 1)
            position = next(
                channel
                for channel in episode["channels"]
                if channel["path"] == "ee_pose/position"
            )
            self.assertAlmostEqual(position["max_abs"], 0.005, places=6)
            self.assertFalse(position["within_tolerance"])


if __name__ == "__main__":
    unittest.main()
