from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from manda_cli.interesting import compare_runs
from manda_cli.results import validate_results


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class ResultTests(unittest.TestCase):
    def test_validation_detects_duplicate_episode(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            row = {"task_name": "TaskA", "episode": 0, "success": True}
            _write_jsonl(root / "episode_results.jsonl", [row, row])
            report = validate_results(root)
            self.assertFalse(report["valid"])
            self.assertTrue(any("duplicate" in error for error in report["errors"]))

    def test_validation_accepts_complete_minimal_results(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task_dir = root / "TaskA"
            task_dir.mkdir()
            (task_dir / "run_0.hdf5").touch()
            rows = [
                {"task_name": "TaskA", "episode": 0, "success": True, "score": 1.0},
                {"task_name": "TaskA", "episode": 1, "success": False, "score": 0.5},
            ]
            _write_jsonl(root / "episode_results.jsonl", rows)
            report = validate_results(root, expected_episodes=2, expected_tasks=["TaskA"])
            self.assertTrue(report["valid"])
            self.assertEqual(report["success_rate"], 0.5)

    def test_comparison_surfaces_task_rank_reversal(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            left = root / "left.jsonl"
            right = root / "right.jsonl"
            _write_jsonl(
                left,
                [
                    {"task_name": "TaskA", "episode": 0, "success": True, "score": 1.0},
                    {"task_name": "TaskB", "episode": 0, "success": False, "score": 0.5},
                ],
            )
            _write_jsonl(
                right,
                [
                    {"task_name": "TaskA", "episode": 0, "success": False, "score": 0.2},
                    {"task_name": "TaskB", "episode": 0, "success": True, "score": 1.0},
                ],
            )
            rows = compare_runs(left, right, left_policy="left", right_policy="right")
            self.assertEqual({row["task_name"] for row in rows}, {"TaskA", "TaskB"})
            self.assertTrue(all(row["components"]["success_rate_gap"] == 1 for row in rows))


if __name__ == "__main__":
    unittest.main()
