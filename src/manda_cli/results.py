from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class ResultsError(ValueError):
    """Raised when a results file cannot be parsed or validated."""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ResultsError(f"Invalid JSON at {source}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ResultsError(f"Expected an object at {source}:{line_number}")
            rows.append(value)
    return rows


def task_name(row: dict[str, Any]) -> str | None:
    for key in ("task_name", "env_name", "task", "environment"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def episode_number(row: dict[str, Any]) -> int | str | None:
    for key in ("episode", "episode_id", "env_id"):
        value = row.get(key)
        if isinstance(value, (int, str)) and not isinstance(value, bool):
            return value
    return None


def success_value(row: dict[str, Any]) -> bool | None:
    for key in ("success", "succeeded", "is_success"):
        value = row.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in {0, 1}:
            return bool(value)
    return None


def numeric_value(row: dict[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def validate_results(
    run_dir: str | Path,
    *,
    expected_episodes: int | None = None,
    expected_tasks: Iterable[str] | None = None,
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    results_path = root / "episode_results.jsonl"
    errors: list[str] = []
    warnings: list[str] = []
    if not results_path.is_file():
        return {
            "valid": False,
            "run_dir": str(root),
            "errors": [f"Missing {results_path}"],
            "warnings": [],
        }
    rows = read_jsonl(results_path)
    names = [task_name(row) for row in rows]
    missing_names = sum(name is None for name in names)
    if missing_names:
        errors.append(f"{missing_names} rows have no recognized task field")
    missing_episode = sum(episode_number(row) is None for row in rows)
    if missing_episode:
        warnings.append(f"{missing_episode} rows have no recognized episode identifier")
    missing_success = sum(success_value(row) is None for row in rows)
    if missing_success:
        warnings.append(f"{missing_success} rows have no recognized binary success field")

    keys = [(task_name(row), episode_number(row)) for row in rows]
    duplicates = [key for key, count in Counter(keys).items() if count > 1 and None not in key]
    if duplicates:
        errors.append(f"Found {len(duplicates)} duplicate task/episode keys")
    if expected_episodes is not None and len(rows) != expected_episodes:
        errors.append(f"Expected {expected_episodes} episodes, found {len(rows)}")

    actual_tasks = {name for name in names if name}
    if expected_tasks is not None:
        expected_set = set(expected_tasks)
        missing = sorted(expected_set - actual_tasks)
        unexpected = sorted(actual_tasks - expected_set)
        if missing:
            errors.append(f"Missing {len(missing)} expected tasks: {', '.join(missing)}")
        if unexpected:
            warnings.append(f"Found {len(unexpected)} unexpected tasks: {', '.join(unexpected)}")

    task_counts = Counter(name for name in names if name)
    task_artifact_warnings = 0
    for name in actual_tasks:
        task_dir = root / name
        if not task_dir.is_dir() or not any(task_dir.glob("run_*.hdf5")):
            task_artifact_warnings += 1
    if task_artifact_warnings:
        warnings.append(
            f"{task_artifact_warnings} task directories have no discoverable run_*.hdf5 file"
        )

    successes = sum(success_value(row) is True for row in rows)
    return {
        "valid": not errors,
        "run_dir": str(root),
        "episode_count": len(rows),
        "task_count": len(actual_tasks),
        "successes": successes,
        "success_rate": successes / len(rows) if rows else None,
        "task_episode_counts": dict(sorted(task_counts.items())),
        "errors": errors,
        "warnings": warnings,
    }


def summarize_for_comparison(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        name = task_name(row)
        if name:
            grouped[name].append(row)

    summary: dict[str, dict[str, float]] = {}
    for name, task_rows in grouped.items():
        successes = [success_value(row) for row in task_rows]
        known_successes = [value for value in successes if value is not None]
        scores = [
            value
            for row in task_rows
            if (value := numeric_value(row, ("score", "subtask_score"))) is not None
        ]
        failure_scores = [
            value
            for row in task_rows
            if success_value(row) is False
            and (value := numeric_value(row, ("score", "subtask_score"))) is not None
        ]
        summary[name] = {
            "n": float(len(task_rows)),
            "success_rate": (
                sum(value is True for value in known_successes) / len(known_successes)
                if known_successes
                else float("nan")
            ),
            "mean_score": sum(scores) / len(scores) if scores else float("nan"),
            "failure_progress": (
                sum(failure_scores) / len(failure_scores) if failure_scores else float("nan")
            ),
        }
    return summary
