from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .results import read_jsonl, summarize_for_comparison


def _difference(left: float, right: float) -> float | None:
    if math.isnan(left) or math.isnan(right):
        return None
    return abs(left - right)


def _serializable(values: dict[str, float]) -> dict[str, float | None]:
    return {key: value if math.isfinite(value) else None for key, value in values.items()}


def compare_runs(
    left_path: str | Path,
    right_path: str | Path,
    *,
    left_policy: str,
    right_policy: str,
) -> list[dict[str, Any]]:
    left = summarize_for_comparison(read_jsonl(left_path))
    right = summarize_for_comparison(read_jsonl(right_path))
    shared = sorted(set(left) & set(right))
    if not shared:
        raise ValueError("No shared tasks found between the two result files")

    left_rates = [
        row["success_rate"] for row in left.values() if math.isfinite(row["success_rate"])
    ]
    right_rates = [
        row["success_rate"] for row in right.values() if math.isfinite(row["success_rate"])
    ]
    if not left_rates or not right_rates:
        raise ValueError("Both result files need recognized binary success fields")
    left_overall = sum(left_rates) / len(left_rates)
    right_overall = sum(right_rates) / len(right_rates)
    overall_direction = (
        math.copysign(1, left_overall - right_overall) if left_overall != right_overall else 0
    )

    rows: list[dict[str, Any]] = []
    for task in shared:
        left_row = left[task]
        right_row = right[task]
        if not math.isfinite(left_row["success_rate"]) or not math.isfinite(
            right_row["success_rate"]
        ):
            continue
        task_delta = left_row["success_rate"] - right_row["success_rate"]
        task_direction = math.copysign(1, task_delta) if task_delta else 0
        rank_reversal = bool(
            overall_direction and task_direction and task_direction != overall_direction
        )
        success_gap = abs(task_delta)
        score_gap = _difference(left_row["mean_score"], right_row["mean_score"])
        progress_gap = _difference(left_row["failure_progress"], right_row["failure_progress"])
        # This is only a transparent triage aid. Claims must use the components and uncertainty.
        triage_score = success_gap + 0.5 * (score_gap or 0) + 0.5 * (progress_gap or 0)
        if rank_reversal:
            triage_score += 0.25
        rows.append(
            {
                "task_name": task,
                "left_policy": left_policy,
                "right_policy": right_policy,
                "left": _serializable(left_row),
                "right": _serializable(right_row),
                "components": {
                    "success_rate_gap": success_gap,
                    "mean_score_gap": score_gap,
                    "failure_progress_gap": progress_gap,
                    "rank_reversal": rank_reversal,
                },
                "triage_score": triage_score,
            }
        )
    return sorted(rows, key=lambda row: row["triage_score"], reverse=True)


def write_comparison(rows: list[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
    return output
