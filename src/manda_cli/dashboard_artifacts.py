from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .results import read_jsonl, task_name

DASHBOARD_ARTIFACT_SCHEMA_VERSION = 1
DASHBOARD_ARTIFACT_DIRECTORY = "_dashboard/v1"
DEFAULT_MAX_POINTS = 300
DEFAULT_MAX_SERIES = 24
DEFAULT_MAX_DIMENSIONS = 16

_OBS_SIGNALS = (
    ("proprio_obs/arm_joint_pos", "arm_joint_pos"),
    ("proprio_obs/gripper_pos", "gripper_pos"),
    ("proprio_obs/eef_pos", "eef_pos"),
    ("proprio_obs/eef_quat", "eef_quat"),
    ("proprio_obs/ee_pos", "ee_pos"),
    ("proprio_obs/ee_quat", "ee_quat"),
)
_EE_POSE_SIGNALS = (
    ("ee_pose/position", "ee_position"),
    ("ee_pose/orientation", "ee_orientation"),
    ("ee_pose/linear_velocity", "ee_linear_velocity"),
    ("ee_pose/angular_velocity", "ee_angular_velocity"),
)
_STATE_BOILERPLATE = {"articulation", "rigid_object"}


class DashboardArtifactError(ValueError):
    """Raised when dashboard-friendly artifacts cannot be derived safely."""


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, allow_nan=False, separators=(",", ":"), sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _integer(row: dict[str, Any], *keys: str, default: int = 0) -> int:
    for key in keys:
        value = row.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
            return int(value)
    return default


def _safe_task_directory(root: Path, name: str) -> Path:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise DashboardArtifactError(f"Unsafe task name in episode results: {name!r}")
    task_directory = (root / name).resolve()
    try:
        task_directory.relative_to(root)
    except ValueError as exc:
        raise DashboardArtifactError(f"Task escapes the run directory: {name!r}") from exc
    return task_directory


def _source_task_directory(root: Path, row: dict[str, Any], logical_name: str) -> Path:
    candidates = [
        value
        for key in ("env_name", "task_name", "task", "environment")
        if isinstance((value := row.get(key)), str) and value
    ]
    for candidate in dict.fromkeys([*candidates, logical_name]):
        directory = _safe_task_directory(root, candidate)
        if directory.is_dir():
            return directory
    return _safe_task_directory(root, logical_name)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DashboardArtifactError(f"Cannot parse {path}: {exc}") from exc


def _event_log_candidates(task_directory: Path, env_id: int, run_index: int) -> tuple[Path, ...]:
    return (
        task_directory / f"log_{run_index}_env{env_id}.json",
        task_directory / f"log_{env_id}.json",
    )


def _resolve_dt(task_directory: Path, env_id: int, run_index: int) -> float | None:
    config_path = task_directory / "env_cfg.json"
    if config_path.is_file():
        try:
            config = _load_json(config_path)
            simulation = config.get("sim") if isinstance(config, dict) else None
            simulation_dt = simulation.get("dt") if isinstance(simulation, dict) else None
            decimation = config.get("decimation") if isinstance(config, dict) else None
            if (
                isinstance(simulation_dt, (int, float))
                and not isinstance(simulation_dt, bool)
                and isinstance(decimation, (int, float))
                and not isinstance(decimation, bool)
            ):
                value = float(simulation_dt) * float(decimation)
                return value if math.isfinite(value) and value > 0 else None
        except DashboardArtifactError:
            pass
    for candidate in _event_log_candidates(task_directory, env_id, run_index):
        if not candidate.is_file():
            continue
        try:
            value = _load_json(candidate)
        except DashboardArtifactError:
            continue
        raw_dt = value.get("dt") if isinstance(value, dict) else None
        if isinstance(raw_dt, (int, float)) and not isinstance(raw_dt, bool):
            dt = float(raw_dt)
            if math.isfinite(dt) and dt > 0:
                return dt
    return None


def _event_severity(name: str) -> str:
    normalized = (name or "").upper()
    if normalized.endswith("_SUCCESS") or normalized == "OK":
        return "success"
    if (
        normalized.endswith("_FAILURE")
        or "DROPPED" in normalized
        or "WRONG" in normalized
        or "HIT" in normalized
    ):
        return "failure"
    return "neutral"


def _finite(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return int(value) if isinstance(value, int) else number


def _events(
    task_directory: Path,
    env_id: int,
    run_index: int,
    dt: float | None,
) -> list[dict[str, Any]]:
    log_path = next(
        (
            path
            for path in _event_log_candidates(task_directory, env_id, run_index)
            if path.is_file()
        ),
        None,
    )
    if log_path is None:
        return []
    value = _load_json(log_path)
    events: list[dict[str, Any]] = []
    if isinstance(value, dict):
        raw_events = value.get("events")
        if not isinstance(raw_events, list):
            return []
        for raw_event in raw_events:
            if not isinstance(raw_event, dict):
                continue
            step = _finite(raw_event.get("step"))
            if step is None:
                continue
            step_number = int(step)
            name = str(raw_event.get("name") or "")
            events.append(
                {
                    "step": step_number,
                    "time_s": step_number * dt if dt else None,
                    "code": raw_event.get("code"),
                    "name": name,
                    "info": str(raw_event.get("info") or ""),
                    "score": _finite(raw_event.get("score")),
                    "severity": _event_severity(name),
                }
            )
        return events
    if not isinstance(value, list):
        return []
    previous_status: Any = object()
    for step_number, raw_event in enumerate(value):
        if not isinstance(raw_event, dict):
            continue
        status = raw_event.get("status")
        if status is None or status == previous_status:
            previous_status = status
            continue
        name = f"STATUS_{status}"
        events.append(
            {
                "step": step_number,
                "time_s": step_number * dt if dt else None,
                "code": status,
                "name": name,
                "info": str(raw_event.get("info") or "").strip(),
                "score": _finite(raw_event.get("score")),
                "severity": _event_severity(name),
            }
        )
        previous_status = status
    return events


def _state_label(parts: tuple[str, ...]) -> str:
    cleaned = [part for part in parts if part not in _STATE_BOILERPLATE]
    return "state." + ".".join(cleaned)


def _walk_state_group(group: Any, h5py: Any, prefix: tuple[str, ...] = ()) -> list[tuple[str, Any]]:
    datasets: list[tuple[str, Any]] = []
    for key in sorted(group.keys()):
        child = group[key]
        child_prefix = (*prefix, key)
        if isinstance(child, h5py.Group):
            datasets.extend(_walk_state_group(child, h5py, child_prefix))
        elif isinstance(child, h5py.Dataset) and child.ndim >= 2:
            datasets.append((_state_label(child_prefix), child))
    return datasets


def _series_payload(
    array: Any,
    np: Any,
    *,
    dt: float | None,
    max_points: int,
    max_dimensions: int,
) -> dict[str, Any] | None:
    values = np.asarray(array)
    if values.ndim < 1 or values.shape[0] == 0:
        return None
    values = (
        values.reshape(-1, 1)
        if values.ndim == 1
        else values.reshape(values.shape[0], -1)
    )
    source_steps = int(values.shape[0])
    total_dimensions = int(values.shape[1])
    values = values[:, :max_dimensions]
    stride = max(1, math.ceil(values.shape[0] / max_points))
    sampled = values[::stride]
    time_values = np.arange(0, values.shape[0], stride, dtype=float)
    if dt:
        time_values *= dt
    rows: list[list[float | None]] = []
    for time_value, row in zip(time_values.tolist(), sampled.tolist(), strict=True):
        output_row: list[float | None] = [float(time_value)]
        for value in row:
            number = float(value)
            output_row.append(number if math.isfinite(number) else None)
        rows.append(output_row)
    return {
        "labels": [f"d{index}" for index in range(int(values.shape[1]))],
        "data": rows,
        "source_steps": source_steps,
        "total_dimensions": total_dimensions,
    }


def _episode_key(handle: Any, env_id: int, run_index: int) -> str | None:
    data = handle.get("data")
    if data is None:
        return None
    candidates = (f"demo_{env_id}", f"demo_{run_index}_{env_id}", f"episode_{env_id}")
    for candidate in candidates:
        if candidate in data:
            return candidate
    keys = sorted(data.keys())
    if not keys:
        return None
    return keys[env_id] if 0 <= env_id < len(keys) else keys[0]


def _timeseries(
    hdf5_path: Path,
    env_id: int,
    run_index: int,
    *,
    dt: float | None,
    max_points: int,
    max_series: int,
    max_dimensions: int,
) -> dict[str, Any] | None:
    try:
        import h5py  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        return None
    series: dict[str, Any] = {}
    with h5py.File(hdf5_path, "r") as handle:
        key = _episode_key(handle, env_id, run_index)
        if key is None:
            return None
        episode = handle["data"][key]
        resolved_dt = dt
        if not resolved_dt:
            raw_dt = episode.attrs.get("dt", 0.0)
            if isinstance(raw_dt, (int, float)) and float(raw_dt) > 0:
                resolved_dt = float(raw_dt)

        def add(label: str, dataset: Any) -> None:
            if len(series) >= max_series:
                return
            payload = _series_payload(
                dataset[...],
                np,
                dt=resolved_dt,
                max_points=max_points,
                max_dimensions=max_dimensions,
            )
            if payload is not None:
                series[label] = payload

        observations = episode.get("obs")
        if observations is not None:
            for source, label in _OBS_SIGNALS:
                try:
                    dataset = observations[source]
                except KeyError:
                    continue
                if isinstance(dataset, h5py.Dataset) and dataset.ndim >= 2:
                    add(label, dataset)
        actions = episode.get("actions")
        if isinstance(actions, h5py.Dataset) and actions.ndim >= 2:
            add("actions", actions)
        states = episode.get("states")
        if isinstance(states, h5py.Dataset) and states.ndim >= 2:
            add("state", states)
        elif isinstance(states, h5py.Group):
            for label, dataset in _walk_state_group(states, h5py):
                add(label, dataset)
        for source, label in _EE_POSE_SIGNALS:
            try:
                dataset = episode[source]
            except KeyError:
                continue
            if isinstance(dataset, h5py.Dataset) and dataset.ndim >= 2:
                add(label, dataset)
        source_steps = max(
            (int(payload["source_steps"]) for payload in series.values()),
            default=0,
        )
        return {
            "dt": resolved_dt,
            "num_steps": source_steps,
            "series": series,
            "truncated": len(series) >= max_series,
        }


def _hdf5_path(task_directory: Path, run_index: int) -> Path | None:
    candidates = (
        task_directory / f"run_{run_index}.hdf5",
        task_directory / f"run_{run_index}.h5",
        task_directory / "data.hdf5",
        task_directory / "data.h5",
    )
    return next((path for path in candidates if path.is_file()), None)


def prepare_dashboard_artifacts(
    run_dir: str | Path,
    *,
    max_points: int = DEFAULT_MAX_POINTS,
    max_series: int = DEFAULT_MAX_SERIES,
    max_dimensions: int = DEFAULT_MAX_DIMENSIONS,
) -> dict[str, Any]:
    """Create deterministic browser diagnostics inside an unfinished artifact run."""
    root = Path(run_dir).expanduser().resolve()
    results_path = root / "episode_results.jsonl"
    if not results_path.is_file():
        raise DashboardArtifactError(f"Missing {results_path}")
    if max_points < 2 or max_series < 1 or max_dimensions < 1:
        raise DashboardArtifactError("Dashboard derivation limits must be positive")
    rows = read_jsonl(results_path)
    generated: list[str] = []
    warnings: list[str] = []
    event_count = 0
    timeseries_count = 0
    hdf5_available = True
    try:
        import h5py  # noqa: F401  # type: ignore[import-not-found]
        import numpy  # noqa: F401  # type: ignore[import-not-found]
    except ImportError:
        hdf5_available = False

    for row_number, row in enumerate(rows, start=1):
        name = task_name(row)
        if name is None:
            raise DashboardArtifactError(f"Episode row {row_number} has no task name")
        source_task_directory = _source_task_directory(root, row, name)
        output_task_directory = _safe_task_directory(root, name)
        env_id = _integer(row, "env_id", "episode", default=row_number - 1)
        run_index = _integer(row, "run_index", "run", default=0)
        dt = _resolve_dt(source_task_directory, env_id, run_index)
        try:
            events = _events(source_task_directory, env_id, run_index, dt)
        except DashboardArtifactError as exc:
            events = []
            warnings.append(str(exc))
        event_count += len(events)
        timeseries = None
        source_hdf5 = _hdf5_path(source_task_directory, run_index)
        if source_hdf5 is not None and hdf5_available:
            try:
                timeseries = _timeseries(
                    source_hdf5,
                    env_id,
                    run_index,
                    dt=dt,
                    max_points=max_points,
                    max_series=max_series,
                    max_dimensions=max_dimensions,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                warnings.append(f"Cannot derive {name} run {run_index} env {env_id}: {exc}")
        if timeseries and timeseries.get("series"):
            timeseries_count += 1
        output = (
            output_task_directory
            / DASHBOARD_ARTIFACT_DIRECTORY
            / f"run_{run_index}_env{env_id}.json"
        )
        _write_json(
            output,
            {
                "schema_version": DASHBOARD_ARTIFACT_SCHEMA_VERSION,
                "task_name": name,
                "run_index": run_index,
                "env_id": env_id,
                "dt": dt,
                "events": events,
                "timeseries": timeseries,
            },
        )
        generated.append(output.relative_to(root).as_posix())

    if not hdf5_available and any(root.glob("*/run_*.hdf5")):
        warnings.append(
            "HDF5 timeseries were not derived; install manda-cli[analysis] to include them"
        )
    return {
        "schema_version": DASHBOARD_ARTIFACT_SCHEMA_VERSION,
        "episode_count": len(rows),
        "event_count": event_count,
        "timeseries_count": timeseries_count,
        "generated": generated,
        "warnings": warnings,
    }
