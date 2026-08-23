from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


class AnalysisDependencyError(RuntimeError):
    """Raised when optional trajectory dependencies are unavailable."""


def _dependencies() -> tuple[Any, Any]:
    try:
        import h5py  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        raise AnalysisDependencyError(
            "Trajectory extraction requires the analysis extra: uv sync --extra analysis"
        ) from exc
    return h5py, np


def _dataset(group: Any, path: str) -> Any | None:
    try:
        return group[path][:]
    except (KeyError, TypeError):
        return None


def _initial_state_hash(group: Any) -> str | None:
    initial = group.get("initial_state")
    if initial is None:
        return None
    digest = hashlib.sha256()

    def visit(name: str, item: Any) -> None:
        if hasattr(item, "shape") and hasattr(item, "dtype"):
            digest.update(name.encode())
            digest.update(str(item.shape).encode())
            digest.update(str(item.dtype).encode())
            digest.update(item[:].tobytes())

    initial.visititems(visit)
    return digest.hexdigest()


def _metrics(group: Any, np: Any) -> dict[str, int | float | str | None]:
    positions = _dataset(group, "ee_pose/position")
    velocities = _dataset(group, "ee_pose/linear_velocity")
    actions = _dataset(group, "actions")
    scores = _dataset(group, "subtask/score")

    result: dict[str, int | float | str | None] = {
        "step_count": int(positions.shape[0]) if positions is not None else None,
        "initial_state_sha256": _initial_state_hash(group),
    }
    if positions is not None and len(positions) > 1:
        deltas = np.diff(positions.astype(np.float64), axis=0)
        result["ee_path_length_m"] = float(np.linalg.norm(deltas, axis=1).sum())
    if velocities is not None and len(velocities):
        velocity = velocities.astype(np.float64)
        speed = np.linalg.norm(velocity, axis=1)
        result["ee_speed_mean_mps"] = float(np.mean(speed))
        result["ee_speed_std_mps"] = float(np.std(speed))
        if len(velocity) > 1:
            delta = np.diff(velocity, axis=0)
            result["ee_velocity_delta_rms_per_step"] = float(np.sqrt(np.mean(np.square(delta))))
    if actions is not None and len(actions) > 1:
        action_values = actions.astype(np.float64)
        action_delta = np.diff(action_values, axis=0)
        result["action_total_variation_per_step"] = float(
            np.abs(action_delta).sum() / len(action_delta)
        )
        signs = np.sign(action_delta)
        reversals = (signs[1:] * signs[:-1]) < 0
        result["action_direction_reversals"] = int(np.count_nonzero(reversals))
        gripper = action_values[:, -1]
        if float(np.max(gripper) - np.min(gripper)) > 1e-6:
            threshold = float((np.max(gripper) + np.min(gripper)) / 2)
            states = gripper > threshold
            result["gripper_state_switches"] = int(np.count_nonzero(states[1:] != states[:-1]))
    if scores is not None and len(scores):
        score_values = scores.astype(np.float64).reshape(-1)
        result["subtask_max_score"] = float(np.max(score_values))
        result["subtask_final_score"] = float(score_values[-1])
        result["subtask_score_regressions"] = int(np.count_nonzero(np.diff(score_values) < -1e-6))
    return result


def extract_trajectory_metrics(run_dir: str | Path, policy: str) -> list[dict[str, Any]]:
    h5py, np = _dependencies()
    root = Path(run_dir).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for hdf5_path in sorted(root.glob("*/run_*.hdf5")):
        task = hdf5_path.parent.name
        match = re.search(r"run_(\d+)\.hdf5$", hdf5_path.name)
        run_index = int(match.group(1)) if match else 0
        with h5py.File(hdf5_path, "r") as handle:
            data = handle.get("data")
            if data is None:
                continue
            demo_names = sorted(
                (name for name in data if name.startswith("demo_")),
                key=lambda value: int(value.split("_")[-1]),
            )
            num_envs = len(demo_names)
            for demo_name in demo_names:
                env_id = int(demo_name.split("_")[-1])
                row: dict[str, Any] = {
                    "policy": policy,
                    "task_name": task,
                    "run_index": run_index,
                    "env_id": env_id,
                    "episode": run_index * num_envs + env_id,
                    "source_hdf5": str(hdf5_path),
                    "demo": demo_name,
                }
                row.update(_metrics(data[demo_name], np))
                rows.append(row)
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return output


_TRAJECTORY_PREFIXES = (
    "states/",
    "initial_state/",
    "initial_camera_extrinsics/",
    "ee_pose/",
    "robot_root_pose/",
    "subtask/",
)


def _is_trajectory_dataset(path: str) -> bool:
    """Keep compact control/state channels and exclude image-sized observations."""
    return path == "actions" or path.startswith(_TRAJECTORY_PREFIXES)


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item"):
        return value.item()
    return value


def _data_provenance(data_group: Any) -> dict[str, Any]:
    keys = ("isaaclab_version", "isaacsim_version", "recorded_at", "env_args")
    return {key: _json_value(data_group.attrs[key]) for key in keys if key in data_group.attrs}


def _demo_attributes(group: Any) -> dict[str, Any]:
    keys = ("seed", "success", "num_samples")
    return {key: _json_value(group.attrs[key]) for key in keys if key in group.attrs}


def _discover_episodes(root: Path, h5py: Any) -> tuple[dict[str, tuple[Path, str]], dict[str, Any]]:
    episodes: dict[str, tuple[Path, str]] = {}
    files: dict[str, Any] = {}
    for hdf5_path in sorted(root.glob("*/run_*.hdf5")):
        match = re.search(r"run_(\d+)\.hdf5$", hdf5_path.name)
        if match is None:
            continue
        run_index = int(match.group(1))
        with h5py.File(hdf5_path, "r") as handle:
            data = handle.get("data")
            if data is None:
                continue
            files[str(hdf5_path)] = _data_provenance(data)
            for demo_name in sorted(data):
                if not demo_name.startswith("demo_"):
                    continue
                env_id = int(demo_name.split("_")[-1])
                key = f"{hdf5_path.parent.name}/run_{run_index}/env_{env_id}"
                if key in episodes:
                    raise ValueError(f"Duplicate trajectory episode key {key!r} below {root}")
                episodes[key] = (hdf5_path, demo_name)
    return episodes, files


def _trajectory_arrays(group: Any, h5py: Any, np: Any) -> dict[str, Any]:
    arrays: dict[str, Any] = {}

    def visit(name: str, item: Any) -> None:
        if not isinstance(item, h5py.Dataset) or not _is_trajectory_dataset(name):
            return
        if np.issubdtype(item.dtype, np.number) or np.issubdtype(item.dtype, np.bool_):
            arrays[name] = item[()]

    group.visititems(visit)
    return arrays


def _first_exceed_index(mask: Any, np: Any) -> int | None:
    values = np.asarray(mask)
    if values.ndim == 0:
        return 0 if bool(values) else None
    per_step = values.reshape(values.shape[0], -1).any(axis=1)
    indices = np.flatnonzero(per_step)
    return int(indices[0]) if len(indices) else None


def _shape_prefix(reference: Any, candidate: Any) -> tuple[Any, Any, bool, bool, int]:
    reference = reference if reference.ndim else reference.reshape(1)
    candidate = candidate if candidate.ndim else candidate.reshape(1)
    trailing_shape_match = reference.shape[1:] == candidate.shape[1:]
    length_match = reference.shape[0] == candidate.shape[0]
    return (
        reference,
        candidate,
        trailing_shape_match,
        length_match,
        min(len(reference), len(candidate)),
    )


def _numeric_comparison(
    path: str,
    reference: Any,
    candidate: Any,
    *,
    tolerance: float,
    kind: str,
    np: Any,
) -> dict[str, Any]:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    raw_reference_shape = list(reference.shape)
    raw_candidate_shape = list(candidate.shape)
    reference, candidate, trailing_match, length_match, length = _shape_prefix(
        reference, candidate
    )
    result: dict[str, Any] = {
        "path": path,
        "kind": kind,
        "reference_shape": raw_reference_shape,
        "candidate_shape": raw_candidate_shape,
        "shape_match": trailing_match and length_match,
        "length_match": length_match,
        "tolerance": tolerance,
    }
    if not trailing_match:
        result.update(
            {
                "exact": False,
                "within_tolerance": False,
                "first_exceed_step": 0,
                "error": "Trailing shapes differ",
            }
        )
        return result

    reference = reference[:length]
    candidate = candidate[:length]
    floating = np.issubdtype(reference.dtype, np.floating) or np.issubdtype(
        candidate.dtype, np.floating
    )
    if floating:
        exact = length_match and np.array_equal(reference, candidate, equal_nan=True)
    else:
        exact = length_match and np.array_equal(reference, candidate)
    if length == 0:
        result.update(
            {
                "exact": exact,
                "within_tolerance": length_match,
                "first_exceed_step": None if length_match else 0,
                "max_abs": 0.0,
                "rmse": 0.0,
                "severity": 0.0,
            }
        )
        return result

    ref64 = reference.astype(np.float64)
    cand64 = candidate.astype(np.float64)
    equal_nonfinite = (ref64 == cand64) | (np.isnan(ref64) & np.isnan(cand64))
    difference = np.where(equal_nonfinite, 0.0, np.abs(cand64 - ref64))
    finite_difference = np.where(np.isfinite(difference), difference, np.inf)
    exceed = finite_difference > tolerance
    first_exceed = _first_exceed_index(exceed, np)
    if not length_match and first_exceed is None:
        first_exceed = length
    max_abs = float(np.max(finite_difference))
    rmse = float(np.sqrt(np.mean(np.square(finite_difference))))
    severity = max_abs / tolerance if tolerance > 0 else float(max_abs > 0)
    result.update(
        {
            "exact": exact,
            "within_tolerance": length_match and not bool(np.any(exceed)),
            "first_exceed_step": first_exceed,
            "max_abs": max_abs,
            "rmse": rmse,
            "severity": severity,
        }
    )
    return result


def _quaternion_comparison(
    path: str,
    reference: Any,
    candidate: Any,
    *,
    tolerance_deg: float,
    np: Any,
) -> dict[str, Any]:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    raw_reference_shape = list(reference.shape)
    raw_candidate_shape = list(candidate.shape)
    reference, candidate, trailing_match, length_match, length = _shape_prefix(
        reference, candidate
    )
    quaternion_shape = trailing_match and reference.shape[-1] == 4
    result: dict[str, Any] = {
        "path": path,
        "kind": "orientation",
        "reference_shape": raw_reference_shape,
        "candidate_shape": raw_candidate_shape,
        "shape_match": quaternion_shape and length_match,
        "length_match": length_match,
        "tolerance_deg": tolerance_deg,
    }
    if not quaternion_shape:
        result.update(
            {
                "exact": False,
                "within_tolerance": False,
                "first_exceed_step": 0,
                "error": "Quaternion trailing shapes differ or do not end in four values",
            }
        )
        return result

    reference = reference[:length]
    candidate = candidate[:length]
    exact = length_match and np.array_equal(reference, candidate, equal_nan=True)
    if length == 0:
        result.update(
            {
                "exact": exact,
                "within_tolerance": length_match,
                "first_exceed_step": None if length_match else 0,
                "max_angle_deg": 0.0,
                "rms_angle_deg": 0.0,
                "severity": 0.0,
            }
        )
        return result

    ref64 = reference.astype(np.float64)
    cand64 = candidate.astype(np.float64)
    dot = np.sum(ref64 * cand64, axis=-1)
    norms = np.linalg.norm(ref64, axis=-1) * np.linalg.norm(cand64, axis=-1)
    normalized_dot = np.divide(
        np.abs(dot),
        norms,
        out=np.zeros_like(dot, dtype=np.float64),
        where=norms > 0,
    )
    angle_deg = np.degrees(2.0 * np.arccos(np.clip(normalized_dot, 0.0, 1.0)))
    exceed = angle_deg > tolerance_deg
    first_exceed = _first_exceed_index(exceed, np)
    if not length_match and first_exceed is None:
        first_exceed = length
    max_angle = float(np.max(angle_deg))
    rms_angle = float(np.sqrt(np.mean(np.square(angle_deg))))
    severity = max_angle / tolerance_deg if tolerance_deg > 0 else float(max_angle > 0)
    result.update(
        {
            # Raw representation equality remains useful, while angular comparison
            # correctly treats q and -q as the same physical orientation.
            "exact": exact,
            "within_tolerance": length_match and not bool(np.any(exceed)),
            "first_exceed_step": first_exceed,
            "max_angle_deg": max_angle,
            "rms_angle_deg": rms_angle,
            "severity": severity,
        }
    )
    return result


def _kind_and_tolerance(path: str, tolerances: dict[str, float]) -> tuple[str, float]:
    lowered = path.lower()
    if path == "actions":
        return "action", tolerances["action_atol"]
    if "joint_position" in lowered or "joint_pos" in lowered:
        return "joint_position", tolerances["joint_atol_rad"]
    if "velocity" in lowered or lowered.endswith("vel"):
        return "velocity", tolerances["velocity_atol"]
    if "position" in lowered:
        return "position", tolerances["position_atol_m"]
    return "value", tolerances["value_atol"]


def _compare_channel(
    path: str,
    reference: Any,
    candidate: Any,
    *,
    tolerances: dict[str, float],
    np: Any,
) -> list[dict[str, Any]]:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    lowered = path.lower()
    if (
        reference.shape
        and candidate.shape
        and reference.shape[-1] == candidate.shape[-1] == 7
        and (lowered.endswith("root_pose") or lowered.endswith("pose"))
    ):
        return [
            _numeric_comparison(
                f"{path}/position",
                reference[..., :3],
                candidate[..., :3],
                tolerance=tolerances["position_atol_m"],
                kind="position",
                np=np,
            ),
            _quaternion_comparison(
                f"{path}/orientation",
                reference[..., 3:7],
                candidate[..., 3:7],
                tolerance_deg=tolerances["rotation_atol_deg"],
                np=np,
            ),
        ]
    if (
        reference.shape
        and candidate.shape
        and reference.shape[-1] == candidate.shape[-1] == 4
        and any(token in lowered for token in ("orientation", "quat", "rotation"))
    ):
        return [
            _quaternion_comparison(
                path,
                reference,
                candidate,
                tolerance_deg=tolerances["rotation_atol_deg"],
                np=np,
            )
        ]
    kind, tolerance = _kind_and_tolerance(path, tolerances)
    if not (
        np.issubdtype(reference.dtype, np.floating)
        or np.issubdtype(candidate.dtype, np.floating)
    ):
        kind, tolerance = "discrete", 0.0
    return [
        _numeric_comparison(
            path,
            reference,
            candidate,
            tolerance=tolerance,
            kind=kind,
            np=np,
        )
    ]


def _all_within(channels: list[dict[str, Any]], prefixes: tuple[str, ...]) -> bool | None:
    selected = [channel for channel in channels if channel["path"].startswith(prefixes)]
    if not selected:
        return None
    return all(bool(channel["within_tolerance"]) for channel in selected)


def _compare_episode(
    key: str,
    reference_location: tuple[Path, str],
    candidate_location: tuple[Path, str],
    *,
    tolerances: dict[str, float],
    h5py: Any,
    np: Any,
) -> dict[str, Any]:
    reference_path, reference_demo = reference_location
    candidate_path, candidate_demo = candidate_location
    with h5py.File(reference_path, "r") as reference_handle, h5py.File(
        candidate_path, "r"
    ) as candidate_handle:
        reference_group = reference_handle[f"data/{reference_demo}"]
        candidate_group = candidate_handle[f"data/{candidate_demo}"]
        reference_arrays = _trajectory_arrays(reference_group, h5py, np)
        candidate_arrays = _trajectory_arrays(candidate_group, h5py, np)
        reference_attrs = _demo_attributes(reference_group)
        candidate_attrs = _demo_attributes(candidate_group)
        reference_hash = _initial_state_hash(reference_group)
        candidate_hash = _initial_state_hash(candidate_group)

    reference_channels = set(reference_arrays)
    candidate_channels = set(candidate_arrays)
    channels: list[dict[str, Any]] = []
    for path in sorted(reference_channels & candidate_channels):
        channels.extend(
            _compare_channel(
                path,
                reference_arrays[path],
                candidate_arrays[path],
                tolerances=tolerances,
                np=np,
            )
        )

    channels.sort(key=lambda channel: (-float(channel.get("severity", 0.0)), channel["path"]))
    first_exceed_steps = [
        int(channel["first_exceed_step"])
        for channel in channels
        if channel.get("first_exceed_step") is not None
    ]
    action_channels = [channel for channel in channels if channel["path"] == "actions"]
    missing_from_candidate = sorted(reference_channels - candidate_channels)
    missing_from_reference = sorted(candidate_channels - reference_channels)
    structure_matches = not missing_from_candidate and not missing_from_reference and all(
        bool(channel["shape_match"]) for channel in channels
    )
    within_tolerance = structure_matches and all(
        bool(channel["within_tolerance"]) for channel in channels
    )
    return {
        "episode": key,
        "reference_hdf5": str(reference_path),
        "candidate_hdf5": str(candidate_path),
        "reference_demo": reference_demo,
        "candidate_demo": candidate_demo,
        "reference_attributes": reference_attrs,
        "candidate_attributes": candidate_attrs,
        "outcome_matches": reference_attrs.get("success") == candidate_attrs.get("success"),
        "initial_state_sha256": {
            "reference": reference_hash,
            "candidate": candidate_hash,
            "exact": reference_hash is not None and reference_hash == candidate_hash,
        },
        "missing_from_candidate": missing_from_candidate,
        "missing_from_reference": missing_from_reference,
        "structure_matches": structure_matches,
        "actions_exact": bool(action_channels) and all(
            bool(channel["exact"]) for channel in action_channels
        ),
        "actions_within_tolerance": bool(action_channels) and all(
            bool(channel["within_tolerance"]) for channel in action_channels
        ),
        "initial_state_within_tolerance": _all_within(channels, ("initial_state/",)),
        "physics_state_within_tolerance": _all_within(
            channels, ("states/", "ee_pose/", "robot_root_pose/")
        ),
        "within_tolerance": within_tolerance,
        "first_exceed_step": min(first_exceed_steps) if first_exceed_steps else None,
        "channels": channels,
    }


def compare_trajectory_runs(
    reference_dir: str | Path,
    candidate_dir: str | Path,
    *,
    position_atol_m: float = 1.0e-3,
    joint_atol_rad: float = 1.0e-3,
    rotation_atol_deg: float = 0.25,
    velocity_atol: float = 1.0e-2,
    action_atol: float = 1.0e-5,
    value_atol: float = 1.0e-4,
) -> dict[str, Any]:
    """Compare matching RoboLab HDF5 episodes from two complete run folders.

    Episodes are paired by task directory, run index, and environment id. The
    report deliberately separates reset-state, policy-action, and physics-state
    agreement so renderer-induced policy drift is not mistaken for PhysX drift.
    Quaternion comparisons are sign-invariant.
    """
    h5py, np = _dependencies()
    reference_root = Path(reference_dir).expanduser().resolve()
    candidate_root = Path(candidate_dir).expanduser().resolve()
    if not reference_root.is_dir():
        raise ValueError(f"Reference run directory does not exist: {reference_root}")
    if not candidate_root.is_dir():
        raise ValueError(f"Candidate run directory does not exist: {candidate_root}")

    tolerances = {
        "position_atol_m": position_atol_m,
        "joint_atol_rad": joint_atol_rad,
        "rotation_atol_deg": rotation_atol_deg,
        "velocity_atol": velocity_atol,
        "action_atol": action_atol,
        "value_atol": value_atol,
    }
    if any(value < 0 for value in tolerances.values()):
        raise ValueError("Trajectory tolerances must be non-negative")

    reference_episodes, reference_files = _discover_episodes(reference_root, h5py)
    candidate_episodes, candidate_files = _discover_episodes(candidate_root, h5py)
    if not reference_episodes:
        raise ValueError(f"No */run_*.hdf5 episodes found below {reference_root}")
    if not candidate_episodes:
        raise ValueError(f"No */run_*.hdf5 episodes found below {candidate_root}")

    reference_keys = set(reference_episodes)
    candidate_keys = set(candidate_episodes)
    matched_keys = sorted(reference_keys & candidate_keys)
    episodes = [
        _compare_episode(
            key,
            reference_episodes[key],
            candidate_episodes[key],
            tolerances=tolerances,
            h5py=h5py,
            np=np,
        )
        for key in matched_keys
    ]
    missing_from_candidate = sorted(reference_keys - candidate_keys)
    missing_from_reference = sorted(candidate_keys - reference_keys)
    passed = (
        not missing_from_candidate
        and not missing_from_reference
        and bool(episodes)
        and all(bool(episode["within_tolerance"]) for episode in episodes)
        and all(bool(episode["outcome_matches"]) for episode in episodes)
    )
    summary = {
        "reference_episode_count": len(reference_episodes),
        "candidate_episode_count": len(candidate_episodes),
        "matched_episode_count": len(episodes),
        "missing_from_candidate_count": len(missing_from_candidate),
        "missing_from_reference_count": len(missing_from_reference),
        "initial_state_exact_count": sum(
            bool(episode["initial_state_sha256"]["exact"]) for episode in episodes
        ),
        "initial_state_within_tolerance_count": sum(
            episode["initial_state_within_tolerance"] is True for episode in episodes
        ),
        "actions_exact_count": sum(bool(episode["actions_exact"]) for episode in episodes),
        "actions_within_tolerance_count": sum(
            bool(episode["actions_within_tolerance"]) for episode in episodes
        ),
        "physics_state_within_tolerance_count": sum(
            episode["physics_state_within_tolerance"] is True for episode in episodes
        ),
        "outcome_match_count": sum(bool(episode["outcome_matches"]) for episode in episodes),
        "episode_within_tolerance_count": sum(
            bool(episode["within_tolerance"]) for episode in episodes
        ),
        "passed": passed,
    }
    return {
        "reference_root": str(reference_root),
        "candidate_root": str(candidate_root),
        "tolerances": tolerances,
        "reference_files": reference_files,
        "candidate_files": candidate_files,
        "missing_from_candidate": missing_from_candidate,
        "missing_from_reference": missing_from_reference,
        "summary": summary,
        "episodes": episodes,
    }


def write_trajectory_comparison(report: dict[str, Any], path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output
