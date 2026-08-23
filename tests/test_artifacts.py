from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from manda_cli.artifacts import (
    ArtifactError,
    ArtifactManager,
    create_manifest,
    verify_local_run,
)
from manda_cli.catalog import publish_catalog
from manda_cli.storage import StorageSettings


class FilesystemRcloneClient:
    """Minimal rclone-compatible test double backed by a temporary directory."""

    def __init__(self, root: Path):
        self.root = root

    def _path(self, value: str | Path) -> Path:
        raw = str(value)
        if ":" in raw and not raw.startswith("/"):
            _, object_path = raw.split(":", 1)
            return self.root / object_path.lstrip("/")
        return Path(raw)

    def configuration_report(self):
        return {"remote": "test:bucket", "producer_id": "node-a"}

    def version(self):
        return "rclone test-double"

    def list_files(self, remote: str, *, include: str | None = None):
        root = self._path(remote)
        if not root.exists():
            return []
        rows = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if include == "**/_COMPLETE" and not relative.endswith("/_COMPLETE"):
                continue
            rows.append((relative, path.stat().st_size))
        return sorted(rows)

    def cat(self, remote: str, *, missing_ok: bool = False):
        path = self._path(remote)
        if not path.is_file():
            if missing_ok:
                return None
            raise ArtifactError(f"missing: {path}")
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _argument(extra: tuple[str, ...], flag: str) -> str | None:
        if flag not in extra:
            return None
        return extra[extra.index(flag) + 1]

    def copy(self, source: str | Path, destination: str, *extra: str):
        src = self._path(source)
        dst = self._path(destination)
        dst.mkdir(parents=True, exist_ok=True)
        files_from = self._argument(extra, "--files-from-raw")
        excluded = self._argument(extra, "--exclude")
        if files_from:
            relatives = Path(files_from).read_text(encoding="utf-8").splitlines()
        else:
            relatives = [p.relative_to(src).as_posix() for p in src.rglob("*") if p.is_file()]
        for relative in relatives:
            if excluded and Path(relative).match(excluded):
                continue
            source_path = src / relative
            if not source_path.is_file():
                continue
            target = dst / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)

    def copyto(self, source: str | Path, destination: str):
        src = self._path(source)
        dst = self._path(destination)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    def check(self, source: str | Path, destination: str, *extra: str):
        src = self._path(source)
        dst = self._path(destination)
        excluded = self._argument(extra, "--exclude")
        files_from = self._argument(extra, "--files-from-raw")
        if files_from:
            relatives = Path(files_from).read_text(encoding="utf-8").splitlines()
        else:
            relatives = [p.relative_to(src).as_posix() for p in src.rglob("*") if p.is_file()]
        for relative in relatives:
            if excluded and Path(relative).match(excluded):
                continue
            path = src / relative
            target = dst / relative
            if not target.is_file() or target.read_bytes() != path.read_bytes():
                raise ArtifactError(f"check failed: {relative}")


def _write_run(root: Path, *, policy: str = "pi05") -> None:
    task = root / "TaskA"
    task.mkdir(parents=True)
    (task / "run_0.hdf5").write_bytes(b"trajectory")
    (task / "episode_0.mp4").write_bytes(b"video")
    (task / "log_0_env0.json").write_text('{"events": []}\n', encoding="utf-8")
    rows = [
        {
            "env_name": "TaskA",
            "episode": 0,
            "success": True,
            "score": 1.0,
            "policy": policy,
        },
        {
            "env_name": "TaskA",
            "episode": 1,
            "success": False,
            "score": 0.5,
            "policy": policy,
        },
    ]
    with (root / "episode_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class ArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.remote = self.root / "remote"
        self.settings = StorageSettings(
            remote="test:bucket",
            prefix="robolab/v1",
            producer_id="node-a",
        )
        self.client = FilesystemRcloneClient(self.remote)
        self.manager = ArtifactManager(self.settings, client=self.client)  # type: ignore[arg-type]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_manifest_roles_and_local_corruption_detection(self) -> None:
        run = self.root / "run"
        run.mkdir()
        _write_run(run)
        record = {"run_id": "run-1", "study_id": "study", "producer_id": "node-a"}
        (run / "run.json").write_text(json.dumps(record), encoding="utf-8")
        manifest = create_manifest(run, record)
        roles = {entry["path"]: entry["role"] for entry in manifest["files"]}
        self.assertEqual(roles["TaskA/run_0.hdf5"], "trajectory")
        self.assertEqual(roles["TaskA/episode_0.mp4"], "video")
        (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.assertTrue(verify_local_run(run)["valid"])
        (run / "TaskA" / "run_0.hdf5").write_bytes(b"corrupt")
        report = verify_local_run(run)
        self.assertFalse(report["valid"])
        self.assertTrue(any("mismatch" in error for error in report["errors"]))

    def test_finalize_list_pull_profiles_and_idempotence(self) -> None:
        run = self.root / "run"
        run.mkdir()
        _write_run(run)
        report = self.manager.finalize(
            run,
            study_id="study",
            run_id="run-1",
            stage="pilot",
            policy="pi05",
            expected_episodes=2,
            expected_tasks=["TaskA"],
        )
        self.assertTrue(report["valid"])
        self.assertTrue(report["committed"])
        self.assertFalse(report["idempotent"])
        repeated = self.manager.finalize(run, study_id="study", run_id="run-1")
        self.assertTrue(repeated["idempotent"])

        listed = self.manager.list_runs(study_id="study")
        self.assertEqual([item["run_id"] for item in listed], ["run-1"])

        metadata = self.root / "metadata"
        self.manager.pull("run-1", destination=metadata, profile="metadata")
        self.assertTrue((metadata / "episode_results.jsonl").is_file())
        self.assertTrue((metadata / "TaskA" / "log_0_env0.json").is_file())
        self.assertFalse((metadata / "TaskA" / "run_0.hdf5").exists())
        self.assertFalse((metadata / "TaskA" / "episode_0.mp4").exists())

        analysis = self.root / "analysis"
        self.manager.pull("run-1", destination=analysis, profile="analysis")
        self.assertTrue((analysis / "TaskA" / "run_0.hdf5").is_file())
        self.assertFalse((analysis / "TaskA" / "episode_0.mp4").exists())

        full = self.root / "full"
        self.manager.pull("run-1", destination=full, profile="full")
        self.assertTrue((full / "TaskA" / "episode_0.mp4").is_file())
        self.assertTrue(verify_local_run(full)["valid"])

    def test_incomplete_checkpoint_is_not_discoverable(self) -> None:
        run = self.root / "run"
        run.mkdir()
        _write_run(run)
        result = self.manager.checkpoint(run, run_id="partial-1", min_age_seconds=0)
        self.assertFalse(result["committed"])
        self.assertEqual(self.manager.list_runs(), [])
        restored = self.root / "restored"
        self.manager.resume(
            "partial-1", destination=restored, producer_id="node-a"
        )
        self.assertEqual(
            (restored / ".artifact-run-id").read_text(encoding="utf-8").strip(),
            "partial-1",
        )
        self.assertTrue((restored / "TaskA" / "run_0.hdf5").is_file())

    def test_remote_corruption_is_detected(self) -> None:
        run = self.root / "run"
        run.mkdir()
        _write_run(run)
        result = self.manager.finalize(run, study_id="study", run_id="run-1")
        remote_run = self.client._path(result["remote"])
        (remote_run / "TaskA" / "run_0.hdf5").write_bytes(b"short")
        report = self.manager.verify_remote(
            "run-1", study_id="study", producer_id="node-a"
        )
        self.assertFalse(report["valid"])

    def test_catalog_publication_without_parquet(self) -> None:
        run = self.root / "run"
        run.mkdir()
        _write_run(run)
        self.manager.finalize(run, study_id="study", run_id="run-1")
        report = publish_catalog(
            self.manager,
            output_dir=self.root / "catalog-output",
            study_id="study",
            version="catalog-1",
            parquet=False,
            upload=True,
        )
        self.assertEqual(report["run_count"], 1)
        self.assertEqual(report["episode_count"], 2)
        output = Path(report["output_dir"])
        self.assertTrue((output / "runs.json").is_file())
        self.assertTrue((output / "assets.jsonl").is_file())
        self.assertGreater(report["asset_count"], 0)
        self.assertTrue((output / "episodes.jsonl").is_file())
        latest = self.remote / "bucket" / "robolab" / "v1" / "catalog" / "latest.json"
        self.assertEqual(json.loads(latest.read_text())["catalog_version"], "catalog-1")

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "PyArrow is not installed")
    def test_catalog_publication_with_parquet(self) -> None:
        run = self.root / "run"
        run.mkdir()
        _write_run(run)
        self.manager.finalize(run, study_id="study", run_id="run-1")
        report = publish_catalog(
            self.manager,
            output_dir=self.root / "catalog-output",
            study_id="study",
            version="catalog-parquet",
            parquet=True,
            upload=False,
        )
        output = Path(report["output_dir"])
        self.assertTrue((output / "episodes.parquet").is_file())
        self.assertTrue((output / "tasks.parquet").is_file())
        self.assertTrue((output / "assets.parquet").is_file())


if __name__ == "__main__":
    unittest.main()
