from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

from manda_cli.cli import _parser, main


class CliTests(unittest.TestCase):
    def test_program_name_and_command_groups(self) -> None:
        parser = _parser()
        self.assertEqual(parser.prog, "manda")
        help_text = parser.format_help()
        self.assertIn("artifacts", help_text)
        self.assertIn("extract-trajectories", help_text)
        self.assertIn("execute", help_text)
        self.assertIn("dashboard", help_text)

    def test_artifact_push_has_no_repository_relative_config_default(self) -> None:
        args = _parser().parse_args(
            ["artifacts", "push", "--run-dir", "/tmp/run", "--study-id", "study"]
        )
        self.assertIsNone(args.config)
        self.assertEqual(args.study_id, "study")

    def test_version(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as raised:
            main(["--version"])
        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(output.getvalue().strip(), "manda 0.1.0")

    def test_push_requires_study_id_or_config_before_storage_access(self) -> None:
        stderr = io.StringIO()
        with (
            patch("manda_cli.cli.ArtifactManager.from_environment") as manager,
            contextlib.redirect_stderr(stderr),
        ):
            result = main(["artifacts", "push", "--run-dir", "/tmp/run"])
        self.assertEqual(result, 2)
        self.assertIn("Pass --study-id or --config", stderr.getvalue())
        manager.assert_not_called()


if __name__ == "__main__":
    unittest.main()
