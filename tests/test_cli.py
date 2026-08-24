from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

from manda_cli.cli import _login_welcome, _parser, main


class CliTests(unittest.TestCase):
    def test_program_name_and_command_groups(self) -> None:
        parser = _parser()
        self.assertEqual(parser.prog, "manda")
        help_text = parser.format_help()
        self.assertIn("auth", help_text)
        self.assertIn("teams", help_text)
        self.assertIn("studies", help_text)
        self.assertIn("quota", help_text)
        self.assertIn("artifacts", help_text)
        self.assertIn("extract-trajectories", help_text)
        self.assertIn("execute", help_text)
        self.assertNotIn("dashboard", help_text)

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

    def test_login_welcome_summarizes_the_workspace(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _login_welcome(
                {
                    "authenticated": True,
                    "user": {"email": "user@example.com"},
                    "producer_id": "producer-a",
                    "teams": [],
                    "studies": [],
                    "quota": {
                        "available_bytes": 5 * 1024**3,
                        "quota_bytes": 5 * 1024**3,
                    },
                },
                color=False,
            )
        value = output.getvalue()
        self.assertIn("MANDA", value)
        self.assertIn("CONNECTED", value)
        self.assertIn("Welcome, user@example.com", value)
        self.assertIn("5 GiB available of 5 GiB", value)
        self.assertIn("0 studies · 0 teams", value)
        self.assertIn("manda teams create", value)
        self.assertNotIn('"authenticated"', value)

    def test_auth_login_supports_machine_readable_output(self) -> None:
        args = _parser().parse_args(["auth", "login", "--json"])
        self.assertTrue(args.json)

    def test_push_validates_local_run_before_storage_access(self) -> None:
        stderr = io.StringIO()
        with (
            patch("manda_cli.cli.MandaAccessClient.from_environment") as access,
            contextlib.redirect_stderr(stderr),
        ):
            result = main(["artifacts", "push", "--run-dir", "/tmp/run"])
        self.assertEqual(result, 2)
        self.assertIn("Run directory does not exist", stderr.getvalue())
        access.assert_not_called()

    def test_team_invite_and_study_grant_arguments(self) -> None:
        invite = _parser().parse_args(
            ["teams", "invite", "robotics", "user@example.com", "--role", "admin"]
        )
        self.assertEqual(invite.team_id, "robotics")
        self.assertEqual(invite.email, "user@example.com")
        self.assertEqual(invite.role, "admin")
        grant = _parser().parse_args(
            ["studies", "grant", "pilot", "--team", "robotics", "--role", "contributor"]
        )
        self.assertEqual(grant.team_id, "robotics")
        self.assertEqual(grant.role, "contributor")

    def test_artifact_study_id_is_optional_for_auto_discovery(self) -> None:
        args = _parser().parse_args(["artifacts", "doctor"])
        self.assertIsNone(args.study_id)


if __name__ == "__main__":
    unittest.main()
