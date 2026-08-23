from __future__ import annotations

import unittest
from pathlib import Path

from manda_cli.config import ConfigError, load_config
from manda_cli.planning import build_jobs

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples" / "experiment.toml"
ISAAC60_CONFIG = ROOT / "examples" / "isaac60_port_regression.toml"


class ConfigTests(unittest.TestCase):
    def test_repository_config_is_valid(self) -> None:
        config = load_config(CONFIG)
        self.assertEqual(config.study.expected_task_count, 120)
        self.assertEqual([policy.name for policy in config.enabled_policies()], ["pi05"])
        self.assertEqual(config.stages["core"].expected_episodes, 1200)
        server_command = config.policies["pi05"].server_command
        self.assertLess(
            server_command.index("--port=8000"),
            server_command.index("policy:checkpoint"),
        )

    def test_smoke_plan_has_expected_contract(self) -> None:
        config = load_config(CONFIG)
        jobs = build_jobs(config, "smoke")
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertEqual(job.policy, "pi05")
        self.assertEqual(job.expected_episodes, 10)
        self.assertIn("--video-mode", job.command)
        self.assertIn("BananaInBowlTask", job.command)
        self.assertNotIn("--kit_args=--/UJITSO/enabled=false", job.command)
        self.assertEqual(job.output_folder, f"{config.study.id}_smoke_pi05")

    def test_isaac60_port_regression_matches_collected_reference(self) -> None:
        candidate = load_config(ISAAC60_CONFIG)
        self.assertEqual(candidate.study.isaac_stack, "isaac60")
        self.assertEqual(
            candidate.stages["smoke"].tasks,
            (
                "RubiksCubeThenBananaTask",
                "RubiksCubeAndBananaTask",
                "UnstackRubiksCubeTask",
                "SauceBottlesCrateTask",
                "ReorientWhiteMugsTask",
            ),
        )
        self.assertEqual(candidate.stages["smoke"].num_envs, 1)
        jobs = build_jobs(candidate, "smoke")
        self.assertIn("--kit_args=--/UJITSO/enabled=false", jobs[0].command)
        self.assertEqual(jobs[0].output_folder, "robolab-isaac60-port-regression_smoke_pi05")

    def test_pilot_can_be_sharded_without_overlap(self) -> None:
        config = load_config(CONFIG)
        jobs = build_jobs(config, "pilot", shards=3)
        flattened = [task for job in jobs for task in job.tasks]
        self.assertEqual(len(flattened), 12)
        self.assertEqual(len(set(flattened)), 12)

    def test_core_sharding_is_blocked_until_metadata_import(self) -> None:
        config = load_config(CONFIG)
        with self.assertRaisesRegex(ConfigError, "FEATURE-003"):
            build_jobs(config, "core", shards=2)

    def test_disabled_blocked_policy_cannot_be_planned(self) -> None:
        config = load_config(CONFIG)
        with self.assertRaisesRegex(ConfigError, "blocked by"):
            build_jobs(
                config,
                "smoke",
                policy_names=["dreamzero"],
                include_disabled=True,
            )


if __name__ == "__main__":
    unittest.main()
