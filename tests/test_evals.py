"""Hosted evaluation jobs: the specification, batch planning and the dry-run plan."""

import json

import pytest

from manda_cli import cli
from manda_cli.evals import EvalError, build_spec, easiest, parallel_plan


def test_a_reference_job_matches_what_the_dashboard_submits():
    spec = build_spec(tasks=["BananaInBowlTask"], episodes=10, policy="pi05")
    assert spec["num_envs"] == 5
    assert spec["policy"] == {
        "kind": "reference", "name": "pi05", "version": "droid-jointpos@aa64205"
    }
    assert spec["simulation"] == {"physics": "newton_mjwarp", "renderer": "ovrtx", "seed": 0,
                                  "sim_rate_hz": None, "friction": "upstream"}


def test_rung_runs_one_environment_at_a_time_against_mandas_endpoint():
    spec = build_spec(tasks=["BananaInBowlTask"], episodes=10, policy="rung")
    assert spec["num_envs"] == 1
    assert spec["policy"]["protocol"] == "rung"
    assert spec["policy"]["uri"] == "wss://policy.mandarobotics.com"


@pytest.mark.parametrize(("episodes", "envs"), [(10, 5), (12, 4), (7, 1), (1, 1), (200, 5)])
def test_batches_divide_the_episodes(episodes, envs):
    assert parallel_plan("RTX PRO 5000", "pi05", episodes)["num_envs"] == envs


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"tasks": []}, "1 to 200"),
        ({"tasks": ["A", "A"]}, "1 to 200"),
        ({"tasks": ["bad; rm"]}, "invalid task"),
        ({"episodes": 0}, "1 to 200"),
        ({"gpu": "A100"}, "GPU"),
        ({"friction": "$(id)"}, "friction"),
        ({"policy": "openpi"}, "--name"),
        ({"policy": "openpi", "name": "m", "uri": "https://x"}, "ws://"),
        ({"policy": "openpi", "name": "m", "uri": "wss://x", "version": "a b"}, "version"),
    ],
)
def test_invalid_jobs_are_explained_before_anything_is_held(change, message):
    with pytest.raises(EvalError, match=message):
        build_spec(**{"tasks": ["BananaInBowlTask"], "episodes": 5, "policy": "pi05", **change})


def test_the_easiest_tasks_follow_the_catalog_difficulty():
    catalog = [{"task_name": "Hard", "difficulty_score": 9},
               {"task_name": "B", "difficulty_score": 1},
               {"task_name": "A", "difficulty_score": 1}, {"task_name": "Unscored"}]
    assert easiest(catalog, 3) == ["A", "B", "Hard"]


def test_submit_without_yes_only_plans(monkeypatch, capsys):
    class NoNetwork:
        def submit(self, *args):
            raise AssertionError("must not submit without --yes")

    monkeypatch.setattr(cli.EvalsClient, "from_environment", classmethod(lambda cls: NoNetwork()))
    code = cli.main(["evals", "submit", "--study", "s", "--task", "BananaInBowlTask",
                     "--episodes", "10", "--budget", "3"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0 and report["submitted"] is False
    assert report["episodes"] == 10 and report["batches_per_task"] == "2 of 5"
