"""Hosted evaluations: submit a benchmark job, follow it, and stop it.

A job is the same ``manda.hosted-eval.v1`` specification the dashboard submits. The
service holds the budget from your credit, rents a GPU node, and streams verified
episodes into the chosen study. Every rule here is checked again by the service.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .auth import AuthManager

SPEC_SCHEMA = "manda.hosted-eval.v1"
# The node image runs this RoboLab commit; its catalog names the tasks a job may ask for.
ROBOLAB_COMMIT = "3680bc7b8d8bca7507d796bf10e2d5128aeb9743"
TASK_CATALOG_URL = (
    "https://raw.githubusercontent.com/Manda-Robotics/RoboLab-Verified/"
    f"{ROBOLAB_COMMIT}/robolab/tasks/_metadata/task_metadata.json"
)
REFERENCE_VERSION = "droid-jointpos@aa64205"
RUNG_ENDPOINT = "wss://policy.mandarobotics.com"
RUNG_MODELS = ("gemini-4-argon", "claude-fable-5-2", "gpt-6.1-sol")
"""Models Manda's endpoint serves for Rung jobs, over each vendor's paid API."""
RUNG_MODEL = RUNG_MODELS[0]
GPU_NODES = ("RTX PRO 5000", "RTX PRO 4000")
POLICIES = ("pi05", "rung", "openpi")
TERMINAL = ("succeeded", "failed", "cancelled")

# Parallel environments per node, as measured; the dashboard plans from the same table.
_LIMITS = {
    ("RTX PRO 5000", "pi05"): 5,
    ("RTX PRO 5000", "openpi"): 8,
    ("RTX PRO 4000", "pi05"): 4,
    ("RTX PRO 4000", "openpi"): 8,
}
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/+=-]{0,119}")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._ -]{0,119}")
_URI = re.compile(r"wss?://[A-Za-z0-9.-]+(:[0-9]{1,5})?(/[A-Za-z0-9._~/-]*)?")
_TASK = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,127}")
JOB_COLUMNS = (
    "job_id,study_id,benchmark,spec,policy_label,expected_episodes,status,status_detail,"
    "max_cost_micros,cost_micros,charged_micros,run_key,gpu_name,created_at,finished_at"
)


class EvalError(ValueError):
    """A job the service would refuse, or a failed request to it."""


def parallel_plan(gpu: str, policy: str, episodes: int) -> dict[str, Any]:
    """The largest batch that divides the episodes within what the node is known to handle."""
    limit = 1 if policy == "rung" else _LIMITS.get((gpu, policy), 1)
    envs = next(n for n in range(min(limit, episodes), 0, -1) if episodes % n == 0)
    return {"num_envs": envs, "batches": episodes // envs, "max_envs": limit}


def easiest(catalog: list[dict[str, Any]], count: int) -> list[str]:
    ordered = sorted(
        catalog, key=lambda t: (t.get("difficulty_score", math.inf), t["task_name"])
    )
    return [task["task_name"] for task in ordered[:count]]


def build_spec(
    *,
    tasks: list[str],
    episodes: int,
    policy: str,
    gpu: str = "RTX PRO 5000",
    seed: int = 0,
    friction: str = "upstream",
    sim_rate_hz: float | None = None,
    name: str | None = None,
    uri: str | None = None,
    version: str | None = None,
    checkpoint: str | None = None,
    training_config: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    if not tasks or len(tasks) > 200 or len(set(tasks)) != len(tasks):
        raise EvalError("choose 1 to 200 distinct tasks")
    if bad := [task for task in tasks if not _TASK.fullmatch(task)]:
        raise EvalError(f"invalid task names: {', '.join(bad)}")
    if not 1 <= episodes <= 200:
        raise EvalError("episodes per task must be 1 to 200")
    if gpu not in GPU_NODES:
        raise EvalError(f"GPU node must be one of: {', '.join(GPU_NODES)}")
    if not 0 <= seed <= 2147483647:
        raise EvalError("seed is out of range")
    if not re.fullmatch(r"upstream|realistic|[0-9]{1,3}(\.[0-9]{1,6})?", friction):
        raise EvalError("friction must be upstream, realistic or a coefficient")
    if sim_rate_hz is not None and not 1 <= sim_rate_hz <= 10000:
        raise EvalError("simulation frequency must be 1 to 10000 Hz")
    if model is not None and policy != "rung":
        raise EvalError("--model applies to --policy rung only")
    if policy == "pi05":
        policy_spec: dict[str, Any] = {
            "kind": "reference", "name": "pi05", "version": REFERENCE_VERSION
        }
    elif policy == "rung":
        model = model or RUNG_MODEL
        if model not in RUNG_MODELS:
            raise EvalError(f"Rung serves these models: {', '.join(RUNG_MODELS)}")
        # The version names the model; the service issues the job a key for that model.
        policy_spec = {
            "kind": "endpoint", "protocol": "rung", "name": "rung",
            "version": model, "uri": RUNG_ENDPOINT,
        }
    elif policy == "openpi":
        if not name or not _NAME.fullmatch(name):
            raise EvalError("an OpenPI endpoint needs --name, shown in results")
        if not uri or not _URI.fullmatch(uri):
            raise EvalError("--endpoint must be a ws:// or wss:// URL without credentials")
        policy_spec = {"kind": "endpoint", "protocol": "openpi", "name": name, "uri": uri}
        for field, value in (
            ("version", version), ("checkpoint", checkpoint), ("training_config", training_config)
        ):
            if value is None:
                continue
            if not _LABEL.fullmatch(value):
                raise EvalError(f"invalid {field.replace('_', ' ')}: {value!r}")
            policy_spec[field] = value
    else:
        raise EvalError(f"policy must be one of: {', '.join(POLICIES)}")
    plan = parallel_plan(gpu, policy, episodes)
    return {
        "schema": SPEC_SCHEMA,
        "benchmark": "robolab",
        "tasks": list(tasks),
        "episodes_per_task": episodes,
        "num_envs": plan["num_envs"],
        "policy": policy_spec,
        "simulation": {
            "physics": "newton_mjwarp",
            "renderer": "ovrtx",
            "seed": seed,
            "sim_rate_hz": sim_rate_hz,
            "friction": friction,
        },
        "node": {"gpu": gpu},
    }


def usd(micros: int | None) -> str:
    return f"${(micros or 0) / 1_000_000:.2f}"


class EvalsClient:
    """The signed-in user's view of hosted jobs. Row-level security limits it to their own."""

    def __init__(self, auth: AuthManager):
        self.auth = auth

    @classmethod
    def from_environment(cls) -> EvalsClient:
        return cls(AuthManager.from_environment())

    def _request(self, method: str, path: str, *, query=None, payload=None, single=False):
        session = self.auth.ensure_session()
        settings = self.auth.settings
        suffix = f"?{urlencode(query)}" if query else ""
        headers = {
            "apikey": settings.publishable_key,
            "Authorization": f"Bearer {session.access_token}",
            "Accept": "application/vnd.pgrst.object+json" if single else "application/json",
            "User-Agent": "manda-cli/0.1",
        }
        data = None
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{settings.supabase_url}/rest/v1/{path}{suffix}", data=data, headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=45) as response:  # noqa: S310 - validated URL
                raw = response.read()
        except HTTPError as exc:
            try:
                message = json.loads(exc.read()).get("message")
            except (ValueError, AttributeError):
                message = None
            raise EvalError(message or f"service answered HTTP {exc.code}") from None
        return json.loads(raw) if raw else None

    def credit(self) -> dict[str, Any]:
        rows = self._request(
            "GET", "credit_accounts", query={"select": "balance_micros,held_micros"}
        ) or []
        account = rows[0] if rows else {"balance_micros": 0, "held_micros": 0}
        available = max(0, account["balance_micros"] - account["held_micros"])
        return {**account, "available_micros": available, "available": usd(available)}

    def jobs(self, *, active: bool = False, limit: int = 50) -> list[dict[str, Any]]:
        query = {"select": JOB_COLUMNS, "order": "created_at.desc", "limit": str(limit)}
        if active:
            query["status"] = f"not.in.({','.join(TERMINAL)})"
        return self._request("GET", "eval_jobs", query=query) or []

    def job(self, job_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"[0-9a-f-]{36}", job_id):
            raise EvalError("a job ID is a UUID")
        job = self._request(
            "GET", "eval_jobs", query={"select": JOB_COLUMNS, "job_id": f"eq.{job_id}"},
            single=True,
        )
        if job.get("run_key"):
            rows = self._request(
                "GET", "runs",
                query={"select": "run_id,episode_count,expected_episodes,index_status",
                       "run_key": f"eq.{job['run_key']}"},
            ) or []
            job["run"] = rows[0] if rows else None
        return job

    def submit(self, study_id: str, spec: dict[str, Any], budget_usd: float) -> str:
        if budget_usd < 1:
            raise EvalError("the budget must be at least $1.00")
        return self._request(
            "POST", "rpc/manda_submit_eval_job",
            payload={"target_study_id": study_id, "job_spec": spec,
                     "max_cost_micros": round(budget_usd * 1_000_000)},
        )

    def cancel(self, job_id: str) -> str:
        return self._request(
            "POST", "rpc/manda_cancel_eval_job", payload={"target_job_id": job_id}
        )


def load_catalog(url: str = TASK_CATALOG_URL) -> list[dict[str, Any]]:
    with urlopen(Request(url, headers={"User-Agent": "manda-cli/0.1"}), timeout=30) as response:  # noqa: S310
        return json.loads(response.read())


def summarize(job: dict[str, Any]) -> dict[str, Any]:
    """The fields worth reading, with costs in dollars."""
    run = job.get("run") or {}
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "detail": job.get("status_detail"),
        "study_id": job["study_id"],
        "policy": job["policy_label"],
        "tasks": len(job["spec"].get("tasks") or []),
        "episodes": f"{run.get('episode_count', 0)}/{job['expected_episodes']}",
        "gpu": job.get("gpu_name") or job["spec"].get("node", {}).get("gpu"),
        "cost": usd(job["charged_micros"] if job.get("charged_micros") is not None
                    else job["cost_micros"]),
        "budget": usd(job["max_cost_micros"]),
        "run_id": run.get("run_id"),
        "created_at": job["created_at"],
    }
