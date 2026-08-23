from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .storage import StorageError, StorageSettings, safe_identifier

_B2_AUTHORIZE_URL = "https://api.backblazeb2.com/b2api/v4/b2_authorize_account"


@dataclass(frozen=True)
class _NativeSession:
    account_id: str
    api_url: str
    authorization_token: str
    capabilities: frozenset[str]


def desired_lifecycle_rules(prefix: str, producer_ids: list[str]) -> list[dict[str, Any]]:
    """Return B2-native rules so the working and archive prefixes may overlap safely."""
    root = prefix.strip("/")
    rules = []
    for producer in sorted(set(producer_ids)):
        safe_identifier(producer, "producer ID")
        rules.append(
            {
                "fileNamePrefix": f"{root}/producers/{producer}/working/",
                "daysFromUploadingToHiding": 14,
                "daysFromHidingToDeleting": 1,
                "daysFromStartingToCancelingUnfinishedLargeFiles": 7,
            }
        )
    rules.append(
        {
            "fileNamePrefix": f"{root}/",
            "daysFromUploadingToHiding": None,
            "daysFromHidingToDeleting": 30,
            "daysFromStartingToCancelingUnfinishedLargeFiles": 7,
        }
    )
    return rules


def configuration_plan(settings: StorageSettings, producer_ids: list[str]) -> dict[str, Any]:
    bucket = settings.remote.split(":", 1)[1].split("/", 1)[0]
    return {
        "bucket": bucket,
        "endpoint": settings.endpoint,
        "encryption": {
            "enabled": True,
            "algorithm": "AES256 (Backblaze-managed SSE-B2)",
            "note": "Applies to new uploads only; existing objects are unchanged.",
        },
        "lifecycle_rules": desired_lifecycle_rules(settings.prefix, producer_ids),
        "lifecycle_api": "Backblaze B2 Native API v4 (supports overlapping prefixes)",
        "warning": (
            "Lifecycle rules permanently delete expired working objects and noncurrent versions."
        ),
    }


def _client(settings: StorageSettings):
    try:
        import boto3
    except ImportError as exc:
        raise StorageError(
            "B2 provisioning requires the storage extra: uv sync --extra storage"
        ) from exc
    if not settings.endpoint or not settings.key_id or settings.application_key_file is None:
        raise StorageError(
            "B2 provisioning requires direct endpoint, key ID, and key file settings"
        )
    secret = Path(settings.application_key_file).read_text(encoding="utf-8").strip()
    return boto3.client(
        "s3",
        endpoint_url=settings.endpoint,
        region_name=settings.region,
        aws_access_key_id=settings.key_id,
        aws_secret_access_key=secret,
    )


def _bucket_name(settings: StorageSettings) -> str:
    return settings.remote.split(":", 1)[1].split("/", 1)[0]


def _json_response(request: Request, operation: str) -> dict[str, Any]:
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed HTTPS/API URLs
            value = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            error = json.loads(exc.read().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            error = {}
        code = error.get("code") or f"HTTP {exc.code}"
        message = error.get("message") or exc.reason
        raise StorageError(f"Backblaze {operation} failed: {code}: {message}") from exc
    except (URLError, TimeoutError) as exc:
        raise StorageError(f"Backblaze {operation} failed: {exc.reason}") from exc
    if not isinstance(value, dict):
        raise StorageError(f"Backblaze {operation} returned an invalid response")
    return value


def _native_session(settings: StorageSettings) -> _NativeSession:
    if not settings.key_id or settings.application_key_file is None:
        raise StorageError("B2 provisioning requires a key ID and key file")
    secret = settings.application_key_file.read_text(encoding="utf-8").strip()
    credentials = base64.b64encode(f"{settings.key_id}:{secret}".encode()).decode("ascii")
    request = Request(
        _B2_AUTHORIZE_URL,
        headers={"Authorization": f"Basic {credentials}"},
        method="GET",
    )
    response = _json_response(request, "account authorization")
    try:
        storage_api = response["apiInfo"]["storageApi"]
        allowed = storage_api["allowed"]
        return _NativeSession(
            account_id=str(response["accountId"]),
            api_url=str(storage_api["apiUrl"]),
            authorization_token=str(response["authorizationToken"]),
            capabilities=frozenset(str(item) for item in allowed["capabilities"]),
        )
    except (KeyError, TypeError) as exc:
        raise StorageError("Backblaze authorization response is missing storage API data") from exc


def _native_post(
    session: _NativeSession,
    operation: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    request = Request(
        f"{session.api_url}/b2api/v4/{operation}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": session.authorization_token,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    return _json_response(request, operation)


def _native_bucket(settings: StorageSettings, session: _NativeSession) -> dict[str, Any]:
    if "listBuckets" not in session.capabilities:
        raise StorageError("The B2 application key needs the listBuckets capability")
    bucket_name = _bucket_name(settings)
    payload: dict[str, Any] = {"accountId": session.account_id, "bucketName": bucket_name}
    if settings.bucket_id:
        payload = {"accountId": session.account_id, "bucketId": settings.bucket_id}
    response = _native_post(session, "b2_list_buckets", payload)
    buckets = response.get("buckets")
    if not isinstance(buckets, list) or len(buckets) != 1:
        raise StorageError(f"Backblaze did not return exactly one bucket for {bucket_name!r}")
    bucket = buckets[0]
    if not isinstance(bucket, dict):
        raise StorageError("Backblaze returned invalid bucket metadata")
    if bucket.get("bucketName") != bucket_name:
        raise StorageError("Configured Backblaze bucket ID and bucket name do not match")
    if settings.bucket_id and bucket.get("bucketId") != settings.bucket_id:
        raise StorageError("Backblaze returned a different bucket ID than configured")
    return bucket


def _managed_lifecycle_rule(rule: dict[str, Any], root: str) -> bool:
    prefix = str(rule.get("fileNamePrefix", ""))
    if prefix == f"{root}/":
        return True
    producer_root = f"{root}/producers/"
    return prefix.startswith(producer_root) and prefix.endswith("/working/")


def _without_metadata(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "ResponseMetadata"}


def inspect_bucket(settings: StorageSettings) -> dict[str, Any]:
    client = _client(settings)
    bucket = _bucket_name(settings)
    from botocore.exceptions import ClientError

    result: dict[str, Any] = {"bucket": bucket}
    try:
        encryption = client.get_bucket_encryption(Bucket=bucket)
        result["encryption"] = {"configured": True, "value": _without_metadata(encryption)}
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code == "ServerSideEncryptionConfigurationNotFoundError":
            result["encryption"] = {"configured": False, "value": None}
        else:
            raise StorageError(f"Cannot inspect B2 encryption: {code or exc}") from exc

    native_bucket = _native_bucket(settings, _native_session(settings))
    lifecycle_rules = native_bucket.get("lifecycleRules") or []
    result["lifecycle"] = {
        "configured": bool(lifecycle_rules),
        "value": {"Rules": lifecycle_rules, "api": "B2 Native v4"},
    }
    return result


def apply_bucket_configuration(
    settings: StorageSettings,
    producer_ids: list[str],
) -> dict[str, Any]:
    client = _client(settings)
    plan = configuration_plan(settings, producer_ids)
    bucket = str(plan["bucket"])

    session = _native_session(settings)
    if "writeBuckets" not in session.capabilities:
        raise StorageError("The B2 application key needs the writeBuckets capability")
    native_bucket = _native_bucket(settings, session)
    existing_rules = native_bucket.get("lifecycleRules") or []
    root = settings.prefix.strip("/")
    preserved = [
        rule
        for rule in (existing_rules or [])
        if not _managed_lifecycle_rule(rule, root)
    ]
    rules = [*preserved, *plan["lifecycle_rules"]]

    client.put_bucket_encryption(
        Bucket=bucket,
        ServerSideEncryptionConfiguration={
            "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
        },
    )
    _native_post(
        session,
        "b2_update_bucket",
        {
            "accountId": session.account_id,
            "bucketId": native_bucket["bucketId"],
            "ifRevisionIs": native_bucket["revision"],
            "lifecycleRules": rules,
        },
    )
    verified = inspect_bucket(settings)
    return {
        "applied": True,
        "bucket": bucket,
        "preserved_rule_count": len(preserved),
        "managed_rule_count": len(plan["lifecycle_rules"]),
        "verified": verified,
    }


def plan_as_json(settings: StorageSettings, producer_ids: list[str]) -> str:
    return json.dumps(configuration_plan(settings, producer_ids), indent=2, sort_keys=True)
