from __future__ import annotations

import base64
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


class AuthError(ValueError):
    """Raised when user authentication or local session handling fails."""


class AuthHttpError(AuthError):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(f"Authentication request failed ({status}): {message}")


def _validated_https_url(value: str, label: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise AuthError(f"{label} must be an absolute HTTP(S) URL")
    if parsed.scheme != "https" and parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise AuthError(f"{label} must use HTTPS except for a local development server")
    return value.rstrip("/")


def _default_session_file() -> Path:
    override = os.environ.get("MANDA_SESSION_FILE", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    config_root = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(config_root).expanduser() if config_root else Path.home() / ".config"
    return (base / "manda" / "session.json").resolve()


@dataclass(frozen=True)
class AuthSettings:
    supabase_url: str
    publishable_key: str = field(repr=False)
    api_url: str = ""
    session_file: Path = Path("session.json")

    @classmethod
    def from_environment(cls) -> AuthSettings:
        supabase_url = os.environ.get("MANDA_SUPABASE_URL", "").strip()
        publishable_key = os.environ.get("MANDA_SUPABASE_PUBLISHABLE_KEY", "").strip()
        if not supabase_url or not publishable_key:
            raise AuthError("Set MANDA_SUPABASE_URL and MANDA_SUPABASE_PUBLISHABLE_KEY")
        supabase_url = _validated_https_url(supabase_url, "MANDA_SUPABASE_URL")
        api_url = os.environ.get("MANDA_API_URL", "").strip()
        if not api_url:
            api_url = f"{supabase_url}/functions/v1/manda-access"
        api_url = _validated_https_url(api_url, "MANDA_API_URL")
        return cls(
            supabase_url=supabase_url,
            publishable_key=publishable_key,
            api_url=api_url,
            session_file=_default_session_file(),
        )

    def report(self) -> dict[str, Any]:
        return {
            "supabase_url": self.supabase_url,
            "api_url": self.api_url,
            "publishable_key_configured": True,
            "session_file": str(self.session_file),
        }


@dataclass(frozen=True)
class AuthSession:
    access_token: str
    refresh_token: str | None
    expires_at: int
    user_id: str
    email: str | None
    producer_id: str | None = None
    producer_label: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AuthSession:
        required = ("access_token", "expires_at", "user_id")
        if any(key not in value for key in required):
            raise AuthError("Stored Manda session is incomplete")
        access_token = value["access_token"]
        user_id = value["user_id"]
        if not isinstance(access_token, str) or not access_token:
            raise AuthError("Stored Manda access token is invalid")
        if not isinstance(user_id, str) or not user_id:
            raise AuthError("Stored Manda user ID is invalid")
        try:
            expires_at = int(value["expires_at"])
        except (TypeError, ValueError) as exc:
            raise AuthError("Stored Manda session expiry is invalid") from exc
        return cls(
            access_token=access_token,
            refresh_token=value.get("refresh_token") or None,
            expires_at=expires_at,
            user_id=user_id,
            email=value.get("email") or None,
            producer_id=value.get("producer_id") or None,
            producer_label=value.get("producer_label") or None,
        )

    def report(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "email": self.email,
            "expires_at": self.expires_at,
            "refreshable": self.refresh_token is not None,
            "producer_id": self.producer_id,
            "producer_label": self.producer_label,
        }


class SessionStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> AuthSession:
        if not self.path.is_file():
            raise AuthError("No Manda user session; run `manda auth login`")
        mode = self.path.stat().st_mode & 0o777
        if mode & 0o077:
            raise AuthError(f"Manda session file must have mode 600: {self.path}")
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AuthError(f"Cannot read Manda session: {self.path}") from exc
        if not isinstance(value, dict):
            raise AuthError("Stored Manda session must be a JSON object")
        return AuthSession.from_dict(value)

    def save(self, session: AuthSession) -> None:
        parent_exists = self.path.parent.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not parent_exists:
            os.chmod(self.path.parent, 0o700)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=".session-",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(asdict(session), handle, sort_keys=True)
                handle.write("\n")
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


class SupabaseAuthClient:
    def __init__(self, settings: AuthSettings):
        self.settings = settings

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        access_token: str | None = None,
        query: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        suffix = f"?{urlencode(query)}" if query else ""
        url = f"{self.settings.supabase_url}/auth/v1/{path.lstrip('/')}{suffix}"
        headers = {
            "apikey": self.settings.publishable_key,
            "Accept": "application/json",
            "User-Agent": "manda-cli/0.1",
        }
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        data = None
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=30) as response:  # noqa: S310 - validated URL
                raw = response.read()
        except HTTPError as exc:
            raw = exc.read()
            try:
                error = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                error = {}
            message = str(
                error.get("msg")
                or error.get("message")
                or error.get("error_description")
                or exc.reason
            )
            raise AuthHttpError(exc.code, message) from exc
        except (URLError, TimeoutError) as exc:
            reason = getattr(exc, "reason", exc)
            raise AuthError(f"Cannot reach Manda authentication: {reason}") from exc
        if not raw:
            return {}
        try:
            value = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise AuthError("Authentication service returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise AuthError("Authentication service returned an invalid response")
        return value

    def start_email_otp(self, email: str) -> None:
        normalized = email.strip()
        if "@" not in normalized or len(normalized) > 320:
            raise AuthError("Enter a valid email address")
        self._request(
            "POST",
            "otp",
            payload={"email": normalized, "create_user": False},
        )

    def verify_email_otp(self, email: str, token: str) -> AuthSession:
        value = self._request(
            "POST",
            "verify",
            payload={"email": email.strip(), "token": token.strip(), "type": "email"},
        )
        return _session_from_response(value)

    def refresh(self, refresh_token: str) -> AuthSession:
        value = self._request(
            "POST",
            "token",
            query={"grant_type": "refresh_token"},
            payload={"refresh_token": refresh_token},
        )
        return _session_from_response(value)

    def user(self, access_token: str) -> dict[str, Any]:
        return self._request("GET", "user", access_token=access_token)

    def logout(self, access_token: str) -> None:
        self._request(
            "POST",
            "logout",
            access_token=access_token,
            query={"scope": "local"},
        )


def _jwt_claims(access_token: str) -> dict[str, Any]:
    try:
        encoded = access_token.split(".")[1]
        encoded += "=" * (-len(encoded) % 4)
        value = json.loads(base64.urlsafe_b64decode(encoded).decode("utf-8"))
    except (IndexError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthError("Access token is not a valid JWT") from exc
    if not isinstance(value, dict):
        raise AuthError("Access token has invalid JWT claims")
    return value


def _session_from_response(value: dict[str, Any]) -> AuthSession:
    access_token = value.get("access_token")
    refresh_token = value.get("refresh_token")
    if not isinstance(access_token, str) or not access_token:
        raise AuthError("Authentication response has no access token")
    claims = _jwt_claims(access_token)
    user = value.get("user") if isinstance(value.get("user"), dict) else {}
    user_id = user.get("id") or claims.get("sub")
    email = user.get("email") or claims.get("email")
    expires_at = value.get("expires_at") or claims.get("exp")
    if not isinstance(user_id, str) or not user_id:
        raise AuthError("Authentication response has no user ID")
    if not isinstance(expires_at, (int, float)):
        expires_in = value.get("expires_in")
        if not isinstance(expires_in, (int, float)):
            raise AuthError("Authentication response has no expiry")
        expires_at = time.time() + float(expires_in)
    return AuthSession(
        access_token=access_token,
        refresh_token=refresh_token if isinstance(refresh_token, str) else None,
        expires_at=int(expires_at),
        user_id=user_id,
        email=email if isinstance(email, str) else None,
    )


class AuthManager:
    def __init__(
        self,
        settings: AuthSettings,
        *,
        store: SessionStore | None = None,
        client: SupabaseAuthClient | None = None,
    ):
        self.settings = settings
        self.store = store or SessionStore(settings.session_file)
        self.client = client or SupabaseAuthClient(settings)

    @classmethod
    def from_environment(cls) -> AuthManager:
        return cls(AuthSettings.from_environment())

    def ensure_session(self) -> AuthSession:
        session = self.store.load()
        if session.expires_at > int(time.time()) + 60:
            return session
        if not session.refresh_token:
            raise AuthError("Manda access token expired; run `manda auth login` again")
        refreshed = self.client.refresh(session.refresh_token)
        refreshed = replace(
            refreshed,
            producer_id=session.producer_id,
            producer_label=session.producer_label,
        )
        self.store.save(refreshed)
        return refreshed

    def login_email(self, email: str, otp: str) -> AuthSession:
        session = self.client.verify_email_otp(email, otp)
        self.store.save(session)
        return session

    def login_access_token(self, access_token: str) -> AuthSession:
        claims = _jwt_claims(access_token.strip())
        user = self.client.user(access_token.strip())
        session = AuthSession(
            access_token=access_token.strip(),
            refresh_token=None,
            expires_at=int(claims.get("exp") or 0),
            user_id=str(user.get("id") or claims.get("sub") or ""),
            email=user.get("email") if isinstance(user.get("email"), str) else None,
        )
        if not session.user_id or session.expires_at <= int(time.time()):
            raise AuthError("Access token is expired or has no user identity")
        self.store.save(session)
        return session

    def update_producer(self, producer_id: str, label: str) -> AuthSession:
        session = self.store.load()
        updated = replace(session, producer_id=producer_id, producer_label=label)
        self.store.save(updated)
        return updated

    def logout(self) -> None:
        try:
            session = self.ensure_session()
            self.client.logout(session.access_token)
        finally:
            self.store.clear()
