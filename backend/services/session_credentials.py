from __future__ import annotations

from threading import RLock
from time import monotonic
from urllib.parse import urlparse
from uuid import UUID


PROVIDERS = ("openai", "azure-openai", "gemini", "anthropic", "huggingface", "cohere")
SESSION_TTL_SECONDS = 12 * 60 * 60
MAX_SESSIONS = 1024


class SessionCredentialStore:
    """Short-lived provider credentials; deliberately never persisted to disk."""

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, dict]] = {}
        self._lock = RLock()

    def save(self, session_id: str, provider: str, payload: dict) -> dict:
        session_id = _session_id(session_id)
        provider = _provider(provider)
        api_key = str(payload.get("api_key") or "").strip()
        if not api_key or len(api_key) > 8192:
            raise ValueError("Enter a valid API key (maximum 8192 characters).")

        endpoint = str(payload.get("endpoint") or "").strip().rstrip("/")
        deployment = str(payload.get("deployment") or "").strip()
        model = str(payload.get("model") or "").strip()
        if provider == "azure-openai":
            parsed = urlparse(endpoint)
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or not parsed.hostname.lower().endswith(".openai.azure.com")
                or parsed.username
                or parsed.password
                or parsed.path not in ("", "/")
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("Use an HTTPS Azure OpenAI endpoint ending in .openai.azure.com.")
            if not deployment:
                raise ValueError("Enter your Azure OpenAI deployment name.")

        with self._lock:
            self._purge_expired()
            if session_id not in self._entries and len(self._entries) >= MAX_SESSIONS:
                oldest = min(self._entries, key=lambda item: self._entries[item]["_updated"])
                self._entries.pop(oldest, None)
            session = self._entries.setdefault(session_id, {})
            session[provider] = {
                "api_key": api_key,
                "endpoint": endpoint,
                "deployment": deployment,
                "model": model,
                "verified": False,
            }
            session["_updated"] = monotonic()
        return self.status(session_id)[provider]

    def get(self, session_id: str, provider: str) -> dict | None:
        session_id = _session_id(session_id)
        provider = _provider(provider)
        with self._lock:
            self._purge_expired()
            session = self._entries.get(session_id)
            if not session or provider not in session:
                return None
            session["_updated"] = monotonic()
            return dict(session[provider])

    def status(self, session_id: str) -> dict:
        session_id = _session_id(session_id)
        with self._lock:
            self._purge_expired()
            session = self._entries.get(session_id, {})
            return {
                provider: {
                    "connected": provider in session,
                    "verified": bool(session.get(provider, {}).get("verified")),
                    "model": session.get(provider, {}).get("model") or None,
                    "endpoint": session.get(provider, {}).get("endpoint") or None,
                    "deployment": session.get(provider, {}).get("deployment") or None,
                }
                for provider in PROVIDERS
            }

    def clear_provider(self, session_id: str, provider: str) -> None:
        session_id = _session_id(session_id)
        provider = _provider(provider)
        with self._lock:
            session = self._entries.get(session_id)
            if session:
                session.pop(provider, None)
                if not any(name in session for name in PROVIDERS):
                    self._entries.pop(session_id, None)

    def mark_verified(self, session_id: str, provider: str) -> None:
        session_id = _session_id(session_id)
        provider = _provider(provider)
        with self._lock:
            session = self._entries.get(session_id)
            if not session or provider not in session:
                raise ValueError("Save this provider's key in the session first.")
            session[provider]["verified"] = True
            session["_updated"] = monotonic()

    def clear_session(self, session_id: str) -> None:
        session_id = _session_id(session_id)
        with self._lock:
            self._entries.pop(session_id, None)

    def _purge_expired(self) -> None:
        now = monotonic()
        expired = [
            session_id
            for session_id, session in self._entries.items()
            if now - session.get("_updated", now) > SESSION_TTL_SECONDS
        ]
        for session_id in expired:
            self._entries.pop(session_id, None)


def _provider(provider: str) -> str:
    normalized = provider.lower()
    if normalized not in PROVIDERS:
        raise ValueError(f"Choose one of: {', '.join(PROVIDERS)}.")
    return normalized


def _session_id(value: str) -> str:
    try:
        return str(UUID(value))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Session ID must be a valid UUID.") from exc


session_credentials = SessionCredentialStore()
