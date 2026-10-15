from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import httpx
from fastapi import Header, HTTPException

from utils.config import settings


@dataclass(frozen=True)
class CurrentUser:
    id: str
    email: str | None
    access_token: str | None
    authenticated: bool


async def get_current_user(authorization: str | None = Header(default=None)) -> CurrentUser:
    configured = bool(settings.supabase_url and settings.supabase_publishable_key)
    auth_required = settings.auth_required or bool(settings.supabase_url or settings.supabase_publishable_key)
    if not auth_required:
        return CurrentUser("local-demo", None, None, False)
    if not configured:
        raise HTTPException(status_code=503, detail="Supabase auth is required but SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY are not configured.")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Sign in to access this workspace.", headers={"WWW-Authenticate": "Bearer"})

    token = authorization[7:].strip()
    if not token or len(token) > 16384:
        raise HTTPException(status_code=401, detail="The access token is invalid.", headers={"WWW-Authenticate": "Bearer"})
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            response = await client.get(
                f"{settings.supabase_url}/auth/v1/user",
                headers={"apikey": settings.supabase_publishable_key, "Authorization": f"Bearer {token}"},
            )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="Supabase Auth could not be reached. Try again shortly.") from exc

    if response.status_code != 200:
        raise HTTPException(status_code=401, detail="Your session is invalid or expired. Sign in again.", headers={"WWW-Authenticate": "Bearer"})
    try:
        claims = response.json()
        user_id = str(UUID(claims["id"]))
    except (ValueError, TypeError, KeyError):
        raise HTTPException(status_code=401, detail="Supabase returned an invalid user identity.") from None
    return CurrentUser(user_id, claims.get("email"), token, True)
