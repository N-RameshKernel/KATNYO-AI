from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import quote

import httpx

from utils.config import settings


async def upload_user_object(path: Path, object_path: str, access_token: str) -> None:
    if not settings.supabase_url or not settings.supabase_publishable_key:
        raise RuntimeError("Supabase Storage is not configured.")

    bucket = quote(settings.supabase_storage_bucket, safe="")
    encoded_path = quote(object_path, safe="/")

    async def chunks():
        with path.open("rb") as source:
            while chunk := await asyncio.to_thread(source.read, 1024 * 1024):
                yield chunk

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=10)) as client:
            response = await client.post(
                f"{settings.supabase_url}/storage/v1/object/{bucket}/{encoded_path}",
                headers={
                    "apikey": settings.supabase_publishable_key,
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "text/csv",
                    "Content-Length": str(path.stat().st_size),
                    "x-upsert": "true",
                },
                content=chunks(),
            )
    except httpx.HTTPError as exc:
        raise RuntimeError("Could not save this CSV to Supabase Storage. Check your connection and retry.") from exc
    if response.is_error:
        raise RuntimeError("Supabase Storage rejected this CSV. Confirm the private bucket and storage policies are installed.")
