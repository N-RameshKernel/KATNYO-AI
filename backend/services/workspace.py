from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from uuid import UUID

from fastapi import Depends

from pipelines.training import AutoMLPipeline
from services.dataset_service import DatasetService
from services.state_store import StateStore
from services.supabase_auth import CurrentUser, get_current_user
from utils.config import settings


@dataclass(frozen=True)
class Workspace:
    user_id: str
    email: str | None
    authenticated: bool
    state_store: StateStore
    dataset_service: DatasetService
    pipeline: AutoMLPipeline
    upload_dir: Path
    artifact_dir: Path


_workspaces: dict[str, Workspace] = {}
_lock = RLock()


def workspace_for(user: CurrentUser) -> Workspace:
    user_id = user.id
    if user_id != "local-demo":
        user_id = str(UUID(user_id))
    with _lock:
        existing = _workspaces.get(user_id)
        if existing:
            return Workspace(user_id, user.email, user.authenticated, existing.state_store, existing.dataset_service, existing.pipeline, existing.upload_dir, existing.artifact_dir)

        if user_id == "local-demo":
            root = settings.storage_dir
            state_store = StateStore(settings.state_path)
            upload_dir = settings.upload_dir
            artifact_dir = settings.artifact_dir
        else:
            root = settings.storage_dir / "users" / user_id
            state_store = StateStore(root / "state.json")
            upload_dir = root / "uploads"
            artifact_dir = root / "artifacts"
        upload_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        workspace = Workspace(
            user_id=user_id,
            email=user.email,
            authenticated=user.authenticated,
            state_store=state_store,
            dataset_service=DatasetService(state_store, upload_dir),
            pipeline=AutoMLPipeline(state_store, artifact_dir, user_id),
            upload_dir=upload_dir,
            artifact_dir=artifact_dir,
        )
        _workspaces[user_id] = workspace
        return workspace


def get_workspace(user: CurrentUser = Depends(get_current_user)) -> Workspace:
    return workspace_for(user)
