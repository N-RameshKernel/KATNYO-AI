from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response

from models.schemas import ChatRequest, FeedbackRequest, PredictionRequest, ProviderCredentialsRequest, RegistryRequest, TrainRequest
from pipelines.training import AutoMLPipeline
from services.ai_provider import answer_dataset_question, test_provider_connection
from services.profiler import DatasetProfiler
from services.session_credentials import session_credentials
from services.state_store import StateStore
from services.supabase_auth import CurrentUser, get_current_user
from services.supabase_storage import upload_user_object
from services.workspace import Workspace, get_workspace
from utils.config import settings
from utils.io import slugify, utc_now

router = APIRouter(tags=["katnyo-ai"])
profiler = DatasetProfiler()
executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="katnyo-train")


@router.post("/upload")
async def upload_csv(
    file: UploadFile = File(...),
    target: str | None = Form(default=None),
    workspace: Workspace = Depends(get_workspace),
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    dataset = None
    try:
        dataset = await workspace.dataset_service.save_upload(file)
        profile = profiler.profile_dataset(dataset["path"], target=target)
        dataset["profile"] = profile
        if user.authenticated:
            object_path = f"{user.id}/{dataset['id']}/{dataset['revisionId']}/{dataset['filename']}"
            try:
                await upload_user_object(Path(dataset["path"]), object_path, user.access_token or "")
            except RuntimeError as exc:
                Path(dataset["path"]).unlink(missing_ok=True)
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            dataset["storagePath"] = object_path
        workspace.state_store.add_dataset(dataset)
        return {"datasetId": dataset["id"], "profile": profile, "storage": "private-supabase" if user.authenticated else "local-demo"}
    except HTTPException:
        raise
    except ValueError as exc:
        if dataset:
            Path(dataset["path"]).unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        if dataset:
            Path(dataset["path"]).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc


@router.get("/profile")
def get_profile(dataset_id: str | None = Query(default=None), workspace: Workspace = Depends(get_workspace)) -> dict:
    dataset = _resolve_dataset(workspace, dataset_id)
    if not dataset.get("profile"):
        dataset["profile"] = profiler.profile_dataset(dataset["path"])
        workspace.state_store.add_dataset(dataset)
    return {"datasetId": dataset["id"], "profile": dataset["profile"]}


@router.post("/train")
def train(payload: TrainRequest, workspace: Workspace = Depends(get_workspace)) -> dict:
    dataset = _resolve_revision(workspace, _resolve_dataset(workspace, payload.dataset_id), payload.revision_id)
    try:
        run = workspace.pipeline.train(dataset=dataset, target=payload.target, max_rows=payload.max_rows, optuna_trials=payload.optuna_trials, include_models=payload.include_models)
        workspace.state_store.add_run(run)
        return run
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc


@router.post("/train/jobs", status_code=202)
def start_training_job(payload: TrainRequest, workspace: Workspace = Depends(get_workspace)) -> dict:
    dataset = _resolve_revision(workspace, _resolve_dataset(workspace, payload.dataset_id), payload.revision_id)
    job_id = f"job_{uuid4().hex[:12]}"
    workspace.state_store.add_job({"jobId": job_id, "datasetId": dataset["id"], "status": "queued", "progress": 0, "logs": [], "createdAt": utc_now()})
    executor.submit(_run_training_job, workspace, job_id, dataset, payload)
    return workspace.state_store.get_job(job_id)


@router.get("/train/jobs")
def list_training_jobs(dataset_id: str | None = Query(default=None), workspace: Workspace = Depends(get_workspace)) -> dict:
    return {"jobs": workspace.state_store.list_jobs(dataset_id)}


@router.get("/train/jobs/{job_id}")
def get_training_job(job_id: str, workspace: Workspace = Depends(get_workspace)) -> dict:
    job = workspace.state_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found.")
    return job


@router.get("/leaderboard")
def leaderboard(dataset_id: str | None = Query(default=None), workspace: Workspace = Depends(get_workspace)) -> dict:
    rows = []
    for run in workspace.state_store.list_runs(dataset_id):
        for model in run.get("models", []) + run.get("ensembles", []):
            rows.append({"runId": run["runId"], "datasetId": run["datasetId"], "createdAt": run["createdAt"], **model})
    rows.sort(key=lambda item: item.get("score") if item.get("score") is not None else float("-inf"), reverse=True)
    return {"models": rows}


@router.get("/runs")
def list_runs(dataset_id: str | None = Query(default=None), workspace: Workspace = Depends(get_workspace)) -> dict:
    return {"runs": [{"runId": run["runId"], "datasetId": run["datasetId"], "target": run["target"], "problemType": run["problemType"], "createdAt": run["createdAt"], "bestModel": run["bestModel"]} for run in workspace.state_store.list_runs(dataset_id)]}


@router.get("/integrations")
def integration_status() -> dict:
    return settings.integrations()


@router.get("/sessions/{session_id}/providers")
def provider_session_status(session_id: str) -> dict:
    try:
        return {"providers": session_credentials.status(session_id)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/sessions/{session_id}/providers/{provider}")
def save_provider_session(session_id: str, provider: str, payload: ProviderCredentialsRequest) -> dict:
    try:
        status = session_credentials.save(session_id, provider, payload.model_dump())
        return {"provider": provider, **status}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/sessions/{session_id}/providers/{provider}/test")
def test_provider_session(session_id: str, provider: str) -> dict:
    try:
        credentials = session_credentials.get(session_id, provider)
        if not credentials:
            raise HTTPException(status_code=404, detail="Save this provider's key in the session first.")
        result = test_provider_connection(provider, credentials)
        session_credentials.mark_verified(session_id, provider)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.delete("/sessions/{session_id}/providers/{provider}")
def clear_provider_session(session_id: str, provider: str) -> dict:
    try:
        session_credentials.clear_provider(session_id, provider)
        return {"provider": provider, "connected": False}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/sessions/{session_id}")
def end_provider_session(session_id: str) -> dict:
    try:
        session_credentials.clear_session(session_id)
        return {"sessionId": session_id, "ended": True}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/experiments/compare")
def compare_experiments(run_ids: list[str] = Query(..., alias="runIds"), workspace: Workspace = Depends(get_workspace)) -> dict:
    runs = [workspace.state_store.get_run(run_id) for run_id in run_ids]
    if any(run is None for run in runs):
        raise HTTPException(status_code=404, detail="One or more run IDs were not found.")
    return {"runs": [{"runId": run["runId"], "datasetId": run["datasetId"], "target": run["target"], "problemType": run["problemType"], "createdAt": run["createdAt"], "bestModel": run["bestModel"], "models": run["models"] + run["ensembles"]} for run in runs]}


@router.get("/datasets/{dataset_id}/revisions")
def dataset_revisions(dataset_id: str, workspace: Workspace = Depends(get_workspace)) -> dict:
    if not workspace.state_store.get_dataset(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found.")
    revisions = [{key: value for key, value in item.items() if key != "path"} for item in workspace.state_store.dataset_revisions(dataset_id)]
    return {"datasetId": dataset_id, "revisions": revisions}


@router.get("/datasets/{dataset_id}/revisions/{revision_id}")
def get_dataset_revision(dataset_id: str, revision_id: str, workspace: Workspace = Depends(get_workspace)) -> dict:
    if not workspace.state_store.get_dataset(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found.")
    revision = next((item for item in workspace.state_store.dataset_revisions(dataset_id) if item.get("revisionId") == revision_id), None)
    if not revision:
        raise HTTPException(status_code=404, detail="Dataset revision not found.")
    return {"datasetId": dataset_id, "revision": {key: value for key, value in revision.items() if key != "path"}}


@router.post("/predict/{run_id}")
def predict(run_id: str, payload: PredictionRequest, workspace: Workspace = Depends(get_workspace)) -> dict:
    run = workspace.state_store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found.")
    candidate = payload.model_name or run["bestModel"]["name"]
    if payload.model_name and payload.model_name not in run.get("artifacts", {}).get("models", {}):
        raise HTTPException(status_code=404, detail="Model artifact not found for this run.")
    model_path = workspace.artifact_dir / run_id / "model.joblib"
    if payload.model_name:
        model_path = workspace.artifact_dir / run_id / "models" / f"{slugify(candidate)}.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model artifact is unavailable.")
    try:
        model = joblib.load(model_path)
        frame = pd.DataFrame(payload.rows)
        predictions = model.predict(frame)
        result = {"predictions": [value.item() if hasattr(value, "item") else value for value in predictions]}
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(frame)
            result["probabilities"] = probabilities.tolist()
            result["classes"] = [value.item() if hasattr(value, "item") else value for value in model.classes_]
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction rows do not match the trained features: {exc}") from exc


@router.post("/registry")
def register_model(payload: RegistryRequest, workspace: Workspace = Depends(get_workspace)) -> dict:
    entry = workspace.state_store.register_model(payload.run_id, payload.name, payload.stage)
    if not entry:
        raise HTTPException(status_code=404, detail="Training run not found.")
    return entry


@router.get("/registry")
def model_registry(workspace: Workspace = Depends(get_workspace)) -> dict:
    return {"models": workspace.state_store.list_registry()}


@router.get("/reports/{run_id}")
def training_report(run_id: str, workspace: Workspace = Depends(get_workspace)):
    run = workspace.state_store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found.")
    lines = ["# KATNYO AI Training Report", "", f"- Run: `{run['runId']}`", f"- Dataset: `{run['datasetId']}`", f"- Target: `{run['target']}`", f"- Problem: `{run['problemType']}`", f"- Created: {run['createdAt']}", "", "## Best model", "", f"**{run['bestModel']['name']}**", "", "| Metric | Value |", "| --- | ---: |"]
    lines.extend(f"| {name} | {value} |" for name, value in run["bestModel"].get("metrics", {}).items())
    lines.extend(["", "## Model leaderboard", "", "| Model | Type | Primary metric | Value |", "| --- | --- | --- | ---: |"])
    lines.extend(f"| {model['name']} | {model['type']} | {model['primaryMetric']} | {model['primaryValue']} |" for model in run.get("models", []) + run.get("ensembles", []))
    return Response(content="\n".join(lines), media_type="text/markdown", headers={"Content-Disposition": f'attachment; filename="katnyo-report-{run_id}.md"'})


@router.get("/artifacts/{run_id}/{asset_path:path}")
def get_artifact(run_id: str, asset_path: str, workspace: Workspace = Depends(get_workspace)):
    if not workspace.state_store.get_run(run_id):
        raise HTTPException(status_code=404, detail="Training run not found.")
    run_dir = (workspace.artifact_dir / run_id).resolve()
    artifact = (run_dir / asset_path).resolve()
    if run_dir not in artifact.parents or not artifact.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return FileResponse(artifact)


@router.post("/chat")
def dataset_chat(payload: ChatRequest, workspace: Workspace = Depends(get_workspace)) -> dict:
    dataset = _resolve_dataset(workspace, payload.dataset_id)
    profile = dataset.get("profile") or profiler.profile_dataset(dataset["path"])
    try:
        provider = (payload.provider or settings.chat_provider or "local").lower()
        credentials = None
        if provider != "local":
            if not payload.session_id:
                raise HTTPException(status_code=400, detail="Open a provider session before using an external AI provider.")
            credentials = session_credentials.get(payload.session_id, provider)
        answer = answer_dataset_question(payload.question, profile, provider, credentials)
        return {**answer, "datasetId": dataset["id"], "dataSent": "profile metadata only"}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/results")
def get_results(dataset_id: str | None = Query(default=None), run_id: str | None = Query(default=None), workspace: Workspace = Depends(get_workspace)) -> dict:
    return _resolve_run(workspace, dataset_id=dataset_id, run_id=run_id)


@router.get("/explain")
def get_explainability(dataset_id: str | None = Query(default=None), run_id: str | None = Query(default=None), workspace: Workspace = Depends(get_workspace)) -> dict:
    run = _resolve_run(workspace, dataset_id=dataset_id, run_id=run_id)
    return {"datasetId": run["datasetId"], "runId": run["runId"], "target": run["target"], "problemType": run["problemType"], "explainability": run.get("explainability", {})}


@router.post("/feedback")
def submit_feedback(payload: FeedbackRequest, workspace: Workspace = Depends(get_workspace)) -> dict:
    run = _resolve_run(workspace, dataset_id=payload.dataset_id, run_id=payload.run_id)
    feedback = workspace.state_store.add_feedback({"datasetId": run["datasetId"], "runId": run["runId"], "rating": payload.rating, "comment": payload.comment, "acceptedFeatures": payload.accepted_features})
    return {"feedback": feedback, "timeline": workspace.state_store.feedback_timeline(run["datasetId"])}


def _resolve_dataset(workspace: Workspace, dataset_id: str | None) -> dict:
    dataset = workspace.state_store.get_dataset(dataset_id) if dataset_id else workspace.state_store.latest_dataset()
    if not dataset:
        raise HTTPException(status_code=404, detail="No dataset found. Upload a CSV first.")
    return dataset


def _resolve_revision(workspace: Workspace, dataset: dict, revision_id: str | None) -> dict:
    if not revision_id:
        return dataset
    revision = next((item for item in workspace.state_store.dataset_revisions(dataset["id"]) if item.get("revisionId") == revision_id), None)
    if not revision:
        raise HTTPException(status_code=404, detail="Dataset revision not found.")
    return {**dataset, **revision, "id": dataset["id"]}


def _resolve_run(workspace: Workspace, dataset_id: str | None, run_id: str | None) -> dict:
    run = workspace.state_store.get_run(run_id) if run_id else workspace.state_store.latest_run(dataset_id)
    if not run:
        raise HTTPException(status_code=404, detail="No training run found yet.")
    return run


def _run_training_job(workspace: Workspace, job_id: str, dataset: dict, payload: TrainRequest) -> None:
    store = workspace.state_store
    store.update_job(job_id, status="running", progress=5, logs=["Training started."])
    try:
        def progress(percent: int, message: str) -> None:
            current = store.get_job(job_id) or {}
            store.update_job(job_id, progress=percent, logs=current.get("logs", []) + [message])

        run = workspace.pipeline.train(dataset, payload.target, payload.max_rows, payload.optuna_trials, payload.include_models, progress_callback=progress)
        store.add_run(run)
        store.update_job(job_id, status="completed", progress=100, runId=run["runId"], result=run, logs=run.get("logs", []))
    except Exception as exc:
        store.update_job(job_id, status="failed", progress=100, error=str(exc), logs=["Training failed."])
