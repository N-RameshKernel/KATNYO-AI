from __future__ import annotations

import json
import os
from threading import RLock
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from utils.config import settings
from utils.io import make_json_safe, utc_now


class StateStore:
    def __init__(self, state_path: Path | None = None) -> None:
        self.state_path = state_path or settings.state_path
        self._lock = RLock()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load()
        interrupted = [job_id for job_id, job in self._state.get("jobs", {}).items() if job.get("status") in {"queued", "running"}]
        for job_id in interrupted:
            self._state["jobs"][job_id].update({"status": "failed", "error": "Training process restarted before the job completed."})
        if interrupted:
            self._persist()

    def add_dataset(self, dataset: dict) -> dict:
        with self._lock:
            self._state["datasets"][dataset["id"]] = make_json_safe(dataset)
            self._state["latestDatasetId"] = dataset["id"]
            revisions = self._state.setdefault("datasetRevisions", {}).setdefault(dataset["id"], [])
            if not any(item.get("revisionId") == dataset.get("revisionId") for item in revisions):
                revisions.append(make_json_safe({key: dataset.get(key) for key in ("revisionId", "filename", "path", "sizeBytes", "createdAt", "profile")}))
            self._persist()
        return deepcopy(dataset)

    def dataset_by_filename(self, filename: str) -> dict | None:
        for dataset in self._state["datasets"].values():
            if dataset.get("filename", "").casefold() == filename.casefold():
                return deepcopy(dataset)
        return None

    def get_dataset(self, dataset_id: str | None) -> dict | None:
        if not dataset_id:
            return None
        dataset = self._state["datasets"].get(dataset_id)
        return deepcopy(dataset) if dataset else None

    def latest_dataset(self) -> dict | None:
        return self.get_dataset(self._state.get("latestDatasetId"))

    def dataset_revisions(self, dataset_id: str) -> list[dict]:
        return deepcopy(self._state.setdefault("datasetRevisions", {}).get(dataset_id, []))

    def add_job(self, job: dict) -> dict:
        with self._lock:
            self._state.setdefault("jobs", {})[job["jobId"]] = make_json_safe(job)
            self._persist()
        return deepcopy(job)

    def update_job(self, job_id: str, **updates) -> dict | None:
        with self._lock:
            job = self._state.setdefault("jobs", {}).get(job_id)
            if not job:
                return None
            job.update(make_json_safe(updates))
            self._persist()
        return deepcopy(job)

    def get_job(self, job_id: str) -> dict | None:
        job = self._state.setdefault("jobs", {}).get(job_id)
        return deepcopy(job) if job else None

    def list_jobs(self, dataset_id: str | None = None) -> list[dict]:
        jobs = [job for job in self._state.setdefault("jobs", {}).values() if dataset_id is None or job.get("datasetId") == dataset_id]
        return deepcopy(sorted(jobs, key=lambda job: job["createdAt"], reverse=True))

    def list_runs(self, dataset_id: str | None = None) -> list[dict]:
        runs = [run for run in self._state["runs"].values() if dataset_id is None or run.get("datasetId") == dataset_id]
        return deepcopy(sorted(runs, key=lambda run: run["createdAt"], reverse=True))

    def register_model(self, run_id: str, name: str, stage: str = "staging") -> dict:
        run = self.get_run(run_id)
        if not run:
            return None
        entry = {"modelId": f"model_{uuid4().hex[:12]}", "name": name, "runId": run_id, "datasetId": run["datasetId"], "stage": stage, "createdAt": utc_now(), "artifact": run["artifacts"]["model"], "metrics": run["bestModel"].get("metrics", {})}
        with self._lock:
            self._state.setdefault("registry", []).append(make_json_safe(entry))
            self._persist()
        return deepcopy(entry)

    def list_registry(self) -> list[dict]:
        return deepcopy(self._state.setdefault("registry", []))

    def add_run(self, run: dict) -> dict:
        with self._lock:
            self._state["runs"][run["runId"]] = make_json_safe(run)
            self._state["latestRunId"] = run["runId"]
            self._persist()
        return deepcopy(run)

    def get_run(self, run_id: str | None) -> dict | None:
        if not run_id:
            return None
        run = self._state["runs"].get(run_id)
        return deepcopy(run) if run else None

    def latest_run(self, dataset_id: str | None = None) -> dict | None:
        if dataset_id:
            dataset_runs = [
                run
                for run in self._state["runs"].values()
                if run.get("datasetId") == dataset_id
            ]
            if not dataset_runs:
                return None
            return deepcopy(sorted(dataset_runs, key=lambda run: run["createdAt"])[-1])
        return self.get_run(self._state.get("latestRunId"))

    def add_feedback(self, feedback: dict) -> dict:
        feedback_record = {
            "id": f"fb_{uuid4().hex[:12]}",
            "createdAt": utc_now(),
            **feedback,
        }
        self._state["feedback"].append(make_json_safe(feedback_record))
        self._persist()
        return deepcopy(feedback_record)

    def feedback_timeline(self, dataset_id: str | None = None) -> list[dict]:
        runs = [
            run
            for run in self._state["runs"].values()
            if dataset_id is None or run.get("datasetId") == dataset_id
        ]
        runs = sorted(runs, key=lambda run: run["createdAt"])
        return [
            {
                "run": index + 1,
                "runId": run["runId"],
                "bestModel": run["bestModel"]["name"],
                "metric": run["bestModel"]["primaryMetric"],
                "value": run["bestModel"]["primaryValue"],
                "createdAt": run["createdAt"],
            }
            for index, run in enumerate(runs)
        ]

    def _load(self) -> dict:
        if not self.state_path.exists():
            return {"datasets": {}, "runs": {}, "feedback": [], "latestDatasetId": None, "latestRunId": None, "datasetRevisions": {}, "jobs": {}, "registry": []}
        with self.state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        for key, default in (("datasetRevisions", {}), ("jobs", {}), ("registry", [])):
            state.setdefault(key, default)
        return state

    def _persist(self) -> None:
        with self._lock:
            tmp_path = self.state_path.with_suffix(".tmp")
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(make_json_safe(self._state), handle, indent=2)
            os.replace(tmp_path, self.state_path)
