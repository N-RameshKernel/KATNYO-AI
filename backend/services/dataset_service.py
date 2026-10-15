from pathlib import Path
import hashlib
from uuid import uuid4

from fastapi import UploadFile

from services.state_store import StateStore
from utils.config import settings
from utils.io import safe_filename, utc_now


class DatasetService:
    def __init__(self, state_store: StateStore, upload_dir: Path | None = None) -> None:
        self.state_store = state_store
        self.upload_dir = upload_dir or settings.upload_dir

    async def save_upload(self, file: UploadFile) -> dict:
        if not file.filename:
            raise ValueError("A CSV filename is required.")
        if not file.filename.lower().endswith(".csv"):
            raise ValueError("KATNYO AI accepts CSV files only.")

        filename = safe_filename(file.filename)
        existing = self.state_store.dataset_by_filename(filename)
        dataset_id = existing["id"] if existing else f"ds_{uuid4().hex[:12]}"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        destination = self.upload_dir / f"{dataset_id}_{uuid4().hex[:8]}_{filename}"

        with destination.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                output.write(chunk)

        hasher = hashlib.sha256()
        with destination.open("rb") as saved_file:
            for chunk in iter(lambda: saved_file.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()[:12]
        dataset = {
            "id": dataset_id,
            "revisionId": f"rev_{digest}",
            "filename": filename,
            "path": str(destination),
            "sizeBytes": destination.stat().st_size,
            "createdAt": utc_now(),
        }
        return dataset

    def sample_dataset_record(self) -> dict:
        sample_path = settings.project_dir / "data" / "sample_m5_sales.csv"
        return {
            "id": "sample_m5_sales",
            "filename": sample_path.name,
            "path": str(Path(sample_path)),
            "sizeBytes": sample_path.stat().st_size if sample_path.exists() else 0,
            "createdAt": utc_now(),
        }
