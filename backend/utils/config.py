from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _base_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _project_dir() -> Path:
    base = _base_dir()
    return base.parent if base.name == "backend" else base


load_dotenv(_project_dir() / ".env")


@dataclass(frozen=True)
class Settings:
    base_dir: Path = field(default_factory=_base_dir)
    project_dir: Path = field(default_factory=_project_dir)
    storage_dir: Path = field(default_factory=lambda: Path(os.getenv("KATNYO_STORAGE_DIR", _base_dir() / "storage")))
    supabase_url: str = os.getenv("SUPABASE_URL", "").rstrip("/")
    supabase_publishable_key: str = os.getenv("SUPABASE_PUBLISHABLE_KEY", "")
    supabase_storage_bucket: str = os.getenv("KATNYO_SUPABASE_STORAGE_BUCKET", "katnyo-user-data")
    auth_required: bool = os.getenv("KATNYO_AUTH_REQUIRED", "false").lower() == "true"
    allowed_origins: list[str] = field(
        default_factory=lambda: os.getenv(
            "KATNYO_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
    )
    max_profile_rows: int = int(os.getenv("KATNYO_MAX_PROFILE_ROWS", "50000"))
    max_train_rows: int = int(os.getenv("KATNYO_MAX_TRAIN_ROWS", "30000"))
    exact_row_count_limit_mb: int = int(os.getenv("KATNYO_EXACT_ROW_COUNT_LIMIT_MB", "80"))
    optuna_trials: int = int(os.getenv("KATNYO_OPTUNA_TRIALS", "8"))
    random_state: int = int(os.getenv("KATNYO_RANDOM_STATE", "42"))
    mlflow_experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "KATNYO AI")
    chat_provider: str = os.getenv("KATNYO_CHAT_PROVIDER", "local").lower()
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
    huggingface_model: str = os.getenv("HUGGINGFACE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    cohere_model: str = os.getenv("COHERE_MODEL", "command-a-plus-05-2026")
    wandb_api_key: str | None = os.getenv("WANDB_API_KEY")
    wandb_project: str = os.getenv("WANDB_PROJECT", "katnyo-ai")
    neptune_api_token: str | None = os.getenv("NEPTUNE_API_TOKEN")
    neptune_project: str | None = os.getenv("NEPTUNE_PROJECT")
    ray_tune_enabled: bool = os.getenv("RAY_TUNE_ENABLED", "false").lower() == "true"
    h2o_enabled: bool = os.getenv("H2O_AUTOML_ENABLED", "false").lower() == "true"
    aws_access_key_id: str | None = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = os.getenv("AWS_SECRET_ACCESS_KEY")
    azure_ml_endpoint: str | None = os.getenv("AZURE_ML_ENDPOINT")
    azure_ml_key: str | None = os.getenv("AZURE_ML_KEY")
    vertex_project: str | None = os.getenv("VERTEX_AI_PROJECT")
    vertex_location: str | None = os.getenv("VERTEX_AI_LOCATION")

    def integrations(self) -> dict:
        import importlib.util

        return {
            "ai": {
                "credentialMode": "browser-session",
                "providers": ["openai", "azure-openai", "gemini", "anthropic", "huggingface", "cohere"],
            },
            "tracking": {
                "mlflow": importlib.util.find_spec("mlflow") is not None,
                "wandbCredentialsConfigured": bool(self.wandb_api_key),
                "wandbPackageInstalled": importlib.util.find_spec("wandb") is not None,
                "neptuneCredentialsConfigured": bool(self.neptune_api_token and self.neptune_project),
                "neptunePackageInstalled": importlib.util.find_spec("neptune") is not None,
            },
            "automl": {"optuna": importlib.util.find_spec("optuna") is not None, "rayTuneConfigured": self.ray_tune_enabled, "h2oAutoMLConfigured": self.h2o_enabled},
            "cloud": {"amazonForecastCredentialsConfigured": bool(self.aws_access_key_id and self.aws_secret_access_key), "azureMLCredentialsConfigured": bool(self.azure_ml_endpoint and self.azure_ml_key), "vertexAIProjectConfigured": bool(self.vertex_project and self.vertex_location)},
            "chatProvider": self.chat_provider,
            "accounts": {
                "authRequired": self.auth_required or bool(self.supabase_url or self.supabase_publishable_key),
                "supabaseConfigured": bool(self.supabase_url and self.supabase_publishable_key),
                "storageCredentialsConfigured": bool(self.supabase_url and self.supabase_publishable_key),
            },
        }

    @property
    def upload_dir(self) -> Path:
        return self.storage_dir / "uploads"

    @property
    def artifact_dir(self) -> Path:
        return self.storage_dir / "artifacts"

    @property
    def state_path(self) -> Path:
        return self.storage_dir / "state.json"

    @property
    def mlflow_tracking_uri(self) -> str:
        return os.getenv("MLFLOW_TRACKING_URI") or f"file://{self.project_dir / 'mlruns'}"

    def ensure_directories(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        (self.project_dir / "mlruns").mkdir(parents=True, exist_ok=True)


settings = Settings()
