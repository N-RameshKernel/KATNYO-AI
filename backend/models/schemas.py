from typing import Literal

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    dataset_id: str | None = Field(default=None, alias="datasetId")
    revision_id: str | None = Field(default=None, alias="revisionId")
    target: str | None = None
    max_rows: int | None = Field(default=None, alias="maxRows", ge=200)
    optuna_trials: int | None = Field(default=None, alias="optunaTrials", ge=0, le=100)
    include_models: list[str] | None = Field(default=None, alias="includeModels")

    class Config:
        populate_by_name = True


class FeedbackRequest(BaseModel):
    dataset_id: str | None = Field(default=None, alias="datasetId")
    run_id: str | None = Field(default=None, alias="runId")
    rating: int = Field(ge=1, le=5)
    comment: str | None = None
    accepted_features: list[str] = Field(default_factory=list, alias="acceptedFeatures")

    class Config:
        populate_by_name = True


class PredictionRequest(BaseModel):
    rows: list[dict]
    model_name: str | None = Field(default=None, alias="modelName")


class RegistryRequest(BaseModel):
    run_id: str = Field(alias="runId")
    name: str
    stage: Literal["staging", "production", "archived"] = "staging"

    class Config:
        populate_by_name = True


class ChatRequest(BaseModel):
    dataset_id: str | None = Field(default=None, alias="datasetId")
    session_id: str | None = Field(default=None, alias="sessionId")
    question: str
    provider: str | None = None

    class Config:
        populate_by_name = True


class ProviderCredentialsRequest(BaseModel):
    api_key: str = Field(alias="apiKey", min_length=1, max_length=8192)
    endpoint: str | None = Field(default=None, max_length=512)
    deployment: str | None = Field(default=None, max_length=128)
    model: str | None = Field(default=None, max_length=200)

    class Config:
        populate_by_name = True
