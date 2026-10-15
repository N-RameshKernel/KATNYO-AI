from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier, StackingRegressor, VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from pipelines.explainability import generate_explainability
from pipelines.model_factory import ModelCandidate, available_model_candidates, suggest_params
from pipelines.preprocessing import build_preprocessor, prepare_frame
from services.profiler import DatasetProfiler
from services.state_store import StateStore
from services.tracking import ExperimentTracker
from utils.config import settings
from utils.io import make_json_safe, slugify, utc_now


class AutoMLPipeline:
    def __init__(self, state_store: StateStore, artifact_dir: Path | None = None, workspace_id: str = "local-demo") -> None:
        self.state_store = state_store
        self.artifact_dir = artifact_dir or settings.artifact_dir
        self.workspace_id = workspace_id
        self.profiler = DatasetProfiler()

    def train(
        self,
        dataset: dict,
        target: str | None = None,
        max_rows: int | None = None,
        optuna_trials: int | None = None,
        include_models: list[str] | None = None,
        progress_callback=None,
    ) -> dict:
        run_id = f"run_{uuid4().hex[:12]}"
        artifact_dir = self.artifact_dir / run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        row_limit = max_rows or settings.max_train_rows
        profile = self.profiler.profile_dataset(dataset["path"], target=target, max_rows=min(row_limit, settings.max_profile_rows))
        target_column = target or profile["target"]
        problem_type = profile["problemType"]

        logs = [
            f"Loaded profile for {dataset['filename']}.",
            f"Detected {problem_type} target '{target_column}'.",
            f"Reading up to {row_limit:,} rows for adaptive training.",
        ]

        df = pd.read_csv(dataset["path"], nrows=row_limit, low_memory=False)
        X, y, prep_meta = prepare_frame(
            df,
            target_column=target_column,
            problem_type=problem_type,
            date_columns=profile["featureTypes"].get("datetime", []),
        )
        if len(X) < 30:
            raise ValueError("At least 30 usable rows are required for training.")

        X_train, X_valid, y_train, y_valid = self._split(X, y, problem_type)
        candidates, unavailable = available_model_candidates(problem_type, settings.random_state)
        if include_models:
            requested = {name.lower() for name in include_models}
            candidates = [candidate for candidate in candidates if candidate.name.lower() in requested]
        if not candidates:
            raise ValueError("No trainable model candidates are available.")

        experiment_name = settings.mlflow_experiment_name if self.workspace_id == "local-demo" else f"{settings.mlflow_experiment_name}-{self.workspace_id}"
        tracker = ExperimentTracker(experiment_name)
        model_records: list[dict] = []
        fitted_models: dict[str, Pipeline] = {}
        candidate_params: dict[str, dict] = {}

        with tracker.start_run(f"KATNYO-{run_id}"):
            tracker.log_params(
                {
                    "dataset": dataset["filename"],
                    "target": target_column,
                    "problem_type": problem_type,
                    "row_limit": row_limit,
                    "feature_count": X.shape[1],
                }
            )

            for candidate in candidates:
                if progress_callback:
                    progress_callback(min(10 + int(65 * len(model_records) / max(len(candidates), 1)), 85), f"Training {candidate.name}...")
                logs.append(f"Training {candidate.name}...")
                best_params = self._optimize_candidate(
                    candidate,
                    X_train,
                    X_valid,
                    y_train,
                    y_valid,
                    problem_type,
                    optuna_trials,
                    logs,
                )
                pipeline = self._build_pipeline(candidate, best_params, X_train)
                pipeline.fit(X_train, y_train)
                predictions = pipeline.predict(X_valid)
                metrics, primary_metric, primary_value, score, higher_is_better = _metrics(
                    y_valid,
                    predictions,
                    problem_type,
                )
                tracker.log_metrics({f"{slugify(candidate.name)}_{key}": value for key, value in metrics.items()})
                logs.append(_metric_log(candidate.name, primary_metric, primary_value, higher_is_better))

                record = {
                    "name": candidate.name,
                    "type": "model",
                    "metrics": metrics,
                    "primaryMetric": primary_metric,
                    "primaryValue": primary_value,
                    "score": score,
                    "higherIsBetter": higher_is_better,
                    "hyperparameters": best_params,
                }
                model_records.append(make_json_safe(record))
                fitted_models[candidate.name] = pipeline
                candidate_params[candidate.name] = best_params

            ensemble_records, ensemble_models = self._train_ensembles(
                candidates=candidates,
                candidate_params=candidate_params,
                fitted_models=fitted_models,
                X_train=X_train,
                X_valid=X_valid,
                y_train=y_train,
                y_valid=y_valid,
                problem_type=problem_type,
                logs=logs,
            )
            if progress_callback:
                progress_callback(90, "Evaluating ensembles and generating explanations...")
            all_records = model_records + ensemble_records
            best_record = max(all_records, key=lambda record: record["score"])
            best_pipeline = ensemble_models.get(best_record["name"]) or fitted_models.get(best_record["name"])
            if best_pipeline is None:
                best_pipeline = fitted_models[model_records[0]["name"]]

            model_path = artifact_dir / "model.joblib"
            joblib.dump(best_pipeline, model_path)
            models_dir = artifact_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            model_artifacts = {}
            for record in all_records:
                model = ensemble_models.get(record["name"]) or fitted_models.get(record["name"])
                if model is None:
                    continue
                candidate_path = models_dir / f"{slugify(record['name'])}.joblib"
                if candidate_path != model_path:
                    joblib.dump(model, candidate_path)
                model_artifacts[record["name"]] = f"/artifacts/{run_id}/models/{candidate_path.name}"
            tracker.log_artifact(str(model_path))
            tracker.log_model(best_pipeline)

            explainability = generate_explainability(
                best_pipeline,
                X_valid,
                y_valid,
                artifact_dir,
                problem_type,
                target_column,
            )
            if explainability.get("featureImportanceChart"):
                tracker.log_artifact(str(artifact_dir / "feature_importance.png"))
            if explainability.get("shapChart"):
                tracker.log_artifact(str(artifact_dir / "shap_summary.png"))

        run = {
            "runId": run_id,
            "datasetId": dataset["id"],
            "createdAt": utc_now(),
            "target": target_column,
            "problemType": problem_type,
            "profile": profile,
            "sampleRows": int(len(X)),
            "featureCount": int(X.shape[1]),
            "preprocessing": prep_meta,
            "models": model_records,
            "ensembles": ensemble_records,
            "unavailableModels": unavailable,
            "bestModel": best_record,
            "explainability": explainability,
            "logs": logs,
            "artifacts": {
                "model": f"/artifacts/{run_id}/model.joblib",
                "models": model_artifacts,
                "directory": f"/artifacts/{run_id}",
            },
        "tracking": {
            "mlflowAvailable": tracker.available,
            "trackingUri": settings.mlflow_tracking_uri if tracker.available else None,
            "wandbEnabled": any(provider == "wandb" for provider, _ in getattr(tracker, "external_runs", [])) or bool(settings.wandb_api_key and not tracker.external_reasons.get("wandb")),
            "neptuneEnabled": any(provider == "neptune" for provider, _ in getattr(tracker, "external_runs", [])) or bool(settings.neptune_api_token and settings.neptune_project and not tracker.external_reasons.get("neptune")),
            "externalErrors": getattr(tracker, "external_reasons", {}),
            "message": "Configured tracking providers were initialized when available." if tracker.available else getattr(tracker, "reason", "MLflow unavailable."),
        },
            "feedbackTimeline": self.state_store.feedback_timeline(dataset["id"]),
        }
        return make_json_safe(run)

    def _optimize_candidate(
        self,
        candidate: ModelCandidate,
        X_train,
        X_valid,
        y_train,
        y_valid,
        problem_type: str,
        optuna_trials: int | None,
        logs: list[str],
    ) -> dict:
        trials = settings.optuna_trials if optuna_trials is None else optuna_trials
        if trials <= 0:
            return candidate.default_params

        try:
            import optuna
        except Exception as exc:
            logs.append(f"Optuna unavailable for {candidate.name}: {exc}. Using defaults.")
            return candidate.default_params

        def objective(trial):
            params = {**candidate.default_params, **suggest_params(trial, candidate.name, problem_type)}
            pipeline = self._build_pipeline(candidate, params, X_train)
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_valid)
            return _metrics(y_valid, predictions, problem_type)[3]

        logs.append(f"Optimizing {candidate.name} hyperparameters with Optuna...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=trials, show_progress_bar=False)
        logs.append(f"{candidate.name} optimization best score: {study.best_value:.4f}.")
        return {**candidate.default_params, **study.best_params}

    def _build_pipeline(self, candidate: ModelCandidate, params: dict, X_train: pd.DataFrame) -> Pipeline:
        preprocessor, _ = build_preprocessor(X_train)
        estimator = candidate.estimator_factory(params)
        return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])

    def _split(self, X: pd.DataFrame, y: pd.Series, problem_type: str):
        if problem_type == "time-series":
            cutoff = max(int(len(X) * 0.8), 1)
            return X.iloc[:cutoff], X.iloc[cutoff:], y.iloc[:cutoff], y.iloc[cutoff:]

        stratify = None
        if problem_type == "classification":
            class_counts = y.value_counts()
            if len(class_counts) > 1 and class_counts.min() >= 2:
                stratify = y
        return train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=settings.random_state,
            stratify=stratify,
        )

    def _train_ensembles(
        self,
        candidates: list[ModelCandidate],
        candidate_params: dict[str, dict],
        fitted_models: dict[str, Pipeline],
        X_train,
        X_valid,
        y_train,
        y_valid,
        problem_type: str,
        logs: list[str],
    ) -> tuple[list[dict], dict[str, Pipeline]]:
        top_candidates = sorted(
            candidates,
            key=lambda candidate: fitted_models[candidate.name].score(X_valid, y_valid),
            reverse=True,
        )[:3]
        if len(top_candidates) < 2:
            return [], {}

        is_classification = problem_type == "classification"
        ensemble_records: list[dict] = []
        ensemble_models: dict[str, Pipeline] = {}
        estimator_tuples = [
            (
                slugify(candidate.name),
                self._build_pipeline(candidate, candidate_params[candidate.name], X_train),
            )
            for candidate in top_candidates
        ]

        ensemble_specs = []
        if is_classification:
            ensemble_specs.extend(
                [
                    (
                        "Voting Ensemble",
                        VotingClassifier(estimators=estimator_tuples, voting="soft"),
                    ),
                    (
                        "Stacking Ensemble",
                        StackingClassifier(
                            estimators=estimator_tuples,
                            final_estimator=LogisticRegression(max_iter=1000),
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
        else:
            ensemble_specs.extend(
                [
                    ("Voting Ensemble", VotingRegressor(estimators=estimator_tuples, n_jobs=-1)),
                    (
                        "Stacking Ensemble",
                        StackingRegressor(
                            estimators=estimator_tuples,
                            final_estimator=Ridge(),
                            n_jobs=-1,
                        ),
                    ),
                ]
            )

        for name, ensemble in ensemble_specs:
            try:
                logs.append(f"Training {name}...")
                ensemble.fit(X_train, y_train)
                predictions = ensemble.predict(X_valid)
                metrics, primary_metric, primary_value, score, higher_is_better = _metrics(y_valid, predictions, problem_type)
                record = {
                    "name": name,
                    "type": "ensemble",
                    "metrics": metrics,
                    "primaryMetric": primary_metric,
                    "primaryValue": primary_value,
                    "score": score,
                    "higherIsBetter": higher_is_better,
                    "hyperparameters": {"members": [candidate.name for candidate in top_candidates]},
                }
                ensemble_records.append(make_json_safe(record))
                ensemble_models[name] = ensemble
                logs.append(_metric_log(name, primary_metric, primary_value, higher_is_better))
            except Exception as exc:
                logs.append(f"{name} skipped: {exc}")

        blend_record, blend_model = self._blend(top_candidates, fitted_models, X_valid, y_valid, problem_type, logs)
        if blend_record:
            ensemble_records.append(blend_record)
            ensemble_models[blend_record["name"]] = blend_model
        return ensemble_records, ensemble_models

    def _blend(self, top_candidates, fitted_models, X_valid, y_valid, problem_type: str, logs: list[str]) -> tuple[dict | None, object | None]:
        try:
            logs.append("Blending top model predictions...")
            top_models = [fitted_models[candidate.name] for candidate in top_candidates]
            blend_model = PredictionBlend(top_models, problem_type)
            predictions = blend_model.predict(X_valid)
            metrics, primary_metric, primary_value, score, higher_is_better = _metrics(y_valid, predictions, problem_type)
            logs.append(_metric_log("Blending Ensemble", primary_metric, primary_value, higher_is_better))
            return make_json_safe(
                {
                    "name": "Blending Ensemble",
                    "type": "ensemble",
                    "metrics": metrics,
                    "primaryMetric": primary_metric,
                    "primaryValue": primary_value,
                    "score": score,
                    "higherIsBetter": higher_is_better,
                    "hyperparameters": {"members": [candidate.name for candidate in top_candidates]},
                }
            ), blend_model
        except Exception as exc:
            logs.append(f"Blending skipped: {exc}")
            return None, None


class PredictionBlend:
    def __init__(self, models: list[Pipeline], problem_type: str) -> None:
        self.models = models
        self.problem_type = problem_type
        self.classes_ = getattr(models[0], "classes_", None) if problem_type == "classification" else None

    def predict_proba(self, X):
        probabilities = [model.predict_proba(X) for model in self.models if hasattr(model, "predict_proba")]
        if not probabilities:
            raise AttributeError("No blended classifier exposes predict_proba.")
        return np.mean(probabilities, axis=0)

    def predict(self, X):
        if self.problem_type == "classification":
            if all(hasattr(model, "predict_proba") for model in self.models):
                probabilities = self.predict_proba(X)
                classes = self.classes_ if self.classes_ is not None else np.arange(probabilities.shape[1])
                return classes[np.argmax(probabilities, axis=1)]
            predictions = np.asarray([model.predict(X) for model in self.models]).T
            return np.asarray([max(dict.fromkeys(row, 0), key=list(row).count) for row in predictions])
        return np.mean([model.predict(X) for model in self.models], axis=0)


def _metrics(y_true, predictions, problem_type: str) -> tuple[dict, str, float, float, bool]:
    if problem_type == "classification":
        metrics = {
            "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
            "precision": round(float(precision_score(y_true, predictions, average="weighted", zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, predictions, average="weighted", zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, predictions, average="weighted", zero_division=0)), 4),
        }
        return metrics, "accuracy", metrics["accuracy"], metrics["accuracy"], True

    rmse = float(np.sqrt(mean_squared_error(y_true, predictions)))
    metrics = {
        "rmse": round(rmse, 4),
        "mae": round(float(mean_absolute_error(y_true, predictions)), 4),
        "r2": round(float(r2_score(y_true, predictions)), 4),
    }
    if problem_type == "time-series":
        return metrics, "rmse", metrics["rmse"], -metrics["rmse"], False
    return metrics, "r2", metrics["r2"], metrics["r2"], True


def _metric_log(model_name: str, metric_name: str, value: float, higher_is_better: bool) -> str:
    direction = "improved to" if higher_is_better else "optimized at"
    return f"{model_name} {metric_name} {direction} {value}."
