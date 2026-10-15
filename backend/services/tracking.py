from contextlib import ExitStack, contextmanager
from pathlib import Path

from utils.config import settings


class ExperimentTracker:
    def __init__(self, experiment_name: str | None = None) -> None:
        self.wandb = None
        self.neptune = None
        self.external_runs = []
        self.external_reasons = {}
        try:
            import mlflow
            import mlflow.sklearn

            self.mlflow = mlflow
            self.mlflow_sklearn = mlflow.sklearn
            self.available = True
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(experiment_name or settings.mlflow_experiment_name)
        except Exception as exc:
            self.mlflow = None
            self.mlflow_sklearn = None
            self.available = False
            self.reason = str(exc)

        if settings.wandb_api_key:
            try:
                import wandb

                self.wandb = wandb
            except Exception as exc:
                self.external_reasons["wandb"] = str(exc)
        if settings.neptune_api_token and settings.neptune_project:
            try:
                import neptune

                self.neptune = neptune
            except Exception as exc:
                self.external_reasons["neptune"] = str(exc)

    @contextmanager
    def start_run(self, run_name: str):
        external = []
        with ExitStack() as stack:
            run = stack.enter_context(self.mlflow.start_run(run_name=run_name)) if self.available else None
            if self.wandb:
                try:
                    external.append(("wandb", self.wandb.init(project=settings.wandb_project, name=run_name, reinit=True)))
                except Exception as exc:
                    self.external_reasons["wandb"] = str(exc)
            if self.neptune:
                try:
                    external.append(("neptune", self.neptune.init_run(project=settings.neptune_project, api_token=settings.neptune_api_token, name=run_name)))
                except Exception as exc:
                    self.external_reasons["neptune"] = str(exc)
            self.external_runs = external
            try:
                yield run
            finally:
                for provider, active_run in external:
                    try:
                        if provider == "wandb":
                            active_run.finish()
                        else:
                            active_run.stop()
                    except Exception as exc:
                        self.external_reasons[provider] = str(exc)
                self.external_runs = []

    def log_params(self, params: dict) -> None:
        if self.available:
            self.mlflow.log_params(params)
        for provider, run in self.external_runs:
            try:
                if provider == "wandb":
                    run.config.update(params, allow_val_change=True)
                else:
                    for key, value in params.items():
                        run[f"parameters/{key}"] = value
            except Exception as exc:
                self.external_reasons[provider] = str(exc)

    def log_metrics(self, metrics: dict) -> None:
        if self.available:
            safe_metrics = {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float)) and value == value
            }
            if safe_metrics:
                self.mlflow.log_metrics(safe_metrics)
        for provider, run in self.external_runs:
            try:
                if provider == "wandb":
                    run.log(metrics)
                else:
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)) and value == value:
                            run[f"metrics/{key}"].append(float(value))
            except Exception as exc:
                self.external_reasons[provider] = str(exc)

    def log_artifact(self, path: str) -> None:
        if self.available and path:
            self.mlflow.log_artifact(path)
        for provider, run in self.external_runs:
            try:
                if provider == "wandb":
                    artifact = self.wandb.Artifact(name=Path(path).stem, type="model" if path.endswith(".joblib") else "artifact")
                    artifact.add_file(path)
                    run.log_artifact(artifact)
                else:
                    run[f"artifacts/{Path(path).name}"].upload(path)
            except Exception as exc:
                self.external_reasons[provider] = str(exc)

    def log_model(self, model, artifact_path: str = "model") -> None:
        if self.available:
            self.mlflow_sklearn.log_model(model, artifact_path=artifact_path)
