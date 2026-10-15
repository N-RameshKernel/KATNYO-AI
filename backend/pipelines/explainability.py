from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipelines.preprocessing import get_feature_names


def generate_explainability(
    pipeline,
    X_reference: pd.DataFrame,
    y_reference: pd.Series,
    artifact_dir: Path,
    problem_type: str,
    target_column: str,
) -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(X_reference.head(600))
    feature_names = get_feature_names(preprocessor)

    importances, method = _feature_importances(model, transformed, feature_names)
    if not importances:
        importances, method = _permutation_importances(pipeline, X_reference.head(300), y_reference.head(300), problem_type)

    importances = sorted(importances, key=lambda item: item["importance"], reverse=True)[:20]
    feature_chart_path = artifact_dir / "feature_importance.png"
    _plot_feature_importance(importances[:12], feature_chart_path)

    shap_path, shap_status = _try_shap_summary(model, transformed, feature_names, artifact_dir)
    top_three = [item["feature"] for item in importances[:3]]

    return {
        "method": method,
        "featureImportance": importances,
        "topFeatures": top_three,
        "featureImportanceChart": f"/artifacts/{artifact_dir.name}/feature_importance.png",
        "shapChart": f"/artifacts/{artifact_dir.name}/shap_summary.png" if shap_path else None,
        "shapStatus": shap_status,
        "narrative": _narrative(top_three, target_column),
    }


def _feature_importances(model, transformed, feature_names: list[str]) -> tuple[list[dict], str]:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        return _importance_records(values, feature_names), "model_feature_importance"
    if hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float))
        if values.ndim > 1:
            values = values.mean(axis=0)
        return _importance_records(values, feature_names), "linear_coefficients"
    return [], "unavailable"


def _permutation_importances(pipeline, X_sample, y_sample, problem_type: str) -> tuple[list[dict], str]:
    try:
        from sklearn.inspection import permutation_importance

        scoring = "accuracy" if problem_type == "classification" else "r2"
        result = permutation_importance(
            pipeline,
            X_sample,
            y_sample,
            n_repeats=5,
            random_state=42,
            scoring=scoring,
        )
        return _importance_records(result.importances_mean, X_sample.columns.tolist()), "permutation_importance"
    except Exception:
        return [], "unavailable"


def _importance_records(values, feature_names: list[str]) -> list[dict]:
    values = np.nan_to_num(values, nan=0.0)
    if len(values) != len(feature_names):
        feature_names = [f"feature_{index}" for index in range(len(values))]
    total = float(np.sum(np.abs(values))) or 1.0
    records = []
    for feature, value in zip(feature_names, values):
        records.append(
            {
                "feature": str(feature),
                "importance": round(float(abs(value)), 6),
                "share": round(float(abs(value)) / total, 4),
            }
        )
    return records


def _try_shap_summary(model, transformed, feature_names: list[str], artifact_dir: Path) -> tuple[Path | None, str]:
    try:
        import shap

        sample = transformed[: min(180, transformed.shape[0])]
        explainer = shap.Explainer(model, sample, feature_names=feature_names)
        shap_values = explainer(sample)
        plt.figure(figsize=(9, 5))
        shap.plots.beeswarm(shap_values, show=False, max_display=12)
        shap_path = artifact_dir / "shap_summary.png"
        plt.tight_layout()
        plt.savefig(shap_path, dpi=160, transparent=True)
        plt.close()
        return shap_path, "generated"
    except Exception as exc:
        return None, f"SHAP visualization unavailable: {exc}"


def _plot_feature_importance(importances: list[dict], destination: Path) -> None:
    if not importances:
        importances = [{"feature": "No importances available", "importance": 0.0}]
    features = [item["feature"] for item in importances][::-1]
    values = [item["importance"] for item in importances][::-1]

    plt.figure(figsize=(9, 5))
    ax = plt.gca()
    ax.set_facecolor("#0B0F1A")
    plt.gcf().patch.set_facecolor("#0B0F1A")
    ax.barh(features, values, color="#38BDF8")
    ax.tick_params(colors="#E5EDF8", labelsize=8)
    ax.set_xlabel("Influence", color="#E5EDF8")
    ax.set_title("Feature Importance", color="#E5EDF8", pad=12)
    for spine in ax.spines.values():
        spine.set_color("#1E293B")
    plt.tight_layout()
    plt.savefig(destination, dpi=160, transparent=True)
    plt.close()


def _narrative(top_features: list[str], target_column: str) -> str:
    if not top_features:
        return "The model did not expose stable feature influence scores for this run."
    if len(top_features) == 1:
        return f"The strongest signal for predicting {target_column} is {top_features[0]}."
    if len(top_features) == 2:
        features = " and ".join(top_features)
    else:
        features = ", ".join(top_features[:-1]) + f", and {top_features[-1]}"
    return f"Top factors influencing prediction for {target_column}: {features}."
