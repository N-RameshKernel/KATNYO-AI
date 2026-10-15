from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge


@dataclass
class ModelCandidate:
    name: str
    estimator_factory: Callable[[dict], object]
    default_params: dict
    optional: bool = False


def available_model_candidates(problem_type: str, random_state: int) -> tuple[list[ModelCandidate], list[dict]]:
    is_classification = problem_type == "classification"
    unavailable: list[dict] = []
    candidates: list[ModelCandidate] = [
        _random_forest(is_classification, random_state),
    ]

    if is_classification:
        candidates.append(
            ModelCandidate(
                name="Logistic Regression",
                estimator_factory=lambda params: LogisticRegression(
                    max_iter=1500,
                    n_jobs=-1,
                    **params,
                ),
                default_params={"C": 1.0},
            )
        )
    else:
        candidates.append(
            ModelCandidate(
                name="Ridge Regression",
                estimator_factory=lambda params: Ridge(**params),
                default_params={"alpha": 1.0},
            )
        )

    xgb_candidate, xgb_error = _xgboost(is_classification, random_state)
    if xgb_candidate:
        candidates.append(xgb_candidate)
    else:
        unavailable.append({"name": "XGBoost", "reason": xgb_error})

    lgbm_candidate, lgbm_error = _lightgbm(is_classification, random_state)
    if lgbm_candidate:
        candidates.append(lgbm_candidate)
    else:
        unavailable.append({"name": "LightGBM", "reason": lgbm_error})

    return candidates, unavailable


def suggest_params(trial, model_name: str, problem_type: str) -> dict:
    if model_name == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 80, 260),
            "max_depth": trial.suggest_int("max_depth", 3, 22),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        }
    if model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 80, 260),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        }
    if model_name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 80, 280),
            "num_leaves": trial.suggest_int("num_leaves", 16, 96),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        }
    if model_name == "Logistic Regression":
        return {"C": trial.suggest_float("C", 0.01, 10.0, log=True)}
    if model_name == "Ridge Regression":
        return {"alpha": trial.suggest_float("alpha", 0.01, 20.0, log=True)}
    return {}


def _random_forest(is_classification: bool, random_state: int) -> ModelCandidate:
    if is_classification:
        return ModelCandidate(
            name="RandomForest",
            estimator_factory=lambda params: RandomForestClassifier(
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced",
                **params,
            ),
            default_params={"n_estimators": 160, "max_depth": 12},
        )
    return ModelCandidate(
        name="RandomForest",
        estimator_factory=lambda params: RandomForestRegressor(
            random_state=random_state,
            n_jobs=-1,
            **params,
        ),
        default_params={"n_estimators": 160, "max_depth": 14},
    )


def _xgboost(is_classification: bool, random_state: int) -> tuple[ModelCandidate | None, str | None]:
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except Exception as exc:
        return None, str(exc)

    if is_classification:
        return (
            ModelCandidate(
                name="XGBoost",
                estimator_factory=lambda params: XGBClassifier(
                    random_state=random_state,
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=-1,
                    **params,
                ),
                default_params={"n_estimators": 180, "max_depth": 4, "learning_rate": 0.08},
                optional=True,
            ),
            None,
        )
    return (
        ModelCandidate(
            name="XGBoost",
            estimator_factory=lambda params: XGBRegressor(
                random_state=random_state,
                objective="reg:squarederror",
                tree_method="hist",
                n_jobs=-1,
                **params,
            ),
            default_params={"n_estimators": 180, "max_depth": 4, "learning_rate": 0.08},
            optional=True,
        ),
        None,
    )


def _lightgbm(is_classification: bool, random_state: int) -> tuple[ModelCandidate | None, str | None]:
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except Exception as exc:
        return None, str(exc)

    if is_classification:
        return (
            ModelCandidate(
                name="LightGBM",
                estimator_factory=lambda params: LGBMClassifier(
                    random_state=random_state,
                    n_jobs=-1,
                    verbose=-1,
                    **params,
                ),
                default_params={"n_estimators": 180, "num_leaves": 48, "learning_rate": 0.08},
                optional=True,
            ),
            None,
        )
    return (
        ModelCandidate(
            name="LightGBM",
            estimator_factory=lambda params: LGBMRegressor(
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
                **params,
            ),
            default_params={"n_estimators": 180, "num_leaves": 48, "learning_rate": 0.08},
            optional=True,
        ),
        None,
    )
