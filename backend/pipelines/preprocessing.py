from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

try:
    from sklearn.preprocessing import OneHotEncoder
except Exception:  # pragma: no cover
    OneHotEncoder = None


def prepare_frame(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    date_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    df = _drop_index_columns(df).copy()
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    df = df.dropna(subset=[target_column])
    date_columns = date_columns or _detect_date_columns(df)

    if problem_type == "time-series" and date_columns:
        df = df.sort_values(date_columns[0])

    y = df[target_column]
    X = df.drop(columns=[target_column])
    engineered_dates: list[str] = []

    for column in date_columns:
        if column not in X.columns:
            continue
        parsed = pd.to_datetime(X[column], errors="coerce")
        X[f"{column}_year"] = parsed.dt.year
        X[f"{column}_month"] = parsed.dt.month
        X[f"{column}_day"] = parsed.dt.day
        X[f"{column}_dayofweek"] = parsed.dt.dayofweek
        X = X.drop(columns=[column])
        engineered_dates.append(column)

    for column in X.select_dtypes(include=["bool"]).columns:
        X[column] = X[column].astype(int)

    if problem_type == "classification" and not pd.api.types.is_numeric_dtype(y):
        y = y.astype("category").cat.codes

    return X, y, {"dateColumns": engineered_dates}


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, dict]:
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    low_cardinality = [
        column
        for column in categorical_features
        if X[column].nunique(dropna=True) <= 40
    ]
    high_cardinality = [
        column
        for column in categorical_features
        if column not in low_cardinality
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    low_card_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _one_hot_encoder()),
        ]
    )
    high_card_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "label",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if low_cardinality:
        transformers.append(("onehot", low_card_pipeline, low_cardinality))
    if high_cardinality:
        transformers.append(("label", high_card_pipeline, high_cardinality))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    feature_groups = {
        "numeric": numeric_features,
        "oneHotEncoded": low_cardinality,
        "labelEncoded": high_cardinality,
    }
    return preprocessor, feature_groups


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return [clean_feature_name(name) for name in preprocessor.get_feature_names_out()]
    except Exception:
        names: list[str] = []
        for name, _, columns in preprocessor.transformers_:
            if name == "remainder":
                continue
            names.extend([clean_feature_name(f"{name}__{column}") for column in columns])
        return names


def clean_feature_name(name: str) -> str:
    cleaned = name.split("__", maxsplit=1)[-1]
    return cleaned.replace("_", " ").strip()


def _one_hot_encoder():
    if OneHotEncoder is None:
        raise RuntimeError("scikit-learn OneHotEncoder is unavailable.")
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _detect_date_columns(df: pd.DataFrame) -> list[str]:
    detected: list[str] = []
    for column in df.columns:
        name = str(column).lower()
        if not any(token in name for token in ("date", "time", "timestamp", "ds")):
            continue
        parsed = pd.to_datetime(df[column].dropna().head(200), errors="coerce")
        if len(parsed) and parsed.notna().mean() >= 0.85:
            detected.append(column)
    return detected


def _drop_index_columns(df: pd.DataFrame) -> pd.DataFrame:
    index_columns = [
        column
        for column in df.columns
        if str(column).startswith("Unnamed") or str(column).strip() == ""
    ]
    return df.drop(columns=index_columns, errors="ignore")
