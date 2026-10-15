from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import settings


TARGET_CANDIDATES = (
    "target",
    "label",
    "y",
    "sales",
    "demand",
    "value",
    "price",
    "revenue",
    "churn",
    "class",
)

DATE_HINTS = ("date", "timestamp", "time", "day", "ds")


class DatasetProfiler:
    def profile_dataset(
        self,
        path: str,
        target: str | None = None,
        max_rows: int | None = None,
    ) -> dict:
        sample_limit = max_rows or settings.max_profile_rows
        df = pd.read_csv(path, nrows=sample_limit, low_memory=False)
        df = _drop_index_columns(df)
        if df.empty:
            raise ValueError("The CSV is empty.")

        target_column = target or self._guess_target(df)
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' was not found in the CSV.")

        date_columns = self._detect_date_columns(df)
        feature_columns = [column for column in df.columns if column != target_column]
        numeric_columns = df[feature_columns].select_dtypes(include=np.number).columns.tolist()
        categorical_columns = [
            column
            for column in feature_columns
            if column not in numeric_columns and column not in date_columns
        ]
        problem_type = self._detect_problem_type(df, target_column, date_columns)

        missing_cells = int(df.isna().sum().sum())
        total_cells = int(df.shape[0] * df.shape[1])
        missing_percent = round((missing_cells / total_cells) * 100, 2) if total_cells else 0

        rows, estimated = _row_count(path, sample_limit)
        return {
            "filename": Path(path).name,
            "rows": rows,
            "rowCountEstimated": estimated,
            "sampleRows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "target": target_column,
            "problemType": problem_type,
            "missingCells": missing_cells,
            "missingPercent": missing_percent,
            "featureTypes": {
                "numeric": numeric_columns,
                "categorical": categorical_columns,
                "datetime": date_columns,
            },
            "targetSummary": self._target_summary(df[target_column]),
            "message": (
                f"Detected {problem_type} problem with {missing_percent}% missing values "
                f"and target '{target_column}'."
            ),
        }

    def _guess_target(self, df: pd.DataFrame) -> str:
        lower_to_column = {str(column).lower(): column for column in df.columns}
        for candidate in TARGET_CANDIDATES:
            if candidate in lower_to_column:
                return lower_to_column[candidate]

        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_columns:
            return numeric_columns[-1]
        return df.columns[-1]

    def _detect_date_columns(self, df: pd.DataFrame) -> list[str]:
        detected: list[str] = []
        for column in df.columns:
            column_name = str(column).lower()
            looks_like_date = any(hint in column_name for hint in DATE_HINTS)
            if not looks_like_date and df[column].dtype != object:
                continue
            if df[column].nunique(dropna=True) < 3:
                continue
            parsed = pd.to_datetime(df[column].dropna().head(200), errors="coerce")
            if len(parsed) and parsed.notna().mean() >= 0.85:
                detected.append(column)
        return detected

    def _detect_problem_type(
        self,
        df: pd.DataFrame,
        target_column: str,
        date_columns: list[str],
    ) -> str:
        target = df[target_column].dropna()
        target_is_numeric = pd.api.types.is_numeric_dtype(target)
        target_name = str(target_column).lower()
        has_temporal_signal = bool(date_columns) and target_is_numeric
        if has_temporal_signal and target_name in {"sales", "demand", "value", "revenue"}:
            return "time-series"

        unique_values = target.nunique(dropna=True)
        unique_ratio = unique_values / max(len(target), 1)
        if not target_is_numeric or unique_values <= 20 or unique_ratio <= 0.05:
            return "classification"
        return "regression"

    def _target_summary(self, target: pd.Series) -> dict:
        if pd.api.types.is_numeric_dtype(target):
            return {
                "kind": "numeric",
                "min": _safe_float(target.min()),
                "max": _safe_float(target.max()),
                "mean": _safe_float(target.mean()),
                "unique": int(target.nunique(dropna=True)),
            }
        return {
            "kind": "categorical",
            "unique": int(target.nunique(dropna=True)),
            "topValues": target.astype(str).value_counts().head(5).to_dict(),
        }


def _drop_index_columns(df: pd.DataFrame) -> pd.DataFrame:
    index_columns = [
        column
        for column in df.columns
        if str(column).startswith("Unnamed") or str(column).strip() == ""
    ]
    return df.drop(columns=index_columns, errors="ignore")


def _row_count(path: str, sample_limit: int) -> tuple[int, bool]:
    file_size_mb = Path(path).stat().st_size / (1024 * 1024)
    if file_size_mb > settings.exact_row_count_limit_mb:
        return sample_limit, True

    with open(path, "rb") as handle:
        rows = max(sum(1 for _ in handle) - 1, 0)
    return rows, False


def _safe_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)
